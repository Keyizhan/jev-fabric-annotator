import json
import threading
import time
from contextlib import contextmanager
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from jevfabric.client import JevClient


@contextmanager
def server(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        assert self.path == "/v1/systemone"
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"answers": {"echo": payload["state"]}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_real_http_roundtrip_and_single_owner_contract():
    with server(Handler) as port:
        client = JevClient(port=port)
        try:
            assert client.ask("hello", {}, [Image.new("RGB", (10, 10))]) == {
                "echo": "hello"
            }
            errors = []

            def wrong_thread():
                try:
                    client.ask("again", {})
                except RuntimeError as exc:
                    errors.append(exc)

            thread = threading.Thread(target=wrong_thread)
            thread.start()
            thread.join(1)
            assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
        finally:
            client.close()


def test_cancel_interrupts_http10_body_read_and_owner_cleans_up():
    entered, release = threading.Event(), threading.Event()

    class Slow(Handler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "10000")
            self.end_headers()
            self.wfile.write(b"{")
            self.wfile.flush()
            entered.set()
            release.wait(2)

    with server(Slow) as port:
        client = JevClient(port=port, timeout_s=5)
        errors = []

        def request():
            try:
                client.ask("test", {})
            except (HTTPException, OSError, ValueError) as exc:
                errors.append(exc)
            finally:
                client.close()

        thread = threading.Thread(target=request)
        thread.start()
        try:
            assert entered.wait(1)
            start = time.monotonic()
            client.cancel()
            thread.join(0.5)
            assert not thread.is_alive()
            assert time.monotonic() - start < 0.5
            assert errors
        finally:
            release.set()
            thread.join(2)


def test_response_size_is_bounded():
    with server(Handler) as port:
        client = JevClient(port=port, max_response_bytes=2)
        try:
            with pytest.raises(ValueError, match="size limit"):
                client.ask("hello", {})
        finally:
            client.close()
