import http.client
import http.server
import json
import socket
import sys
import threading

import pytest

from jevfabric.serve import _persistent_http, main


def test_persistent_transport_reuses_connection_and_enables_nodelay():
    original_server = http.server.ThreadingHTTPServer
    original_protocol = http.server.BaseHTTPRequestHandler.protocol_version
    with _persistent_http():

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                payload = json.dumps(
                    {
                        "port": self.client_address[1],
                        "nodelay": self.connection.getsockopt(
                            socket.IPPROTO_TCP, socket.TCP_NODELAY
                        ),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}
        )
        thread.start()
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        replies = []
        try:
            for _ in range(2):
                client.request("POST", "/", b"{}")
                response = client.getresponse()
                assert response.version == 11
                replies.append(json.loads(response.read()))
                response.close()
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(2)
        assert not thread.is_alive()
        assert replies[0]["port"] == replies[1]["port"]
        assert all(reply["nodelay"] == 1 for reply in replies)
    assert http.server.ThreadingHTTPServer is original_server
    assert http.server.BaseHTTPRequestHandler.protocol_version == original_protocol


def test_native_server_failure_restores_interpreter_and_transport(
    tmp_path, monkeypatch
):
    package = tmp_path / "startlux_decision"
    package.mkdir()
    (package / "server.py").write_text("")
    original_argv, original_path = sys.argv[:], sys.path[:]
    original_server = http.server.ThreadingHTTPServer

    def native_server(module, run_name):
        assert module == "startlux_decision.server"
        assert run_name == "__main__"
        assert sys.path[0] == str(tmp_path)
        assert sys.argv[-2:] == ["--max-pixels", "262144"]
        assert http.server.BaseHTTPRequestHandler.protocol_version == "HTTP/1.1"
        raise RuntimeError("simulated model load failure")

    monkeypatch.setattr("jevfabric.serve.runpy.run_module", native_server)
    with pytest.raises(RuntimeError, match="simulated model load failure"):
        main(["--model", str(tmp_path), "--max-pixels", "262144"])
    assert sys.argv == original_argv
    assert sys.path == original_path
    assert http.server.ThreadingHTTPServer is original_server
