"""Minimal client for a jev (typed-decision) server: POST /v1/systemone with a state, typed questions and images."""
import base64, http.client, io, json
import math
import socket
import threading


def to_data_uri(img, quality=85):
    """PIL image -> JPEG data URI."""
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class JevClient:
    """Single keep-alive connection. The server answers one request at a time, so do not share one client between threads."""

    def __init__(self, host="127.0.0.1", port=8091, *, timeout_s=2.0, max_response_bytes=1 << 20):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.host, self.port = host, port
        self.timeout_s, self.max_response_bytes = timeout_s, max_response_bytes
        self.conn = None
        self._active_socket = None
        self._owner = None
        self._lock = threading.Lock()

    def _check_owner(self):
        ident = threading.get_ident()
        with self._lock:
            if self._owner is None:
                self._owner = ident
            if self._owner != ident:
                raise RuntimeError("JevClient.ask/close must use one owning thread")

    def cancel(self):
        """Wake blocked I/O from another thread; only the owner closes/reuses HTTP."""
        with self._lock:
            sock = self._active_socket or (self.conn.sock if self.conn is not None else None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # Already closed by the I/O owner.

    def close(self):
        self._check_owner()
        with self._lock:
            connection, self.conn = self.conn, None
            self._active_socket = None
        if connection is not None:
            connection.close()

    def ask(self, state, questions, images=(), quality=85):
        self._check_owner()
        body = json.dumps({"state": state, "questions": questions, "images": [to_data_uri(i, quality) for i in images]})
        with self._lock:
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
            connection = self.conn
        response = None
        try:
            connection.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
            # HTTP/1.0 getresponse() can detach conn.sock while its response is
            # still reading. Keep the socket available to cancel() until done.
            with self._lock:
                self._active_socket = connection.sock
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"JEV returned HTTP {response.status}")
            payload = response.read(self.max_response_bytes + 1)
            if len(payload) > self.max_response_bytes:
                raise ValueError("JEV response exceeds size limit")
            answers = json.loads(payload)["answers"]
            if not isinstance(answers, dict):
                raise ValueError("JEV answers must be an object")
            return answers
        except Exception:
            self.close()
            raise
        finally:
            if response is not None:
                response.close()
            with self._lock:
                self._active_socket = None
