"""Launch the external JEV server with persistent HTTP and TCP_NODELAY.

Run this CLI in the model's own Python environment, in a separate process.
The model package, weights and inference implementation remain external.
"""

from __future__ import annotations

import argparse
import http.server
import runpy
import socket
import sys
from contextlib import contextmanager
from pathlib import Path


class _NoDelayHTTPServer(http.server.ThreadingHTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection, address


@contextmanager
def _persistent_http():
    """Scope transport overrides to the external server's process lifetime."""
    original_server = http.server.ThreadingHTTPServer
    original_protocol = http.server.BaseHTTPRequestHandler.protocol_version
    http.server.ThreadingHTTPServer = _NoDelayHTTPServer
    http.server.BaseHTTPRequestHandler.protocol_version = "HTTP/1.1"
    try:
        yield
    finally:
        http.server.ThreadingHTTPServer = original_server
        http.server.BaseHTTPRequestHandler.protocol_version = original_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--device", default="cuda")
    args, native_options = parser.parse_known_args(argv)
    model = args.model.expanduser().resolve()
    if not (model / "startlux_decision/server.py").is_file():
        parser.error(
            "model must contain the external startlux_decision/server.py package"
        )
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    original_argv, original_path = sys.argv[:], sys.path[:]
    sys.path.insert(0, str(model))
    sys.argv = [
        "startlux_decision.server",
        "--model",
        str(model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device",
        args.device,
        *native_options,
    ]
    try:
        with _persistent_http():
            runpy.run_module("startlux_decision.server", run_name="__main__")
    finally:
        sys.argv, sys.path = original_argv, original_path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
