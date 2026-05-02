from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from worldsim.phase_0c_verify_http import verify_http


class _Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        if self.headers.get("X-Worldsim-Token") != "secret":
            self.send_response(403)
        else:
            self.send_response(204)
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or "0"))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"accepted:" + body[:20])

    def log_message(self, format, *args):
        return


def test_verify_http_reports_status_and_auth_header():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        result = verify_http(
            url=f"http://{host}:{port}/probe",
            method="HEAD",
            auth_header="X-Worldsim-Token: secret",
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result["outcome"] == "ok"
    assert result["status"] == 204
    assert result["bytes_read"] == 0


def test_verify_http_bounds_post_body_preview():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        result = verify_http(
            url=f"http://{host}:{port}/submit",
            method="POST",
            data="field=value",
            max_bytes=9,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert result["outcome"] == "ok"
    assert result["status"] == 200
    assert result["bytes_read"] == 9
    assert result["body_preview"] == "accepted:"
