"""Minimal client for a jev (typed-decision) server: POST /v1/systemone with a state, typed questions and images."""
import base64, http.client, io, json


def to_data_uri(img, quality=85):
    """PIL image -> JPEG data URI."""
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class JevClient:
    """Single keep-alive connection. The server answers one request at a time, so do not share one client between threads."""

    def __init__(self, host="127.0.0.1", port=8091):
        self.conn = http.client.HTTPConnection(host, port)

    def ask(self, state, questions, images=(), quality=85):
        body = json.dumps({"state": state, "questions": questions, "images": [to_data_uri(i, quality) for i in images]})
        self.conn.request("POST", "/v1/systemone", body, {"Content-Type": "application/json"})
        return json.loads(self.conn.getresponse().read())["answers"]
