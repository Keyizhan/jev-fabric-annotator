import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from jevfabric.replay import main


def test_wall_clock_capture_stays_30hz_with_slow_real_http(tmp_path):
    class SlowDecision(BaseHTTPRequestHandler):
        active = maximum = 0

        def log_message(self, *args):
            pass

        def do_POST(self):
            cls = type(self)
            cls.active += 1
            cls.maximum = max(cls.maximum, cls.active)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                time.sleep(0.09)
                answer = (
                    {"support": {"probabilities": {"mat": 0.99}}}
                    if "support" in body["questions"]
                    else {"piece_shape": {"choice": "C", "confidence": 0.9}}
                )
                payload = json.dumps({"answers": answer}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            finally:
                cls.active -= 1

    data = tmp_path / "data"
    (data / "meta").mkdir(parents=True)
    (data / "data/chunk-000").mkdir(parents=True)
    (data / "videos/chunk-000/observation.images.cam_front").mkdir(parents=True)
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (data / "meta/info.json").write_text(json.dumps(info))
    states = np.zeros((120, 14), np.float32)
    states[:, [6, 13]] = 0.05
    states[30:90, 6] = 0
    pq.write_table(
        pa.table({"observation.state": states.tolist()}),
        data / "data/chunk-000/episode_000000.parquet",
    )
    writer = cv2.VideoWriter(
        str(data / "videos/chunk-000/observation.images.cam_front/episode_000000.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (640, 480),
    )
    assert writer.isOpened()
    for _ in range(120):
        writer.write(np.zeros((480, 640, 3), np.uint8))
    writer.release()
    reference = tmp_path / "reference.png"
    Image.new("RGB", (360, 648)).save(reference)
    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowDecision)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}
    )
    thread.start()
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": server.server_address[1],
                "request_interval_s": 0.03,
            }
        )
    )
    out = tmp_path / "report"
    try:
        assert (
            main(
                [
                    "--dataset",
                    str(data),
                    "--episode",
                    "0",
                    "--reference",
                    str(reference),
                    "--config",
                    str(cfg),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
    report = json.loads((out / "report.json").read_text())
    events = [
        json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()
    ]
    assert report["processed_frames"] >= 115
    assert report["capture_interval_p95_ms"] < 45
    assert report["callback_p95_ms"] < 10
    assert report["metrics"]["releases"] >= 1
    assert report["metrics"]["running_workers"] == 0
    assert SlowDecision.maximum == 1
    assert any(event["event"] == "box_on" for event in events)
    assert any(event["event"] == "released" for event in events)
