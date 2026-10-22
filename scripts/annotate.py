#!/usr/bin/env python3
"""Replay episodes as a live system would see them and write, per episode, when each target basket should be boxed.

For every episode of a LeRobot v2.1 dataset: find when a piece lies flat on the staging mat (jev), decide its shape (jev
votes + geometry rules), box the target basket, and remove the box when the picking gripper releases the piece.
jev is single-flight: a call is issued on the current frame, takes its measured latency, and its answer only becomes
visible that much later, so decisions are as stale as they would be live.

  python scripts/annotate.py --dataset DS --reference labelled_pieces.png --out OUT [--episodes 0 1 2] [--overlay]
Needs a running jev server (see README), ffmpeg, OpenCV, pyarrow and Pillow. Finished episodes are skipped (resumable).
"""
import argparse, collections, json, os, subprocess, sys, time
import numpy as np
from PIL import Image, ImageDraw
import cv2
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jevfabric.client import JevClient
from jevfabric.decide import decide_shape
from jevfabric.geometry import geometry_vote
from jevfabric.prompts import (BASKET_BOXES, FLAT_THRESHOLD, MAP, MAT_BOX, Q_SHAPE, Q_SUPPORT, STATE_SHAPE, STATE_SUPPORT)
from jevfabric.release import ReleaseDetector

FPS, W, H = 30, 640, 480


def run_episode(ep, a, client, ref):
    g = np.array(pq.read_table(f"{a.dataset}/data/chunk-000/episode_{ep:06d}.parquet", columns=[a.state_key])[a.state_key].to_pylist(), np.float32)
    rd = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", f"{a.dataset}/videos/chunk-000/{a.video_key}/episode_{ep:06d}.mp4",
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    wr = None
    if a.overlay:
        wr = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", f"{a.out}/ep{ep:03d}_overlay.mp4"], stdin=subprocess.PIPE)
    lat = collections.defaultdict(list); events, votes, geo = [], [], []
    mode, busy_until, pending = "SEARCH", 0.0, None           # pending = (ready_time, kind, payload)
    target = shape = t_flat = t_on = release = None; n = 0
    while True:
        buf = rd.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        t = n / FPS; im = Image.frombytes("RGB", (W, H), buf)
        if pending and t >= pending[0]:                        # 1) a jev answer becomes visible
            _, kind, p = pending; pending = None
            if kind == "support" and mode == "SEARCH" and p >= FLAT_THRESHOLD:
                mode, t_flat, votes, geo = "SHAPE", t_flat or t, [], []; events.append({"ev": "flat", "t": round(t, 2), "f": n})
            elif kind == "shape" and mode == "SHAPE":
                votes.append(p); res = decide_shape(votes, geo)
                if res:
                    shape, src = res; target = MAP[shape]; mode = "TRACK"; t_on = t; release = ReleaseDetector()
                    events.append({"ev": "box_on", "t": round(t, 2), "f": n, "shape": shape, "target": target, "src": src,
                                   "flat_to_box_s": round(t - t_flat, 2), "votes": [v for v, _ in votes], "geo_votes": dict(collections.Counter(geo))})
                elif len(votes) >= 6: mode, t_flat = "SEARCH", None
        if mode == "SHAPE" and n % 2 == 0:                     # 1b) geometry vote, cheap
            v = geometry_vote(cv2.cvtColor(np.asarray(im.crop(MAT_BOX)), cv2.COLOR_RGB2BGR))
            if v: geo.append(v)
        if not pending and t >= busy_until:                    # 2) one jev call at a time
            s = time.perf_counter()
            if mode == "SEARCH":
                p = client.ask(STATE_SUPPORT, Q_SUPPORT, [im.resize((320, 240))])["support"]["probabilities"]["mat"]; kind = "support"
            elif mode == "SHAPE":
                crop = im.crop(MAT_BOX); crop = crop.resize((crop.width * 2 // 3, crop.height * 2 // 3))
                r = client.ask(STATE_SHAPE, Q_SHAPE, [crop, ref])["piece_shape"]; p = (r["choice"], r["confidence"]); kind = "shape"
            else: kind = None
            if kind:
                L = time.perf_counter() - s; lat[kind].append(L * 1000); busy_until = t + L; pending = (t + L, kind, p)
        if mode == "TRACK":                                    # 3) the picking arm is the one on the target basket's side
            if release.update(n, float(g[n, a.left_gripper if target.startswith("left") else a.right_gripper])):
                events.append({"ev": "placed_box_off", "t": round(t, 2), "f": n, "target": target, "box_on_s": round(t - t_on, 2)})
                mode, target, shape, t_flat = "SEARCH", None, None, None
        if wr:
            d = ImageDraw.Draw(im)
            if mode == "TRACK" and target: d.rectangle(BASKET_BOXES[target], outline=(255, 40, 40), width=4)
            d.rectangle((0, 0, 300, 20), fill=(0, 0, 0)); d.text((4, 4), f"f{n} t={t:5.1f}s {mode} {shape or ''}", fill=(255, 255, 0))
            wr.stdin.write(im.tobytes())
        n += 1
    rd.wait()
    if wr: wr.stdin.close(); wr.wait()
    summary = {"episode": ep, "frames": n, "events": events, "latency_ms": {
        k: {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(max(v))} for k, v in lat.items()}}
    json.dump(summary, open(f"{a.out}/ep{ep:03d}_summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--reference", required=True, help="image of the six pieces with letters A-F")
    ap.add_argument("--out", required=True); ap.add_argument("--episodes", type=int, nargs="*"); ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--state-key", default="state"); ap.add_argument("--video-key", default="images.cam_front")
    ap.add_argument("--left-gripper", type=int, default=6); ap.add_argument("--right-gripper", type=int, default=13)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ref = Image.open(a.reference); ref.thumbnail((360, 648)); client = JevClient(a.host, a.port)
    eps = a.episodes or range(sum(1 for _ in open(f"{a.dataset}/meta/episodes.jsonl")))
    for ep in eps:
        if os.path.exists(f"{a.out}/ep{ep:03d}_summary.json"): continue
        t0 = time.time(); run_episode(ep, a, client, ref); print(f"[ep{ep}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
