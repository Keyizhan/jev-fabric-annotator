#!/usr/bin/env python3
"""Draw ONLY a red rectangle (no text) on the target basket in the original front-camera videos, from annotate.py summaries.
Same frame count, size and fps as the original; re-encoded with x264 (crf 18). No jev calls.

  python scripts/render_boxed.py --dataset DS --summaries OUT --out BOXED [--episodes 4 6]
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jevfabric.prompts import BASKET_BOXES

W, H, FPS, RED, THICK = 640, 480, 30, (40, 40, 255), 4      # BGR


def intervals(events):
    ev = sorted(events, key=lambda e: e["f"]); out = []
    for e in ev:
        if e["ev"] != "box_on": continue
        off = next((x for x in ev if x["ev"] == "placed_box_off" and x["f"] > e["f"] and x["target"] == e["target"]), None)
        out.append({"start": e["f"], "end": off["f"] if off else None, "target": e["target"], "shape": e["shape"]})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--summaries", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--episodes", type=int, nargs="*"); ap.add_argument("--video-key", default="images.cam_front")
    a = ap.parse_args(); vdir = f"{a.out}/videos/chunk-000/{a.video_key}_boxed"; os.makedirs(vdir, exist_ok=True); meta = {}
    for f in sorted(glob.glob(f"{a.summaries}/ep*_summary.json")):
        s = json.load(open(f)); ep = s["episode"]
        if a.episodes and ep not in a.episodes: continue
        iv = intervals(s["events"])
        rd = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", f"{a.dataset}/videos/chunk-000/{a.video_key}/episode_{ep:06d}.mp4",
                               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        wr = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", f"{vdir}/episode_{ep:06d}.mp4"], stdin=subprocess.PIPE)
        n = 0
        while True:
            b = rd.stdout.read(W * H * 3)
            if len(b) < W * H * 3: break
            fr = np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()
            for x in iv:
                if x["start"] <= n and (x["end"] is None or n < x["end"]):
                    x0, y0, x1, y1 = BASKET_BOXES[x["target"]]; cv2.rectangle(fr, (x0, y0), (x1, y1), RED, THICK)
            wr.stdin.write(fr.tobytes()); n += 1
        wr.stdin.close(); wr.wait(); rd.wait(); meta[ep] = {"frames": n, "intervals": iv}; print("ep", ep, n, len(iv), flush=True)
    json.dump({"fps": FPS, "frame_size": [W, H], "basket_boxes_xyxy": BASKET_BOXES, "episodes": meta}, open(f"{a.out}/box_annotations.json", "w"), indent=1)


if __name__ == "__main__":
    main()
