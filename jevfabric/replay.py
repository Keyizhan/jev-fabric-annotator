"""Exercise the real-time workers against a recorded episode at wall-clock FPS."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .deployment import LivePolicyBridge
from .live import AnnotationUnavailable, LiveAnnotator, LiveConfig


def load_config(path):
    if path is None:
        return LiveConfig()
    values = json.loads(Path(path).read_text())
    for key in ("frame_size", "mat_box"):
        if key in values:
            values[key] = tuple(values[key])
    if "basket_boxes" in values:
        values["basket_boxes"] = tuple(
            (k, tuple(v)) for k, v in values["basket_boxes"].items()
        )
    return LiveConfig(**values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--front-key", default="observation.images.cam_front")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args(argv)
    if (
        args.episode < 0
        or args.seconds is not None
        and (not np.isfinite(args.seconds) or args.seconds <= 0)
    ):
        parser.error(
            "episode must be nonnegative and seconds must be finite and positive"
        )
    info = json.loads((args.dataset / "meta/info.json").read_text())
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        parser.error("dataset fps must be finite and positive")
    chunk = args.episode // int(info.get("chunks_size", 1000))
    parquet = args.dataset / info["data_path"].format(
        episode_chunk=chunk, episode_index=args.episode
    )
    video = args.dataset / info["video_path"].format(
        episode_chunk=chunk, episode_index=args.episode, video_key=args.front_key
    )
    states = np.asarray(
        pq.read_table(parquet, columns=[args.state_key])[args.state_key].to_pylist(),
        dtype=np.float32,
    )
    if (
        states.ndim != 2
        or len(states) == 0
        or states.shape[1] != 14
        or not np.isfinite(states).all()
    ):
        parser.error("state must be finite raw [frames,14], with grippers in metres")
    config = load_config(args.config)
    with Image.open(args.reference) as reference:
        live = LiveAnnotator(reference, config)
    bridge = LivePolicyBridge(live)
    args.out.mkdir(parents=True, exist_ok=True)
    counts, durations, capture_times = Counter(), [], []
    limit = (
        len(states)
        if args.seconds is None
        else min(len(states), int(args.seconds * fps))
    )
    if limit < 1:
        parser.error("seconds must include at least one video frame")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"cannot open recorded front video: {video}")
    started = time.monotonic()
    frame_index = 0
    try:
        with (args.out / "events.jsonl").open("w") as events, live:
            last_drain = started
            while frame_index < limit:
                delay = started + frame_index / fps - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                # A slow decoder must not catch up by replaying old frames faster
                # than real time. Skip expired frames and their matching state.
                wanted = min(limit - 1, int((time.monotonic() - started) * fps))
                while frame_index < wanted:
                    if not cap.grab():
                        raise ValueError("video ended before its matching state")
                    frame_index += 1
                    counts["skipped_frames"] += 1
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError("video ended before its matching state")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = time.monotonic_ns()
                observation = {
                    "cam_front": rgb,
                    "state": states[frame_index],
                    "observation_timestamp_ns": timestamp,
                }
                before = time.monotonic()
                bridge.on_feedback(observation["state"], observed_at_ns=timestamp)
                bridge.on_observation(observation)
                durations.append((time.monotonic() - before) * 1000)
                capture_times.append(timestamp)
                try:
                    _, annotation = bridge.prepare(observation)
                    counts["healthy_frames"] += 1
                    counts["boxed_frames"] += int(annotation.box is not None)
                except AnnotationUnavailable:
                    counts["unavailable_frames"] += 1
                if time.monotonic() - last_drain >= 0.5:
                    for event in live.drain_events():
                        events.write(json.dumps(event) + "\n")
                    events.flush()
                    last_drain = time.monotonic()
                frame_index += 1
            for event in live.drain_events():
                events.write(json.dumps(event) + "\n")
            last_state = asdict(live.snapshot())
    finally:
        cap.release()
        # Both context managers have closed: include the shutdown event, too.
        with (args.out / "events.jsonl").open("a") as events:
            for event in live.drain_events():
                events.write(json.dumps(event) + "\n")
    intervals = np.diff(capture_times) / 1e6
    report = dict(
        episode=args.episode,
        started_at_s=started,
        processed_frames=len(capture_times),
        **counts,
        elapsed_s=time.monotonic() - started,
        callback_p95_ms=float(np.percentile(durations, 95)),
        capture_interval_p95_ms=float(np.percentile(intervals, 95))
        if len(intervals)
        else None,
        final_annotation=last_state,
        metrics=live.metrics(),
        hardware_access=False,
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if counts["healthy_frames"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
