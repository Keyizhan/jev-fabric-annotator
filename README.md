# jevfabric

Label dual-arm fabric-sorting episodes with a typed-decision model (**jev**) plus cheap geometry rules. The output is a red box on the target basket from the moment the piece is classified until the gripper releases it — overlay only, no caption text — so a VLA can be trained with a prompt like:

> Sort fabric pieces from the pile into the basket marked by the red box.

This repository is the cleaned pipeline. It does **not** include:

- the StartLux / jev model weights (CC BY-NC)
- recorded robot data or the labelled reference photo
- training code (`soma-fluxvla` no-subtask config lives in that repo)

## Shape → basket (front camera, image coordinates)

`near` = bottom of the image (robot side). `far` = top (pile side).

| Letter | Basket        | Outline |
|--------|---------------|---------|
| A      | left_near     | wedge / blade; zigzag hem on the **left** |
| B      | left_middle   | wide body + one thin horn; may appear mirrored face-down |
| C      | left_far      | narrow U strip with a large hole |
| D      | right_near    | small irregular patch |
| E      | right_middle  | wide body, notch upper-right, thin tail; may appear mirrored |
| F      | right_far     | large C-shaped hook |

Blue vs beige can tell face-up vs face-down; if the thin feature points down, chirality flips. Geometry rules only arbitrate **A / B / E**; C, D, F trust jev.

Basket rectangles in `jevfabric/prompts.py` (`BASKET_BOXES`) are for a **640×480** front camera and a fixed table layout. Re-measure them if the camera or baskets move.

## Pipeline

1. Replay each LeRobot v2.1 episode at 30 fps (`scripts/annotate.py`).
2. jev is **single-flight**: one HTTP call at a time; the answer is only applied after the measured latency, so boxes appear as late as they would live.
3. SEARCH: is a piece lying flat on the staging mat?
4. SHAPE: jev votes + `geometry_vote` → `decide_shape` → map letter to basket, draw the box.
5. TRACK until `ReleaseDetector` sees the picking gripper open.
6. `scripts/render_boxed.py` burns **only** the red rectangle into the original videos (same length / size / fps).
7. `scripts/compare_to_labels.py` scores against human `subtask` text if you have it.

## Setup

```bash
pip install -e ".[test]"
# jev HTTP server must already be running (POST /v1/systemone)
pytest -q
```

```bash
python scripts/annotate.py \
  --dataset /path/to/lerobot_v2_1 \
  --reference /path/to/labelled_pieces.png \
  --out /path/to/summaries \
  --host 127.0.0.1 --port 8091

python scripts/render_boxed.py \
  --dataset /path/to/lerobot_v2_1 \
  --summaries /path/to/summaries \
  --out /path/to/boxed_dataset
```

`--overlay` on `annotate.py` writes debug videos *with* HUD text; do not use those for VLA training. Use `render_boxed.py`.

## Real-time annotation

`jevfabric.live.LiveAnnotator` runs the same SEARCH → SHAPE → TRACK decisions
against live observations. `jevfabric.deployment.LivePolicyBridge` connects it to
an existing camera/feedback owner and VLA inference callback. These modules open
no cameras, CAN devices or robot controllers. The hardware runtime must install
the hooks below; importing the library does not change an existing deployment.

Thread ownership:

| Owner | Work |
|---|---|
| Existing capture thread | `on_observation`: copy raw RGB into a capacity-one latest-frame slot |
| Existing feedback thread | `on_feedback`: update raw gripper openings and release state |
| `jev-coordinate` | State transitions, geometry at up to 15 Hz, deadlines and stale-result rejection |
| `jev-http` | Own the HTTP client; one outstanding request, JPEG encoding outside the state lock |
| Existing VLA producer | `predict`: draw the front-camera box on a copy, then call the existing policy |
| Existing action consumer | Check the annotation token before accepting a queued action chunk |

Frames are replaced rather than queued behind slow inference. HTTP, geometry,
rendering, policy calls and file I/O never run under the annotation state lock.
Workers are non-daemon and explicitly joined. `close()` interrupts HTTP reads,
reports worker errors, and raises if any worker remains alive. Call it from the
deployment owner **after stopping ingestion callbacks**.

All timestamps must be acquisition timestamps in the local process's
`time.monotonic_ns()` domain. Do not pass wall-clock time, dataset timestamps,
another machine's monotonic clock or the time an old frame was retrieved.
Publish camera frames and feedback at their existing rates (normally 30 Hz),
independent of the slower VLA producer. No extra camera or hardware polling
thread is required. Inputs are **uint8 RGB 640×480** and **raw metre gripper
openings**, with default indices 6 and 13 in the 14-dimensional raw state.
Normalized states, action commands and already boxed images are not inputs.

```python
from PIL import Image
from jevfabric.deployment import LivePolicyBridge
from jevfabric.live import AnnotationUnavailable, LiveAnnotator, LiveConfig

with Image.open("/path/to/labelled_pieces.png") as reference:
    annotator = LiveAnnotator(reference, LiveConfig(port=8091)).start()
bridge = LivePolicyBridge(annotator)

# Install these hooks in the existing owners, once:
# capture:  bridge.on_observation({"cam_front": raw_rgb,
#             "camera_timestamps_ns": {"cam_front": acquired_ns}})
# feedback: bridge.on_feedback(raw_state, observed_at_ns=feedback_ns)
# If capture already supplies aligned raw state, bridge.on_sample(obs) combines them.

def policy_producer(obs):
    # obs has the existing policy keys plus observation_timestamp_ns or
    # camera_timestamps_ns["cam_front"]. Wrist images/state are passed unchanged.
    try:
        candidate = bridge.predict(obs, infer=existing_policy_infer)
    except AnnotationUnavailable:
        return  # Skip this candidate; use the runtime's existing no-valid-action handling.
    existing_queue.put(candidate)  # Preserve value AND annotation, not just value.

def action_consumer(candidate):
    if not bridge.is_current(candidate):
        return  # Drop invalid buffered chunks through the existing queue path.
    existing_execute(candidate.value)

# Episode boundary: annotator.reset() invalidates previous labels/results/tokens.
# Shutdown: stop/join existing ingestion callbacks, then annotator.close().
```

`existing_policy_infer`, `existing_queue` and `existing_execute` above denote
the deployment's own components, not additional workers supplied by this repo.
For RTC, retain the annotation token through resampling and chunk buffering,
and check it when selecting the next action, including partially consumed
chunks. A check only at inference completion does not invalidate actions
already buffered. This library does not implement a robot stop/hold command;
the existing runtime owns that behavior when no valid candidate is available.

Only the front image receives the training-compatible red rectangle
`RGB(255,40,40)`, thickness 4, with no HUD or generated subtask text. Use the
no-subtask VLA instruction shown at the top of this README. JEV sees the raw
front image; SHAPE receives the mat crop and the labelled reference. TRACK
keeps the selected basket until the target-side gripper completes
open → closed/carry → open, matching the offline annotation's side assumption.
Release timing uses elapsed seconds, with continuity checks on feedback, so
changing feedback rate does not change the intended hold duration. Defaults
match the original 30 Hz detector (six-sample hold, 45-frame carry).

Results from before reset, the wrong phase, expired requests or stale source
images are rejected. A decision is never painted onto an image acquired before
that decision became available. Missing/stale camera or feedback clears the
target and invalidates policy tokens. Missing JEV service prevents new valid
policy candidates; it does not silently supply an unboxed fallback.

### Recorded replay with the real server

Start the external JEV model server separately with image support and point
`configs/live.example.json` at its host/port. The model assets/reference are
not bundled. Run a recorded episode at wall-clock FPS, without hardware:

```bash
python scripts/replay_live.py \
  --dataset /path/to/raw_lerobot_v2_1 --episode 105 \
  --front-key images.cam_front --state-key state \
  --reference /path/to/labelled_pieces.png \
  --config configs/live.example.json --seconds 40 \
  --out output/live-ep105
```

The default dataset keys are `observation.images.cam_front` and
`observation.state`; override them as above for datasets without that prefix.
`report.json` records capture/callback timings, request/drop/error counts and
remaining workers after shutdown. `events.jsonl` records phase changes,
shape/basket/source decisions, request timestamps/latencies, rejected replies,
release and errors. The owner drains a bounded event buffer; worker threads
never write logs to disk. An overflow increments `dropped_events`.

`max_frame_age_s` and `max_feedback_age_s` default to 0.25 s. HTTP deadlines and
source-result age default to 2 s. These bound input validity, **not guaranteed
inference latency**. Measure with the intended JEV/VLA models running together
before choosing deployment rates. Geometry/basket rectangles must match the
actual camera layout; configure them rather than resizing an arbitrary view.

## Layout

```
jevfabric/     client, prompts, geometry, decide, release, live, deployment, replay
configs/       live.example.json
scripts/       annotate.py, render_boxed.py, compare_to_labels.py, replay_live.py
tests/
```
