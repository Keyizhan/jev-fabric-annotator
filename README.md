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

## Layout

```
jevfabric/     client, prompts, geometry, decide, release
scripts/       annotate.py, render_boxed.py, compare_to_labels.py
tests/
```
