#!/usr/bin/env python3
"""Score annotate.py summaries against human per-frame subtask text, place by place.

The human text must contain a phrase matched by --pattern (default: "Place fabric X in the ... basket using the left|right arm").
A human place is matched to the jev box that overlaps it most; reports shape accuracy, misses, extra boxes and release timing.

  python scripts/compare_to_labels.py --dataset DS --summaries OUT --report report.json
"""
import argparse, collections, glob, json, re
import numpy as np, pyarrow.parquet as pq

DEFAULT = r"Place fabric ([A-F]) in the [\w-]+ basket using the (left|right) arm"


def human_places(dataset, ep, pat, column):
    sub = pq.read_table(f"{dataset}/data/chunk-000/episode_{ep:06d}.parquet", columns=[column])[column].to_pylist()
    cur, runs = {}, []
    for i, text in enumerate(sub + [""]):
        now = {(m.group(1), m.group(2)) for part in text.split(" || ") for m in [pat.search(part)] if m}
        for k in list(cur):
            if k not in now: runs.append((cur.pop(k), i, k[0], k[1]))
        for k in now: cur.setdefault(k, i)
    return sorted(runs)          # (start, end, letter, arm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--summaries", required=True)
    ap.add_argument("--report", required=True); ap.add_argument("--pattern", default=DEFAULT); ap.add_argument("--column", default="subtask")
    a = ap.parse_args(); pat = re.compile(a.pattern); rows, tot, conf = [], collections.Counter(), collections.Counter()
    for f in sorted(glob.glob(f"{a.summaries}/ep*_summary.json")):
        s = json.load(open(f)); ep = s["episode"]; gold = human_places(a.dataset, ep, pat, a.column)
        ons = [e for e in s["events"] if e["ev"] == "box_on"]; offs = [e for e in s["events"] if e["ev"] == "placed_box_off"]
        boxes = [{"on": e["f"], "off": next((o["f"] for o in offs if o["f"] > e["f"] and o["target"] == e["target"]), None), "shape": e["shape"]} for e in ons]
        used, places = set(), []
        for g0, g1, letter, _ in gold:
            best, bo = None, 0
            for i, b in enumerate(boxes):
                ov = min(g1, b["off"] or g1) - max(g0, b["on"])
                if i not in used and ov > bo: best, bo = i, ov
            if best is None: tot["missed"] += 1; places.append({"gold": letter, "g0": g0, "g1": g1, "match": None}); continue
            used.add(best); b = boxes[best]; ok = b["shape"] == letter; tot["matched"] += 1; tot["correct"] += ok; conf[(letter, b["shape"])] += 1
            places.append({"gold": letter, "g0": g0, "g1": g1, "jev": b["shape"], "ok": ok, "on": b["on"], "off": b["off"],
                           "off_minus_g1": (b["off"] - g1) if b["off"] else None})
        tot["extra_boxes"] += len(boxes) - len(used)
        rows.append({"episode": ep, "places": places, "extras": [b for i, b in enumerate(boxes) if i not in used]})
    d = np.array([p["off_minus_g1"] for e in rows for p in e["places"] if p.get("off_minus_g1") is not None])
    rep = {**tot, "shape_accuracy": tot["correct"] / max(1, tot["matched"]), "episodes": len(rows),
           "release_minus_human_end_frames": {"median": float(np.median(d)), "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90))} if len(d) else None,
           "confusion": {f"{x}->{y}": n for (x, y), n in sorted(conf.items())}, "per_episode": rows}
    json.dump(rep, open(a.report, "w"), indent=1); print({k: v for k, v in rep.items() if k != "per_episode"})


if __name__ == "__main__":
    main()
