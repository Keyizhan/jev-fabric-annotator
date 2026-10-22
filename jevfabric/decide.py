"""Combine jev votes and geometry votes into one shape decision."""
import collections
from .prompts import FAMILY, MAP


def decide_shape(votes, geo, max_votes=6, min_geo=3, geo_agree=0.67):
    """votes: list of (letter, confidence) from jev, one per call. geo: list of "A"/"B"/"E" votes from geometry_vote().
    Returns (letter, source) when decided, else None. source is "jev", "geo" or "jev_uncertain".
    The geometry rules only arbitrate inside the A/B/E family: jev is trusted for C, D and F."""
    c = collections.Counter(v for v, _ in votes if v in MAP); top = c.most_common(1)[0] if c else None
    n_family = sum(v in FAMILY for v, _ in votes)
    if n_family >= 2 and (top is None or top[0] in FAMILY):
        g = collections.Counter(geo); gt = g.most_common(1)[0] if g else None
        if gt and gt[1] >= min_geo and gt[1] >= geo_agree * len(geo): return gt[0], "geo"
        if len(votes) >= max_votes: return top[0], "jev_uncertain"
        return None
    if top and (top[1] >= 2 or len(votes) >= 4): return top[0], "jev"
    return None
