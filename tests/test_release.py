import numpy as np
from jevfabric.release import ReleaseDetector, find_release


def trace(*parts):
    return np.concatenate([np.full(n, v, np.float32) for n, v in parts])


def test_regrasp_is_ignored_real_carry_ends_the_box():
    g = trace((20, 0.05), (20, 0.0), (30, 0.05), (100, 0.0), (40, 0.05))     # short closure, re-open, then a 100-frame carry
    assert find_release(g, 0) == 175                                         # closure ends at 170, confirmed after 6 open frames


def test_only_a_short_closure_never_releases():
    assert find_release(trace((20, 0.05), (20, 0.0), (60, 0.05)), 0) is None


def test_no_release_without_a_prior_open():
    assert find_release(trace((200, 0.0), (50, 0.05)), 0) is None            # starts closed: waits for an open arm first


def test_streaming_matches_offline():
    g = trace((10, 0.05), (80, 0.0), (30, 0.05)); d = ReleaseDetector(); hit = [n for n in range(len(g)) if d.update(n, float(g[n]))]
    assert hit and hit[0] == find_release(g, 0)
