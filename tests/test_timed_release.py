import numpy as np
import pytest

from jevfabric.release import ReleaseDetector, TimedReleaseDetector


def test_seconds_rule_matches_original_frame_rule_at_30hz():
    values = np.concatenate(
        [
            np.full(n, v)
            for n, v in [(20, 0.05), (20, 0), (30, 0.05), (100, 0), (40, 0.05)]
        ]
    )
    original, timed = ReleaseDetector(), TimedReleaseDetector()
    frame_hits, time_hits = [], []
    for n, value in enumerate(values):
        if original.update(n, value):
            frame_hits.append(n)
        if timed.update(n / 30, value):
            time_hits.append(n)
    assert time_hits == [frame_hits[0]] == [175]


@pytest.mark.parametrize("hz", [25, 30, 60, 100])
def test_release_duration_does_not_depend_on_feedback_frequency(hz):
    detector = TimedReleaseDetector()
    hits = []
    for n in range(int(hz * 3.3)):
        t = n / hz
        value = 0.05 if t < 0.5 or t >= 2.5 else 0
        if detector.update(t, value):
            hits.append(t)
    assert len(hits) == 1
    assert 2.5 + 5 / 30 - 1e-9 <= hits[0] <= 2.5 + 5 / 30 + 2 / hz


def test_feedback_gap_cannot_be_counted_as_a_verified_carry():
    d = TimedReleaseDetector()
    for n in range(30):
        d.update(n / 30, 0.05 if n < 15 else 0)
    assert not any(d.update(10 + n / 30, 0.05) for n in range(30))


def test_duplicate_timestamps_do_not_accumulate_confirmation():
    d = TimedReleaseDetector()
    for _ in range(100):
        assert not d.update(0.0, 0.05)
    assert d.phase == 0
