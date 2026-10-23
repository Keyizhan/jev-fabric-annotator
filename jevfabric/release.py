"""Detect when the piece has been put into the basket from the picking gripper's opening width.

A real release is a gripper that closes on the piece, carries it, and opens again. Short closures (about 0.7 s) are
re-grasp attempts and must not end the box; a real carry lasts about 3 s. The closure length is known the moment the
gripper opens, so the rule adds no delay.
"""
from collections import deque
import math


class ReleaseDetector:
    def __init__(self, open_t=0.03, closed_t=0.012, hold=6, min_carry=45):
        self.open_t, self.closed_t, self.hold, self.min_carry = open_t, closed_t, hold, min_carry
        self.buf = deque(maxlen=hold); self.phase = 0; self.closed_since = None

    def update(self, n, gripper):
        """Feed one frame (index n, gripper opening). Returns True on the frame where the release is confirmed."""
        self.buf.append(gripper)
        if len(self.buf) < self.hold: return False
        all_open = all(v > self.open_t for v in self.buf); all_closed = all(v < self.closed_t for v in self.buf)
        if self.phase == 0 and all_open: self.phase = 1                              # arm is open: ready to grasp
        elif self.phase == 1 and all_closed: self.phase = 2; self.closed_since = n - self.hold + 1
        elif self.phase == 2 and all_open:
            if (n - self.hold + 1) - self.closed_since >= self.min_carry: return True
            self.phase = 1                                                          # too short: a re-grasp, wait for the next closure
        return False


def find_release(gripper, start, **kw):
    """Offline helper: first confirmed release frame at or after `start`, or None."""
    d = ReleaseDetector(**kw)
    for n in range(start, len(gripper)):
        if d.update(n, float(gripper[n])): return n
    return None


class TimedReleaseDetector:
    """The streaming release rule in seconds, independent of callback frequency.

    Defaults reproduce hold=6/min_carry=45 at 30 FPS. Missing feedback breaks
    continuity, so a paused stream cannot invent a completed carry.
    """

    def __init__(self, open_t=0.03, closed_t=0.012, hold_s=5 / 30,
                 min_carry_s=45 / 30, max_gap_s=0.2):
        if not 0 <= closed_t < open_t:
            raise ValueError("require 0 <= closed_t < open_t")
        if any(not math.isfinite(x) or x <= 0 for x in (hold_s, min_carry_s, max_gap_s)):
            raise ValueError("release durations must be finite and positive")
        self.open_t, self.closed_t = open_t, closed_t
        self.hold_s, self.min_carry_s, self.max_gap_s = hold_s, min_carry_s, max_gap_s
        self.phase = 0
        self.last_t = self.streak_t = self.closed_t0 = None
        self.kind = None

    def update(self, timestamp_s, gripper):
        if not math.isfinite(timestamp_s) or not math.isfinite(gripper) or gripper < 0:
            raise ValueError("feedback must be a finite timestamp and nonnegative metre opening")
        if self.last_t is not None and timestamp_s <= self.last_t:
            return False
        if self.last_t is not None and timestamp_s - self.last_t > self.max_gap_s:
            self.phase, self.kind, self.streak_t, self.closed_t0 = 0, None, None, None
        self.last_t = timestamp_s
        kind = 'open' if gripper > self.open_t else 'closed' if gripper < self.closed_t else None
        if kind != self.kind or kind is None:
            self.kind, self.streak_t = kind, timestamp_s
        if kind is None or timestamp_s - self.streak_t + 1e-9 < self.hold_s:
            return False
        if self.phase == 0 and kind == 'open':
            self.phase = 1
        elif self.phase == 1 and kind == 'closed':
            self.phase, self.closed_t0 = 2, self.streak_t
        elif self.phase == 2 and kind == 'open':
            if self.streak_t - self.closed_t0 + 1e-9 >= self.min_carry_s:
                self.phase = 3
                return True
            self.phase = 1
        return False
