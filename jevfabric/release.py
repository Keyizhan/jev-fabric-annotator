"""Detect when the piece has been put into the basket from the picking gripper's opening width.

A real release is a gripper that closes on the piece, carries it, and opens again. Short closures (about 0.7 s) are
re-grasp attempts and must not end the box; a real carry lasts about 3 s. The closure length is known the moment the
gripper opens, so the rule adds no delay.
"""
from collections import deque


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
