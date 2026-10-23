"""Real-time annotation over externally owned RGB frames and gripper feedback.

The producer only copies a latest-frame slot. A coordinator owns perception
state/geometry; one I/O thread owns the HTTP client. No device or policy is
opened here, and neither geometry nor network I/O runs under the state lock.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .client import JevClient
from .decide import decide_shape
from .geometry import geometry_vote
from .prompts import (
    BASKET_BOXES,
    FLAT_THRESHOLD,
    MAP,
    MAT_BOX,
    Q_SHAPE,
    Q_SUPPORT,
    STATE_SHAPE,
    STATE_SUPPORT,
)
from .release import TimedReleaseDetector

LOG = logging.getLogger(__name__)


class DecisionClient(Protocol):
    def ask(self, state, questions, images=()): ...
    def cancel(self): ...
    def close(self): ...


class AnnotationUnavailable(RuntimeError):
    """No fresh, healthy annotation is available; the caller must skip inference."""


@dataclass(frozen=True)
class LiveConfig:
    host: str = "127.0.0.1"
    port: int = 8091
    request_timeout_s: float = 2.0
    max_result_age_s: float = 2.0
    max_frame_age_s: float = 0.25
    max_feedback_age_s: float = 0.25
    request_interval_s: float = 0.15
    retry_delay_s: float = 0.5
    geometry_hz: float = 15.0
    frame_size: tuple[int, int] = (640, 480)
    mat_box: tuple[int, int, int, int] = MAT_BOX
    # Tuples keep configuration immutable across threads.
    basket_boxes: tuple = tuple((key, tuple(box)) for key, box in BASKET_BOXES.items())
    hold_s: float = 5 / 30
    min_carry_s: float = 45 / 30
    max_feedback_gap_s: float = 0.2

    def __post_init__(self):
        for name in (
            "request_timeout_s",
            "max_result_age_s",
            "max_frame_age_s",
            "max_feedback_age_s",
            "request_interval_s",
            "retry_delay_s",
            "geometry_hz",
            "hold_s",
            "min_carry_s",
            "max_feedback_gap_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port is out of range")
        boxes = dict(self.basket_boxes)
        if set(boxes) != set(MAP.values()):
            raise ValueError("provide exactly the six target basket rectangles")
        w, h = self.frame_size
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (w, h)):
            raise ValueError("frame_size must contain two positive integers")
        for box in (self.mat_box, *boxes.values()):
            if len(box) != 4 or not all(isinstance(v, int) for v in box):
                raise ValueError("rectangles must have four integer coordinates")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("rectangle is outside frame_size")
        object.__setattr__(self, "frame_size", tuple(self.frame_size))
        object.__setattr__(self, "mat_box", tuple(self.mat_box))
        object.__setattr__(
            self, "basket_boxes", tuple((key, tuple(box)) for key, box in boxes.items())
        )


@dataclass(frozen=True)
class AnnotationSnapshot:
    generation: int
    version: int
    phase: str
    shape: str | None
    target: str | None
    box: tuple[int, int, int, int] | None
    effective_at_s: float
    healthy: bool
    error: str | None

    @property
    def token(self):
        return self.generation, self.version


@dataclass(frozen=True)
class _Frame:
    sequence: int
    captured_at: float
    rgb: np.ndarray


@dataclass(frozen=True)
class _Request:
    generation: int
    phase: str
    frame: _Frame
    started_at: float
    deadline: float


@dataclass(frozen=True)
class _Reply:
    request: _Request
    finished_at: float
    answers: dict | None = None
    error: str | None = None


class LiveAnnotator:
    """Two bounded workers; submit from the existing capture/feedback owner.

    All timestamps use this process's time.monotonic() domain. Feed raw RGB,
    never the previously boxed image. Feedback is in metres, not normalized
    state or commanded actions. Both submission methods are safe to call from
    different producer threads; duplicate/out-of-order timestamps are dropped.
    """

    def __init__(
        self,
        reference: Image.Image,
        config: LiveConfig | None = None,
        *,
        client_factory: Callable[[], DecisionClient] | None = None,
        geometry=geometry_vote,
        clock=time.monotonic,
    ):
        self.config = config or LiveConfig()
        self._reference = reference.convert("RGB").copy()
        self._reference.thumbnail((360, 648))
        self._factory = client_factory or (
            lambda: JevClient(
                self.config.host,
                self.config.port,
                timeout_s=self.config.request_timeout_s,
            )
        )
        self._geometry, self._clock = geometry, clock
        self._condition = threading.Condition(threading.Lock())
        self._stop = threading.Event()
        self._threads = []
        self._started = self._closed = False
        self._client = self._pending = self._job = self._reply = None
        self._cancelled = None
        self._reset_cancel_request = None
        self._frame = self._feedback = None
        self._sequence = self._processed_sequence = 0
        self._was_fresh = False
        self._last_geometry_log = -math.inf
        self._next_request = 0.0
        self._next_geometry = 0.0
        self._geometry_sequence = -1
        self._votes, self._geo = deque(maxlen=6), deque(maxlen=90)
        self._release = None
        self._service_ok = False
        self._error = "waiting for observations and JEV"
        self._fatal = None
        self._events = deque(maxlen=256)
        self._stats = {
            "submitted": 0,
            "dropped_frames": 0,
            "requests": 0,
            "errors": 0,
            "stale_replies": 0,
            "releases": 0,
            "geometry_errors": 0,
            "dropped_events": 0,
            "last_latency_ms": None,
        }
        self._state = AnnotationSnapshot(
            0, 0, "SEARCH", None, None, None, self._clock(), False, self._error
        )
        self._history = deque([self._state], maxlen=128)

    def start(self):
        with self._condition:
            if self._closed:
                raise RuntimeError(
                    "a closed annotator cannot restart; create a new instance"
                )
            if self._started:
                return self
            self._started = True
            self._threads = [
                threading.Thread(target=self._coordinate, name="jev-coordinate"),
                threading.Thread(target=self._network, name="jev-http"),
            ]
        try:
            for thread in self._threads:
                thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def _require_running(self):
        if not self._started or self._closed:
            raise RuntimeError("annotator is not running")

    def submit(self, rgb, grippers=None, *, captured_at_s=None):
        """Publish an owned immutable frame into a capacity-one mailbox."""
        timestamp = self._clock() if captured_at_s is None else float(captured_at_s)
        now = self._clock()
        self._validate_timestamp(timestamp, now)
        frame = np.asarray(rgb)
        w, h = self.config.frame_size
        if frame.dtype != np.uint8 or frame.shape != (h, w, 3):
            raise ValueError(f"expected uint8 RGB {(h, w, 3)}")
        owned = np.array(frame, copy=True, order="C")
        owned.flags.writeable = False
        widths = None if grippers is None else self._widths(grippers)
        with self._condition:
            self._require_running()
            if self._frame is not None and timestamp <= self._frame.captured_at:
                return False
            if (
                self._frame is not None
                and self._frame.sequence > self._processed_sequence
            ):
                self._stats["dropped_frames"] += 1
            self._sequence += 1
            self._frame = _Frame(self._sequence, timestamp, owned)
            self._stats["submitted"] += 1
            if widths is not None:
                self._feedback_locked(timestamp, widths)
            self._condition.notify_all()
        return True

    @staticmethod
    def _widths(grippers):
        values = np.asarray(grippers, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("grippers must be two finite nonnegative metre openings")
        return tuple(float(v) for v in values)

    @staticmethod
    def _validate_timestamp(timestamp, now):
        if not math.isfinite(timestamp) or timestamp > now + 0.05:
            raise ValueError(
                "timestamp must be finite and use the local monotonic clock"
            )

    def update_feedback(self, grippers, *, observed_at_s=None):
        """Cheap feedback path, independent of frame drops and HTTP latency."""
        timestamp = self._clock() if observed_at_s is None else float(observed_at_s)
        self._validate_timestamp(timestamp, self._clock())
        widths = self._widths(grippers)
        with self._condition:
            self._require_running()
            accepted = self._feedback_locked(timestamp, widths)
            self._condition.notify_all()
            return accepted

    def _feedback_locked(self, timestamp, widths):
        if self._feedback is not None and timestamp <= self._feedback[0]:
            return False
        self._feedback = timestamp, widths
        if self._state.phase == "TRACK" and self._release is not None:
            arm = 0 if self._state.target.startswith("left") else 1
            if self._release.update(timestamp, widths[arm]):
                self._stats["releases"] += 1
                self._search_locked("released")
        return True

    def _fresh_locked(self, now):
        return (
            self._frame is not None
            and self._feedback is not None
            and now - self._frame.captured_at <= self.config.max_frame_age_s
            and now - self._feedback[0] <= self.config.max_feedback_age_s
        )

    def _publish_locked(self, event, **fields):
        self._state = replace(
            self._state,
            version=self._state.version + 1,
            effective_at_s=self._clock(),
            healthy=self._service_ok,
            error=self._error,
        )
        self._history.append(self._state)
        self._record_locked(event, **fields)

    def _record_locked(self, event, **fields):
        if len(self._events) == self._events.maxlen:
            self._stats["dropped_events"] += 1
        self._events.append(
            dict(
                event=event,
                timestamp_s=self._clock(),
                generation=self._state.generation,
                version=self._state.version,
                phase=self._state.phase,
                target=self._state.target,
                **fields,
            )
        )

    def _search_locked(self, event):
        self._state = replace(
            self._state,
            generation=self._state.generation + 1,
            phase="SEARCH",
            shape=None,
            target=None,
            box=None,
        )
        self._votes.clear()
        self._geo.clear()
        self._release = None
        self._next_geometry = 0
        self._publish_locked(event)

    def reset(self):
        """Invalidate all in-flight results at an episode boundary."""
        with self._condition:
            self._require_running()
            self._service_ok, self._error = False, "waiting after reset"
            self._search_locked("reset")
            self._frame = self._feedback = None
            self._was_fresh = False
            self._history.clear()
            self._history.append(self._state)
            # Dispatch cancellation through the coordinator. A caller cancelling
            # here could race an old reply and interrupt the next generation.
            self._reset_cancel_request = self._pending
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return self._view_locked(self._state)

    def _view_locked(self, state):
        healthy = (
            not self._closed
            and self._fatal is None
            and self._service_ok
            and self._fresh_locked(self._clock())
        )
        return replace(
            state,
            healthy=healthy and state.healthy,
            error=self._error if not healthy else state.error,
        )

    def snapshot_at(self, captured_at_s):
        """Never apply a decision published after the observation was captured."""
        self._validate_timestamp(float(captured_at_s), self._clock())
        with self._condition:
            candidates = [s for s in self._history if s.effective_at_s <= captured_at_s]
            if not candidates:
                raise AnnotationUnavailable(
                    "observation predates the current annotation generation"
                )
            state = self._view_locked(candidates[-1])
            if self._clock() - captured_at_s > self.config.max_frame_age_s:
                return replace(state, healthy=False, error="observation is stale")
            return state

    def is_current(self, token):
        with self._condition:
            return self._view_locked(self._state).healthy and self._state.token == token

    def render(self, rgb, *, captured_at_s):
        """Copy RGB and draw only the training-compatible red basket rectangle."""
        frame = np.asarray(rgb)
        w, h = self.config.frame_size
        if frame.dtype != np.uint8 or frame.shape != (h, w, 3):
            raise ValueError("render expects the calibrated raw RGB frame size")
        state = self.snapshot_at(captured_at_s)
        if not state.healthy:
            raise AnnotationUnavailable(state.error or "annotation is unavailable")
        output = np.array(frame, copy=True, order="C")
        if state.box is not None:
            x0, y0, x1, y1 = state.box
            cv2.rectangle(output, (x0, y0), (x1, y1), (255, 40, 40), 4)
        return output, state

    def metrics(self):
        with self._condition:
            return dict(
                self._stats,
                in_flight=int(self._pending is not None),
                queued_frames=int(self._frame is not None),
                running_workers=sum(t.is_alive() for t in self._threads),
            )

    def drain_events(self):
        with self._condition:
            events = list(self._events)
            self._events.clear()
            return events

    def _failure(self, exc):
        LOG.exception("live annotation worker failed")
        with self._condition:
            self._fatal = exc
            self._service_ok, self._error = False, f"{type(exc).__name__}: {exc}"
            self._search_locked("worker_failed")
            self._stop.set()
            client = self._client
            self._condition.notify_all()
        if client is not None:
            client.cancel()

    def _accept_reply_locked(self, reply, now):
        request = reply.request
        cancelled = self._cancelled is request
        self._pending = None
        self._cancelled = None
        details = {
            "kind": request.phase,
            "frame_sequence": request.frame.sequence,
            "captured_at_s": request.frame.captured_at,
            "started_at_s": request.started_at,
            "finished_at_s": reply.finished_at,
            "latency_ms": (reply.finished_at - request.started_at) * 1000,
        }
        self._stats["last_latency_ms"] = details["latency_ms"]
        if (
            cancelled
            or reply.finished_at > request.deadline
            or request.generation != self._state.generation
            or request.phase != self._state.phase
            or now - request.frame.captured_at > self.config.max_result_age_s
            or not self._fresh_locked(now)
        ):
            self._stats["stale_replies"] += 1
            self._next_request = now + (
                self.config.retry_delay_s
                if cancelled
                else self.config.request_interval_s
            )
            self._record_locked("reply_discarded", **details)
            return
        self._next_request = now + self.config.request_interval_s
        if reply.error is not None:
            self._stats["errors"] += 1
            changed = self._service_ok
            self._service_ok, self._error = False, reply.error
            self._next_request = now + self.config.retry_delay_s
            self._record_locked("request_failed", reason=reply.error, **details)
            if changed:
                self._publish_locked("service_unavailable")
            return
        try:
            if request.phase == "SEARCH":
                p = float(reply.answers["support"]["probabilities"]["mat"])
                if not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError("invalid support probability")
                decision = None
                details["mat_probability"] = p
            else:
                prediction = reply.answers["piece_shape"]
                letter, confidence = (
                    prediction["choice"],
                    float(prediction["confidence"]),
                )
                if (
                    not isinstance(letter, str)
                    or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1
                ):
                    raise ValueError("invalid shape response")
                if letter not in {*MAP, "none", "unknown"}:
                    raise ValueError("unknown shape response")
                self._votes.append((letter, confidence))
                details.update(shape=letter, confidence=confidence)
                decision = decide_shape(list(self._votes), list(self._geo))
        except (KeyError, TypeError, ValueError) as exc:
            self._stats["errors"] += 1
            self._service_ok, self._error = False, f"invalid JEV answer: {exc}"
            self._publish_locked("invalid_answer")
            self._next_request = now + self.config.retry_delay_s
            return
        recovered = not self._service_ok
        self._service_ok, self._error = True, None
        self._record_locked("reply_applied", **details)
        if request.phase == "SEARCH" and p >= FLAT_THRESHOLD:
            self._state = replace(self._state, phase="SHAPE")
            self._votes.clear()
            self._geo.clear()
            self._publish_locked("flat")
        elif request.phase == "SHAPE" and decision:
            letter, source = decision
            target = MAP[letter]
            self._state = replace(
                self._state,
                phase="TRACK",
                shape=letter,
                target=target,
                box=dict(self.config.basket_boxes)[target],
            )
            self._release = TimedReleaseDetector(
                hold_s=self.config.hold_s,
                min_carry_s=self.config.min_carry_s,
                max_gap_s=self.config.max_feedback_gap_s,
            )
            self._release.update(
                self._feedback[0],
                self._feedback[1][0 if target.startswith("left") else 1],
            )
            self._publish_locked("box_on", shape=letter, source=source)
        elif request.phase == "SHAPE" and len(self._votes) >= 6:
            self._search_locked("uncertain")
        elif recovered:
            self._publish_locked("service_ready")

    def _coordinate(self):
        try:
            while not self._stop.is_set():
                work = cancel = None
                with self._condition:
                    now = self._clock()
                    if self._reset_cancel_request is not None:
                        if (
                            self._reset_cancel_request is self._pending
                            and self._reply is None
                        ):
                            self._cancelled = self._pending
                            cancel = self._client
                        self._reset_cancel_request = None
                    if self._reply is not None:
                        reply, self._reply = self._reply, None
                        self._accept_reply_locked(reply, now)
                    fresh = self._fresh_locked(now)
                    if not fresh and self._was_fresh:
                        self._service_ok, self._error = (
                            False,
                            "stale camera or gripper feedback",
                        )
                        self._search_locked("input_stale")
                        cancel = self._client
                    self._was_fresh = fresh
                    if (
                        self._pending is not None
                        and now >= self._pending.deadline
                        and self._cancelled is not self._pending
                    ):
                        self._cancelled = self._pending
                        self._service_ok, self._error = (
                            False,
                            "JEV request deadline exceeded",
                        )
                        self._publish_locked("request_timeout")
                        cancel = self._client
                    frame = self._frame
                    if fresh and frame is not None:
                        self._processed_sequence = frame.sequence
                        if (
                            self._state.phase == "SHAPE"
                            and frame.sequence != self._geometry_sequence
                            and now >= self._next_geometry
                        ):
                            work = self._state.generation, frame
                            self._geometry_sequence = frame.sequence
                            self._next_geometry = now + 1 / self.config.geometry_hz
                        if (
                            self._state.phase != "TRACK"
                            and self._pending is None
                            and self._client is not None
                            and now >= self._next_request
                        ):
                            request = _Request(
                                self._state.generation,
                                self._state.phase,
                                frame,
                                now,
                                now + self.config.request_timeout_s,
                            )
                            self._pending = self._job = request
                            self._stats["requests"] += 1
                            self._condition.notify_all()
                if cancel is not None:
                    cancel.cancel()
                if work is not None:
                    generation, frame = work
                    x0, y0, x1, y1 = self.config.mat_box
                    try:
                        vote = self._geometry(
                            cv2.cvtColor(frame.rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
                        )
                    except (cv2.error, ValueError, ArithmeticError) as exc:
                        if self._clock() - self._last_geometry_log >= 1:
                            LOG.warning("geometry vote failed: %s", exc)
                            self._last_geometry_log = self._clock()
                        with self._condition:
                            self._stats["geometry_errors"] += 1
                        vote = None
                    with self._condition:
                        if (
                            generation == self._state.generation
                            and self._state.phase == "SHAPE"
                            and self._clock() - frame.captured_at
                            <= self.config.max_frame_age_s
                            and vote in ("A", "B", "E")
                        ):
                            self._geo.append(vote)
                with self._condition:
                    if not self._stop.is_set():
                        self._condition.wait(timeout=0.02)
        except Exception as exc:  # noqa: BLE001 - worker boundary reports fatal errors to owner.
            self._failure(exc)

    def _network(self):
        client = None
        try:
            client = self._factory()
            for method in ("ask", "cancel", "close"):
                if not callable(getattr(client, method, None)):
                    raise TypeError("client must implement ask/cancel/close")
            with self._condition:
                self._client = client
                self._condition.notify_all()
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop.is_set() or self._job is not None
                    )
                    if self._stop.is_set():
                        break
                    request, self._job = self._job, None
                answers, error = None, None
                try:
                    with self._condition:
                        obsolete = (
                            request.generation != self._state.generation
                            or self._cancelled is request
                        )
                    if obsolete:
                        raise TimeoutError("request invalidated before HTTP dispatch")
                    if self._clock() >= request.deadline:
                        raise TimeoutError("request expired before HTTP dispatch")
                    image = Image.fromarray(request.frame.rgb)
                    if request.phase == "SEARCH":
                        answers = client.ask(
                            STATE_SUPPORT, Q_SUPPORT, [image.resize((320, 240))]
                        )
                    else:
                        crop = image.crop(self.config.mat_box)
                        crop = crop.resize((crop.width * 2 // 3, crop.height * 2 // 3))
                        answers = client.ask(
                            STATE_SHAPE, Q_SHAPE, [crop, self._reference]
                        )
                except Exception as exc:  # noqa: BLE001 - client plugins report failures as replies.
                    error = f"{type(exc).__name__}: {exc}"[:300]
                with self._condition:
                    if not self._stop.is_set():
                        self._reply = _Reply(request, self._clock(), answers, error)
                        self._condition.notify_all()
        except Exception as exc:  # noqa: BLE001 - worker boundary reports fatal errors to owner.
            self._failure(exc)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:  # noqa: BLE001 - surface cleanup failures during join.
                    LOG.error("JEV client cleanup failed: %s", exc)
                    with self._condition:
                        self._fatal = self._fatal or exc

    def raise_if_failed(self):
        with self._condition:
            failure = self._fatal
        if failure is not None:
            raise RuntimeError("annotation worker failed") from failure

    def close(self, timeout_s=None):
        budget = self.config.request_timeout_s + 1 if timeout_s is None else timeout_s
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("shutdown timeout must be finite and positive")
        with self._condition:
            already_closed = self._closed
            self._closed = True
            self._stop.set()
            self._service_ok, self._error = False, "closed"
            if not already_closed:
                self._search_locked("closed")
            client = self._client
            self._condition.notify_all()
        if client is not None:
            client.cancel()
        deadline = time.monotonic() + budget
        for thread in self._threads:
            if thread is threading.current_thread():
                raise RuntimeError("close must be called by the deployment owner")
            if thread.ident is not None:
                thread.join(max(0, deadline - time.monotonic()))
        if any(t.is_alive() for t in self._threads):
            raise RuntimeError(
                "annotation shutdown incomplete; worker references retained"
            )
        with self._condition:
            self._client = self._pending = self._job = self._reply = None
            self._frame = self._feedback = self._reset_cancel_request = None
        self.raise_if_failed()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
