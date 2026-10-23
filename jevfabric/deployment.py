"""Device-free integration into an existing capture and policy pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .live import AnnotationSnapshot, AnnotationUnavailable, LiveAnnotator


class StalePrediction(AnnotationUnavailable):
    """The target changed while VLA inference was running; discard this chunk."""


@dataclass(frozen=True)
class AnnotatedPrediction:
    value: Any
    annotation: AnnotationSnapshot


class LivePolicyBridge:
    """Use the existing observation owner, never a second camera or state client.

    on_observation/on_feedback are ingestion hooks. They do not perform JEV or
    VLA inference. predict runs in the existing VLA producer thread. Preserve
    its annotation token on RTC candidates and call is_current at dequeue to
    invalidate buffered chunks when the target/release/episode changes.
    """

    def __init__(
        self,
        annotator: LiveAnnotator,
        *,
        front_key="cam_front",
        state_key="state",
        timestamp_key="observation_timestamp_ns",
        gripper_indices=(6, 13),
    ):
        if len(gripper_indices) != 2 or any(i < 0 for i in gripper_indices):
            raise ValueError("provide two nonnegative gripper indices")
        self.annotator = annotator
        self.front_key, self.state_key, self.timestamp_key = (
            front_key,
            state_key,
            timestamp_key,
        )
        self.gripper_indices = tuple(gripper_indices)

    def _timestamp(self, observation):
        stamps = observation.get("camera_timestamps_ns", {})
        timestamp = stamps.get(self.front_key, observation.get(self.timestamp_key))
        if timestamp is None or isinstance(timestamp, bool):
            raise ValueError(
                "observation requires a monotonic acquisition timestamp in nanoseconds"
            )
        return float(timestamp) / 1e9

    def on_observation(self, observation: Mapping[str, Any]):
        """Capture-thread hook; raw image publication only, no device reads."""
        return self.annotator.submit(
            observation[self.front_key], captured_at_s=self._timestamp(observation)
        )

    def on_sample(self, observation: Mapping[str, Any]):
        """Convenience hook when capture already provides an aligned raw state."""
        state_timestamp = observation.get(
            "state_timestamp_ns", observation.get(self.timestamp_key)
        )
        if state_timestamp is None:
            raise ValueError("aligned observation requires its state timestamp")
        self.on_feedback(observation[self.state_key], observed_at_ns=state_timestamp)
        return self.on_observation(observation)

    def on_feedback(self, state, *, observed_at_ns):
        """Feedback-thread hook; pass the actual raw metre state, not targets."""
        values = np.asarray(state, dtype=float)
        if isinstance(observed_at_ns, (bool, np.bool_)):
            raise TypeError("feedback timestamp must be monotonic nanoseconds")
        if values.ndim != 1 or len(values) <= max(self.gripper_indices):
            raise ValueError("state does not contain the configured grippers")
        return self.annotator.update_feedback(
            values[list(self.gripper_indices)],
            observed_at_s=float(observed_at_ns) / 1e9,
        )

    def prepare(self, observation: Mapping[str, Any]):
        """Replace only the front image, after a causality/freshness check."""
        image, annotation = self.annotator.render(
            observation[self.front_key], captured_at_s=self._timestamp(observation)
        )
        if not self.annotator.is_current(annotation.token):
            raise StalePrediction("a newer annotation requires a newer observation")
        boxed = dict(observation)
        boxed[self.front_key] = image
        return boxed, annotation

    def predict(self, observation: Mapping[str, Any], infer: Callable[[dict], Any]):
        boxed, annotation = self.prepare(observation)
        value = infer(boxed)  # The existing VLA producer owns this call.
        if not self.annotator.is_current(annotation.token):
            raise StalePrediction(
                "target changed during policy inference; discard the result"
            )
        return AnnotatedPrediction(value, annotation)

    def is_current(self, prediction: AnnotatedPrediction):
        """Nonblocking token check for the existing enqueue/dequeue path."""
        return self.annotator.is_current(prediction.annotation.token)
