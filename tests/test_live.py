import threading
import time
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from jevfabric.deployment import LivePolicyBridge, StalePrediction
from jevfabric.live import AnnotationUnavailable, LiveAnnotator, LiveConfig


def wait_for(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


class Client:
    def __init__(self, delay=0, blocked_shape=False):
        self.delay, self.blocked_shape = delay, blocked_shape
        self.calls = []
        self.entered = threading.Event()
        self.unblock = threading.Event()
        self.active = self.maximum = 0
        self.closed_by = None
        self.old_shape = "C"
        self.first_pixels = []

    def ask(self, state, questions, images=()):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.calls.append(threading.get_ident())
        self.first_pixels.append(images[0].getpixel((0, 0)))
        try:
            if "support" in questions:
                time.sleep(self.delay)
                return {"support": {"probabilities": {"mat": 0.99}}}
            if self.blocked_shape:
                self.blocked_shape = False
                shape = self.old_shape
                self.entered.set()
                assert self.unblock.wait(3)
                return {"piece_shape": {"choice": shape, "confidence": 0.9}}
            time.sleep(self.delay)
            return {"piece_shape": {"choice": self.old_shape, "confidence": 0.9}}
        finally:
            self.active -= 1

    def cancel(self):
        pass  # Tests can deliberately simulate a peer that ignores cancellation.

    def close(self):
        self.closed_by = threading.get_ident()


class Pump:
    def __init__(self, annotator):
        self.annotator = annotator
        self.stop = threading.Event()
        self.widths = [0.05, 0.05]
        self.feedback = True
        self.times = []
        self.frame = np.full((480, 640, 3), 17, np.uint8)
        self.thread = threading.Thread(target=self.run, name="test-capture")

    def run(self):
        while not self.stop.is_set():
            start = time.monotonic()
            self.annotator.submit(self.frame, self.widths if self.feedback else None)
            self.times.append(time.monotonic() - start)
            self.stop.wait(0.01)

    def close(self):
        self.stop.set()
        self.thread.join(2)
        assert not self.thread.is_alive()


@contextmanager
def running(client=None, **overrides):
    client = client or Client()
    values = {
        "request_interval_s": 0.01,
        "retry_delay_s": 0.02,
        "request_timeout_s": 1,
        "max_result_age_s": 1.5,
        "hold_s": 0.025,
        "min_carry_s": 0.1,
    }
    values.update(overrides)
    config = replace(LiveConfig(), **values)
    live = LiveAnnotator(
        Image.new("RGB", (360, 648)),
        config,
        client_factory=lambda: client,
        geometry=lambda frame: None,
    ).start()
    pump = Pump(live)
    pump.thread.start()
    try:
        yield live, pump, client
    finally:
        client.unblock.set()
        pump.close()
        live.close()
        assert live.metrics()["running_workers"] == 0


def test_single_flight_and_capture_does_not_wait_for_http():
    with running(Client(delay=0.08)) as (live, pump, client):
        wait_for(lambda: live.snapshot().phase == "TRACK")
        assert client.maximum == 1
        assert len(set(client.calls)) == 1
        assert len(pump.times) >= 10
        assert np.percentile(pump.times, 95) < 0.04
        assert live.metrics()["queued_frames"] == 1
        assert live.snapshot().shape == "C"
    assert client.closed_by == client.calls[0]


def test_copy_on_submit_and_nonmutating_red_only_render():
    with running() as (live, _pump, client):
        wait_for(lambda: live.snapshot().phase == "TRACK")
        raw = np.full((480, 640, 3), 23, np.uint8)
        captured = time.monotonic()
        boxed, state = live.render(raw, captured_at_s=captured)
        assert state.target == "left_far"
        assert not np.shares_memory(raw, boxed)
        assert (raw == 23).all()
        np.testing.assert_array_equal(boxed[155, 0], [255, 40, 40])
        np.testing.assert_array_equal(boxed[300, 300], raw[300, 300])
        assert all(p == (17, 17, 17) for p in client.first_pixels)


def test_release_uses_feedback_even_when_no_new_image_is_submitted():
    with running(max_frame_age_s=2, max_feedback_age_s=1) as (live, pump, _client):
        wait_for(lambda: live.snapshot().phase == "TRACK")
        pump.close()
        generation = live.snapshot().generation
        # Only feedback continues. No HTTP or image processing is needed to clear.
        for width, seconds in [(0.05, 0.05), (0.0, 0.16), (0.05, 0.05)]:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                live.update_feedback([width, 0.05])
                time.sleep(0.01)
        assert live.metrics()["releases"] == 1
        assert live.snapshot().generation > generation
        assert live.snapshot().target is None


def test_reset_discards_an_old_inflight_shape_result():
    with running(Client(blocked_shape=True)) as (live, _pump, client):
        assert client.entered.wait(1)
        live.reset()
        client.old_shape = "F"
        client.unblock.set()
        wait_for(lambda: live.snapshot().phase == "TRACK")
        assert live.snapshot().shape == "F"
        assert live.snapshot().target == "right_far"
        assert live.metrics()["stale_replies"] >= 1


def test_reset_cancellation_is_owned_by_coordinator_before_new_dispatch():
    class Cancelled(Client):
        def __init__(self):
            super().__init__(blocked_shape=True)
            self.cancelling = threading.Event()
            self.finish_cancel = threading.Event()
            self.cancel_thread = None

        def cancel(self):
            if not self.unblock.is_set():
                self.cancel_thread = threading.current_thread().name
                self.cancelling.set()
                assert self.finish_cancel.wait(1)
                self.unblock.set()

    client = Cancelled()
    with running(client) as (live, _pump, _client):
        assert client.entered.wait(1)
        try:
            live.reset()  # This caller must not wait for cancel/HTTP.
            assert client.cancelling.wait(1)
            assert client.cancel_thread == "jev-coordinate"
            assert len(client.calls) == 2  # SEARCH + old SHAPE only.
            client.old_shape = "F"
        finally:
            client.finish_cancel.set()
        wait_for(lambda: live.snapshot().phase == "TRACK")
        assert live.snapshot().shape == "F"


def test_stale_feedback_removes_the_target_and_blocks_policy_input():
    with running(max_feedback_age_s=0.06) as (live, pump, _client):
        wait_for(lambda: live.snapshot().phase == "TRACK")
        pump.feedback = False
        wait_for(lambda: not live.snapshot().healthy)
        wait_for(lambda: live.snapshot().target is None)
        with pytest.raises(AnnotationUnavailable):
            live.render(pump.frame, captured_at_s=time.monotonic())


def test_old_image_does_not_receive_a_future_box():
    with running(Client(delay=0.04)) as (live, pump, _client):
        wait_for(lambda: live.snapshot().phase == "SHAPE")
        before_box = time.monotonic()
        wait_for(lambda: live.snapshot().phase == "TRACK")
        boxed, state = live.render(pump.frame, captured_at_s=before_box)
        assert state.target is None
        np.testing.assert_array_equal(boxed, pump.frame)
        assert not live.is_current(state.token)


def test_policy_result_invalidated_while_inference_is_running():
    with running() as (live, pump, _client):
        wait_for(lambda: live.snapshot().phase == "TRACK")
        bridge = LivePolicyBridge(live)
        entered, release = threading.Event(), threading.Event()
        errors = []
        wrist = np.zeros((480, 640, 3), np.uint8)
        raw = {
            "cam_front": pump.frame,
            "cam_left_wrist": wrist,
            "cam_right_wrist": wrist,
            "state": np.zeros(14),
            "observation_timestamp_ns": time.monotonic_ns(),
        }

        def infer(boxed):
            assert boxed["cam_left_wrist"] is wrist
            assert boxed["cam_right_wrist"] is wrist
            assert boxed["cam_front"] is not raw["cam_front"]
            entered.set()
            assert release.wait(1)
            return np.zeros((50, 14))

        def predict():
            try:
                bridge.predict(raw, infer)
            except StalePrediction as exc:
                errors.append(exc)

        thread = threading.Thread(target=predict)
        thread.start()
        try:
            assert entered.wait(1)
            live.reset()
        finally:
            release.set()
            thread.join(2)
        assert len(errors) == 1 and isinstance(errors[0], StalePrediction)


def test_stalled_policy_reply_expires_even_with_a_healthy_unchanged_target():
    with running(max_frame_age_s=0.1) as (live, pump, _client):
        wait_for(lambda: live.snapshot().phase == "TRACK")
        bridge = LivePolicyBridge(live)
        raw = {"cam_front": pump.frame, "observation_timestamp_ns": time.monotonic_ns()}
        token = live.snapshot().token

        def stalled(_observation):
            time.sleep(0.15)
            return np.zeros((50, 14))

        with pytest.raises(StalePrediction, match="observation expired"):
            bridge.predict(raw, stalled)
        assert live.is_current(token)


def test_no_images_or_labels_leak_across_an_episode_reset():
    live = LiveAnnotator(Image.new("RGB", (20, 20)), client_factory=Client).start()
    try:
        frame = np.zeros((480, 640, 3), np.uint8)
        timestamp = time.monotonic()
        live.submit(frame, [0.05, 0.05], captured_at_s=timestamp)
        assert not live.submit(frame, captured_at_s=timestamp)
        live.reset()
        with pytest.raises(AnnotationUnavailable):
            live.snapshot_at(timestamp)
    finally:
        live.close()


def test_shutdown_reports_uncooperative_worker_and_can_be_retried():
    client = Client(blocked_shape=True)
    config = replace(LiveConfig(), request_interval_s=0.01)
    live = LiveAnnotator(
        Image.new("RGB", (20, 20)),
        config,
        client_factory=lambda: client,
        geometry=lambda x: None,
    ).start()
    pump = Pump(live)
    pump.thread.start()
    try:
        assert client.entered.wait(1)
        pump.close()
        with pytest.raises(RuntimeError, match="shutdown incomplete"):
            live.close(timeout_s=0.01)
        assert live.metrics()["running_workers"] >= 1
    finally:
        client.unblock.set()
        live.close(timeout_s=2)
    assert live.metrics()["running_workers"] == 0


def test_mutating_the_callers_frame_after_submit_cannot_change_the_request():
    client = Client()
    live = LiveAnnotator(
        Image.new("RGB", (20, 20)),
        client_factory=lambda: client,
        geometry=lambda x: None,
    ).start()
    try:
        frame = np.full((480, 640, 3), 42, np.uint8)
        live.submit(frame, [0.05, 0.05])
        frame[:] = 99
        wait_for(lambda: bool(client.first_pixels))
        assert client.first_pixels[0] == (42, 42, 42)
    finally:
        live.close()


def test_config_rejects_invalid_time_and_out_of_frame_layout():
    with pytest.raises(ValueError):
        LiveConfig(request_timeout_s=float("nan"))
    with pytest.raises(ValueError):
        LiveConfig(mat_box=(0, 0, 1000, 480))


def test_expired_response_is_dropped_even_if_the_client_ignores_cancel():
    class SlowOnce(Client):
        def ask(self, *args, **kwargs):
            if not self.calls:
                time.sleep(0.08)
            return super().ask(*args, **kwargs)

    with running(SlowOnce(), request_timeout_s=0.03) as (live, _pump, _client):
        wait_for(lambda: live.snapshot().phase == "TRACK")
        assert live.metrics()["stale_replies"] >= 1
        events = live.drain_events()
        assert any(e["event"] == "request_timeout" for e in events)
        assert any(e["event"] == "reply_discarded" for e in events)
        assert any(e["event"] == "box_on" for e in events)


def test_invalid_answer_recovers_and_event_storage_stays_bounded():
    class InvalidOnce(Client):
        def ask(self, *args, **kwargs):
            result = super().ask(*args, **kwargs)
            return {} if len(self.calls) == 1 else result

    with running(InvalidOnce()) as (live, _pump, _client):
        wait_for(lambda: live.snapshot().phase == "TRACK")
        assert live.metrics()["errors"] == 1
        with live._condition:
            for _ in range(300):
                live._record_locked("test")
        assert len(live.drain_events()) == 256
        assert live.metrics()["dropped_events"] >= 44
