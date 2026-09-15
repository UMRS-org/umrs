"""Tests for B1 — Streaming Progress Tracker."""

import threading
import time

import pytest

from umrs.shared.progress import (
    CallbackAdapter,
    ConsoleProgressBar,
    ProgressEvent,
    ProgressGroup,
    ProgressState,
    ProgressTracker,
)


class TestProgressEvent:
    def test_fraction_zero_total(self):
        e = ProgressEvent(tracker_id="t1", state=ProgressState.RUNNING, current=5, total=0)
        assert e.fraction == 0.0
        assert e.percent == 0.0

    def test_fraction_normal(self):
        e = ProgressEvent(tracker_id="t1", state=ProgressState.RUNNING, current=50, total=100)
        assert e.fraction == pytest.approx(0.5)
        assert e.percent == pytest.approx(50.0)

    def test_fraction_capped(self):
        e = ProgressEvent(tracker_id="t1", state=ProgressState.RUNNING, current=200, total=100)
        assert e.fraction == 1.0


class TestProgressTracker:
    def test_initial_state(self):
        t = ProgressTracker("t1", total=10)
        assert t.state == ProgressState.PENDING
        assert t.current == 0
        assert t.total == 10
        assert t.fraction == 0.0
        assert t.is_active()

    def test_start_and_advance(self):
        t = ProgressTracker("t1", total=5)
        t.start()
        assert t.state == ProgressState.RUNNING
        t.advance(2, message="step 2")
        assert t.current == 2
        t.advance(3)
        assert t.current == 5
        assert t.fraction == pytest.approx(1.0)

    def test_advance_clamped_to_total(self):
        t = ProgressTracker("t1", total=3)
        t.start()
        t.advance(10)
        assert t.current == 3

    def test_set_progress(self):
        t = ProgressTracker("t1", total=100)
        t.start()
        t.set_progress(42, "at 42")
        assert t.current == 42

    def test_complete(self):
        t = ProgressTracker("t1", total=10)
        t.start()
        t.advance(5)
        t.complete("done")
        assert t.state == ProgressState.COMPLETED
        assert t.current == 10
        assert not t.is_active()

    def test_fail(self):
        t = ProgressTracker("t1", total=10)
        t.start()
        t.fail("error!")
        assert t.state == ProgressState.FAILED
        assert not t.is_active()

    def test_cancel(self):
        t = ProgressTracker("t1", total=10)
        t.start()
        t.cancel()
        assert t.state == ProgressState.CANCELLED

    def test_callback_receives_events(self):
        events = []

        class Recorder:
            def on_progress(self, event):
                events.append(event)

        t = ProgressTracker("t1", total=3)
        t.add_callback(Recorder())
        t.start()
        t.advance(1)
        t.advance(1)
        t.complete()
        assert len(events) == 4  # start + 2 advance + complete

    def test_remove_callback(self):
        events = []

        class Recorder:
            def on_progress(self, event):
                events.append(event)

        r = Recorder()
        t = ProgressTracker("t1", total=3)
        t.add_callback(r)
        t.start()
        t.remove_callback(r)
        t.advance(1)
        assert len(events) == 1  # only start

    def test_history(self):
        t = ProgressTracker("t1", total=2)
        t.start()
        t.advance(1)
        t.complete()
        history = t.get_history()
        assert len(history) == 3
        assert history[0].state == ProgressState.RUNNING
        assert history[-1].state == ProgressState.COMPLETED

    def test_metadata(self):
        t = ProgressTracker("t1", total=1)
        t.set_metadata("key", "value")
        t.start()
        event = t.get_history()[-1]
        assert event.metadata.get("key") == "value"

    def test_total_setter(self):
        t = ProgressTracker("t1", total=10)
        t.total = 20
        assert t.total == 20
        t.total = -5
        assert t.total == 0

    def test_thread_safety(self):
        t = ProgressTracker("t1", total=1000)
        t.start()

        def worker():
            for _ in range(100):
                t.advance(1)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert t.current == 1000


class TestCallbackAdapter:
    def test_wraps_function(self):
        events = []
        adapter = CallbackAdapter(lambda e: events.append(e))
        t = ProgressTracker("t1", total=1)
        t.add_callback(adapter)
        t.start()
        assert len(events) == 1


class TestConsoleProgressBar:
    def test_renders_without_crash(self, capsys):
        import io

        buf = io.StringIO()
        bar = ConsoleProgressBar(width=20, file=buf)
        t = ProgressTracker("t1", total=10)
        t.add_callback(bar)
        t.start()
        t.advance(5)
        t.complete()
        output = buf.getvalue()
        assert "%" in output


class TestProgressGroup:
    def test_create_trackers(self):
        g = ProgressGroup("batch")
        t1 = g.create_tracker("a", total=100)
        t2 = g.create_tracker("b", total=100)
        assert len(g.trackers) == 2
        assert g.fraction == 0.0

    def test_fraction_aggregation(self):
        g = ProgressGroup("batch")
        t1 = g.create_tracker("a", total=100)
        t2 = g.create_tracker("b", total=100)
        t1.start()
        t2.start()
        t1.advance(50)
        t2.advance(100)
        assert g.fraction == pytest.approx(0.75)

    def test_all_completed(self):
        g = ProgressGroup("batch")
        t1 = g.create_tracker("a", total=1)
        t2 = g.create_tracker("b", total=1)
        t1.start()
        t2.start()
        assert not g.all_completed
        t1.complete()
        t2.complete()
        assert g.all_completed

    def test_get_tracker(self):
        g = ProgressGroup("batch")
        g.create_tracker("x", total=5)
        assert g.get_tracker("x") is not None
        assert g.get_tracker("missing") is None
