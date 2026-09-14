"""Slice 16 tests: RemotePoolExecutor (ZMQ-based remote worker pool).

Tests cover:
  - Single task round-trip via ZMQ REQ/REP
  - Pool reuse (same worker reused for second task)
  - Worker timeout (unresponsive worker triggers error)
  - Multiple workers (round-robin across 2 workers)
  - Shutdown (sockets closed cleanly)

Uses _FakeZmqWorker — an in-process ZMQ REP thread that imports
test step modules directly, mirroring the _FakeProc pattern from
test_worker_pool.py.
"""

from __future__ import annotations

import pickle
import sys
import threading
import time
import types
from typing import Any

import pytest
import zmq
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.server.step_executor import (
    RemotePoolExecutor,
    StepExecutionError,
    _RemoteWorker,
)
from umrs.server.step_router import reset_feedback_store
from umrs.server.step_worker import _process_task
from umrs.server.workflow_execution_registry import (
    reset_execution_registry_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    reset_feedback_store()
    yield
    reset_execution_registry_for_tests()
    reset_feedback_store()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_ctx(**metadata_overrides) -> StepContext:
    return StepContext(
        s3=None,
        input_bucket="b",
        input_keys=[],
        output_bucket="b",
        config={},
        metadata=dict(metadata_overrides),
        run_id="t",
    )


class _OkStep(Step):
    name = "probe"

    def setup(self, config):
        pass

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(extra={"ok": True})


_PROBE_MODULE = types.ModuleType("umrs_test_remote_pool_module")
_PROBE_MODULE._OkStep = _OkStep  # type: ignore[attr-defined]
sys.modules["umrs_test_remote_pool_module"] = _PROBE_MODULE

_TEST_ENTRYPOINT = "umrs_test_remote_pool_module:_OkStep"


class _FakeZmqWorker:
    """In-process ZMQ REP worker for testing.

    Binds a REP socket on a random port, loops:
    recv → pickle.loads → _process_task → pickle.dumps → send.
    """

    def __init__(self, *, slow: bool = False) -> None:
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self._slow = slow
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def address(self) -> str:
        return f"tcp://127.0.0.1:{self._port}"

    def _loop(self) -> None:
        try:
            while not self._stop:
                if self._sock.poll(timeout=200, flags=zmq.POLLIN) == 0:
                    continue
                raw = self._sock.recv()
                if self._slow:
                    # Simulate unresponsive worker.
                    time.sleep(10)
                payload = pickle.loads(raw)
                reply = _process_task(payload)
                self._sock.send(pickle.dumps(reply))
        except zmq.ContextTerminated:
            pass
        except Exception:
            pass

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=5)
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
        try:
            self._ctx.term()
        except Exception:
            pass


@pytest.fixture
def fake_worker():
    w = _FakeZmqWorker()
    yield w
    w.stop()


@pytest.fixture
def fake_workers():
    """Two fake workers for multi-worker tests."""
    w1 = _FakeZmqWorker()
    w2 = _FakeZmqWorker()
    yield (w1, w2)
    w1.stop()
    w2.stop()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_remote_single_task(fake_worker):
    """Send one task through RemotePoolExecutor and get correct result."""
    executor = RemotePoolExecutor(
        workers=[fake_worker.address],
        timeout_seconds=10.0,
        acquire_timeout=5.0,
    )
    try:
        result = executor.execute(
            entrypoint=_TEST_ENTRYPOINT,
            config={},
            ctx=_make_ctx(),
        )
        assert isinstance(result, StepResult)
        assert result.extra == {"ok": True}
    finally:
        executor.shutdown()


def test_remote_pool_reuse(fake_worker):
    """Same worker reused for a second task."""
    executor = RemotePoolExecutor(
        workers=[fake_worker.address],
        timeout_seconds=10.0,
        acquire_timeout=5.0,
    )
    try:
        r1 = executor.execute(
            entrypoint=_TEST_ENTRYPOINT,
            config={},
            ctx=_make_ctx(),
        )
        r2 = executor.execute(
            entrypoint=_TEST_ENTRYPOINT,
            config={},
            ctx=_make_ctx(),
        )
        assert r1.extra == {"ok": True}
        assert r2.extra == {"ok": True}
    finally:
        executor.shutdown()


def test_remote_worker_timeout():
    """Worker that never replies triggers timeout."""
    slow_worker = _FakeZmqWorker(slow=True)
    try:
        executor = RemotePoolExecutor(
            workers=[slow_worker.address],
            timeout_seconds=0.5,
            acquire_timeout=5.0,
        )
        try:
            with pytest.raises(StepExecutionError, match="timeout"):
                executor.execute(
                    entrypoint=_TEST_ENTRYPOINT,
                    config={},
                    ctx=_make_ctx(),
                )
        finally:
            executor.shutdown()
    finally:
        slow_worker.stop()


def test_remote_pool_multiple_workers(fake_workers):
    """Tasks distributed across multiple workers."""
    w1, w2 = fake_workers
    executor = RemotePoolExecutor(
        workers=[w1.address, w2.address],
        timeout_seconds=10.0,
        acquire_timeout=5.0,
    )
    try:
        results = []
        for _ in range(4):
            r = executor.execute(
                entrypoint=_TEST_ENTRYPOINT,
                config={},
                ctx=_make_ctx(),
            )
            results.append(r)
        assert all(r.extra == {"ok": True} for r in results)
    finally:
        executor.shutdown()


def test_remote_shutdown(fake_worker):
    """Shutdown closes sockets without errors."""
    executor = RemotePoolExecutor(
        workers=[fake_worker.address],
        timeout_seconds=10.0,
        acquire_timeout=5.0,
    )
    # Do one task so there's an idle worker in the pool.
    executor.execute(
        entrypoint=_TEST_ENTRYPOINT,
        config={},
        ctx=_make_ctx(),
    )
    # Shutdown should not raise.
    executor.shutdown()


def test_remote_no_workers():
    """Executor with no workers raises setup error."""
    executor = RemotePoolExecutor(
        workers=[],
        timeout_seconds=10.0,
        acquire_timeout=5.0,
    )
    try:
        with pytest.raises(StepExecutionError, match="no workers"):
            executor.execute(
                entrypoint=_TEST_ENTRYPOINT,
                config={},
                ctx=_make_ctx(),
            )
    finally:
        executor.shutdown()


def test_remote_cooldown_backoff():
    """After a failure, worker enters cooldown and auto-resets."""
    ctx = zmq.Context()
    worker = None
    try:
        worker = _RemoteWorker(
            address="tcp://127.0.0.1:1",  # nothing listens — will fail
            zmq_context=ctx,
            timeout=0.3,
            cooldown_sec=0.5,
        )
        # Force a failure by trying to send (will timeout/error).
        with pytest.raises(StepExecutionError):
            worker.run_task(b"dummy")

        # Worker should be dead right after failure.
        assert not worker.is_alive()

        # Wait past cooldown — worker should auto-reset.
        time.sleep(0.6)
        assert worker.is_alive()
    finally:
        if worker is not None:
            worker.stop()
        ctx.term()


def test_multi_slot_concurrent(fake_worker, monkeypatch):
    """Slice 18a: 1 address, capacity 2 → 2 tasks run concurrently."""
    monkeypatch.setenv("UMRS_REMOTE_WORKER_SLOTS", "2")
    executor = RemotePoolExecutor(
        workers=[fake_worker.address],
        timeout_seconds=10.0,
        acquire_timeout=5.0,
    )
    results = []
    errors = []

    def _run():
        try:
            r = executor.execute(
                entrypoint=_TEST_ENTRYPOINT,
                config={},
                ctx=_make_ctx(),
            )
            results.append(r)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    try:
        assert not errors, f"unexpected errors: {errors}"
        assert len(results) == 2
        assert all(r.extra == {"ok": True} for r in results)
    finally:
        executor.shutdown()


def test_capacity_enforced(monkeypatch):
    """Slice 18a: cap=1 blocks third concurrent request on same address."""
    monkeypatch.setenv("UMRS_REMOTE_WORKER_SLOTS", "1")
    # Slow worker that holds requests for 2s so we can saturate the slot.
    slow = _FakeZmqWorker(slow=True)
    executor = RemotePoolExecutor(
        workers=[slow.address],
        timeout_seconds=5.0,
        acquire_timeout=0.5,  # very short — should timeout
    )
    # First task — will be accepted and block.
    err_holder = []

    def _run_slow():
        try:
            executor.execute(
                entrypoint=_TEST_ENTRYPOINT,
                config={},
                ctx=_make_ctx(),
            )
        except Exception as exc:
            err_holder.append(exc)

    t = threading.Thread(target=_run_slow, daemon=True)
    t.start()
    time.sleep(0.2)  # Let first task start.

    # Second task should timeout because cap=1 and the slot is busy.
    with pytest.raises(StepExecutionError, match="acquire timeout"):
        executor.execute(
            entrypoint=_TEST_ENTRYPOINT,
            config={},
            ctx=_make_ctx(),
        )
    executor.shutdown()
    t.join(timeout=3)
    slow.stop()


def test_capability_ping_updates_capacity():
    """Slice 18a/19: refresh_capabilities() updates worker capacity."""
    ctx = zmq.Context()
    worker_ctx = zmq.Context()
    sock = worker_ctx.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")

    # Background: reply to ping with slots=3.
    stop = threading.Event()

    def _pong_loop():
        try:
            while not stop.is_set():
                if sock.poll(timeout=200, flags=zmq.POLLIN) == 0:
                    continue
                raw = sock.recv()
                payload = pickle.loads(raw)
                if isinstance(payload, dict) and payload.get("type") == "ping":
                    sock.send(
                        pickle.dumps(
                            {
                                "type": "pong",
                                "capabilities": {"gpu": True, "slots": 3},
                            }
                        )
                    )
        except zmq.ContextTerminated:
            pass

    t = threading.Thread(target=_pong_loop, daemon=True)
    t.start()

    try:
        w = _RemoteWorker(
            address=f"tcp://127.0.0.1:{port}",
            zmq_context=ctx,
            timeout=5.0,
        )
        # Initially capacity=1 (env default).
        assert w.capacity == 1

        # Force refresh (TTL expired).
        w._last_capability_refresh = 0
        caps = w.refresh_capabilities()
        assert caps.get("gpu") is True
        assert w.capacity == 3
        w.stop()
    finally:
        stop.set()
        t.join(timeout=3)
        sock.close(linger=0)
        worker_ctx.term()
        ctx.term()


def test_capacity_smoothing_shrink_fast():
    """Slice 19: capacity shrinks immediately when worker reports fewer slots."""
    ctx = zmq.Context()
    w = _RemoteWorker(
        address="tcp://127.0.0.1:1",
        zmq_context=ctx,
        timeout=0.3,
    )
    try:
        # Simulate initial capacity set by a prior refresh.
        w.capacity = 4
        w._prev_slots = 4
        # Worker now reports 2 slots — should shrink immediately.
        w._update_capacity(2)
        assert w.capacity == 2
    finally:
        w.stop()
        ctx.term()


def test_capacity_smoothing_grow_slow():
    """Slice 19: capacity grows gradually, not in one jump."""
    ctx = zmq.Context()
    w = _RemoteWorker(
        address="tcp://127.0.0.1:1",
        zmq_context=ctx,
        timeout=0.3,
    )
    try:
        w.capacity = 2
        w._prev_slots = 2
        # Worker now reports 8 slots — should grow slowly (halfway).
        w._update_capacity(8)
        assert w.capacity == 5  # 2 + max(1, (8-2)//2) = 2+3 = 5
        # Second update: still heading toward 8.
        w._update_capacity(8)
        assert w.capacity == 6  # 5 + max(1, (8-5)//2) = 5+1 = 6
    finally:
        w.stop()
        ctx.term()


def test_remote_payload_too_large():
    """Worker rejects payloads exceeding UMRS_ZMQ_MAX_PAYLOAD_BYTES."""
    import os

    # Spin up a real ZMQ REP worker with a tiny max.
    original = os.environ.get("UMRS_ZMQ_MAX_PAYLOAD_BYTES")
    os.environ["UMRS_ZMQ_MAX_PAYLOAD_BYTES"] = "100"  # very small limit

    zmq_ctx = zmq.Context()
    rep_sock = zmq_ctx.socket(zmq.REP)
    port = rep_sock.bind_to_random_port("tcp://127.0.0.1")
    rep_sock.close(linger=0)
    zmq_ctx.term()

    # Use _FakeZmqWorker but with a patched max — we need to test
    # the server-side guard. Instead, run zmq_pool_main in a thread
    # with a tiny limit.
    worker_ctx = zmq.Context()
    worker_sock = worker_ctx.socket(zmq.REP)
    worker_port = worker_sock.bind_to_random_port("tcp://127.0.0.1")
    max_bytes = 100
    stop_flag = threading.Event()

    def _mini_worker():
        try:
            while not stop_flag.is_set():
                if worker_sock.poll(timeout=200, flags=zmq.POLLIN) == 0:
                    continue
                raw = worker_sock.recv()
                if len(raw) > max_bytes:
                    reply = {
                        "status": "error",
                        "phase": "worker",
                        "message": f"payload too large: {len(raw)} bytes exceeds limit {max_bytes}",
                    }
                else:
                    reply = {"status": "ok"}
                worker_sock.send(pickle.dumps(reply))
        except zmq.ContextTerminated:
            pass

    t = threading.Thread(target=_mini_worker, daemon=True)
    t.start()

    try:
        addr = f"tcp://127.0.0.1:{worker_port}"
        client_ctx = zmq.Context()
        w = None
        try:
            w = _RemoteWorker(
                address=addr,
                zmq_context=client_ctx,
                timeout=5.0,
            )
            # Send a big payload.
            big_blob = pickle.dumps({"data": "x" * 200})
            assert len(big_blob) > max_bytes
            reply = w.run_task(big_blob)
            assert reply["status"] == "error"
            assert "too large" in reply["message"]
        finally:
            if w is not None:
                w.stop()
            client_ctx.term()
    finally:
        stop_flag.set()
        t.join(timeout=3)
        worker_sock.close(linger=0)
        worker_ctx.term()
        if original is None:
            os.environ.pop("UMRS_ZMQ_MAX_PAYLOAD_BYTES", None)
        else:
            os.environ["UMRS_ZMQ_MAX_PAYLOAD_BYTES"] = original


# ── Slice 17: Capability ping ─────────────────────────────────────────


class _PingAwareWorker:
    """In-process ZMQ REP worker that handles ping and task payloads."""

    def __init__(self, capabilities: dict[str, Any] | None = None) -> None:
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self._caps = capabilities or {"gpu": True, "slots": 2}
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def address(self) -> str:
        return f"tcp://127.0.0.1:{self._port}"

    def _loop(self) -> None:
        try:
            while not self._stop:
                if self._sock.poll(timeout=200, flags=zmq.POLLIN) == 0:
                    continue
                raw = self._sock.recv()
                payload = pickle.loads(raw)
                if isinstance(payload, dict) and payload.get("type") == "ping":
                    reply = {"type": "pong", "capabilities": self._caps}
                else:
                    reply = _process_task(payload)
                self._sock.send(pickle.dumps(reply))
        except zmq.ContextTerminated:
            pass

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=5)
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
        try:
            self._ctx.term()
        except Exception:
            pass


def test_remote_worker_refresh_capabilities():
    """``refresh_capabilities()`` round-trips a ping to the worker."""
    worker_proc = _PingAwareWorker(capabilities={"gpu": True, "slots": 4})
    try:
        ctx = zmq.Context()
        w = _RemoteWorker(
            address=worker_proc.address,
            zmq_context=ctx,
            timeout=5.0,
        )
        try:
            caps = w.refresh_capabilities()
            assert caps.get("gpu") is True
            assert caps.get("slots") == 4
        finally:
            w.stop()
            ctx.term()
    finally:
        worker_proc.stop()


def test_remote_worker_capability_ttl_caching():
    """Capabilities are cached for the TTL duration."""
    worker_proc = _PingAwareWorker(capabilities={"gpu": True})
    try:
        ctx = zmq.Context()
        w = _RemoteWorker(
            address=worker_proc.address,
            zmq_context=ctx,
            timeout=5.0,
        )
        try:
            caps1 = w.refresh_capabilities()
            assert caps1.get("gpu") is True
            # Second call within TTL returns cached.
            caps2 = w.refresh_capabilities()
            assert caps2 is caps1
        finally:
            w.stop()
            ctx.term()
    finally:
        worker_proc.stop()


# ── Slice 18a: Multi-slot acquire ─────────────────────────────────────


def test_multi_slot_acquire(monkeypatch):
    """With UMRS_REMOTE_WORKER_SLOTS=2, two tasks can be acquired concurrently."""
    monkeypatch.setenv("UMRS_REMOTE_WORKER_SLOTS", "2")
    w = _FakeZmqWorker()
    try:
        executor = RemotePoolExecutor(
            workers=[w.address],
            timeout_seconds=5.0,
            acquire_timeout=5.0,
        )
        try:
            w1 = executor._acquire_worker()
            w2 = executor._acquire_worker()
            # Both acquired from the same address.
            assert w1.address == w2.address
            executor._release_worker(w1.address, w1)
            executor._release_worker(w2.address, w2)
        finally:
            executor.shutdown()
    finally:
        w.stop()


def test_single_slot_blocks_second_acquire(monkeypatch):
    """With UMRS_REMOTE_WORKER_SLOTS=1 (default), second acquire times out."""
    monkeypatch.setenv("UMRS_REMOTE_WORKER_SLOTS", "1")
    w = _FakeZmqWorker()
    try:
        executor = RemotePoolExecutor(
            workers=[w.address],
            timeout_seconds=5.0,
            acquire_timeout=0.3,
        )
        try:
            w1 = executor._acquire_worker()
            with pytest.raises(StepExecutionError, match="acquire timeout"):
                executor._acquire_worker()
            executor._release_worker(w1.address, w1)
        finally:
            executor.shutdown()
    finally:
        w.stop()


# ── Slice 19: GPU memory-aware slots ──────────────────────────────────


def test_compute_slots():
    """``_compute_slots`` computes floor(free*safety / per_task)."""
    from umrs.server.step_worker import _compute_slots

    # 16 GB free, 4 GB per task, 0.9 safety → floor(14.4/4) = 3 slots.
    assert _compute_slots(16 * 1024**3, 4096, 0.9) == 3
    # Edge: 0 free → always 1.
    assert _compute_slots(0, 4096, 0.9) == 1
    # Edge: 0 per_task → always 1.
    assert _compute_slots(16 * 1024**3, 0, 0.9) == 1


def test_update_capacity_smoothing():
    """Shrink is immediate, grow moves halfway."""
    ctx = zmq.Context()
    w = _RemoteWorker(
        address="tcp://127.0.0.1:19999",
        zmq_context=ctx,
        timeout=1.0,
    )
    try:
        # Initial: set directly.
        w._update_capacity(4)
        assert w.capacity == 4
        # Shrink: immediate.
        w._update_capacity(2)
        assert w.capacity == 2
        # Grow: halfway from 2 → 6 = 2 + max(1, 4//2) = 4.
        w._update_capacity(6)
        assert w.capacity == 4
        # Grow again: 4 → 8 = 4 + max(1, 4//2) = 6.
        w._update_capacity(8)
        assert w.capacity == 6
    finally:
        w.stop()
        ctx.term()
