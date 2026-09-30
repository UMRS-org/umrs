"""Pins for the data-port oversubscription contract (2026-09-30).

Bursty multi-user lanes (PROCESS_OBJECTS_BATCH) can exceed the server's
data-port pool.  The contract:

1. The server reaps IDLE client data queues at the contention point
   (``reap_idle_data_queues`` at failed allocation, then retries the
   allocation once) — an active client is never harvested (liveness is
   stamped on every data-plane receive; uncompleted tasks veto a reap).
2. The client treats NO_DATA_PORT_AVAILABLE as transient: bounded
   exponential backoff + jitter until ``UMRS_DATA_QUEUE_BUSY_TIMEOUT_S``
   elapses; past the deadline the historical RuntimeError fires.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from umrs.core.MessageQueue import MessageQueue


def _server_queue(task_manager=None) -> MessageQueue:
    """Bare server-role instance without opening ZMQ sockets."""
    mq = MessageQueue.__new__(MessageQueue)
    mq.role = "server"
    mq.type = "control"
    mq.task_manager = task_manager
    mq.client_dictionary = {}
    return mq


class _FakeDataQueue:
    def __init__(self, last_activity: float) -> None:
        self.last_activity_monotonic = last_activity
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _tasks(uncompleted_by_client: dict):
    class _TasksStatus:
        def all_uncompleted_task(self, client_id, _status_key):
            return uncompleted_by_client.get(client_id, [])

    return SimpleNamespace(tasks_status=_TasksStatus())


def test_reap_releases_idle_queue_and_port(monkeypatch):
    mq = _server_queue(task_manager=_tasks({}))
    now = time.monotonic()
    stale = _FakeDataQueue(now - 120)
    fresh = _FakeDataQueue(now)
    mq.client_dictionary["old"] = {"port": 50091, "data_queue": stale}
    mq.client_dictionary["new"] = {"port": 50092, "data_queue": fresh}

    reaped = mq.reap_idle_data_queues(idle_ttl_s=90)

    assert reaped == ["old"]
    assert stale.closed is True
    assert fresh.closed is False
    assert "port" not in mq.client_dictionary["old"]
    assert mq.client_dictionary["new"]["port"] == 50092


def test_reap_never_harvests_a_client_with_uncompleted_tasks():
    mq = _server_queue(task_manager=_tasks({"busy": ["task-1"]}))
    now = time.monotonic()
    queue = _FakeDataQueue(now - 3600)
    mq.client_dictionary["busy"] = {"port": 50091, "data_queue": queue}

    reaped = mq.reap_idle_data_queues(idle_ttl_s=90)

    assert reaped == []
    assert queue.closed is False
    assert mq.client_dictionary["busy"]["port"] == 50091


def test_reap_keeps_queue_when_status_lookup_fails():
    class _FailingStatus:
        def all_uncompleted_task(self, _client_id, _status_key):
            raise RuntimeError("pool exhausted")

    tm = SimpleNamespace(tasks_status=_FailingStatus())
    mq = _server_queue(task_manager=tm)
    queue = _FakeDataQueue(time.monotonic() - 3600)
    mq.client_dictionary["doubt"] = {"port": 50091, "data_queue": queue}

    reaped = mq.reap_idle_data_queues(idle_ttl_s=90)

    assert reaped == []
    assert queue.closed is False


def test_client_retries_busy_reply_then_connects(monkeypatch):
    mq = MessageQueue.__new__(MessageQueue)
    mq.role = "client"
    mq.type = "control"
    mq.tunnel_mode = False
    mq.server_ip = "127.0.0.1"
    mq.context = None

    replies = iter(["NO_DATA_PORT_AVAILABLE:50091-50098"] * 2 + ["tcp://127.0.0.1:50091"])
    monkeypatch.setattr(mq, "send_control_request", lambda *_a, **_k: None)
    monkeypatch.setattr(mq, "recv_string", lambda: (None, next(replies)))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    # No __new__ patching: a real client data MessageQueue only creates a
    # ZMQ PUSH socket on connect() (non-blocking, no server needed) — so
    # constructing one here is harmless and exercises the true path.
    data_mq = mq.get_data_queue("bench01")
    try:
        assert data_mq.port == 50091
        assert data_mq.role == "client"
        assert data_mq.type == "data"
    finally:
        data_mq.close()


def test_client_busy_retry_dies_with_runtime_error_past_deadline(monkeypatch):
    mq = MessageQueue.__new__(MessageQueue)
    mq.role = "client"
    mq.type = "control"
    mq.tunnel_mode = False
    mq.server_ip = "127.0.0.1"

    monkeypatch.setenv("UMRS_DATA_QUEUE_BUSY_TIMEOUT_S", "0")
    monkeypatch.setattr(mq, "send_control_request", lambda *_a, **_k: None)
    monkeypatch.setattr(mq, "recv_string", lambda: (None, "NO_DATA_PORT_AVAILABLE:50091-50098"))

    with pytest.raises(RuntimeError, match="No data ports available"):
        mq.get_data_queue("bench01")
