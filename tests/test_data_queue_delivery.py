"""Data-queue delivery: client close flushes, collectors wait on inactivity and
never read one PULL socket at once (port of upstream 8f9fe001; plan U1 A3/A5 review).

Real ZMQ sockets on 127.0.0.1, no server process; plaintext transport.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import zmq
from umrs.core.MessageQueue import MessageQueue

# The module, not the class that umrs.core re-exports under the same name.
mq_module = importlib.import_module("umrs.core.MessageQueue")


@pytest.fixture(autouse=True)
def _plain_env(monkeypatch):
    for name in ("UMRS_ZMQ_AUTH_TYPE", "UMRS_TUNNEL_MODE", "UMRS_SERVER_HOST",
                 "UMRS_DATA_LINGER_MS"):
        monkeypatch.delenv(name, raising=False)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _queue(role, port, **kw):
    return MessageQueue(role=role, queue_type="data", server_ip="127.0.0.1", port=port,
                        timeout_ms=200, prefer_env_server_host=False, **kw)


def _dispose(mq):
    mq.socket.close(linger=0)
    mq.context.destroy(linger=0)


@pytest.fixture
def server():
    queue = _queue("server", _free_port())
    yield queue
    _dispose(queue)


def test_client_close_flushes_queued_uris():
    """URIs still in the client pipe at close() reach the server (LINGER > 0).

    The server binds only after the client has closed, so every URI is still
    queued client-side when close() runs; LINGER 0 used to discard them all.
    """
    port = _free_port()
    client = _queue("client", port)
    uris = [f"s3://b/k{i}" for i in range(50)]
    for uri in uris:
        client.notify_s3(uri)
    client.close()

    server = _queue("server", port)
    try:
        assert server.wait_for_s3_events(len(uris), timeout_s=5) == uris
    finally:
        _dispose(server)
        client.context.destroy(linger=0)


def test_linger_and_high_water_marks_are_set_on_the_sockets(monkeypatch, server):
    monkeypatch.setenv("UMRS_DATA_LINGER_MS", "1500")
    client = _queue("client", server.port, max_nb_messages=4096)
    try:
        assert client.socket.getsockopt(zmq.LINGER) == 1500
        assert client.socket.getsockopt(zmq.SNDHWM) == 4096
        assert server.socket.getsockopt(zmq.RCVHWM) == server.max_msg
    finally:
        _dispose(client)


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_timeout_is_inactivity_not_wall_clock(monkeypatch, server):
    """A slow but steady stream is read to the end: each message resets the timer."""
    clock = _FakeClock()
    monkeypatch.setattr(mq_module, "time", SimpleNamespace(
        time=clock, monotonic=clock, sleep=time.sleep, perf_counter=time.perf_counter))
    sent = iter(f"s3://b/k{i}" for i in range(10))

    def _slow_recv():
        clock.now += 0.5  # one message every 0.5 s, 5 s in total
        return None, next(sent)

    server.recv_string = _slow_recv
    assert len(server.wait_for_s3_events(10, timeout_s=2)) == 10


def test_inactivity_timeout_returns_what_arrived(monkeypatch, server):
    clock = _FakeClock()
    monkeypatch.setattr(mq_module, "time", SimpleNamespace(
        time=clock, monotonic=clock, sleep=time.sleep, perf_counter=time.perf_counter))
    replies = iter([(None, "s3://b/a"), (None, "s3://b/b")])

    def _recv():
        clock.now += 1.0
        return next(replies, (None, None))

    server.recv_string = _recv
    assert server.wait_for_s3_events(5, timeout_s=3) == ["s3://b/a", "s3://b/b"]


def test_concurrent_collectors_read_one_after_the_other(server):
    """Two PROCESS_OBJECTS batches of one client never recv on the socket at once."""
    active, peak = [0], [0]
    guard = threading.Lock()
    counter = iter(range(10_000))

    def _recv():
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with guard:
            active[0] -= 1
        return None, f"s3://b/k{next(counter)}"

    server.recv_string = _recv
    results: list[list[str]] = []
    threads = [threading.Thread(target=lambda: results.append(server.wait_for_s3_events(5)))
               for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert peak[0] == 1
    assert sorted(len(batch) for batch in results) == [5, 5]
    # Each collector got a contiguous run: batches stay in arrival order.
    for batch in results:
        numbers = [int(uri.rsplit("k", 1)[1]) for uri in batch]
        assert numbers == list(range(numbers[0], numbers[0] + 5))
