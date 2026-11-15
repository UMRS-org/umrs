"""Data-queue delivery: client close flushes, collectors wait on inactivity and
never read one PULL socket at once (port of upstream 8f9fe001; plan U1 A3/A5 review).

Real ZMQ sockets on 127.0.0.1, no server process; plaintext transport.
"""

from __future__ import annotations

import importlib
import logging
import re
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import zmq
from umrs.core.MessageQueue import MessageQueue

# The module, not the class that umrs.core re-exports under the same name.
mq_module = importlib.import_module("umrs.core.MessageQueue")


@pytest.fixture(autouse=True)
def _plain_env(monkeypatch):
    for name in ("UMRS_ZMQ_AUTH_TYPE", "UMRS_TUNNEL_MODE", "UMRS_SERVER_HOST",
                 "UMRS_DATA_LINGER_MS", "UMRS_S3_EVENTS_MAX_WAIT_S"):
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


class _ConcurrencyProbe:
    """recv_string stand-in: records how many callers are inside it at once."""

    def __init__(self):
        self.active = 0
        self.peak = 0
        self._guard = threading.Lock()
        self._counter = iter(range(10_000))

    def __call__(self):
        with self._guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self._guard:
            self.active -= 1
        return None, f"s3://b/k{next(self._counter)}"


def _collect_concurrently(queue, n_collectors, nb_objects):
    results: list[list[str]] = []
    threads = [
        threading.Thread(target=lambda: results.append(queue.wait_for_s3_events(nb_objects)))
        for _ in range(n_collectors)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    return results


def _is_contiguous(batch):
    numbers = [int(uri.rsplit("k", 1)[1]) for uri in batch]
    return numbers == list(range(numbers[0], numbers[0] + len(numbers)))


def test_concurrent_collectors_read_one_after_the_other(server):
    """Two PROCESS_OBJECTS batches of one client never recv on the socket at once."""
    probe = _ConcurrencyProbe()
    server.recv_string = probe

    results = _collect_concurrently(server, 2, 5)

    assert probe.peak == 1
    assert sorted(len(batch) for batch in results) == [5, 5]
    # Each collector got a contiguous run: batches stay in arrival order.
    assert all(_is_contiguous(batch) for batch in results)


# --- one client cannot hold a batch open (review U1) ------------------------------


def test_a_steady_trickle_stops_at_the_wall_clock_cap(monkeypatch, server):
    """The inactivity timeout alone let a client that pushes one URI per
    interval keep its collector (and every tenant batched with it) open."""
    monkeypatch.setenv("UMRS_S3_EVENTS_MAX_WAIT_S", "30")
    clock = _FakeClock()
    monkeypatch.setattr(mq_module, "time", SimpleNamespace(
        time=clock, monotonic=clock, sleep=time.sleep, perf_counter=time.perf_counter))
    counter = iter(range(10_000))

    def _trickle():
        if server.socket.RCVTIMEO == 0:  # the final non-blocking drain: nothing buffered
            return None, None
        clock.now += 1.0  # always under the 5 s inactivity timeout
        return None, f"s3://b/k{next(counter)}"

    server.recv_string = _trickle
    received = server.wait_for_s3_events(1_000, timeout_s=5)

    assert 25 <= len(received) <= 31


def _collect_in_thread(queue, nb_objects):
    results: list[list[str]] = []
    thread = threading.Thread(
        target=lambda: results.append(queue.wait_for_s3_events(nb_objects, timeout_s=1)),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=5)
    return thread, results


def test_a_collector_queued_behind_a_stuck_one_gives_up(monkeypatch, server):
    """The per-queue lock had no timeout: every later PROCESS_OBJECTS of a
    stuck client parked a thread on it for ever."""
    monkeypatch.setenv("UMRS_S3_EVENTS_MAX_WAIT_S", "1")
    server._s3_events_lock.acquire()
    try:
        thread, results = _collect_in_thread(server, 3)
        assert not thread.is_alive()
        assert results == [[]]
    finally:
        server._s3_events_lock.release()


def _parse_count(count):
    control = MagicMock()
    control._CmdError = MessageQueue._CmdError
    control._process_object_cmd.side_effect = lambda _client, n, _sender: ("PROCESS", n)
    return MessageQueue._parse_count_cmd(
        control, ["PROCESS_OBJECTS", "alice", str(count)], b"sender")


def test_process_objects_count_is_bounded():
    """A client-claimed count of 10**9 set no limit on the collector."""
    assert _parse_count(8000) == ("PROCESS", 8000)
    assert _parse_count(mq_module.MAX_PROCESS_OBJECTS) == (
        "PROCESS", mq_module.MAX_PROCESS_OBJECTS)
    for count in (mq_module.MAX_PROCESS_OBJECTS + 1, 10**9, -1):
        with pytest.raises(MessageQueue._CmdError):
            _parse_count(count)


# --- a full data queue says so (port of upstream 8f9fe001, review U1) ----------------


def test_a_full_data_queue_names_itself_and_the_dropped_uri(monkeypatch, caplog):
    monkeypatch.setenv("UMRS_DATA_LINGER_MS", "0")  # nothing to flush to
    port = _free_port()  # nobody binds it: the client pipe fills up to SNDHWM
    client = _queue("client", port, max_nb_messages=4)
    try:
        with caplog.at_level(logging.INFO, logger="umrs.core.MessageQueue"):
            with pytest.raises(RuntimeError) as err:
                for i in range(10_000):
                    client.notify_s3(f"s3://b/k{i}")
            client.close()
    finally:
        _dispose(client)

    message = str(err.value)
    assert "Data send timeout after 200ms" in message
    assert f"port {port}" in message
    assert "SNDHWM=4" in message
    assert "notify_s3 failed for s3://b/k" in caplog.text
    assert "object will NOT be indexed" in caplog.text
    # Review U1: the total counts only accepted sends, not the failed one.
    failed = int(re.search(r"notify_s3 failed for s3://b/k(\d+)", caplog.text).group(1))
    total = int(re.search(r"(\d+) message\(s\) sent in total", caplog.text).group(1))
    assert total == failed
