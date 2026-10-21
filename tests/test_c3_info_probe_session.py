"""Cycle-3 hard lane (2026-10-21): ``umrs info`` never pings the control port
in plaintext in session mode.

Deferred by cycle 2 (ledger 46): the ``info`` probe built its control
MessageQueue after only *loading* a saved CURVE session. With none saved (a
fresh host, another identity's file, a different deployment) the queue was a
NULL socket, so the first readiness ping, ``HANDSHAKE``, went out in clear:
a listener on the control port read it, answered ``READY!``, and ``info``
reported the server reachable. Only then did it refresh the session. It now
sets the session up first (reuse or register, failing closed) and, when that
fails, reports the server as not probed instead of pinging it.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from unittest import mock

import pytest

zmq = pytest.importorskip("zmq")

_ZMQ_ENV = (
    "UMRS_ZMQ_AUTH_TYPE",
    "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY",
    "UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY",
    "UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY",
    "UMRS_JWT_TOKEN",
    "UMRS_API_KEY",
)
JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhbGljZSIsImlzcyI6Imh0dHBzOi8vaWRwIn0.c2ln"


class _Listener:
    """A NULL-security ROUTER that answers READY! to anything (a rogue server)."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            if self._sock.poll(50):
                frames = self._sock.recv_multipart()
                self.received.append(frames[-1].decode("utf-8", "replace"))
                self._sock.send_multipart([frames[0], b"READY!"])

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2)
        self._sock.close(0)
        self._ctx.term()


class _Conn:
    def __init__(self, port: int) -> None:
        self.gateway_url = "https://gw.example/rccs"
        self.zmq_auth_type = "session"
        self.profile_path = None
        self.profile_name = "default"
        self.server_host = "127.0.0.1"
        self.control_port = port
        self.tunnel_mode = False
        self.health_check_timeout_ms = 600


class _Info:
    session_id = "s-1"
    expires_at = time.time() + 3600

    def __init__(self) -> None:
        server_pk, _ = zmq.curve_keypair()
        client_pk, client_sk = zmq.curve_keypair()
        self.server_public_key = server_pk.decode()
        self.client_public_key = client_pk.decode()
        self.client_secret_key = client_sk.decode()


def _register(**_kw: str) -> _Info:
    """bootstrap_curve_session's success: keys in the env (set_env=True)."""
    info = _Info()
    os.environ["UMRS_ZMQ_AUTH_TYPE"] = "session"
    os.environ["UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY"] = info.server_public_key
    os.environ["UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY"] = info.client_public_key
    os.environ["UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY"] = info.client_secret_key
    return info


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for name in _ZMQ_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("UMRS_JWT_TOKEN", JWT)
    return importlib.import_module("client_umrs")


@pytest.fixture()
def listener():
    rogue = _Listener()
    yield rogue
    rogue.close()


def _status(client, listener, **patch_kw) -> str:
    with mock.patch("umrs.client.session_bootstrap.bootstrap_curve_session", **patch_kw):
        line = client._info_zmq_status(_Conn(listener.port))
    time.sleep(0.2)
    return line


def test_info_registers_before_it_pings_and_never_pings_in_clear(client, listener):
    line = _status(client, listener, side_effect=_register)
    assert listener.received == [], "a readiness ping went out in plaintext"
    assert "reachable" not in line.replace("not responding", "")


def test_info_does_not_ping_when_no_session_can_be_set_up(client, listener):
    line = _status(client, listener, side_effect=RuntimeError("gateway 502"))
    assert listener.received == []
    assert "not probed" in line and "gateway 502" in line
