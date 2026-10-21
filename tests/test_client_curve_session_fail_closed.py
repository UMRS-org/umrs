"""The umrs CLI's session-mode CURVE bootstrap fails closed, writes its secret
0600 from the first byte, and never reuses another identity's session.

c2-transport lane (2026-10-21), from the security lane's probes
(.stab-cycle2/security/zmq/repro_session.py E and F):

- E: with ``zmq.auth_type=session`` in the profile, a failed registration (or
  no JWT) was logged at DEBUG and the client went on with a NULL MessageQueue;
  a plaintext listener on the control port got ``HANDSHAKE`` and the command.
- F: ``curve-session.json`` was written under the umask (0644 with the CURVE
  secret key in it) and chmod'ed afterwards.
- ``_maybe_load_saved_curve_session`` reused any live file: after ``r4 login``
  as bob, bob's umrs commands rode alice's session (the server binds the
  session's user, so they ran as alice).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib
import json
import os
import stat
import threading
import time
from pathlib import Path
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
GW = "https://gw.example/rccs"


def _jwt(sub: str) -> str:
    def b64(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    claims = {"iss": "https://idp.example", "sub": sub, "exp": int(time.time()) + 3600}
    return f"{b64({'alg': 'RS256'})}.{b64(claims)}.c2ln"


def _subject(jwt: str) -> str:
    """r4.curve_session.session_subject, restated (iss + sub of a JWT)."""
    part = jwt.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    return hashlib.sha256(f"jwt|{claims['iss']}|{claims['sub']}".encode()).hexdigest()[:32]


class _Conn:
    def __init__(self, port: int = 0):
        self.gateway_url = GW
        self.zmq_auth_type = "session"
        self.profile_path = None
        self.profile_name = "default"
        self.server_host = "127.0.0.1"
        self.control_port = port
        self.tunnel_mode = False


class _Info:
    def __init__(self, sid: str):
        self.session_id = sid
        self.server_public_key = "S" * 40
        self.client_public_key = "P" * 40
        self.client_secret_key = sid[:1].upper() * 40
        self.expires_at = time.time() + 3600


@pytest.fixture()
def client(tmp_path, monkeypatch):
    for name in _ZMQ_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return importlib.import_module("client_umrs")


class _PlaintextListener:
    """A NULL-security ROUTER on 127.0.0.1 that records every frame it gets."""

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


def _probe_after_bootstrap(client, conn) -> None:
    """What the CLI does next: build the control MessageQueue and ask if the server is ready."""
    from umrs.core.MessageQueue import MessageQueue

    # The fixed CLI stops here; the probe below must then fail closed too.
    with contextlib.suppress(Exception):
        client._maybe_bootstrap_curve_session(conn)
    try:
        mq = MessageQueue(
            role="client",
            queue_type="control",
            server_ip="127.0.0.1",
            port=conn.control_port,
            tunnel_mode=False,
            prefer_env_server_host=False,
        )
    except Exception:  # noqa: BLE001 - refusing to build a socket is fail-closed
        return
    try:
        mq.is_server_ready(timeout_ms=800)
    finally:
        mq.close()


@pytest.mark.parametrize("jwt", [_jwt("alice"), None], ids=["registration-fails", "no-jwt"])
def test_failed_bootstrap_never_reaches_a_plaintext_listener(client, monkeypatch, jwt):
    if jwt:
        monkeypatch.setenv("UMRS_JWT_TOKEN", jwt)
    listener = _PlaintextListener()
    try:
        with mock.patch(
            "umrs.client.session_bootstrap.bootstrap_curve_session",
            side_effect=RuntimeError("gateway 502"),
        ):
            _probe_after_bootstrap(client, _Conn(listener.port))
        time.sleep(0.2)
    finally:
        listener.close()
    assert listener.received == []


def test_failed_bootstrap_says_why_and_keeps_session_mode(client, monkeypatch):
    monkeypatch.setenv("UMRS_JWT_TOKEN", _jwt("alice"))
    with (
        mock.patch(
            "umrs.client.session_bootstrap.bootstrap_curve_session",
            side_effect=RuntimeError("gateway 502"),
        ),
        pytest.raises(client.CurveSessionUnavailable, match="refusing a plaintext"),
    ):
        client._maybe_bootstrap_curve_session(_Conn())
    assert os.environ["UMRS_ZMQ_AUTH_TYPE"] == "session"


def test_no_jwt_in_session_mode_fails_closed(client):
    with pytest.raises(client.CurveSessionUnavailable, match="no valid JWT"):
        client._maybe_bootstrap_curve_session(_Conn())


@pytest.fixture()
def umask_022():
    old = os.umask(0o022)
    yield
    os.umask(old)


def test_session_file_is_never_on_disk_wider_than_0600(client, tmp_path, monkeypatch, umask_022):
    secret = "K" * 40
    seen: list[tuple[str, int]] = []
    real_chmod, real_replace = os.chmod, os.replace

    def snapshot() -> None:
        for path in Path(tmp_path, "r4").glob("*"):
            if path.is_file() and secret in path.read_text(errors="replace"):
                seen.append((path.name, stat.S_IMODE(path.stat().st_mode)))

    def chmod(path, mode, *a, **kw):
        snapshot()
        return real_chmod(path, mode, *a, **kw)

    def replace(src, dst, *a, **kw):
        snapshot()
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "chmod", chmod)
    monkeypatch.setattr(os, "replace", replace)
    info = _Info("k-session")
    info.client_secret_key = secret
    client._persist_curve_session(info, GW)
    snapshot()
    final = Path(tmp_path, "r4", "curve-session.json")
    assert stat.S_IMODE(final.stat().st_mode) == 0o600
    assert seen and all(mode == 0o600 for _, mode in seen), seen
    assert [p.name for p in final.parent.iterdir()] == ["curve-session.json"]


def _saved_session(tmp_path, owner_jwt: str, sid: str = "alice-session") -> dict:
    data = {
        "session_id": sid,
        "server_public_key": "S" * 40,
        "client_public_key": "P" * 40,
        "client_secret_key": "A" * 40,
        "expires_at": time.time() + 3600,
        "umrs_base_url": GW,
        "subject": _subject(owner_jwt),
    }
    path = Path(tmp_path, "r4", "curve-session.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def test_another_identitys_session_is_not_reused(client, tmp_path, monkeypatch):
    _saved_session(tmp_path, _jwt("alice"))  # r4 login as alice left this
    monkeypatch.setenv("UMRS_JWT_TOKEN", _jwt("bob"))
    with mock.patch(
        "umrs.client.session_bootstrap.bootstrap_curve_session", return_value=_Info("bob-session")
    ) as register:
        client._maybe_bootstrap_curve_session(_Conn())
    register.assert_called_once()
    assert os.environ.get("UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY") != "A" * 40
    saved = json.loads(Path(tmp_path, "r4", "curve-session.json").read_text())
    assert saved["session_id"] == "bob-session"
    assert saved["subject"] == _subject(_jwt("bob"))


def test_the_same_identitys_session_is_reused(client, tmp_path, monkeypatch):
    _saved_session(tmp_path, _jwt("alice"))
    monkeypatch.setenv("UMRS_JWT_TOKEN", _jwt("alice"))
    with mock.patch("umrs.client.session_bootstrap.bootstrap_curve_session") as register:
        client._maybe_bootstrap_curve_session(_Conn())
    register.assert_not_called()
    assert os.environ["UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY"] == "A" * 40


def test_a_session_registered_at_another_deployment_is_not_reused(client, tmp_path, monkeypatch):
    data = _saved_session(tmp_path, _jwt("alice"))
    data["umrs_base_url"] = "https://other.example/ai4s"
    Path(tmp_path, "r4", "curve-session.json").write_text(json.dumps(data))
    monkeypatch.setenv("UMRS_JWT_TOKEN", _jwt("alice"))
    assert client._maybe_load_saved_curve_session(_Conn()) is False


def test_subject_matches_r4():
    curve_session = pytest.importorskip("r4.curve_session")
    client = importlib.import_module("client_umrs")
    for credential in (_jwt("alice"), "opaque-api-key-0123"):
        assert client._session_subject(credential) == curve_session.session_subject(credential)
