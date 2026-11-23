"""D-16 / DEC-016 (2026-11-24) pins: connect-info never emits storage
credentials. ``UMRS_EMIT_S3CFG`` is retired: set, it only logs one warning.

Before DEC-016 the switch made connect-info read (and mint) the caller's
``s3cfg_<user>`` and return it as ``s3cfg_content``.
"""

from __future__ import annotations

import logging

import pytest
import umrs.api.ops as ops
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs_protocol.handshake_hmac import sign_claims

SECRET = "d16-connect-info-secret"
S3_SECRET = "s3-secret-d16"


@pytest.fixture
def server_s3cfg(monkeypatch, tmp_path):
    """A server-side s3cfg the old emission would have returned; switch set."""
    home = tmp_path / "home"
    cfg = home / ".config" / "umrs"
    cfg.mkdir(parents=True)
    (cfg / "s3cfg_alice").write_text(f"[default]\naccess_key = alice\nsecret_key = {S3_SECRET}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("UMRS_ADMIN_S3_CONFIGURATION_FILE", raising=False)
    monkeypatch.setenv("UMRS_EMIT_S3CFG", "1")
    monkeypatch.setattr(ops, "_EMIT_SWITCH_WARNED", [], raising=False)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(ops.router)
    return TestClient(app, raise_server_exceptions=False)


def _signed(user: str) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": "proj-a",
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, "proj-a", False),
    }


def _assert_no_credentials(resp) -> None:
    assert resp.status_code == 200, resp.text
    assert "s3cfg_content" not in resp.json()
    assert S3_SECRET not in resp.text


def test_signed_caller_gets_no_s3cfg_with_switch_set(server_s3cfg, client, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    _assert_no_credentials(client.get("/umrs/ops/connect-info", headers=_signed("alice")))


def test_mvp_header_gets_no_s3cfg_with_switch_set(server_s3cfg, client, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    _assert_no_credentials(client.get("/umrs/ops/connect-info", headers={"X-User-ID": "alice"}))


def test_set_switch_warns_once(server_s3cfg, caplog):
    with caplog.at_level(logging.WARNING, logger=ops.logger.name):
        for _ in range(3):
            ops._add_connect_info_extras({})
    warned = [r for r in caplog.records if "UMRS_EMIT_S3CFG" in r.getMessage()]
    assert len(warned) == 1
    assert "DEC-016" in warned[0].getMessage()


def test_unset_switch_is_silent(monkeypatch, caplog):
    monkeypatch.delenv("UMRS_EMIT_S3CFG", raising=False)
    monkeypatch.setattr(ops, "_EMIT_SWITCH_WARNED", [], raising=False)
    with caplog.at_level(logging.WARNING, logger=ops.logger.name):
        ops._add_connect_info_extras({})
    assert not [r for r in caplog.records if "UMRS_EMIT_S3CFG" in r.getMessage()]
