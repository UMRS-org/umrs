"""Cycle-3 client-server-compat#2 (2026-10-22): the route-level refusal of a
signed export on a server without UMRS_SECRET_KEY names a client command
that exists.

The message told users to pass a bare ``--allow-unsigned``; neither
``r4 project export`` nor ``r4 knowledge export`` had that flag, so a user
who followed the remedy got argparse's "unrecognized arguments" and could
not export at all. r4 now has ``r4 project export --allow-unsigned``; the
parent repo's tests/test_c3_export_unsigned_flag_compat.py parses the
command this message names with the real r4 parser.
"""

from __future__ import annotations

import re

import pytest
from fastapi import HTTPException
from umrs.api.knowledge import ExportRequest, _refuse_unsignable_export


def _refusal(monkeypatch: pytest.MonkeyPatch, **body: object) -> HTTPException | None:
    monkeypatch.delenv("UMRS_SECRET_KEY", raising=False)
    try:
        _refuse_unsignable_export(ExportRequest(project_id="pa", **body))
    except HTTPException as exc:
        return exc
    return None


def test_keyless_signed_export_is_a_typed_503(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = _refusal(monkeypatch)
    assert exc is not None and exc.status_code == 503
    assert (exc.headers or {}).get("x-platform-error-code") == "knowledge.unavailable"


def test_refusal_names_the_real_r4_command(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = _refusal(monkeypatch)
    assert exc is not None
    commands = re.findall(r"`(r4 [^`]+)`", str(exc.detail))
    assert commands == ["r4 project export --allow-unsigned"]
    assert "UMRS_SECRET_KEY" in str(exc.detail)  # the operator remedy stays


def test_explicit_opt_out_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _refusal(monkeypatch, allow_unsigned=True) is None


def test_configured_key_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_SECRET_KEY", "k")
    _refuse_unsignable_export(ExportRequest(project_id="pa"))
