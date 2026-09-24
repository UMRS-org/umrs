"""D-16 paired leg pins: connect-info emits s3cfg only behind
UMRS_EMIT_S3CFG=1 (legacy transition window)."""

from __future__ import annotations

import umrs.api.ops as ops


def test_emission_off_by_default(monkeypatch):
    monkeypatch.delenv("UMRS_EMIT_S3CFG", raising=False)
    monkeypatch.setattr(ops, "_get_or_create_user_s3cfg", lambda _u: "s3cfg-content")
    result: dict = {}
    ops._add_connect_info_extras(result, "alice")
    assert "s3cfg_content" not in result


def test_emission_opt_in_transition(monkeypatch):
    monkeypatch.setenv("UMRS_EMIT_S3CFG", "1")
    monkeypatch.setattr(ops, "_get_or_create_user_s3cfg", lambda _u: "s3cfg-content")
    result: dict = {}
    ops._add_connect_info_extras(result, "alice")
    assert result["s3cfg_content"] == "s3cfg-content"


def test_no_identity_never_emits(monkeypatch):
    monkeypatch.setenv("UMRS_EMIT_S3CFG", "1")
    monkeypatch.setattr(
        ops,
        "_get_or_create_user_s3cfg",
        lambda _u: (_ for _ in ()).throw(AssertionError("must not mint")),
    )
    result: dict = {}
    ops._add_connect_info_extras(result, None)
    assert "s3cfg_content" not in result
