"""Feature-flag registry tests (stability S11): expiry is enforced."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from umrs.shared import feature_flags
from umrs.shared.feature_flags import (
    FeatureFlagError,
    FeatureFlagExpired,
    get_value,
    is_enabled,
)


def _patch_today(monkeypatch: pytest.MonkeyPatch, day: date) -> None:
    monkeypatch.setattr(feature_flags, "_today", lambda: day)


# ── resolution ─────────────────────────────────────────────────────────────


def test_bool_flag_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UMRS_LANE_SCHEMAS_ENABLED", raising=False)
    assert is_enabled("UMRS_LANE_SCHEMAS_ENABLED") is False
    for truthy in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv("UMRS_LANE_SCHEMAS_ENABLED", truthy)
        assert is_enabled("UMRS_LANE_SCHEMAS_ENABLED") is True
    monkeypatch.setenv("UMRS_LANE_SCHEMAS_ENABLED", "0")
    assert is_enabled("UMRS_LANE_SCHEMAS_ENABLED") is False


def test_value_flag_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UMRS_LANE_SCHEMA_BETA", raising=False)
    assert get_value("UMRS_LANE_SCHEMA_BETA") is None
    monkeypatch.setenv("UMRS_LANE_SCHEMA_BETA", "  staged_beta  ")
    assert get_value("UMRS_LANE_SCHEMA_BETA") == "staged_beta"
    monkeypatch.setenv("UMRS_LANE_SCHEMA_BETA", "   ")
    assert get_value("UMRS_LANE_SCHEMA_BETA") is None


def test_kind_mismatch_raises() -> None:
    with pytest.raises(FeatureFlagError, match="value flag"):
        is_enabled("UMRS_LANE_SCHEMA_BETA")
    with pytest.raises(FeatureFlagError, match="bool flag"):
        get_value("UMRS_LANE_SCHEMAS_ENABLED")


def test_unknown_flag_raises() -> None:
    with pytest.raises(FeatureFlagError, match="unregistered"):
        is_enabled("UMRS_TOTALLY_UNKNOWN_FLAG")
    with pytest.raises(FeatureFlagError, match="unregistered"):
        get_value("UMRS_TOTALLY_UNKNOWN_FLAG")


# ── expiry enforcement ─────────────────────────────────────────────────────


def test_expired_flag_hard_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_LANE_SCHEMAS_ENABLED", "true")
    _patch_today(
        monkeypatch, feature_flags.FLAGS["UMRS_LANE_SCHEMAS_ENABLED"].expires + timedelta(days=1)
    )
    with pytest.raises(FeatureFlagExpired, match="expired"):
        is_enabled("UMRS_LANE_SCHEMAS_ENABLED")


def test_unexpired_flag_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_LANE_SCHEMAS_ENABLED", "true")
    _patch_today(monkeypatch, feature_flags.FLAGS["UMRS_LANE_SCHEMAS_ENABLED"].expires)
    assert is_enabled("UMRS_LANE_SCHEMAS_ENABLED") is True


def test_every_registered_flag_has_future_expiry_and_metadata() -> None:
    today = date.today()
    for name, flag in feature_flags.FLAGS.items():
        assert flag.name == name
        assert flag.expires >= today, f"{name}: registry entry already expired — extend or delete"
        assert flag.kind in ("bool", "value"), f"{name}: unknown kind {flag.kind}"
        assert flag.owner and flag.introduced_in and flag.purpose, (
            f"{name}: governance metadata incomplete"
        )


# ── consumer discipline: lanes resolves only through the registry ──────────


def test_lanes_module_reads_flags_only_via_registry() -> None:
    import inspect

    from umrs.shared import lanes

    src = inspect.getsource(lanes)
    assert "feature_flags.is_enabled" in src
    assert "feature_flags.get_value" in src
    assert "os.environ" not in src, "lanes.py must not read env directly (S11 premise)"
