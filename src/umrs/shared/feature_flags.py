"""Platform feature-flag registry with enforced expiry (stability S11).

No permanent feature knobs: every flag registers with an owner, an
introduction reference, and an expiry date. Resolution goes through this
registry only — direct ``os.environ`` reads of registered names are a
premise violation (pinned by tests).

Semantics (fail-closed):
- unknown flag name → :class:`FeatureFlagExpired` is NOT raised; instead
  :class:`FeatureFlagError` — unregistered knobs do not exist;
- past ``expires`` → :class:`FeatureFlagExpired` on every access —
  the knob must be deleted or its expiry extended, with a wave-log entry;
- ``kind="bool"`` resolves via :func:`is_enabled` (truthy parsing);
- ``kind="value"`` resolves via :func:`get_value` (stripped or None).

Date override for tests: monkeypatch :func:`_today`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

_TRUE_VALUES = {"1", "true", "yes", "on"}


class FeatureFlagError(RuntimeError):
    """Unregistered flag, wrong kind, or malformed registry entry."""


class FeatureFlagExpired(FeatureFlagError):
    """Flag past its expiry date — delete the knob or extend it deliberately."""


@dataclass(frozen=True)
class FeatureFlag:
    name: str
    kind: str  # "bool" | "value"
    owner: str
    introduced_in: str
    expires: date
    purpose: str


def _d(iso: str) -> date:
    return date.fromisoformat(iso)


# Registry of flags introduced by the stability wave (S5+). Extending this
# table with legacy env knobs is welcome but out of S11's scope — recorded
# in the wave log.
#
# 2026-09-25: lane-schema knobs extended 2026-09-30 → 2026-12-31 (wave ledger,
# production enablement). S9 has not moved any table; no deployment sets them
# and no runtime path reads them yet, but the lapse would turn every
# lane_db_schema() call into FeatureFlagExpired. Retire with S9, or delete.
FLAGS: dict[str, FeatureFlag] = {
    "UMRS_LANE_SCHEMAS_ENABLED": FeatureFlag(
        name="UMRS_LANE_SCHEMAS_ENABLED",
        kind="bool",
        owner="stability-wave",
        introduced_in="S5 (lane scaffold)",
        expires=_d("2026-12-31"),
        purpose="Rollout switch: per-lane Postgres schema resolution. Must "
        "retire when every deployment runs lane-resolved reads (S9 moves done).",
    ),
    "UMRS_LANE_SCHEMA_STABLE": FeatureFlag(
        name="UMRS_LANE_SCHEMA_STABLE",
        kind="value",
        owner="stability-wave",
        introduced_in="S5 (lane scaffold)",
        expires=_d("2026-12-31"),
        purpose="Staged-cutover schema override for the v1-stable lane.",
    ),
    "UMRS_LANE_SCHEMA_BETA": FeatureFlag(
        name="UMRS_LANE_SCHEMA_BETA",
        kind="value",
        owner="stability-wave",
        introduced_in="S5 (lane scaffold)",
        expires=_d("2026-12-31"),
        purpose="Staged-cutover schema override for the v1beta lane.",
    ),
    "UMRS_LANE_SCHEMA_INTERNAL": FeatureFlag(
        name="UMRS_LANE_SCHEMA_INTERNAL",
        kind="value",
        owner="stability-wave",
        introduced_in="S5 (lane scaffold)",
        expires=_d("2026-12-31"),
        purpose="Staged-cutover schema override for the internal lane.",
    ),
}


def _today() -> date:
    return date.today()


def _flag(name: str) -> FeatureFlag:
    flag = FLAGS.get(name)
    if flag is None:
        raise FeatureFlagError(f"unregistered feature flag: {name!r}")
    if _today() > flag.expires:
        raise FeatureFlagExpired(
            f"feature flag {name!r} expired {flag.expires.isoformat()} "
            f"(introduced {flag.introduced_in}, owner {flag.owner}) — delete the "
            "knob or extend it with a wave-log entry"
        )
    return flag


def is_enabled(name: str) -> bool:
    """Resolve a registered boolean flag from the environment."""
    flag = _flag(name)
    if flag.kind != "bool":
        raise FeatureFlagError(f"{name!r} is a value flag; use get_value()")
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def get_value(name: str) -> str | None:
    """Resolve a registered value flag; stripped, None when unset/empty."""
    flag = _flag(name)
    if flag.kind != "value":
        raise FeatureFlagError(f"{name!r} is a bool flag; use is_enabled()")
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None
