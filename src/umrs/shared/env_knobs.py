"""``UMRS_*`` integer and on/off environment knobs, one rule set for all.

* unset or blank: the default;
* an integer knob whose value is not an integer, or falls outside its
  ``[lo, hi]`` bounds: one WARNING naming the knob, then the default (never a
  silent clamp, so a typo is visible and cannot stop the server);
* an on/off knob: ``0``, ``false``, ``no`` and ``off`` (any case) are off,
  any other value is on.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

OFF_VALUES = frozenset({"0", "false", "no", "off"})


def _parse_int(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _in_bounds(value: int, lo: int | None, hi: int | None) -> bool:
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def _bounds_text(lo: int | None, hi: int | None) -> str:
    if hi is None:
        return f">= {lo}" if lo is not None else "any integer"
    return f"{lo if lo is not None else '-inf'}..{hi}"


def env_int(name: str, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    """The integer value of *name*, or *default* (with a warning when invalid)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = _parse_int(raw)
    if value is None or not _in_bounds(value, lo, hi):
        logger.warning(
            "%s=%r is not an integer %s; using %s",
            name, raw, _bounds_text(lo, hi), default,
        )
        return default
    return value


def env_flag(name: str, default: bool = True) -> bool:
    """On/off knob: ``0/false/no/off`` is off, any other set value is on."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in OFF_VALUES
