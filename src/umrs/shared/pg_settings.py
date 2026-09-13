"""Postgres session hygiene — ONE place for connection kwargs (R-14).

Incident 2026-09-09: the ai4s server exhausted its Postgres connection slots
with blank ``application_name`` backends and 10–23 minute
``idle in transaction`` holders that nobody could attribute to a component.
Every psycopg2 connect / pool and every asyncpg pool in UMRS now takes its
session settings from this module so operators can see WHO holds a slot
(``pg_stat_activity.application_name``) and the server itself reaps
transactions left open by a stalled caller.

Settings:

- ``application_name`` = ``<prefix>-<component>`` where the prefix is
  ``UMRS_PG_APP_NAME`` (default ``umrs``) and the component names the
  construction site (``durable-jobs``, ``knowledge-asyncpg``, …).
- ``idle_in_transaction_session_timeout`` = ``UMRS_PG_IDLE_IN_TX_TIMEOUT_MS``
  (default 600000 ms = 10 min; ``0`` disables the setting entirely).

Deliberately NOT set here: ``statement_timeout``. Migrations, index builds
(``CREATE INDEX CONCURRENTLY``) and backfills legitimately run for longer
than any sane per-statement cap, and a global cap would kill them mid-flight.
A statement timeout, if ever wanted, belongs on the specific short-query
lanes, not in the shared connection kwargs.

Usage::

    psycopg2.connect(dsn, **pg_connect_kwargs("loop-store"))
    pg_pool.ThreadedConnectionPool(1, 10, dsn=dsn, **pg_connect_kwargs("sync-pool"))
    asyncpg.create_pool(dsn=dsn, **asyncpg_pool_kwargs("knowledge-asyncpg"))
"""

from __future__ import annotations

import os
import re
from typing import Any

ENV_APP_NAME_PREFIX = "UMRS_PG_APP_NAME"
ENV_IDLE_IN_TX_TIMEOUT_MS = "UMRS_PG_IDLE_IN_TX_TIMEOUT_MS"

DEFAULT_APP_NAME_PREFIX = "umrs"
DEFAULT_IDLE_IN_TX_TIMEOUT_MS = 600_000  # 10 minutes

# Postgres NAMEDATALEN is 64 → application_name is silently truncated at 63.
_APP_NAME_MAX_LEN = 63
_APP_NAME_SAFE = re.compile(r"[^A-Za-z0-9._:-]+")


def application_name(component: str) -> str:
    """``<prefix>-<component>``, sanitised to a safe identifier, ≤ 63 chars."""
    prefix = os.environ.get(ENV_APP_NAME_PREFIX, "").strip() or DEFAULT_APP_NAME_PREFIX
    component = _APP_NAME_SAFE.sub("-", component.strip()).strip("-") or "unnamed"
    return f"{prefix}-{component}"[:_APP_NAME_MAX_LEN]


def idle_in_transaction_timeout_ms() -> int:
    """Configured idle-in-transaction timeout in ms (``0`` = disabled).

    Non-integer or negative values fall back to the default rather than
    silently disabling the reaper.
    """
    raw = os.environ.get(ENV_IDLE_IN_TX_TIMEOUT_MS, "").strip()
    if not raw:
        return DEFAULT_IDLE_IN_TX_TIMEOUT_MS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_IDLE_IN_TX_TIMEOUT_MS
    return value if value >= 0 else DEFAULT_IDLE_IN_TX_TIMEOUT_MS


def pg_connect_kwargs(component: str) -> dict[str, Any]:
    """libpq keyword arguments for psycopg2 ``connect`` / pool constructors.

    Always carries ``application_name``; carries ``options`` with the
    idle-in-transaction timeout unless it is disabled (``0``).
    """
    kwargs: dict[str, Any] = {"application_name": application_name(component)}
    timeout_ms = idle_in_transaction_timeout_ms()
    if timeout_ms > 0:
        kwargs["options"] = f"-c idle_in_transaction_session_timeout={timeout_ms}"
    return kwargs


def asyncpg_server_settings(component: str) -> dict[str, str]:
    """``server_settings`` mapping for ``asyncpg.create_pool`` / ``connect``."""
    settings = {"application_name": application_name(component)}
    timeout_ms = idle_in_transaction_timeout_ms()
    if timeout_ms > 0:
        settings["idle_in_transaction_session_timeout"] = str(timeout_ms)
    return settings


def asyncpg_pool_kwargs(component: str, **overrides: Any) -> dict[str, Any]:
    """``create_pool`` kwargs with hygiene ``server_settings`` merged in.

    ``overrides`` are passed through; a caller-supplied ``server_settings``
    dict is merged on top of the hygiene settings (caller wins per key) so
    construction sites that already forward ``**kwargs`` stay compatible.
    """
    settings = asyncpg_server_settings(component)
    settings.update(overrides.pop("server_settings", None) or {})
    return {**overrides, "server_settings": settings}
