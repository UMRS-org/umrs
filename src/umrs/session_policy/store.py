"""``session_policy_rules`` storage (migration 0125) and the limit lookups.

In memory when no pool is given (tests, dev), Postgres-backed otherwise:
the same dual shape as ``venues.venue_sessions.VenueSessionStore``. The
pool is psycopg2-shaped (``getconn``/``putconn``).

:class:`LimitsCache` and :func:`effective_limits_cached` serve the
inference proxy (ADR-099 D-5): 30 s per ``(venue, user, project)``,
bounded, read-only (``umrs_infer`` holds SELECT on the table).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .model import PolicyRule, check_rule_key
from .resolve import effective_limits, site_defaults

logger = logging.getLogger(__name__)

_COLUMNS = "venue, scope, subject, layer, doc, set_by, note, updated_at"
_SELECT = f"SELECT {_COLUMNS} FROM session_policy_rules"
_KEY_WHERE = "venue = %s AND scope = %s AND subject = %s AND layer = %s"


def _doc(value: Any) -> dict[str, Any]:
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def _rule(row: Any) -> PolicyRule:
    venue, scope, subject, layer, doc, set_by, note, updated_at = row
    return PolicyRule(
        venue=str(venue or ""),
        scope=str(scope),
        subject=str(subject or ""),
        layer=str(layer),
        doc=_doc(doc),
        set_by=str(set_by or ""),
        note=str(note or ""),
        updated_at=updated_at if isinstance(updated_at, datetime) else None,
    )


def _applies(rule: PolicyRule, venue: str, user_id: str, project_id: str) -> bool:
    if rule.venue not in (venue, ""):
        return False
    if rule.scope == "site":
        return True
    wanted = project_id if rule.scope == "project" else user_id
    return bool(wanted) and rule.subject == wanted


class PolicyRuleStore:
    """In-memory (``pool=None``) or Postgres-backed rule store."""

    def __init__(self, pool: Any | None = None) -> None:
        self._pool = pool
        self._rows: dict[tuple[str, str, str, str], PolicyRule] = {}
        self._lock = threading.Lock()

    # ── reads ─────────────────────────────────────────────────────────

    def get(
        self, venue: str, scope: str, subject: str, layer: str
    ) -> PolicyRule | None:
        if self._pool is None:
            return self._rows.get((venue, scope, subject, layer))
        rows = self._query(
            f"{_SELECT} WHERE {_KEY_WHERE}", (venue, scope, subject, layer)
        )
        return _rule(rows[0]) if rows else None

    def applicable(
        self, *, venue: str, user_id: str, project_id: str
    ) -> list[PolicyRule]:
        """Every rule that can bind a session of ``user_id`` in ``project_id``
        at ``venue`` (this site's rows and the all-sites rows)."""
        if self._pool is None:
            return [
                r
                for r in self._rows.values()
                if _applies(r, venue, user_id, project_id)
            ]
        rows = self._query(
            f"{_SELECT} WHERE venue IN (%s, '') AND ("
            "scope = 'site' OR (scope = 'project' AND subject = %s AND %s <> '') "
            "OR (scope = 'user' AND subject = %s AND %s <> ''))",
            (venue, project_id, project_id, user_id, user_id),
        )
        return [_rule(r) for r in rows]

    def list(
        self, *, venue: str | None = None, scope: str | None = None
    ) -> list[PolicyRule]:
        """Rules, filtered by venue (``''`` = the all-sites rows) and scope."""
        if self._pool is None:
            picked = [
                r
                for r in self._rows.values()
                if (venue is None or r.venue == venue)
                and (scope is None or r.scope == scope)
            ]
            return sorted(picked, key=lambda r: r.key)
        rows = self._query(
            f"{_SELECT} WHERE (%s IS NULL OR venue = %s) AND (%s IS NULL OR scope = %s) "
            "ORDER BY venue, scope, subject, layer",
            (venue, venue, scope, scope),
        )
        return [_rule(r) for r in rows]

    # ── writes ────────────────────────────────────────────────────────

    def put(self, rule: PolicyRule) -> PolicyRule:
        """Insert or replace the rule at its key (``updated_at`` = now)."""
        check_rule_key(rule.scope, rule.subject, rule.layer)
        if self._pool is None:
            stored = replace(rule, updated_at=datetime.now(timezone.utc))
            with self._lock:
                self._rows[rule.key] = stored
            return stored
        rows = self._query(
            "INSERT INTO session_policy_rules "
            "(venue, scope, subject, layer, doc, set_by, note, updated_at) "
            "VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, now()) "
            "ON CONFLICT (venue, scope, subject, layer) DO UPDATE SET "
            "doc = EXCLUDED.doc, set_by = EXCLUDED.set_by, note = EXCLUDED.note, "
            f"updated_at = now() RETURNING {_COLUMNS}",
            (*rule.key, json.dumps(rule.doc), rule.set_by, rule.note),
            commit=True,
        )
        return _rule(rows[0])

    def delete(self, venue: str, scope: str, subject: str, layer: str) -> bool:
        if self._pool is None:
            with self._lock:
                return self._rows.pop((venue, scope, subject, layer), None) is not None
        rows = self._query(
            f"DELETE FROM session_policy_rules WHERE {_KEY_WHERE} RETURNING 1",
            (venue, scope, subject, layer),
            commit=True,
        )
        return bool(rows)

    # ── pool plumbing ─────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = self._pool.getconn()
        try:
            yield conn
        except Exception:
            rollback = getattr(conn, "rollback", None)
            if callable(rollback):
                rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _query(self, sql: str, params: tuple, *, commit: bool = False) -> list:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = list(cur.fetchall())  # every statement here returns rows
            done = getattr(conn, "commit" if commit else "rollback", None)
            if callable(done):
                done()  # a read must not leave the conn idle in transaction
            return rows


def effective_limits_for(
    store: PolicyRuleStore,
    *,
    venue: str,
    user_id: str,
    project_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The effective limits of a session (store errors propagate: the
    caller fails closed with ``sessionpolicy.unavailable``)."""
    rules = store.applicable(venue=venue, user_id=user_id, project_id=project_id)
    return effective_limits(
        rules,
        venue=venue,
        user_id=user_id,
        project_id=project_id,
        defaults=site_defaults(env),
    )


class LimitsCache:
    """Effective limits per ``(venue, user, project)`` for ``ttl_s`` seconds;
    at most ``max_keys`` entries (the oldest goes first).

    A key whose rules cannot be re-read serves its last good limits (stale)
    instead of failing (ADR-099 D-5): a database blip must not turn every
    proxied call into a 503 while the limits it would read are unchanged.
    Only a key never read before fails (the caller fails closed). Both
    events are logged and counted in :attr:`stats` (``stale_served``,
    ``unavailable``)."""

    def __init__(
        self,
        store: PolicyRuleStore,
        ttl_s: float = 30.0,
        *,
        max_keys: int = 1024,
        clock: Callable[[], float] = time.monotonic,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._store = store
        self._ttl = ttl_s
        self._max = max_keys
        self._clock = clock
        self._env = env
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str, str], tuple[float, dict]] = (
            OrderedDict()
        )
        self.stats: dict[str, int] = {"stale_served": 0, "unavailable": 0}

    def get(self, *, venue: str, user_id: str, project_id: str) -> dict[str, Any]:
        key = (venue, user_id, project_id)
        now = self._clock()
        with self._lock:
            hit = self._entries.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        try:
            limits = effective_limits_for(
                self._store,
                venue=venue,
                user_id=user_id,
                project_id=project_id,
                env=self._env,
            )
        except Exception as exc:  # noqa: BLE001 — any read failure: last good, else re-raised
            return self._last_good(key, hit, exc)
        self._remember(key, now, limits)
        return limits

    def _last_good(
        self,
        key: tuple[str, str, str],
        hit: tuple[float, dict] | None,
        exc: Exception,
    ) -> dict[str, Any]:
        """The stale entry of ``key`` after a failed read, else re-raise."""
        name = "unavailable" if hit is None else "stale_served"
        with self._lock:
            self.stats[name] += 1
        logger.warning(
            "session policy rules unreadable for %s; %s",
            "/".join(key),
            "no cached limits (fail closed)"
            if hit is None
            else "serving last good limits",
            exc_info=exc,
        )
        if hit is None:
            raise exc
        return hit[1]

    def _remember(
        self, key: tuple[str, str, str], now: float, limits: dict[str, Any]
    ) -> None:
        with self._lock:
            self._entries[key] = (now + self._ttl, limits)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)


def effective_limits_cached(pool: Any) -> Callable[[str, str, str], dict[str, Any]]:
    """``(venue, user_id, project_id) -> effective limits``: the proxy's
    provider (ADR-099 D-5), cached 30 s."""
    cache = LimitsCache(PolicyRuleStore(pool))

    def limits(venue: str, user_id: str, project_id: str) -> dict[str, Any]:
        return cache.get(venue=venue, user_id=user_id, project_id=project_id)

    return limits
