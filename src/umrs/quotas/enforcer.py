"""Token and rate quotas at the inference proxy (ADR-099 D-5).

Before the proxy unseals a provider key it asks ``QuotaEnforcer.check`` with
the call's lease. The lease owner's effective limits (ADR-099 D-3; cached
30 s per ``(venue, user, project)`` by the session-policy provider) decide:

- ``budget.tokens_per_day_user`` set and the user's ``infer_proxy`` tokens
  today at this venue >= it: ``quota.tokens_per_day_exceeded``, scope user;
- ``budget.tokens_per_day_project`` set and the project's >= it: the same
  code, scope project;
- ``budget.rpm_user`` set and the user's calls in the last 60 s >= it:
  ``quota.rpm_exceeded`` (an in-process sliding window per ``(venue, user)``;
  the bound is per proxy process, as for the per-session limiter).

All three answer 429 with ``x-platform-error-code`` and ``Retry-After``.
"Today" is the calendar day in ``UMRS_QUOTA_DAY_TZ`` (default UTC). With no
limit set (the default) nothing is queried. A limit that is set but whose
usage (or the limits themselves) cannot be read refuses as
``quota.unavailable`` (503): fail closed. The check runs before a call; a
call already in flight can end above the limit.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any

from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

ERROR_CODE_HEADER = "x-platform-error-code"
TOKENS_EXCEEDED = "quota.tokens_per_day_exceeded"
RPM_EXCEEDED = "quota.rpm_exceeded"
UNAVAILABLE = "quota.unavailable"

RPM_WINDOW_S = 60.0
UNAVAILABLE_RETRY_S = 30
_MAX_WINDOWS = 4096

LimitsFn = Callable[[str, str, str], dict[str, Any]]


@dataclass(frozen=True)
class QuotaDenial:
    code: str  # "quota.tokens_per_day_exceeded" | "quota.rpm_exceeded" | "quota.unavailable"
    status: int  # 429 | 503
    scope: str
    subject: str
    limit: int
    used: int
    window: str  # "day" | "minute"
    resets_at: str
    retry_after_s: int
    message: str

    def body(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "scope": self.scope,
            "subject": self.subject,
            "limit": self.limit,
            "used": self.used,
            "window": self.window,
            "resets_at": self.resets_at,
        }

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            self.body(),
            status_code=self.status,
            headers={
                ERROR_CODE_HEADER: self.code,
                "Retry-After": str(self.retry_after_s),
            },
        )


@dataclass(frozen=True)
class _Caller:
    venue: str
    user: str
    project: str

    @classmethod
    def of(cls, lease: Any) -> _Caller:
        return cls(
            venue=str(getattr(lease, "venue", "") or ""),
            user=str(getattr(lease, "owner_id", "") or ""),
            project=str(getattr(lease, "project", "") or ""),
        )

    @property
    def where(self) -> str:
        return self.venue or "this site"


def _limit(value: Any) -> int | None:
    """A set limit (int >= 0), else None (``null`` or absent = no limit)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_zone(name: str) -> tzinfo:
    """``UMRS_QUOTA_DAY_TZ`` as a tzinfo; unknown names fall back to UTC."""
    name = (name or "UTC").strip()
    if name.upper() == "UTC":
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — a bad name must not stop the proxy
        logger.warning("UMRS_QUOTA_DAY_TZ=%r unknown; quota days are UTC", name)
        return timezone.utc


def day_window(now_ts: float, zone: tzinfo) -> tuple[datetime, datetime]:
    """(start, end) of the calendar day holding ``now_ts`` in ``zone``."""
    now = datetime.fromtimestamp(now_ts, tz=zone)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Aware arithmetic is wall-clock: the next local midnight, DST included.
    return start, start + timedelta(days=1)


_SUBJECT_COLUMNS = {"user": "user_id", "project": "project_id"}


def sum_tokens(
    pool: Any, *, venue: str, scope: str, subject: str, since: datetime
) -> int:
    """``infer_proxy`` tokens of one user or project at ``venue`` since ``since``."""
    if pool is None:
        raise RuntimeError("usage ledger unavailable: no database pool")
    column = _SUBJECT_COLUMNS[scope]
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage_log"
                f" WHERE origin = 'infer_proxy' AND venue = %s AND {column} = %s"
                " AND event_time >= %s",
                (venue, subject, since),
            )
            row = cur.fetchone()
        conn.commit()
    finally:
        pool.putconn(conn)
    return int(row[0]) if row and row[0] is not None else 0


def policy_limits(
    pool: Any, *, venue: str, user_id: str, project_id: str = ""
) -> dict[str, Any]:
    """Effective limits from the session policy (lane P's resolver)."""
    from umrs.session_policy.store import PolicyRuleStore, effective_limits_for

    return effective_limits_for(
        PolicyRuleStore(pool), venue=venue, user_id=user_id, project_id=project_id
    )


class QuotaEnforcer:
    """The proxy's per-call quota check (one instance per proxy process)."""

    def __init__(
        self,
        pool: Any,
        limits_fn: LimitsFn,
        *,
        day_tz: str = "UTC",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._pool = pool
        self._limits_fn = limits_fn
        self._zone = day_zone(day_tz)
        self._clock = clock
        self._windows: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    # ── the check ─────────────────────────────────────────────────────

    def check(self, lease: Any) -> QuotaDenial | None:
        caller = _Caller.of(lease)
        try:
            limits = self._limits_fn(caller.venue, caller.user, caller.project)
        except Exception as exc:  # noqa: BLE001 — fail closed, named
            logger.warning("quota limits unreadable for %s: %s", caller.user, exc)
            return self._unavailable(caller, "session policy unreadable")
        budget = (limits or {}).get("budget") or {}
        denial = self._token_denial(caller, budget)
        if denial is None:
            denial = self._rpm_denial(caller, _limit(budget.get("rpm_user")))
        return denial

    def _token_denial(
        self, caller: _Caller, budget: dict[str, Any]
    ) -> QuotaDenial | None:
        for scope, subject in (("user", caller.user), ("project", caller.project)):
            limit = _limit(budget.get(f"tokens_per_day_{scope}"))
            if limit is None or not subject:
                continue
            try:
                used = self.tokens_today(
                    venue=caller.venue, scope=scope, subject=subject
                )
            except Exception as exc:  # noqa: BLE001 — fail closed, named
                logger.warning(
                    "token usage unreadable for %s %s: %s", scope, subject, exc
                )
                return self._unavailable(caller, "token usage unreadable")
            if used >= limit:
                return self._tokens_exceeded(caller, scope, subject, limit, used)
        return None

    def tokens_today(self, *, venue: str, scope: str, subject: str) -> int:
        start, _end = day_window(self._clock(), self._zone)
        return sum_tokens(
            self._pool, venue=venue, scope=scope, subject=subject, since=start
        )

    def _rpm_denial(self, caller: _Caller, limit: int | None) -> QuotaDenial | None:
        if limit is None:
            return None
        now = self._clock()
        with self._lock:
            window = self._window(caller, now)
            used = len(window)
            if used < limit:
                window.append(now)
                return None
            oldest = window[0] if window else now
        retry = max(1, math.ceil(oldest + RPM_WINDOW_S - now))
        return self._rpm_exceeded(caller, limit, used, retry, now)

    def _window(self, caller: _Caller, now: float) -> deque[float]:
        if len(self._windows) > _MAX_WINDOWS:
            self._prune(now)
        window = self._windows.setdefault((caller.venue, caller.user), deque())
        while window and window[0] <= now - RPM_WINDOW_S:
            window.popleft()
        return window

    def _prune(self, now: float) -> None:
        for key, window in list(self._windows.items()):
            if not window or window[-1] <= now - RPM_WINDOW_S:
                del self._windows[key]

    # ── the denials ───────────────────────────────────────────────────

    def _tokens_exceeded(
        self, caller: _Caller, scope: str, subject: str, limit: int, used: int
    ) -> QuotaDenial:
        _start, end = day_window(self._clock(), self._zone)
        resets_at = _iso(end)
        return QuotaDenial(
            code=TOKENS_EXCEEDED,
            status=429,
            scope=scope,
            subject=subject,
            limit=limit,
            used=used,
            window="day",
            resets_at=resets_at,
            retry_after_s=max(1, math.ceil(end.timestamp() - self._clock())),
            message=(
                f"daily token limit reached for {scope} {subject} at {caller.where}: "
                f"{used} of {limit} tokens; resets {resets_at}"
            ),
        )

    def _rpm_exceeded(
        self, caller: _Caller, limit: int, used: int, retry: int, now: float
    ) -> QuotaDenial:
        resets_at = _iso(datetime.fromtimestamp(now + retry, tz=timezone.utc))
        return QuotaDenial(
            code=RPM_EXCEEDED,
            status=429,
            scope="user",
            subject=caller.user,
            limit=limit,
            used=used,
            window="minute",
            resets_at=resets_at,
            retry_after_s=retry,
            message=(
                f"request rate limit reached for user {caller.user} at "
                f"{caller.where}: {used} of {limit} requests per minute; "
                f"retry in {retry}s"
            ),
        )

    def _unavailable(self, caller: _Caller, why: str) -> QuotaDenial:
        return QuotaDenial(
            code=UNAVAILABLE,
            status=503,
            scope="user",
            subject=caller.user,
            limit=0,
            used=0,
            window="day",
            resets_at="",
            retry_after_s=UNAVAILABLE_RETRY_S,
            message=(
                f"quota check unavailable for user {caller.user} at {caller.where}: "
                f"{why}; refused while a limit may apply"
            ),
        )


__all__ = [
    "ERROR_CODE_HEADER",
    "RPM_EXCEEDED",
    "TOKENS_EXCEEDED",
    "UNAVAILABLE",
    "LimitsFn",
    "QuotaDenial",
    "QuotaEnforcer",
    "day_window",
    "day_zone",
    "policy_limits",
    "sum_tokens",
]
