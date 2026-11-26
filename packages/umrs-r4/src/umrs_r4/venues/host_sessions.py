"""The r4 extension's sessions on hosts (ADR-098 D-4, slice SH-H1a).

Core's hosts janitor (``umrs.hosts.janitor``) asks this provider which live
frontend-venue sessions run on a down host, and ends each ``host_lost``:

- the ``companion_venue_sessions`` row turns ``host_lost`` (compare-and-set on
  the live states, so a session that ended meanwhile keeps its own end);
- the session snapshot projects ``error``, like any failed session;
- open approval questions are cancelled (no worker polls them any more);
- one ``venue.session.stop`` audit row with cause ``host_lost``.

The janitor revokes the leases itself. A session belongs to a site through
its durable job (``jobs.venue``), and to a host through ``host_id`` (rows from
before the hosts registry read ``local``, the inline host). No teardown here:
the lost host is not this process's to drive; the session's status check
ends the durable job as failed.
"""

from __future__ import annotations

from typing import Any

from umrs.hosts.janitor import HostSession
from umrs.venues.session_audit import audit_session_stop

from umrs_r4.trust.chat_turn_questions import cancel_session_questions
from umrs_r4.venues.venue_sessions import project_session_state

LIVE_STATES = ("provisioning", "running", "paused")

_ON_HOST_SQL = (
    "SELECT s.session_id, s.user_id, COALESCE(j.project_id, '')"
    " FROM companion_venue_sessions s JOIN jobs j ON j.job_id = s.job_id"
    " WHERE s.host_id = %s AND j.venue = %s AND s.state = ANY(%s)"
    " ORDER BY s.started_at"
)
_LOSE_SQL = (
    "UPDATE companion_venue_sessions SET state = 'host_lost', reason = %s,"
    " updated_at = now(), stopped_at = now()"
    " WHERE session_id = %s AND state = ANY(%s) RETURNING user_id, job_id"
)


class VenueHostSessions:
    """``umrs.hosts.janitor.HostSessions`` over the frontend-venue rows."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    def sessions_on_host(self, venue: str, host_id: str) -> list[HostSession]:
        rows = self._query(_ON_HOST_SQL, (host_id, venue, list(LIVE_STATES)))
        return [HostSession(str(r[0]), str(r[1]), str(r[2])) for r in rows]

    def mark_host_lost(self, session_id: str, reason: str) -> bool:
        rows = self._query(_LOSE_SQL, (reason, session_id, list(LIVE_STATES)))
        if not rows:
            return False
        owner, job_id = str(rows[0][0]), str(rows[0][1] or "")
        project_session_state(self.pool, owner, session_id, "error")
        cancel_session_questions(self.pool, session_id=session_id, user_id=owner)
        audit_session_stop(
            cause="host_lost",
            actor="hosts-janitor",
            session_id=session_id,
            owner=owner,
            detail=reason,
            job_id=job_id,
        )
        return True

    def _query(self, sql: str, params: tuple) -> list[Any]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = list(cur.fetchall())
            conn.commit()
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)


def build_host_sessions(pool: Any) -> VenueHostSessions | None:
    """The ``add_host_sessions`` factory: None without a database."""
    return VenueHostSessions(pool) if pool is not None else None


__all__ = ["LIVE_STATES", "VenueHostSessions", "build_host_sessions"]
