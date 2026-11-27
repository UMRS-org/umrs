"""Live frontend-venue sessions per host at one site (ADR-098 D-9, SH-H2).

Placement's "least loaded" stage reads this map. A session belongs to a site
through its durable job (``jobs.venue``) and to a host through
``companion_venue_sessions.host_id``; live = provisioning, running, paused
(the states the hosts janitor ends ``host_lost``). Read under the venue's
start lock, so a parallel start counts this one.
"""

from __future__ import annotations

from typing import Any

LIVE_STATES = ("provisioning", "running", "paused")

_LOAD_SQL = (
    "SELECT s.host_id, count(*) FROM companion_venue_sessions s"
    " JOIN jobs j ON j.job_id = s.job_id"
    " WHERE j.venue = %s AND s.state = ANY(%s) GROUP BY s.host_id"
)


def host_load(pool: Any, venue: str) -> dict[str, int]:
    """``{host_id: live sessions}`` at ``venue`` ({} without a pool)."""
    if pool is None:
        return {}
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_LOAD_SQL, (venue, list(LIVE_STATES)))
            rows = cur.fetchall()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
    return {str(host): int(count) for host, count in rows}


__all__ = ["LIVE_STATES", "host_load"]
