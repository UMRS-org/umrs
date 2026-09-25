"""Snapshot-journal LWW on real Postgres (live remote-session receipt 2026-09-25).

``update_marker::timestamptz`` reads a naive marker as UTC, but r4 clients
wrote naive LOCAL time until 2026-09-25: a JST workstation's naive ``18:41``
outranked the UTC venue worker's ``09:41:50+00:00`` written ten seconds later,
so every worker save came back ``applied=False``. Against an offset-carrying
write a stored naive marker now reads in that write's offset (a session keeps
its owner's offset); everything else keeps the instant compare.

Env-gated like ``test_catalog_pg_integration``: runs where the CI job's
``POSTGRES_*`` variables (or ``UMRS_TEST_DB_URL``) point at a migrated DB.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from umrs.api.companion_sessions import upsert_snapshot_row

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_USER = "lww-pg-test"


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


pytestmark = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pytest.fixture()
def conn():
    import psycopg2

    connection = psycopg2.connect(_db_url())
    with connection.cursor() as cur:
        for name in ("0080_companion_session_snapshots.sql", "0084_companion_session_registry.sql"):
            cur.execute((_MIGRATIONS / name).read_text())
        cur.execute("DELETE FROM companion_session_snapshots WHERE user_id = %s", (_USER,))
    connection.commit()
    yield connection
    with connection.cursor() as cur:
        cur.execute("DELETE FROM companion_session_snapshots WHERE user_id = %s", (_USER,))
    connection.commit()
    connection.close()


def _put(conn, sid: str, marker: str) -> bool:
    return upsert_snapshot_row(
        conn, _USER, sid, name="n", payload=marker, updated_at=marker, venue="", state=""
    )


def _stored(conn, sid: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT update_marker FROM companion_session_snapshots"
            " WHERE user_id = %s AND session_id = %s",
            (_USER, sid),
        )
        return cur.fetchone()[0]


@pytest.mark.parametrize(
    ("first", "second", "applied"),
    [
        # the live defect: naive JST row, the worker's later write in the
        # owner's offset (a session keeps its offset)
        ("2026-09-25T18:41:40.123456", "2026-09-25T18:41:50.5+09:00", True),
        # the client's first save attaches its offset to the naive marker
        ("2026-09-25T18:41:40", "2026-09-25T18:41:40+09:00", True),
        # a stale offset write never clobbers newer naive content
        ("2026-09-25T19:00:00", "2026-09-25T18:50:00+09:00", False),
        ("2026-09-25T18:41:40.1+09:00", "2026-09-25T09:41:50.5+00:00", True),
        ("2026-09-25T18:41:40.1+09:00", "2026-09-25T09:41:30+00:00", False),
        ("2026-09-25T18:41:40", "2026-09-25T18:41:30", False),
        ("2026-09-25T18:41:40", "2026-09-25T18:41:50", True),
        ("2026-09-25T18:41:40+09:00", "2026-09-25T09:41:30", False),
        ("2026-09-25T18:41:40", "2026-09-25T18:41:50Z", True),
        ("2026-09-25T18:41:40", "2026-09-25T18:41:50+0900", True),
        ("", "2026-09-25T09:41:50+00:00", True),
        # the regex and the concat both see the trimmed value (a padded
        # offset marker once concatenated a second offset and raised)
        ("2026-09-25T18:41:40+09:00 ", "2026-09-25T18:41:50+09:00", True),
    ],
)
def test_lww_marker_upgrade(conn, first, second, applied):
    sid = f"s-{abs(hash((first, second)))}"
    _put(conn, sid, first)
    assert _put(conn, sid, second) is applied
    assert _stored(conn, sid) == (second if applied else first)
