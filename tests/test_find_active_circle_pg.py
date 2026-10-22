"""``find_active_circle`` on real Postgres (2026-10-22).

The access check refuses a ``circle:`` share naming no active circle and
hints at the one a case-only typo meant (``BDR/opossum`` → ``bdr/opossum``).
Runs with ``--db-url`` / ``UMRS_TEST_DB_URL``; a TEMP ``circles`` table
shadows the schema one, and the connection rolls back.
"""

from __future__ import annotations

import pytest
from umrs.api.circles import find_active_circle

pytestmark = pytest.mark.integration


@pytest.fixture
def conn(db_connection):
    cur = db_connection.cursor()
    cur.execute(
        "CREATE TEMP TABLE circles (project_id text, circle_id text, status text)"
        " ON COMMIT DROP"
    )
    cur.executemany(
        "INSERT INTO circles VALUES (%s, %s, %s)",
        [
            ("bdr", "opossum", "active"),
            ("lab", "old", "retired"),
            ("Mixed", "Case", "active"),
            ("mixed", "case", "active"),
        ],
    )
    return db_connection


def test_an_exact_active_circle_is_itself(conn):
    assert find_active_circle(conn, "bdr/opossum") == "bdr/opossum"


def test_a_case_only_typo_names_the_circle_it_meant(conn):
    assert find_active_circle(conn, "BDR/opossum") == "bdr/opossum"


def test_an_exact_match_wins_over_a_case_twin(conn):
    assert find_active_circle(conn, "Mixed/Case") == "Mixed/Case"
    assert find_active_circle(conn, "mixed/case") == "mixed/case"


def test_retired_and_unknown_circles_are_none(conn):
    assert find_active_circle(conn, "lab/old") is None
    assert find_active_circle(conn, "lab/none") is None
