"""KN-S6 task kn-s6-02: reading index state, planning and applying index work.

``umrs.knowledge.embedding_space_index`` over a fake psycopg2 connection;
the action function is injected, so these tests do not depend on task
kn-s6-01. The real DDL against Postgres is exercised in
``test_embedding_space_hnsw.py``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from kn_s6_pending import pending
from umrs.knowledge import embedding_space_index as esi
from umrs.knowledge.embedding_spaces import (
    CODE_INVALID,
    EmbeddingSpace,
    EmbeddingSpaceError,
    IndexState,
)

pytestmark = pending(
    "kn-s6-02",
    esi.read_index_states,
    esi.plan_index_work,
    esi.apply_index_work,
    esi.stored_model_names,
    esi.coverage_warnings,
)

A = EmbeddingSpace(space_id="aa", model_name="/models/aa", dimension=4)
B = EmbeddingSpace(space_id="bb", model_name="/models/bb", dimension=8)
C = EmbeddingSpace(space_id="cc", model_name="/models/cc", dimension=8, status="retired")


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    def execute(self, sql: str, params: Any = None) -> None:
        self.conn.executed.append((sql, params))
        if sql in self.conn.fail_on:
            raise RuntimeError(f"boom: {sql}")

    def fetchall(self) -> list[tuple]:
        return list(self.conn.rows)


class _Conn:
    def __init__(self, rows=(), *, autocommit=True, fail_on=()) -> None:
        self.rows = list(rows)
        self.autocommit = autocommit
        self.fail_on = set(fail_on)
        self.executed: list[tuple[str, Any]] = []
        self.cursors = 0

    def cursor(self) -> _Cursor:
        self.cursors += 1
        return _Cursor(self)


# ── read_index_states ───────────────────────────────────────────────────────


def test_states_come_from_one_catalog_read_keyed_by_space():
    conn = _Conn(rows=[("hnsw_aa", True, "sig-a"), ("hnsw_bb", False, None)])
    states = esi.read_index_states(conn, [A, B, C])
    assert conn.executed == [(esi.INDEX_STATE_SQL, (["hnsw_aa", "hnsw_bb", "hnsw_cc"],))]
    assert states == {
        "aa": IndexState(exists=True, valid=True, comment="sig-a"),
        "bb": IndexState(exists=True, valid=False, comment=None),
        "cc": IndexState(),
    }


def test_no_spaces_reads_nothing():
    conn = _Conn()
    assert esi.read_index_states(conn, []) == {}
    assert conn.executed == [] and conn.cursors == 0


def test_rows_of_other_indexes_are_ignored():
    conn = _Conn(rows=[("hnsw_zz", True, "x"), ("hnsw_aa", 1, "sig")])
    states = esi.read_index_states(conn, [A])
    assert states == {"aa": IndexState(exists=True, valid=True, comment="sig")}


def test_the_state_sql_reads_index_validity_and_comment():
    assert "pg_index" in esi.INDEX_STATE_SQL and "indisvalid" in esi.INDEX_STATE_SQL
    assert "obj_description" in esi.INDEX_STATE_SQL and "ANY(%s)" in esi.INDEX_STATE_SQL


# ── plan_index_work ─────────────────────────────────────────────────────────


def _actions(space: EmbeddingSpace, state: IndexState) -> tuple[str, ...]:
    if space.space_id == "bb":
        return ()
    return (f"do {space.space_id} exists={state.exists}",)


def test_the_plan_follows_the_space_order_and_skips_current_spaces():
    states = {"aa": IndexState(exists=True)}
    work = esi.plan_index_work([A, B, C], states, actions=_actions)
    assert work == [
        esi.IndexWork("aa", "hnsw_aa", ("do aa exists=True",)),
        esi.IndexWork("cc", "hnsw_cc", ("do cc exists=False",)),
    ]


def test_a_space_without_a_state_is_planned_as_missing():
    work = esi.plan_index_work([A], {}, actions=_actions)
    assert work[0].statements == ("do aa exists=False",)


@pytest.mark.parametrize("only", [None, []])
def test_no_selection_plans_every_space(only):
    assert len(esi.plan_index_work([A, C], {}, only=only, actions=_actions)) == 2


def test_only_restricts_the_plan():
    work = esi.plan_index_work([A, B, C], {}, only=["cc"], actions=_actions)
    assert [w.space_id for w in work] == ["cc"]


def test_an_unknown_space_in_only_is_refused_by_name():
    with pytest.raises(EmbeddingSpaceError) as raised:
        esi.plan_index_work([A, B], {}, only=["aa", "nope"], actions=_actions)
    assert raised.value.code == CODE_INVALID and "nope" in str(raised.value)


def test_the_default_actions_are_the_registry_rule():
    import inspect

    default = inspect.signature(esi.plan_index_work).parameters["actions"].default
    from umrs.knowledge.embedding_spaces import index_actions

    assert default is index_actions


# ── apply_index_work ────────────────────────────────────────────────────────

WORK = [
    esi.IndexWork("aa", "hnsw_aa", ("s1", "s2")),
    esi.IndexWork("cc", "hnsw_cc", ("s3",)),
]


def test_apply_runs_every_statement_in_order_without_params():
    conn = _Conn()
    assert esi.apply_index_work(conn, WORK) == ["s1", "s2", "s3"]
    assert conn.executed == [("s1", None), ("s2", None), ("s3", None)]


def test_apply_needs_an_autocommit_connection():
    conn = _Conn(autocommit=False)
    with pytest.raises(RuntimeError) as raised:
        esi.apply_index_work(conn, WORK)
    assert str(raised.value).startswith(esi.CODE_NEEDS_AUTOCOMMIT)
    assert conn.executed == []


def test_apply_refuses_a_connection_without_the_attribute():
    class Bare:
        def cursor(self):  # pragma: no cover - never reached
            raise AssertionError("must not run")

    with pytest.raises(RuntimeError):
        esi.apply_index_work(Bare(), WORK)


def test_a_failing_statement_stops_the_run():
    conn = _Conn(fail_on={"s2"})
    with pytest.raises(RuntimeError, match="boom"):
        esi.apply_index_work(conn, WORK)
    assert [sql for sql, _p in conn.executed] == ["s1", "s2"]


def test_no_work_runs_nothing():
    conn = _Conn()
    assert esi.apply_index_work(conn, []) == []
    assert conn.executed == []


# ── stored names and coverage ───────────────────────────────────────────────


def test_stored_model_names_reads_the_distinct_names():
    conn = _Conn(rows=[("/models/aa",), ("/x/aa",)])
    assert esi.stored_model_names(conn) == ["/models/aa", "/x/aa"]
    assert conn.executed == [(esi.STORED_NAMES_SQL, None)]


def _uncovered(space: EmbeddingSpace, stored) -> tuple[str, ...]:
    stored = list(stored)
    return tuple(n for n in stored if n.endswith(space.space_id) and n != space.model_name)


def test_coverage_warns_once_per_active_space_with_uncovered_names():
    stored = iter(["/models/aa", "/x/aa", "/y/aa", "/x/cc"])
    lines = esi.coverage_warnings([A, B, C], stored, uncovered=_uncovered)
    assert lines == [
        "aa: vectors stored under /x/aa, /y/aa are outside hnsw_aa; "
        "register them as legacy names and rerun index"
    ]


def test_coverage_is_quiet_when_everything_is_covered():
    assert esi.coverage_warnings([A, B], ["/models/aa"], uncovered=_uncovered) == []


def test_the_default_coverage_rule_is_the_registry_one():
    import inspect

    from umrs.knowledge.embedding_spaces import uncovered_names

    default = inspect.signature(esi.coverage_warnings).parameters["uncovered"].default
    assert default is uncovered_names


def test_a_retired_space_is_never_warned_about():
    active_c = replace(C, status="active")
    lines = esi.coverage_warnings([C], ["/x/cc"], uncovered=_uncovered)
    assert lines == []
    assert esi.coverage_warnings([active_c], ["/x/cc"], uncovered=_uncovered) != []
