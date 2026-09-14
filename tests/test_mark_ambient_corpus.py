"""tools/mark_ambient_corpus.py pins (ADR-050 operator tool, R-02 wave 2).

Contracts locked with a mocked connection (no database):

- all three planes are audited and marked (knowledge_nodes,
  sat_knowledge_objects, catalog_objects — 0092 + 0093);
- nothing is ever marked without an explicit project list (audit-only default;
  no-action invocation is an error);
- ``--mark-project`` restores rows to the owner-only default tier;
- per-plane counts are printed for the audit and for the marks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "mark_ambient_corpus.py"


def _load():
    spec = importlib.util.spec_from_file_location("_mark_ambient_corpus", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Cursor:
    def __init__(self, audit_rows, update_rows):
        self.audit_rows = audit_rows  # {table: [(project, visibility, count)]}
        self.update_rows = update_rows  # {table: n rows updated}
        self.executed: list[tuple[str, tuple | None]] = []
        self._result: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        table = sql.split(" FROM ")[1].split()[0] if " FROM " in sql else sql.split()[1]
        if sql.startswith("SELECT"):
            self._result = list(self.audit_rows.get(table, []))
        else:
            self._result = [(1,)] * self.update_rows.get(table, 0)

    def fetchall(self):
        return self._result


class _Conn:
    def __init__(self, audit_rows=None, update_rows=None):
        self.cur = _Cursor(audit_rows or {}, update_rows or {})
        self.commits = 0
        self.closed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


@pytest.fixture
def tool(monkeypatch):
    module = _load()
    state = {"conn": _Conn()}
    monkeypatch.setattr(module, "_connect", lambda _dsn: state["conn"])
    module._test_state = state
    return module


def test_planes_include_catalog_objects(tool):
    assert [table for _plane, table in tool.PLANES] == [
        "knowledge_nodes",
        "sat_knowledge_objects",
        "catalog_objects",
    ]


def test_audit_covers_every_plane_and_reports_counts(tool):
    conn = _Conn(
        audit_rows={
            "knowledge_nodes": [("proj-a", "project", 5), ("proj-b", "ambient", 2)],
            "catalog_objects": [("proj-a", "system", 1)],
        }
    )
    rows = tool.audit(conn)
    assert ("knowledge_nodes", "proj-b", "ambient", 2) in rows
    assert ("catalog_objects", "proj-a", "system", 1) in rows
    tables = [sql.split(" FROM ")[1].split()[0] for sql, _ in conn.cur.executed]
    assert tables == ["knowledge_nodes", "sat_knowledge_objects", "catalog_objects"]
    counts = tool.plane_counts(rows)
    assert counts["knowledge_nodes"] == {"project": 5, "system": 0, "ambient": 2}
    assert counts["catalog_objects"] == {"project": 0, "system": 1, "ambient": 0}
    assert counts["sat_catalog"] == {"project": 0, "system": 0, "ambient": 0}


def test_no_action_is_an_error_and_never_connects(tool, capsys):
    rc = tool.main(["--dsn", "postgresql://x"])
    assert rc == 2
    assert "no action requested" in capsys.readouterr().err
    assert tool._test_state["conn"].cur.executed == []


def test_audit_only_marks_nothing(tool, capsys):
    conn = _Conn(audit_rows={"knowledge_nodes": [("proj-a", "project", 3)]})
    tool._test_state["conn"] = conn
    assert tool.main(["--dsn", "postgresql://x", "--audit"]) == 0
    assert all(sql.startswith("SELECT") for sql, _ in conn.cur.executed)
    assert conn.commits == 0
    out = capsys.readouterr().out
    assert "per-plane totals" in out and "knowledge_nodes    project=3" in out


def _updates(conn: _Conn) -> list[tuple[str, tuple | None]]:
    """``[(table, params)]`` of every UPDATE the tool executed, in order."""
    return [
        (sql.split()[1], params) for sql, params in conn.cur.executed if sql.startswith("UPDATE")
    ]


def test_mark_ambient_touches_every_plane_with_explicit_projects(tool, capsys):
    conn = _Conn(
        update_rows={"knowledge_nodes": 4, "sat_knowledge_objects": 1, "catalog_objects": 2}
    )
    tool._test_state["conn"] = conn
    assert tool.main(["--dsn", "postgresql://x", "--mark-ambient", "old-demo, seed"]) == 0
    expected_params = ("ambient", ["old-demo", "seed"], "ambient")
    assert _updates(conn) == [
        ("knowledge_nodes", expected_params),
        ("sat_knowledge_objects", expected_params),
        ("catalog_objects", expected_params),
    ]
    assert conn.commits == 1
    out = capsys.readouterr().out
    assert "marked: knowledge_nodes=4, sat_catalog=1, catalog_objects=2" in out
    assert "marked total: 7 row(s)" in out


def test_mark_project_restores_default_tier(tool):
    conn = _Conn(update_rows={"catalog_objects": 3})
    tool._test_state["conn"] = conn
    assert tool.main(["--dsn", "postgresql://x", "--mark-project", "old-demo"]) == 0
    updates = _updates(conn)
    assert updates and all(params[0] == "project" for _table, params in updates)
    assert conn.commits == 1


def test_mark_helpers_refuse_empty_lists_and_unknown_tiers(tool):
    conn = _Conn()
    assert tool.mark_projects(conn, [], "ambient") == {
        "knowledge_nodes": 0,
        "sat_catalog": 0,
        "catalog_objects": 0,
    }
    assert conn.cur.executed == []
    with pytest.raises(ValueError):
        tool._mark(conn, "knowledge_nodes", ["p"], "hidden")


def test_no_changes_means_no_commit(tool, capsys):
    conn = _Conn()  # every UPDATE returns 0 rows
    tool._test_state["conn"] = conn
    assert tool.main(["--dsn", "postgresql://x", "--mark-system", "handbook"]) == 0
    assert conn.commits == 0
    assert "marked" not in capsys.readouterr().out.split("per-plane totals")[1]


def test_lifecycle_report_lists_candidates_and_marks_nothing_by_default(tool, capsys, monkeypatch):
    audit_rows = {"knowledge_nodes": [("proj-dead", "project", 4)], "catalog_objects": []}
    tool2 = tool
    monkeypatch.setattr(tool2, "_connect", lambda _dsn: _Conn(audit_rows=audit_rows))
    # lifecycle fake: proj-dead has rows; NO live default graph row (candidate)
    monkeypatch.setattr(tool2, "lifecycle_candidates", lambda _conn: ["proj-dead"])
    rc = tool2.main(["--dsn", "postgresql://x", "--lifecycle-report"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lifecycle ambient candidates (1)" in out
    assert "proj-dead" in out
    conn = tool2._test_state["conn"]
    assert conn.commits == 0


def test_lifecycle_mark_requires_confirmation_phrase(tool, capsys, monkeypatch):
    tool2 = tool
    monkeypatch.setattr(tool2, "lifecycle_candidates", lambda _conn: ["proj-dead"])
    monkeypatch.setattr(
        tool2, "mark_projects", lambda conn, projects, tier: dict.fromkeys(dict(tool2.PLANES), 0)
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "ambient")
    rc = tool2.main(["--dsn", "postgresql://x", "--lifecycle-report", "--mark-ambient-lifecycle"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "marked (lifecycle)" in out
