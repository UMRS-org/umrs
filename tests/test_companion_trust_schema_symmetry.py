"""U3 S1 dead code: the companion-trust schema creates what it drops.

``ensure_schema`` executed a byte-identical copy of migration 0098's
``chat_turn_approvals`` DDL that ``drop_schema`` never dropped. The copy is
gone (0098 owns the table); the tables created and dropped now match.
"""

from __future__ import annotations

import re

from umrs.companion_trust import schema


class _Cursor:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def execute(self, sql: str, *_args) -> None:
        self.log.append(sql)

    def fetchone(self):
        return None  # no schema_versions row yet: ensure_schema does its work


class _Conn:
    def __init__(self) -> None:
        self.log: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self.log)

    def commit(self) -> None:
        return None


def _tables(sql: list[str], pattern: str) -> set[str]:
    return {m for s in sql for m in re.findall(pattern, s, flags=re.I)}


def test_created_and_dropped_tables_match():
    conn = _Conn()
    assert schema.ensure_schema(conn) is True
    created = _tables(conn.log, r"CREATE TABLE IF NOT EXISTS (\w+)") - {"schema_versions"}
    dropped_conn = _Conn()
    schema.drop_schema(dropped_conn, confirm=True)
    dropped = _tables(dropped_conn.log, r"DROP TABLE IF EXISTS (\w+)")
    assert created == dropped
    assert "chat_turn_approvals" not in created
