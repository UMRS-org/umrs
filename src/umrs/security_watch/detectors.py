"""Deterministic SQL detectors behind a read-only role (ADR-088 D-3, D-4.1, D-6).

Each :class:`~umrs.security_watch.rules.Rule` becomes one parameterised
``SELECT``: the table, its columns and the ``group_by`` expression come from
the fixed maps below; event names, the time bound, ``where`` values, the
outcome and the threshold are bound parameters. No rule text is ever
spliced into SQL.

Read-only twice over:

- :func:`assert_read_only` refuses a connection whose role may write any
  table the rules read (``INSERT``/``UPDATE``/``DELETE``/``TRUNCATE``). The
  watch reads through the ``umrs_audit_reader`` grant
  (docs/admin/security/audit-reader-role.md), never through the owner role.
- Every detection runs inside ``SET TRANSACTION READ ONLY`` and is rolled
  back, so even a mis-granted role writes nothing. The writer check runs
  again inside that same transaction, so a role that only held for an
  earlier transaction (an uncommitted ``SET ROLE``) cannot carry the
  detection over to the owner role.

A hit carries ids and counts only: the rule id, the group value (an actor
id, a project id, or the edge's ``ipfp:`` fingerprint, never an address),
the row count, the time span and up to :data:`MAX_EVIDENCE` evidence ids.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from umrs.security_watch.rules import Rule, RulePack

MAX_EVIDENCE = 20
MAX_GROUPS = 100
WRITE_PRIVILEGES = ("INSERT", "UPDATE", "DELETE", "TRUNCATE")


@dataclass(frozen=True)
class _Table:
    name: str
    event_col: str
    id_expr: str
    ts_col: str
    ts_epoch: str
    since_expr: str
    group_cols: Mapping[str, str]
    outcome_col: str = ""


_TABLES: dict[str, _Table] = {
    "audit_events": _Table(
        name="audit_events",
        event_col="event",
        id_expr="event_id",
        ts_col="ts",
        ts_epoch="ts",
        since_expr="%s",
        group_cols={"actor": "actor", "project_id": "project_id", "outcome": "outcome"},
        outcome_col="outcome",
    ),
    "gateway_audit_log": _Table(
        name="gateway_audit_log",
        event_col="action",
        id_expr="id::text",
        ts_col="timestamp",
        ts_epoch="EXTRACT(EPOCH FROM timestamp)",
        since_expr="to_timestamp(%s)",
        group_cols={"client_ip": "client_ip", "user_id": "user_id", "path": "path"},
    ),
}


class WriterRoleRefused(PermissionError):
    """The detector connection's role may write an audit table (D-6)."""


@dataclass(frozen=True)
class Hit:
    """One rule hit: ids and counts only."""

    rule_id: str
    severity: str
    group_by: str
    group: str
    count: int
    first_ts: float
    last_ts: float
    evidence_ids: tuple[str, ...]


def _group_expr(table: _Table, group_by: str) -> tuple[str, list[Any]]:
    column = table.group_cols.get(group_by)
    if column is not None:
        return f"COALESCE({column}, '')", []
    name = group_by.removeprefix("details.")
    return "COALESCE(details->>%s, '')", [name]


def _filters(table: _Table, rule: Rule, since: float) -> tuple[str, list[Any]]:
    clauses = [f"{table.event_col} = ANY(%s)", f"{table.ts_col} >= {table.since_expr}"]
    params: list[Any] = [list(rule.events), since]
    if rule.outcome and table.outcome_col:
        clauses.append(f"{table.outcome_col} = %s")
        params.append(rule.outcome)
    for name, value in rule.where:
        clauses.append("details->>%s = %s")
        params.extend([name, value])
    return " AND ".join(clauses), params


def build_query(rule: Rule, now: float) -> tuple[str, list[Any]]:
    """The rule's ``SELECT`` and its parameters (no rule text in the SQL)."""
    table = _TABLES[rule.source]
    group, group_params = _group_expr(table, rule.group_by)
    where, where_params = _filters(table, rule, now - rule.window_seconds)
    sql = (
        f"SELECT {group} AS grp, COUNT(*) AS n, "
        f"MIN({table.ts_epoch}) AS first_ts, MAX({table.ts_epoch}) AS last_ts, "
        f"(ARRAY_AGG({table.id_expr} ORDER BY {table.ts_col} DESC))[1:{MAX_EVIDENCE}] AS ids "
        f"FROM {table.name} WHERE {where} "
        f"GROUP BY grp HAVING COUNT(*) >= %s "
        f"ORDER BY n DESC, grp LIMIT {MAX_GROUPS}"
    )
    return sql, [*group_params, *where_params, rule.threshold]


def _hit(rule: Rule, row: tuple[Any, ...]) -> Hit:
    group, count, first_ts, last_ts, ids = row
    return Hit(
        rule_id=rule.id,
        severity=rule.severity,
        group_by=rule.group_by,
        group=str(group or ""),
        count=int(count),
        first_ts=float(first_ts or 0.0),
        last_ts=float(last_ts or 0.0),
        evidence_ids=tuple(str(i) for i in (ids or ())),
    )


def _write_check_sql(tables: Iterable[str]) -> tuple[str, list[str]]:
    pairs = [(t, p) for t in tables for p in WRITE_PRIVILEGES]
    checks = " OR ".join("has_table_privilege(current_user, %s, %s)" for _ in pairs)
    return f"SELECT {checks}", [value for pair in pairs for value in pair]


def _refuse_writer(cur: Any, tables: list[str]) -> None:
    sql, params = _write_check_sql(tables)
    cur.execute(sql, params)
    if bool(cur.fetchone()[0]):
        raise WriterRoleRefused(
            f"security watch refuses a role that can write {tables}"
        )


def assert_read_only(conn: Any, sources: Iterable[str]) -> None:
    """Refuse a connection whose role may write any of ``sources``' tables."""
    tables = sorted({_TABLES[source].name for source in sources})
    if not tables:
        return
    try:
        with conn.cursor() as cur:
            _refuse_writer(cur, tables)
    finally:
        conn.rollback()


def run_rule(conn: Any, rule: Rule, now: float | None = None) -> list[Hit]:
    """Run one rule in a read-only transaction; the transaction is rolled back.

    An autocommit connection is refused: ``SET TRANSACTION READ ONLY`` would
    have no transaction to bind to there. A role that may write the rule's
    table is refused (:class:`WriterRoleRefused`) inside that transaction.
    """
    if getattr(conn, "autocommit", False):
        raise ValueError("security watch detectors need a transactional connection")
    sql, params = build_query(rule, time.time() if now is None else now)
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            _refuse_writer(cur, [_TABLES[rule.source].name])
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.rollback()
    return [_hit(rule, tuple(row)) for row in rows]


@dataclass(frozen=True)
class DetectionResult:
    """Hits of one pass, and the rules skipped for want of a connection."""

    rules_version: str
    hits: tuple[Hit, ...]
    skipped: tuple[str, ...]


def run_pack(
    pack: RulePack, conns: Mapping[str, Any], now: float | None = None
) -> DetectionResult:
    """Run every enabled rule whose source has a connection in ``conns``.

    ``conns`` maps a source (``audit_events``, ``gateway_audit_log``) to a
    connection of the read-only role; each is checked first and a writer
    role is refused (:class:`WriterRoleRefused`). A rule whose source has no
    connection is skipped by id, never silently dropped.
    """
    rules = pack.enabled()
    for source, conn in conns.items():
        assert_read_only(conn, [source])
    at = time.time() if now is None else now
    hits: list[Hit] = []
    for rule in rules:
        if rule.source in conns:
            hits.extend(run_rule(conns[rule.source], rule, at))
    skipped = tuple(rule.id for rule in rules if rule.source not in conns)
    return DetectionResult(
        rules_version=pack.rules_version, hits=tuple(hits), skipped=skipped
    )
