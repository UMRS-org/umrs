"""Lane-schema table motion (stability S9).

Moves lane-owned tables between Postgres schemas per the S5 decision:
metadata-only ``ALTER TABLE ... SET SCHEMA`` inside ONE caller-managed
transaction (atomic: on any failure ``conn.rollback()`` leaves nothing
moved; PostgreSQL DDL is transactional).

Rollback is the symmetric inverse motion back to ``public`` — recorded
in the fresh runbook (docs/operations/lane-migration-runbook.md). Table
moves are per-family batches ordered by the runbook, never a bulk flip:
``UNCLASSIFIED`` families stay on ``public`` until placed (deliberate,
matching lanes.UNCLASSIFIED resolution).
"""

from __future__ import annotations

import re
from typing import Iterable, Protocol, Sequence

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

DEFAULT_LOCK_TIMEOUT_MS = 5000


class TableMotionError(RuntimeError):
    """Provider-agnostic motion failure (preflight or execution)."""


class _CursorLike(Protocol):
    def execute(self, query: str, params: object = None) -> object: ...
    def fetchone(self) -> object: ...
    def fetchall(self) -> list: ...


def _validate_identifier(value: str, kind: str) -> str:
    """Schemas/tables are system identifiers — strict allowlist, no quoting
    games possible beyond this point."""
    if not _IDENTIFIER.match(value or ""):
        raise TableMotionError(f"invalid {kind} identifier: {value!r}")
    return value


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _preflight(cur: _CursorLike, src: str, tgt: str, names: Sequence[str]) -> None:
    cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (tgt,))
    if cur.fetchone() is None:
        raise TableMotionError(
            f"target schema {tgt!r} does not exist — apply migration 0072 first"
        )
    cur.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = %s AND tablename = ANY(%s)",
        (src, list(names)),
    )
    present = {row[0] for row in cur.fetchall()}
    missing = [n for n in names if n not in present]
    if missing:
        raise TableMotionError(f"tables not found in schema {src!r}: {missing}")


def move_tables_to_schema(
    conn,
    tables: Iterable[str],
    *,
    target_schema: str,
    source_schema: str = "public",
    lock_timeout_ms: int = DEFAULT_LOCK_TIMEOUT_MS,
) -> list[str]:
    """Move ``tables`` source→target schema inside the caller's transaction.

    Preflight: target schema exists (0072 applied), every table exists in
    the source schema. Any failure raises ``TableMotionError`` before the
    first ``ALTER``; the caller rolls back. Returns the moved table names
    in the given order.
    """
    src = _validate_identifier(source_schema, "source schema")
    tgt = _validate_identifier(target_schema, "target schema")
    names = [_validate_identifier(t, "table") for t in tables]
    if not names:
        return []
    cur = conn.cursor()
    cur.execute(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}'")
    _preflight(cur, src, tgt, names)
    for name in names:
        cur.execute(
            f"ALTER TABLE {_quote_ident(src)}.{_quote_ident(name)} "
            f"SET SCHEMA {_quote_ident(tgt)}"
        )
    return names


def rollback_tables_to_public(
    conn,
    tables: Iterable[str],
    *,
    current_schema: str,
    lock_timeout_ms: int = DEFAULT_LOCK_TIMEOUT_MS,
) -> list[str]:
    """Inverse motion — the recorded rollback path (lane schema → public)."""
    return move_tables_to_schema(
        conn,
        tables,
        target_schema="public",
        source_schema=current_schema,
        lock_timeout_ms=lock_timeout_ms,
    )
