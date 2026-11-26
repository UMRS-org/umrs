"""Admin-side reads and writes of the embedding-space registry (slice KN-S5).

Synchronous, on a psycopg2 connection (the admin CLI's
``umrs.cli.admin.cascade.cli_db_pool``). The caller owns the transaction:
these functions never call ``conn.commit()`` or ``conn.rollback()``.

SQL rules: placeholders only (``%s``), never values formatted into SQL;
column lists come from ``umrs.knowledge.embedding_spaces.COLUMNS`` and the
table name from ``TABLE``. Rows are read with a plain cursor
(``conn.cursor()``): a row is a tuple in ``COLUMNS`` order, so build the
mapping with ``dict(zip(COLUMNS, row, strict=True))`` before ``space_from_row``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from umrs.knowledge.embedding_spaces import (
    CODE_EXISTS,
    CODE_INVALID,
    COLUMNS,
    TABLE,
    VALID_STATUSES,
    EmbeddingSpace,
    EmbeddingSpaceError,
    require_names_free,
    space_from_row,
    space_to_row,
    validate_space,
)

_COLUMN_LIST = ", ".join(COLUMNS)
_PLACEHOLDERS = ", ".join(["%s"] * len(COLUMNS))
_SELECT_SQL = f"SELECT {_COLUMN_LIST} FROM {TABLE}"
_INSERT_SQL = f"INSERT INTO {TABLE} ({_COLUMN_LIST}) VALUES ({_PLACEHOLDERS})"
_REPLACE_SQL = (
    f"UPDATE {TABLE} SET "
    + ", ".join(f"{column} = %s" for column in COLUMNS[1:])
    + ", updated_at = now() WHERE space_id = %s"
)


def _as_space(row: Any) -> EmbeddingSpace:
    """One plain-cursor row as a validated space."""
    return space_from_row(dict(zip(COLUMNS, row, strict=True)))


def list_spaces(conn: Any, *, include_retired: bool = False) -> list[EmbeddingSpace]:
    """Registry rows ordered by ``space_id``; only active rows unless
    ``include_retired``.

    One statement: ``SELECT <COLUMNS> FROM <TABLE> ORDER BY space_id`` with
    NO parameters; ``fetchall()``; each row goes through
    ``space_from_row(dict(zip(COLUMNS, row, strict=True)))``; retired rows are dropped
    in Python (``space.active``) unless ``include_retired``."""
    cur = conn.cursor()
    cur.execute(f"{_SELECT_SQL} ORDER BY space_id")
    spaces = [_as_space(row) for row in cur.fetchall()]
    if include_retired:
        return spaces
    return [space for space in spaces if space.active]


def get_space(conn: Any, space_id: str) -> EmbeddingSpace | None:
    """The row with this ``space_id`` (any status), or None.

    One statement: ``SELECT <COLUMNS> FROM <TABLE> WHERE space_id = %s``
    with params ``(space_id,)``; ``fetchone()``."""
    cur = conn.cursor()
    cur.execute(f"{_SELECT_SQL} WHERE space_id = %s", (space_id,))
    row = cur.fetchone()
    return None if row is None else _as_space(row)


def register_space(conn: Any, space: EmbeddingSpace, *, replace: bool = False) -> bool:
    """Write ``space`` (``validate_space`` first). Returns True when a new
    row was inserted, False when an existing row was replaced.

    1. ``validate_space(space)``.
    2. Name check: load every row (``list_spaces(conn, include_retired=True)``).
       For every OTHER row (different ``space_id``), if any name of
       ``space_names(space)`` matches that row (``matches(row, name)``) or any
       name of ``space_names(row)`` matches ``space`` (``matches(space, n)``),
       raise ``EmbeddingSpaceError(CODE_NAME_TAKEN, ...)`` naming both
       space ids and the name.
    3. If a row with ``space.space_id`` exists: without ``replace`` raise
       ``EmbeddingSpaceError(CODE_EXISTS, ...)`` naming the id; with
       ``replace`` run one ``UPDATE <TABLE> SET <c> = %s, ... (every
       column of COLUMNS but space_id, in COLUMNS order), updated_at = now()
       WHERE space_id = %s`` with params
       ``(*space_to_row(space)[1:], space.space_id)`` and return False.
       (Existence is checked from the rows loaded in step 2.)
    4. Else one ``INSERT INTO <TABLE> (<COLUMNS>) VALUES (%s, ...)`` with
       ``space_to_row(space)`` and return True.
    """
    validate_space(space)
    rows = list_spaces(conn, include_retired=True)
    require_names_free(space, rows)
    exists = any(row.space_id == space.space_id for row in rows)
    if exists and not replace:
        raise EmbeddingSpaceError(
            CODE_EXISTS, f"embedding space {space.space_id!r} already exists"
        )
    cur = conn.cursor()
    if exists:
        cur.execute(_REPLACE_SQL, (*space_to_row(space)[1:], space.space_id))
        return False
    cur.execute(_INSERT_SQL, space_to_row(space))
    return True


def seed_spaces(conn: Any, spaces: Sequence[EmbeddingSpace]) -> list[str]:
    """Insert each space that is absent; return the inserted ``space_id``s
    in input order.

    Per space: ``validate_space``; one ``INSERT INTO <TABLE> (<COLUMNS>)
    VALUES (%s, ...) ON CONFLICT DO NOTHING`` with ``space_to_row(space)``;
    it counts as inserted when ``cursor.rowcount == 1``. Never updates an
    existing row (a retired row stays retired).

    Before any write, every space is checked against every row (any status)
    with ``require_names_free``: a space reusing another row's name raises
    ``CODE_NAME_TAKEN`` and nothing is inserted (review 2026-11-27: the
    seed made that name ambiguous for every writer, or brought a retired
    space's model back under a new id)."""
    rows = list_spaces(conn, include_retired=True)
    for space in spaces:
        validate_space(space)
        require_names_free(space, rows)
    inserted: list[str] = []
    for space in spaces:
        cur = conn.cursor()
        cur.execute(f"{_INSERT_SQL} ON CONFLICT DO NOTHING", space_to_row(space))
        if cur.rowcount == 1:
            inserted.append(space.space_id)
    return inserted


def set_status(conn: Any, space_id: str, status: str) -> bool:
    """``UPDATE <TABLE> SET status = %s, updated_at = now() WHERE space_id
    = %s`` with params ``(status, space_id)``; True when
    ``cursor.rowcount == 1``. ``status`` outside
    ``VALID_STATUSES`` raises ``EmbeddingSpaceError(CODE_INVALID, ...)``
    before any SQL runs."""
    if status not in VALID_STATUSES:
        raise EmbeddingSpaceError(
            CODE_INVALID, f"status {status!r} must be one of {sorted(VALID_STATUSES)}"
        )
    cur = conn.cursor()
    cur.execute(
        f"UPDATE {TABLE} SET status = %s, updated_at = now() WHERE space_id = %s",
        (status, space_id),
    )
    return bool(cur.rowcount == 1)
