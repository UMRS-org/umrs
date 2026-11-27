"""Build and reconcile the per-space HNSW indexes (ADR-083 D-2, slice KN-S6).

One partial expression index per registered space, ``hnsw_<space_id>``
(DDL in ``umrs.knowledge.embedding_spaces``: ``hnsw_index_ddl``,
``index_actions``). Each index carries a comment
(``embedding_spaces.index_signature``) naming the space, its dimension and
its names; a comment that no longer matches (legacy names changed) or an
invalid index (a failed CONCURRENTLY build) is dropped and rebuilt, and a
retired space's index is dropped.

Builds are an operator step, never a migration: ``umrs-server admin
embedding-space index [--space ID] [--dry-run]`` after a converge (hosted)
or after registering a space (self-hosted). ``CREATE INDEX CONCURRENTLY``
cannot run in a transaction block, so :func:`apply_index_work` needs an
autocommit connection (the CLI's ``index_conn``).

Synchronous, psycopg2-shaped connections; plain cursors (``conn.cursor()``,
rows are tuples). SQL rules: placeholders only for values; the DDL comes
from ``embedding_spaces`` (names validated there).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from umrs.knowledge.embedding_spaces import (
    EmbeddingSpace,
    IndexState,
    index_actions,
    uncovered_names,
)

#: ``(relname, indisvalid, comment)`` of the named indexes; one parameter,
#: the list of index names.
INDEX_STATE_SQL = (
    "SELECT c.relname, i.indisvalid, obj_description(c.oid, 'pg_class') "
    "FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid "
    "WHERE c.relkind = 'i' AND c.relname = ANY(%s)"
)
#: Every model name vectors are stored under; no parameter.
STORED_NAMES_SQL = (
    "SELECT DISTINCT model_name FROM knowledge_node_embeddings ORDER BY model_name"
)
CODE_NEEDS_AUTOCOMMIT = "knowledge.embedding_space_index_needs_autocommit"

Actions = Callable[[EmbeddingSpace, IndexState], tuple[str, ...]]
Uncovered = Callable[[EmbeddingSpace, Iterable[str]], tuple[str, ...]]


@dataclass(frozen=True)
class IndexWork:
    """The statements one space's index needs, in order."""

    space_id: str
    index_name: str
    statements: tuple[str, ...]


def read_index_states(
    conn: Any, spaces: Sequence[EmbeddingSpace]
) -> dict[str, IndexState]:
    """``{space_id: IndexState}`` for every space in ``spaces`` (task
    kn-s6-02).

    With no spaces: return ``{}`` WITHOUT touching ``conn``. Otherwise ONE
    statement: ``cur = conn.cursor()``; ``cur.execute(INDEX_STATE_SQL,
    ([s.index_name for s in spaces],))``; ``cur.fetchall()``. A row
    ``(relname, indisvalid, comment)`` belongs to the space whose
    ``index_name == relname`` and becomes ``IndexState(exists=True,
    valid=bool(indisvalid), comment=comment)`` (``comment`` as returned,
    None stays None). A space without a row gets ``IndexState()`` (not
    existing). Rows naming no space in ``spaces`` are ignored."""
    raise NotImplementedError("kn-s6-02")


def plan_index_work(
    spaces: Sequence[EmbeddingSpace],
    states: dict[str, IndexState],
    *,
    only: Sequence[str] | None = None,
    actions: Actions = index_actions,
) -> list[IndexWork]:
    """The work that brings every space's index up to date (task kn-s6-02).

    ``only`` (when not None/empty) restricts the plan to those space ids;
    an id in ``only`` that names no space in ``spaces`` raises
    ``EmbeddingSpaceError`` with code ``CODE_INVALID`` (import both from
    ``umrs.knowledge.embedding_spaces``) and a message naming the id.
    For each selected space, in the order of ``spaces``:
    ``statements = actions(space, states.get(space.space_id, IndexState()))``;
    a space with no statements is left out; else append
    ``IndexWork(space.space_id, space.index_name, statements)``."""
    raise NotImplementedError("kn-s6-02")


def apply_index_work(conn: Any, work: Sequence[IndexWork]) -> list[str]:
    """Run every statement of ``work`` in order; return them, in order (task
    kn-s6-02).

    ``conn.autocommit`` must be True (``getattr(conn, "autocommit", False)``):
    otherwise raise ``RuntimeError`` whose message starts with
    ``CODE_NEEDS_AUTOCOMMIT`` and run nothing. One ``cur = conn.cursor()``;
    ``cur.execute(statement)`` (no parameters) per statement. A failing
    statement propagates its exception (statements after it do not run)."""
    raise NotImplementedError("kn-s6-02")


def stored_model_names(conn: Any) -> list[str]:
    """Every distinct stored ``model_name`` (task kn-s6-02): ``cur =
    conn.cursor()``; ``cur.execute(STORED_NAMES_SQL)``; ``[str(row[0]) for
    row in cur.fetchall()]``."""
    raise NotImplementedError("kn-s6-02")


def coverage_warnings(
    spaces: Sequence[EmbeddingSpace],
    stored: Iterable[str],
    *,
    uncovered: Uncovered = uncovered_names,
) -> list[str]:
    """One line per ACTIVE space with stored names outside its index (task
    kn-s6-02).

    ``stored`` is read once into a list. For each space in order, skipping
    retired ones (``not space.active``): ``names = uncovered(space,
    stored_list)``; when non-empty append exactly
    ``f"{space.space_id}: vectors stored under {', '.join(names)} are outside "
    f"{space.index_name}; register them as legacy names and rerun index"``."""
    raise NotImplementedError("kn-s6-02")
