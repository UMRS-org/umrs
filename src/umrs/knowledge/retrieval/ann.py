"""Filtered ANN over one embedding space (ADR-083 D-2, slice KN-S6).

The shared read shape of the knowledge vector lanes (the retrieval core's
dense lane and the K3 similarity lane of ``api/knowledge_query.py``):

1. the query model resolves to ONE registered space (:func:`query_space`;
   an empty registry stands for the bundled space, as on the write side);
2. the inner scan filters on that space's names with the exact predicate of
   its partial HNSW index (``embedding_spaces.space_filter_sql``), joins
   ``knowledge_nodes`` for scope, and orders by the index expression
   ``(e.embedding)::vector(<D>) <=> q`` with ``LIMIT max(k, 100)`` (the
   over-fetch);
3. ``hnsw.iterative_scan = relaxed_order`` lets the index scan continue past
   rows the join filters out, so a small project in a big space still fills
   its pool (pgvector >= 0.8.0);
4. the outer query re-sorts the pool by the EXACT cosine distance and keeps
   ``k``, so relaxed order never reaches the caller.

Without a built index the same statement is an exact scan: correct, slower.
The settings travel in the same ``execute`` as the SELECT (``SET LOCAL``
inside the caller's transaction), so a pooled connection never keeps them.

Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from umrs.knowledge.embedding_space_guard import SELECT_SPACES_SQL
from umrs.knowledge.embedding_spaces import (
    BUNDLED_SPACE,
    COLUMNS,
    EmbeddingSpace,
    normalize_name,
    require_space,
    space_filter_sql,
    space_from_row,
    space_names,
    vector_cast_sql,
)

#: ADR-083 D-2: the pool is ``max(k, OVERFETCH_FLOOR)`` rows.
OVERFETCH_FLOOR = 100
#: pgvector's ceiling for ``hnsw.ef_search``.
EF_SEARCH_MAX = 1000
CODE_DIMENSION = "knowledge.embedding_dimension_mismatch"

_FROM = (
    "FROM knowledge_node_embeddings e JOIN knowledge_nodes n ON n.node_id = e.node_id "
)


def overfetch(k: int) -> int:
    """The ANN pool size for a caller who keeps ``k``."""
    return max(int(k), OVERFETCH_FLOOR)


def settings_sql(pool_size: int) -> str:
    """The ``SET LOCAL`` prefix of an ANN read (ends with ``"; "``)."""
    ef_search = min(max(int(pool_size), OVERFETCH_FLOOR), EF_SEARCH_MAX)
    return (
        "SET LOCAL hnsw.iterative_scan = relaxed_order; "
        f"SET LOCAL hnsw.ef_search = {ef_search}; "
    )


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def ann_query(
    space: EmbeddingSpace,
    *,
    columns: Sequence[tuple[str, str]],
    where: str,
    where_params: Sequence[Any],
    vector: Sequence[float],
    k: int,
) -> tuple[str, list[Any]]:
    """``(sql, params)`` of one ANN read in ``space``.

    ``columns`` are ``(sql_expression, alias)`` pairs over ``e``/``n`` (one
    alias must be ``node_id``); the result rows are those columns in order,
    then ``distance`` (exact cosine distance), best first, ties by node id.
    ``where`` is the caller's scope over ``e``/``n`` (it binds
    ``where_params``); the space filter is added here. A vector whose length
    is not the space's dimension raises ``ValueError`` (named code).
    """
    if len(vector) != space.dimension:
        raise ValueError(
            f"{CODE_DIMENSION}: query vector has {len(vector)} dimensions, "
            f"space {space.space_id!r} has {space.dimension}"
        )
    inner_cols = ", ".join(f"{expr} AS {alias}" for expr, alias in columns)
    outer_cols = ", ".join(f"c.{alias}" for _expr, alias in columns)
    pool_size = overfetch(k)
    order = (
        f"{vector_cast_sql(space, 'e.embedding')} <=> "
        f"{vector_cast_sql(space, '%s::vector')}"
    )
    sql = (
        settings_sql(pool_size)
        + f"SELECT {outer_cols}, (c.ann_vec <=> %s::vector) AS distance FROM ("
        f"SELECT {inner_cols}, e.embedding AS ann_vec {_FROM}"
        f"WHERE {space_filter_sql(space, 'e')} AND {where} "
        f"ORDER BY {order} LIMIT %s"
        ") c ORDER BY distance ASC, c.node_id LIMIT %s"
    )
    literal = vector_literal(vector)
    return sql, [literal, *where_params, literal, pool_size, int(k)]


def count_query(
    space: EmbeddingSpace, *, where: str, where_params: Sequence[Any]
) -> tuple[str, list[Any]]:
    """``(sql, params)`` of the scope's vector count and latest
    ``updated_at`` in ``space`` (the dense lane's ``in_scope`` and the
    null's cache stamp)."""
    sql = (
        f"SELECT count(*), max(e.updated_at) {_FROM}"
        f"WHERE {space_filter_sql(space, 'e')} AND {where}"
    )
    return sql, list(where_params)


def fetch(pool: Any, sql: str, params: Sequence[Any]) -> list[tuple]:
    """Run one read on a psycopg2-shaped pool; the transaction ends (and the
    ``SET LOCAL`` settings with it) before the connection goes back."""
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return list(cur.fetchall())
    finally:
        pool.putconn(conn)


def read_spaces(pool: Any) -> list[EmbeddingSpace]:
    """Every registry row, any status (``SELECT_SPACES_SQL``, no params)."""
    rows = fetch(pool, SELECT_SPACES_SQL, ())
    return [space_from_row(dict(zip(COLUMNS, row, strict=True))) for row in rows]


def query_space(pool: Any, model: str) -> EmbeddingSpace:
    """The ACTIVE space the query encoder's ``model`` belongs to.

    An empty registry stands for the bundled space alone (the write side's
    rule), widened by :func:`covering`. No active space →
    ``UnregisteredEmbeddingModel``; two →
    ``EmbeddingSpaceError`` (ambiguous); a missing table raises the
    driver's error. Callers turn each into an ``unavailable`` lane.
    """
    spaces = read_spaces(pool) or [BUNDLED_SPACE]
    return covering(require_space(model, spaces), model)


def covering(space: EmbeddingSpace, model: str) -> EmbeddingSpace:
    """``space`` whose names also hold the query ``model``'s own name.

    The writer admits a model whose path only basename-matches a bare
    registered name and stores its vectors under that path, so the read
    must rank that name too (KN-S6 review). A covered model leaves the space
    unchanged (the read keeps the index predicate exactly); an uncovered one
    reads by exact scan until an admin registers it as a legacy name and
    reruns ``admin embedding-space index`` (which warns about it).
    """
    name = normalize_name(model)
    if not name or name in space_names(space):
        return space
    return replace(space, legacy_names=(*space.legacy_names, name))
