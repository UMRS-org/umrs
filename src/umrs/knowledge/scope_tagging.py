"""Tag the graph rows a scoped understand pass wrote (ADR-108 D-4, S-W3a).

The node and edge upserts (``KnowledgeStore.upsert_node``/``upsert_edge``,
four SQL shapes with typed and search-text variants) are left as they are.
A scoped pass writes its rows and then runs the two statements below **on
the same connection, inside the same transaction**
(``knowledge_adapters._write_scoped_rows``), so a row is never committed
proposed-but-untagged.

Only rows that are still untagged and proposed are tagged: an id that
already held an established or root row keeps it (an upsert never changes
a stored row's tier either), and a row another scope tagged first keeps
that scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: $1 project_id, $2 node ids (text[]), $3 scope path.
TAG_NODES_SQL = (
    "UPDATE knowledge_nodes SET scope_path = $3"
    " WHERE project_id = $1 AND node_id = ANY($2::text[])"
    " AND scope_path IS NULL AND status = 'proposed'"
)
#: $1 project_id, $2 edge ids (text[]), $3 scope path.
TAG_EDGES_SQL = (
    "UPDATE knowledge_edges SET scope_path = $3"
    " WHERE project_id = $1 AND edge_id = ANY($2::text[])"
    " AND scope_path IS NULL AND status = 'proposed'"
)


def updated_count(status: Any) -> int:
    """The row count of an asyncpg command status (``"UPDATE 3"`` gives 3).

    Anything else (None, ``""``, a status whose last word is not a
    non-negative integer such as ``"UPDATE"`` or ``"UPDATE x"``) gives 0.
    Surrounding whitespace is ignored.
    """
    raise NotImplementedError("s-w3a-02")


async def tag_graph_rows(
    db: Any,
    project_id: str,
    node_ids: Sequence[str],
    edge_ids: Sequence[str],
    scope_path: str,
) -> tuple[int, int]:
    """Tag the given untagged proposed rows with ``scope_path``;
    ``(nodes tagged, edges tagged)``.

    ``db`` is an asyncpg connection (or anything with
    ``async execute(sql, *args) -> str``); the caller owns the transaction.
    Runs ``TAG_NODES_SQL`` with ``(project_id, list(node_ids), scope_path)``
    only when ``node_ids`` is non-empty, then ``TAG_EDGES_SQL`` with
    ``(project_id, list(edge_ids), scope_path)`` only when ``edge_ids`` is
    non-empty; an empty list sends no statement and counts 0. Counts come
    from :func:`updated_count` of each command status. An empty
    ``scope_path`` raises ``ValueError`` before any statement (the root
    scope is never written as a tag). Errors from ``db`` propagate.
    """
    raise NotImplementedError("s-w3a-02")


__all__ = ["TAG_EDGES_SQL", "TAG_NODES_SQL", "tag_graph_rows", "updated_count"]
