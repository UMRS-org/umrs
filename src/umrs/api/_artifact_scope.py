"""The graph-readable predicate for artifact paths (knowledge-sharing K1.2).

The retrieve verb leaves out the graphs a caller may not read
(``rag_retrieve.graph_hidden``: another member's personal graph, a graph
whose ADR-080 list does not admit the caller). Artifacts carry no graph of
their own; their contributions (migration 0129) do. An artifact is readable
through a contribution in a graph that is not hidden, so the artifact paths
apply the same rule as retrieval and never list another member's
personal-graph file. Without the registry the rule cannot be applied, and
the path answers a typed 503 rather than an unfiltered list.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from starlette.concurrency import run_in_threadpool

from .typed_errors import PlatformError, raise_typed

_CONTRIBUTION_GRAPHS_SQL = (
    "SELECT DISTINCT graph_id FROM knowledge_artifact_contributions WHERE project_id = %s"
)


def hidden_artifact_graphs(request: Request, db: Any, project_id: str) -> tuple[str, ...]:
    """The graphs holding the project's contributions that the caller may not
    read (sorted). Registry reads only."""
    from umrs.knowledge.retrieval.dense import query_rows

    from .rag_retrieve import graph_hidden

    rows = query_rows(db, _CONTRIBUTION_GRAPHS_SQL, (project_id,))
    graph_ids = sorted({str(row[0]) for row in rows if row and row[0]})
    return tuple(g for g in graph_ids if graph_hidden(request, db, g, project_id))


def _registry_db(request: Request) -> Any:
    db = getattr(request.app.state, "umrs_db_pool", None)
    if db is None:
        raise_typed(
            PlatformError.knowledge2(
                "unavailable",
                "knowledge registry unavailable: artifact visibility cannot be checked",
            )
        )
    return db


async def hidden_graphs_for(request: Request, project_id: str) -> tuple[str, ...]:
    """:func:`hidden_artifact_graphs` off the event loop."""
    db = _registry_db(request)
    return await run_in_threadpool(hidden_artifact_graphs, request, db, project_id)


async def readable_artifact_ids(
    request: Request, project_id: str, artifact_ids: list[str] | None = None
) -> set[str]:
    """The project's artifacts this caller may see (optionally among
    *artifact_ids*): a contribution in a graph that is not hidden."""
    from umrs.knowledge.contributions import readable_artifact_ids as readable

    from ._asyncpg_pool import knowledge_asyncpg_pool

    hidden = await hidden_graphs_for(request, project_id)
    pool = await knowledge_asyncpg_pool(request)
    return await readable(pool, project_id, hidden, artifact_ids)
