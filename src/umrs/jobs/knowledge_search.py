"""Cross-experiment knowledge search (MEM-C5).

Searches across experiment namespaces within a project with source
attribution and relevance scoring.

This module lives in ``umrs.jobs`` because it reads from the same Postgres
database and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSearchHit:
    """A single search result with source attribution."""

    memory_id: str
    content: str
    namespace: str
    project_id: str | None = None
    source: str = ""
    score: float = 0.0
    document_source_uri: str | None = None
    document_source_type: str | None = None
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "namespace": self.namespace,
            "project_id": self.project_id,
            "source": self.source,
            "score": self.score,
            "document_source_uri": self.document_source_uri,
            "document_source_type": self.document_source_type,
            "importance": self.importance,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeSearchResult:
    """Aggregated search results across namespaces."""

    query: str
    scope: str  # project | experiment | user
    project_id: str | None = None
    namespaces_searched: list[str] = field(default_factory=list)
    hits: list[KnowledgeSearchHit] = field(default_factory=list)
    total_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope,
            "project_id": self.project_id,
            "namespaces_searched": self.namespaces_searched,
            "hits": [h.to_dict() for h in self.hits],
            "total_hits": self.total_hits,
        }


def _memory_candidates(
    items: list[dict[str, Any]],
    query_lower: str,
    project_id: str | None,
    namespaces: list[str] | None,
) -> list[dict[str, Any]]:
    """In-memory substring match, narrowed by project and namespace filters."""
    candidates: list[dict[str, Any]] = []
    for item in items:
        if query_lower not in item.get("content", "").lower():
            continue
        if project_id and item.get("project_id") != project_id:
            continue
        if namespaces and item.get("namespace", "default") not in namespaces:
            continue
        candidates.append(item)
    return candidates


def _memory_hit(item: dict[str, Any]) -> KnowledgeSearchHit:
    importance = float(item.get("importance", 0.5))
    return KnowledgeSearchHit(
        memory_id=item.get("memory_id", ""),
        content=item.get("content", ""),
        namespace=item.get("namespace", "default"),
        project_id=item.get("project_id"),
        source=item.get("source", ""),
        score=importance,
        document_source_uri=item.get("document_source_uri"),
        document_source_type=item.get("document_source_type"),
        importance=importance,
        metadata=item.get("metadata") or {},
    )


class KnowledgeSearchEngine:
    """Searches memory_items across namespace partitions.

    Supports project-scoped search (across all experiment namespaces
    within a project) and experiment-scoped search (within a single
    experiment namespace).

    Uses text-based search (ILIKE) with importance-weighted scoring.
    When pgvector is available, can also do vector similarity search.
    """

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url = db_url
        self._psycopg2 = None
        if db_url:
            try:
                import psycopg2

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning("psycopg2 unavailable; search will use in-memory")
        # In-memory index for dev/test
        self._items: list[dict[str, Any]] = []

    @property
    def db_enabled(self) -> bool:
        return self._psycopg2 is not None and self._db_url is not None

    def _get_conn(self):
        from umrs.shared.pg_settings import pg_connect_kwargs

        return self._psycopg2.connect(
            self._db_url, **pg_connect_kwargs("knowledge-search")
        )

    def search(
        self,
        query: str,
        scope: str = "project",
        project_id: str | None = None,
        namespaces: list[str] | None = None,
        limit: int = 20,
    ) -> KnowledgeSearchResult:
        """Search across knowledge namespaces.

        Args:
            query: Search query text.
            scope: Search scope ('project', 'experiment', 'user').
            project_id: Project to scope search within.
            namespaces: Specific namespace names to search. If None,
                searches all namespaces visible to the project.
            limit: Maximum results.

        Returns:
            KnowledgeSearchResult with hits and attribution.
        """
        if self.db_enabled:
            return self._search_db(query, scope, project_id, namespaces, limit)
        return self._search_memory(query, scope, project_id, namespaces, limit)

    def add_item(self, item: dict[str, Any]) -> None:
        """Add an item to the in-memory index (for dev/test)."""
        self._items.append(item)

    # =========================================================================
    # Database search
    # =========================================================================

    def _search_db(
        self,
        query: str,
        scope: str,
        project_id: str | None,
        namespaces: list[str] | None,
        limit: int,
    ) -> KnowledgeSearchResult:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                conditions = ["content ILIKE %s"]
                params: list[Any] = [f"%{query}%"]

                if project_id:
                    conditions.append("project_id = %s")
                    params.append(project_id)

                if namespaces:
                    placeholders = ",".join(["%s"] * len(namespaces))
                    conditions.append(f"namespace IN ({placeholders})")
                    params.extend(namespaces)

                where = " AND ".join(conditions)
                params.append(limit)

                cur.execute(
                    f"""
                    SELECT memory_id, content, namespace, project_id,
                           source, importance, document_source_uri,
                           document_source_type, metadata
                    FROM memory_items
                    WHERE {where}
                    ORDER BY importance DESC, created_at DESC
                    LIMIT %s
                    """,
                    params,
                )

                hits = []
                searched_ns: set[str] = set()
                for row in cur.fetchall():
                    cols = [desc[0] for desc in cur.description]
                    d = dict(zip(cols, row))
                    searched_ns.add(d.get("namespace", "default"))
                    hits.append(
                        KnowledgeSearchHit(
                            memory_id=d["memory_id"],
                            content=d["content"],
                            namespace=d.get("namespace", "default"),
                            project_id=d.get("project_id"),
                            source=d.get("source", ""),
                            score=float(d.get("importance", 0.5)),
                            document_source_uri=d.get("document_source_uri"),
                            document_source_type=d.get("document_source_type"),
                            importance=float(d.get("importance", 0.5)),
                            metadata=d.get("metadata") or {},
                        )
                    )

                return KnowledgeSearchResult(
                    query=query,
                    scope=scope,
                    project_id=project_id,
                    namespaces_searched=sorted(searched_ns)
                    if searched_ns
                    else (namespaces or []),
                    hits=hits,
                    total_hits=len(hits),
                )
        finally:
            conn.close()

    # =========================================================================
    # In-memory search (dev/test fallback)
    # =========================================================================

    def _search_memory(
        self,
        query: str,
        scope: str,
        project_id: str | None,
        namespaces: list[str] | None,
        limit: int,
    ) -> KnowledgeSearchResult:
        candidates = _memory_candidates(
            self._items, query.lower(), project_id, namespaces
        )

        # Score by importance
        candidates.sort(key=lambda x: x.get("importance", 0.5), reverse=True)
        candidates = candidates[:limit]

        hits = [_memory_hit(item) for item in candidates]
        searched_ns = {hit.namespace for hit in hits}

        return KnowledgeSearchResult(
            query=query,
            scope=scope,
            project_id=project_id,
            namespaces_searched=sorted(searched_ns)
            if searched_ns
            else (namespaces or []),
            hits=hits,
            total_hits=len(hits),
        )
