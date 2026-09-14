"""Catalog adapter for knowledge modules (verify, coverage, planner).

Wraps ``data_catalog`` search/read/list APIs behind a stable interface that
matches the ``KnowledgeStore`` protocol used by ``KnowledgeVerifier``.
"""

from __future__ import annotations

import logging
from typing import Any

from .artifacts import (
    artifact_id_for_meta,
    load_artifact_content,
    meta_to_artifact,
    read_artifact_text,
)
from .pg_store import CatalogMetaStore
from .schema import CatalogObjectMeta
from .scoring import allowed
from .store import read_structured_knowledge, write_structured_knowledge

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTENT_BYTES = 10 * 1024 * 1024


def _resolve_meta_store(meta_store: CatalogMetaStore | None) -> CatalogMetaStore:
    if meta_store is not None:
        return meta_store
    from umrs.api.catalog_store import get_catalog_meta_store

    return get_catalog_meta_store()


class CatalogArtifactAdapter:
    """Read-only catalog view for a single project.

    Usage:
        adapter = CatalogArtifactAdapter(s3_client, bucket, project_id="proj-1")
        artifacts = adapter.list_artifacts()
        hits = adapter.search("batch-3 csv")
        text = adapter.read_text("proj-1/uploads/data/batch-3.csv")
    """

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        project_id: str,
        *,
        prefix: str | None = None,
        include_pending: bool = True,
        max_content_bytes: int = _DEFAULT_MAX_CONTENT_BYTES,
        meta_store: CatalogMetaStore | None = None,
    ) -> None:
        self.s3_client = s3_client
        self.bucket = bucket
        self.project_id = project_id
        self.prefix = prefix or f"{project_id}/"
        self.include_pending = include_pending
        self.max_content_bytes = max_content_bytes
        self._meta_store = meta_store
        self._knowledge_graph: dict[str, Any] = {}
        self._cached_metas: list[CatalogObjectMeta] | None = None
        self._meta_by_s3_key: dict[str, CatalogObjectMeta] | None = None
        self._meta_by_artifact_id: dict[str, CatalogObjectMeta] | None = None

    @property
    def meta_store(self) -> CatalogMetaStore:
        if self._meta_store is None:
            self._meta_store = _resolve_meta_store(None)
        return self._meta_store

    # ── Catalog-native API ───────────────────────────────────────────────

    def list_metas(self) -> list[CatalogObjectMeta]:
        """List catalog metadata for objects in this project."""
        return list(self._load_metas())

    def refresh(self) -> None:
        """Drop cached metas so the next read sees newly ingested catalog objects."""
        self._cached_metas = None
        self._meta_by_s3_key = None
        self._meta_by_artifact_id = None

    def get_meta(self, s3_key: str) -> CatalogObjectMeta | None:
        """Return catalog metadata for an object after ACL check."""
        meta = self.meta_store.get(self.project_id, s3_key)
        if meta is None or not allowed(meta, self.project_id):
            return None
        return meta

    def search(self, query: str, *, limit: int = 20) -> list[CatalogObjectMeta]:
        """Lexical search over catalog metadata."""
        hits = self.meta_store.search(self.project_id, query, limit=limit)
        return [hit.meta for hit in hits if hit.meta.s3_key.startswith(self.prefix)]

    def read_text(self, s3_key: str, *, max_bytes: int | None = None) -> str:
        """Read extracted or raw text for a catalog object."""
        meta = self.get_meta(s3_key)
        if meta is None:
            raise PermissionError(f"access denied or missing metadata: {s3_key}")
        return read_artifact_text(
            self.s3_client,
            self.bucket,
            s3_key,
            meta,
            max_bytes=max_bytes or self.max_content_bytes,
        )

    # ── KnowledgeStore-compatible API ──────────────────────────────────

    def list_artifacts(self, artifact_type: str | None = None) -> list[dict[str, Any]]:
        """List artifacts as knowledge-store dicts (metadata only, no content)."""
        artifacts = [meta_to_artifact(meta) for meta in self._load_metas()]
        if artifact_type:
            artifacts = [
                item for item in artifacts if item.get("artifact_type") == artifact_type
            ]
        return artifacts

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Get one artifact by id or ``s3_key``, loading content."""
        self._load_metas()
        assert self._meta_by_s3_key is not None
        assert self._meta_by_artifact_id is not None

        meta = self._meta_by_artifact_id.get(artifact_id)
        if meta is None:
            meta = self._meta_by_s3_key.get(artifact_id)
        if meta is None:
            meta = self.get_meta(artifact_id)
            if meta is not None:
                logger.debug(
                    "get_artifact cache miss resolved via s3_key lookup: %s",
                    artifact_id,
                )
        if meta is None:
            return None
        return self._meta_to_artifact_with_content(meta)

    def search_artifacts(
        self,
        query: str,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search artifacts by query string."""
        artifacts = [meta_to_artifact(meta) for meta in self.search(query)]
        if artifact_type:
            artifacts = [
                item for item in artifacts if item.get("artifact_type") == artifact_type
            ]
        return artifacts

    def get_catalog_text(self, artifact_id: str) -> dict[str, Any] | None:
        """Return catalog flat text/summary for an artifact (Phase 3 output)."""
        artifact = self.get_artifact(artifact_id)
        if artifact is None:
            return None
        content = artifact.get("content", b"")
        text = (
            content.decode("utf-8", errors="replace")
            if isinstance(content, (bytes, bytearray))
            else str(content)
        )
        return {
            "artifact_id": artifact["artifact_id"],
            "summary": artifact.get("summary", ""),
            "extracted_text": text,
            "tags": artifact.get("tags", []),
            "metadata": artifact.get("metadata", {}),
        }

    def get_extracted_knowledge(self, artifact_id: str) -> dict[str, Any] | None:
        """Return persisted Phase 4 structured knowledge for an artifact."""
        return self.load_structured_knowledge(artifact_id)

    def persist_structured_knowledge(
        self,
        artifact_id: str,
        knowledge: dict[str, Any],
    ) -> None:
        """Write Phase 4 structured knowledge to S3 and update catalog meta."""
        self._load_metas()
        assert self._meta_by_artifact_id is not None

        meta = self._meta_by_artifact_id.get(artifact_id)
        if meta is None:
            logger.warning(
                "Cannot persist structured knowledge for unknown artifact %s",
                artifact_id,
            )
            return

        ref = write_structured_knowledge(
            self.s3_client,
            self.bucket,
            meta.s3_key,
            knowledge,
        )
        meta.extracted_knowledge_ref = ref
        self.meta_store.upsert(meta, self.project_id)

    def load_structured_knowledge(self, artifact_id: str) -> dict[str, Any] | None:
        """Load persisted Phase 4 structured knowledge from catalog refs."""
        self._load_metas()
        assert self._meta_by_artifact_id is not None

        meta = self._meta_by_artifact_id.get(artifact_id)
        if meta is None or not meta.extracted_knowledge_ref:
            return None
        return read_structured_knowledge(
            self.s3_client,
            self.bucket,
            meta.extracted_knowledge_ref,
        )

    def set_knowledge_graph(self, graph: dict[str, Any]) -> None:
        """Attach an in-memory knowledge graph for verifier queries."""
        self._knowledge_graph = graph

    def get_knowledge_graph(self) -> dict[str, Any]:
        """Return attached knowledge graph (empty until linked)."""
        return self._knowledge_graph

    # ── Internal ───────────────────────────────────────────────────────

    def _load_metas(self) -> list[CatalogObjectMeta]:
        """Load and cache project-scoped metas from Postgres."""
        if self._cached_metas is not None:
            return self._cached_metas

        metas = self.meta_store.list_by_project(
            self.project_id,
            prefix=self.prefix,
            include_pending=self.include_pending,
            include_error=False,
        )
        metas = [meta for meta in metas if meta.s3_key.startswith(self.prefix)]

        self._cached_metas = metas
        self._meta_by_s3_key = {meta.s3_key: meta for meta in metas}
        self._meta_by_artifact_id = {artifact_id_for_meta(meta): meta for meta in metas}
        return metas

    def _meta_to_artifact_with_content(self, meta: CatalogObjectMeta) -> dict[str, Any]:
        content = load_artifact_content(
            self.s3_client,
            self.bucket,
            meta.s3_key,
            meta,
            max_bytes=self.max_content_bytes,
        )
        return meta_to_artifact(meta, content)
