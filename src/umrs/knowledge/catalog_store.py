"""KnowledgeStore backed by the S3 data catalog (Phase B)."""

from __future__ import annotations

from typing import Any

from ..data_catalog.adapter import CatalogArtifactAdapter


class CatalogKnowledgeStore:
    """``KnowledgeStore`` that reads artifacts from ``CatalogArtifactAdapter``.

    Extracted knowledge and knowledge-graph attachments are held in a small
    in-memory overlay populated by ``ResearchProject.understand()`` and
    ``verify()``. Artifact bytes are loaded from catalog on demand.
    """

    def __init__(self, adapter: CatalogArtifactAdapter) -> None:
        self._adapter = adapter
        self._extracted_knowledge: dict[str, dict[str, Any]] = {}
        self._knowledge_graph: dict[str, Any] = {}

    def refresh(self) -> None:
        """Reload catalog metadata after new ingestions."""
        self._adapter.refresh()

    def add_extracted_knowledge(
        self,
        artifact_id: str,
        knowledge: dict[str, Any],
    ) -> None:
        self._extracted_knowledge[artifact_id] = knowledge

    def list_artifacts(self, artifact_type: str | None = None) -> list[dict[str, Any]]:
        return self._adapter.list_artifacts(artifact_type)

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return self._adapter.get_artifact(artifact_id)

    def get_extracted_knowledge(self, artifact_id: str) -> dict[str, Any] | None:
        overlay = self._extracted_knowledge.get(artifact_id)
        if overlay is not None:
            return overlay
        return self._adapter.get_extracted_knowledge(artifact_id)

    def search_artifacts(
        self,
        query: str,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._adapter.search_artifacts(query, artifact_type)

    def get_knowledge_graph(self) -> dict[str, Any]:
        return self._knowledge_graph or self._adapter.get_knowledge_graph()

    def set_knowledge_graph(self, graph: dict[str, Any]) -> None:
        self._knowledge_graph = graph
        self._adapter.set_knowledge_graph(graph)

    @property
    def catalog_adapter(self) -> CatalogArtifactAdapter:
        return self._adapter
