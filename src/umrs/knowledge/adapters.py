"""UMRS API adapters — replace in-memory implementations with UMRS API calls.

Phase 3 of the RC1 migration plan.

These adapters wrap the new UMRS subsystems (scientific_memory,
knowledge_model, knowledge_graph_api, etc.) and provide the same
interface as the existing in-memory implementations.

The key architectural rule is:
  - R4 orchestrates scientific research.
  - UMRS owns scientific knowledge.

These adapters ensure that all knowledge management operations go
through UMRS APIs, not through local in-memory stores.
"""

from __future__ import annotations

from typing import Any

from umrs.embeddings_api import InMemoryEmbeddingStore
from umrs.embeddings_api import build_default_store as _build_default_embedding_store
from umrs.evidence import (
    Evidence,
    EvidenceSourceType,
    EvidenceStatus,
    InMemoryEvidenceStore,
)
from umrs.evidence import build_default_store as _build_default_evidence_store
from umrs.experiment_registry import (
    Experiment,
    ExperimentStatus,
    InMemoryExperimentRegistry,
)
from umrs.experiment_registry import (
    build_default_store as _build_default_experiment_registry,
)
from umrs.knowledge.graph_types import KnowledgeEdge, KnowledgeNode, RelationType
from umrs.knowledge_model import (
    InMemoryKnowledgeModelStore,
    KnowledgeObject,
    KnowledgeObjectStatus,
    KnowledgeObjectType,
)
from umrs.knowledge_model import (
    build_default_store as _build_default_knowledge_model_store,
)
from umrs.scientific_memory import (
    InMemoryScientificMemoryStore,
    MemoryEntry,
    MemoryEntryStatus,
    MemoryEntryType,
)
from umrs.scientific_memory import (
    build_default_store as _build_default_scientific_memory_store,
)
from umrs.search_api import (
    SearchDepth,
    SearchResults,
    SearchScope,
    UnifiedSearch,
)

from .canonical_graph_view import CanonicalGraphView as TransactionCanonicalGraphView
from .canonical_graph_view import default_view

__all__ = [
    "UMRSEmbeddingsAdapter",
    "UMRSEvidenceAdapter",
    "UMRSKnowledgeGraphAdapter",
    "UMRSKnowledgeModelAdapter",
    "UMRSScientificExperimentRegistryAdapter",
    "UMRSScientificMemoryAdapter",
    "UMRSSearchAdapter",
]


# ── Scientific Memory Adapter ─────────────────────────────────────────────


class UMRSScientificMemoryAdapter:
    """Adapter that wraps the UMRS scientific_memory subsystem.

    This adapter provides the same interface as the existing
    ScientificMemoryStore but delegates all operations to the UMRS
    scientific_memory subsystem.

    Architecture: UMRS owns scientific memory. R4 orchestrates.
    """

    def __init__(self, store: InMemoryScientificMemoryStore | None = None) -> None:
        # Default backend comes from the satellite factory: Postgres when a
        # database pool/env is available (migration 0059), else in-memory.
        self._store = store or _build_default_scientific_memory_store()

    def add(self, entry: MemoryEntry) -> str:
        """Add a new entry to the store."""
        return self._store.add(entry)

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Get an entry by ID."""
        return self._store.get(entry_id)

    def list_entries(
        self,
        *,
        project_id: str | None = None,
        entry_type: MemoryEntryType | None = None,
        status: MemoryEntryStatus | None = None,
        experiment_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List entries with optional filters."""
        return self._store.list_entries(
            project_id=project_id,
            entry_type=entry_type,
            status=status,
            experiment_id=experiment_id,
            limit=limit,
        )

    def search(
        self,
        query: str,
        *,
        project_id: str | None = None,
        entry_type: MemoryEntryType | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Search entries by content, summary, or tags."""
        return self._store.search(
            query,
            project_id=project_id,
            entry_type=entry_type,
            limit=limit,
        )

    def update_status(
        self,
        entry_id: str,
        status: MemoryEntryStatus,
    ) -> bool:
        """Update the status of an entry."""
        return self._store.update_status(entry_id, status)

    def count(
        self,
        *,
        project_id: str | None = None,
        entry_type: MemoryEntryType | None = None,
        status: MemoryEntryStatus | None = None,
    ) -> int:
        """Count entries with optional filters."""
        return self._store.count(
            project_id=project_id,
            entry_type=entry_type,
            status=status,
        )

    def get_timeline(self, project_id: str) -> list[MemoryEntry]:
        """Get a timeline of all entries for a project."""
        return self._store.get_timeline(project_id)


# ── Knowledge Model Adapter ──────────────────────────────────────────────


class UMRSKnowledgeModelAdapter:
    """Adapter that wraps the UMRS knowledge_model subsystem.

    This adapter provides the same interface as the existing
    KnowledgeModelStore but delegates all operations to the UMRS
    knowledge_model subsystem.

    Architecture: UMRS owns the knowledge model. R4 orchestrates.
    """

    def __init__(self, store: InMemoryKnowledgeModelStore | None = None) -> None:
        # Default backend via satellite factory (PG when available).
        self._store = store or _build_default_knowledge_model_store()

    def create_object(
        self,
        *,
        project_id: str,
        object_type: KnowledgeObjectType,
        content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        is_immutable: bool = False,
        created_by: str = "",
    ) -> str:
        """Create a new knowledge object."""
        return self._store.create_object(
            project_id=project_id,
            object_type=object_type,
            content=content,
            metadata=metadata,
            is_immutable=is_immutable,
            created_by=created_by,
        )

    def get_object(self, object_id: str) -> KnowledgeObject | None:
        """Get an object by ID."""
        return self._store.get_object(object_id)

    def update_object(
        self,
        object_id: str,
        *,
        content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an object's content and/or metadata."""
        return self._store.update_object(
            object_id,
            content=content,
            metadata=metadata,
        )

    def delete_object(
        self,
        object_id: str,
        *,
        deleted_by: str = "",
        deletion_reason: str = "",
    ) -> bool:
        """Delete an object (soft delete via tombstone)."""
        return self._store.delete_object(
            object_id,
            deleted_by=deleted_by,
            deletion_reason=deletion_reason,
        )

    def list_objects(
        self,
        *,
        project_id: str | None = None,
        object_type: KnowledgeObjectType | None = None,
        status: KnowledgeObjectStatus | None = None,
        limit: int = 100,
    ) -> list[KnowledgeObject]:
        """List objects with optional filters."""
        return self._store.list_objects(
            project_id=project_id,
            object_type=object_type,
            status=status,
            limit=limit,
        )

    def search_objects(
        self,
        query: str,
        *,
        project_id: str | None = None,
        object_type: KnowledgeObjectType | None = None,
        limit: int = 20,
    ) -> list[KnowledgeObject]:
        """Search objects by content, metadata, or object_id."""
        return self._store.search_objects(
            query,
            project_id=project_id,
            object_type=object_type,
            limit=limit,
        )

    def get_object_history(self, object_id: str) -> list[Any]:
        """Get the version history of an object."""
        return self._store.get_object_history(object_id)

    def get_tombstone(self, object_id: str) -> Any | None:
        """Get the tombstone for a deleted object."""
        return self._store.get_tombstone(object_id)


# ── Knowledge Graph Adapter ─────────────────────────────────────────────


class UMRSKnowledgeGraphAdapter:
    """Adapter that wraps the UMRS knowledge_graph_api subsystem.

    This adapter provides the same interface as the existing
    KnowledgeGraph but delegates all operations to the UMRS
    knowledge_graph_api subsystem.

    Architecture: UMRS owns the knowledge graph. R4 orchestrates.
    """

    def __init__(self, graph: TransactionCanonicalGraphView | None = None) -> None:
        # ADR-026 migration: canonical-backed view (durable when resolvable).
        if graph is None:
            from .canonical_graph_view import default_view

            graph = default_view()
        self._graph = graph

    @property
    def graph(self) -> TransactionCanonicalGraphView:
        """Underlying UMRS knowledge graph store."""
        return self._graph

    def load_from_linker_graph(self, linker_graph: Any) -> None:
        """Replace adapter contents with linker output from ``understand()``."""
        # ADR-026 exit wave A: same semantics via the canonical view
        # (store=None → in-memory dicts, same dedupe/traversal).
        fresh = default_view(store=None)
        for node in linker_graph.nodes.values():
            fresh.add_node(
                node_id=node.node_id,
                node_type=node.node_type,
                title=node.title,
                metadata=dict(node.metadata),
            )
        for edge in linker_graph.edges:
            relation_value = (
                edge.relation_type.value
                if hasattr(edge.relation_type, "value")
                else str(edge.relation_type)
            )
            try:
                relation_type = RelationType(relation_value)
            except ValueError:
                relation_type = RelationType.RELATED_TO
            fresh.add_edge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation_type=relation_type,
                weight=edge.weight,
                confidence=edge.confidence,
                metadata=dict(edge.metadata),
            )
        self._graph = fresh

    def add_node(
        self,
        *,
        node_id: str,
        node_type: str,
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a node to the graph."""
        self._graph.add_node(
            node_id=node_id,
            node_type=node_type,
            title=title,
            metadata=metadata,
        )

    def add_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add an edge to the graph."""
        return self._graph.add_edge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            metadata=metadata,
        )

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Get a node by ID."""
        return self._graph.get_node(node_id)

    def get_outgoing(self, node_id: str) -> list[KnowledgeEdge]:
        """Get all outgoing edges from a node."""
        return self._graph.get_outgoing(node_id)

    def get_incoming(self, node_id: str) -> list[KnowledgeEdge]:
        """Get all incoming edges to a node."""
        return self._graph.get_incoming(node_id)

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all neighbor node IDs."""
        return self._graph.get_neighbors(node_id)

    def get_related(
        self,
        node_id: str,
        relation_type: RelationType,
    ) -> list[str]:
        """Get all node IDs related to the given node with the given relation type."""
        return self._graph.get_related(node_id, relation_type)

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Find a path between two nodes using BFS."""
        return self._graph.find_path(source_id, target_id, max_depth=max_depth)

    def query(
        self,
        *,
        node_type: str | None = None,
        relation_type: RelationType | None = None,
    ) -> list[KnowledgeNode]:
        """Query the graph for nodes matching the given criteria."""
        return self._graph.query(node_type=node_type, relation_type=relation_type)


# ── Experiment Registry Adapter ─────────────────────────────────────────


class UMRSScientificExperimentRegistryAdapter:
    """Adapter that wraps the UMRS experiment_registry subsystem.

    This adapter provides the same interface as the existing
    ExperimentRegistry but delegates all operations to the UMRS
    experiment_registry subsystem.

    Architecture: UMRS owns the experiment registry. R4 orchestrates.
    """

    def __init__(self, registry: InMemoryExperimentRegistry | None = None) -> None:
        # Default backend via satellite factory (PG when available).
        self._registry = registry or _build_default_experiment_registry()

    def create_experiment(
        self,
        *,
        project_id: str,
        hypothesis_id: str,
        protocol: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        validation_method: str = "statistical_test",
        created_by: str = "",
        notes: str = "",
    ) -> str:
        """Create a new experiment."""
        return self._registry.create_experiment(
            project_id=project_id,
            hypothesis_id=hypothesis_id,
            protocol=protocol,
            parameters=parameters,
            validation_method=validation_method,
            created_by=created_by,
            notes=notes,
        )

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID."""
        return self._registry.get_experiment(experiment_id)

    def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
    ) -> bool:
        """Update the status of an experiment."""
        return self._registry.update_experiment_status(experiment_id, status)

    def add_experiment_result(
        self,
        experiment_id: str,
        *,
        metrics: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        logs: str = "",
        status: str = "completed",
        created_by: str = "",
    ) -> str:
        """Add a result to an experiment."""
        return self._registry.add_experiment_result(
            experiment_id,
            metrics=metrics,
            artifacts=artifacts,
            logs=logs,
            status=status,
            created_by=created_by,
        )

    def list_experiments(
        self,
        *,
        project_id: str | None = None,
        hypothesis_id: str | None = None,
        status: ExperimentStatus | None = None,
        limit: int = 100,
    ) -> list[Experiment]:
        """List experiments with optional filters."""
        return self._registry.list_experiments(
            project_id=project_id,
            hypothesis_id=hypothesis_id,
            status=status,
            limit=limit,
        )

    def get_experiment_history(self, project_id: str) -> list[Experiment]:
        """Get the experiment history for a project."""
        return self._registry.get_experiment_history(project_id)


# ── Evidence Adapter ────────────────────────────────────────────────────


class UMRSEvidenceAdapter:
    """Adapter that wraps the UMRS evidence subsystem.

    This adapter provides the same interface as the existing
    EvidenceStore but delegates all operations to the UMRS
    evidence subsystem.

    Architecture: UMRS owns evidence. R4 orchestrates.
    """

    def __init__(self, store: InMemoryEvidenceStore | None = None) -> None:
        # Default backend via satellite factory (PG when available).
        self._store = store or _build_default_evidence_store()

    def add_evidence(self, evidence: Evidence) -> str:
        """Add a new evidence item."""
        return self._store.add_evidence(evidence)

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Get an evidence item by ID."""
        return self._store.get_evidence(evidence_id)

    def list_evidence(
        self,
        *,
        project_id: str | None = None,
        source_type: EvidenceSourceType | None = None,
        status: EvidenceStatus | None = None,
        claim_id: str | None = None,
        limit: int = 100,
    ) -> list[Evidence]:
        """List evidence items with optional filters."""
        return self._store.list_evidence(
            project_id=project_id,
            source_type=source_type,
            status=status,
            claim_id=claim_id,
            limit=limit,
        )

    def search_evidence(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[Evidence]:
        """Search evidence items by content or citation."""
        return self._store.search_evidence(
            query,
            project_id=project_id,
            limit=limit,
        )

    def get_evidence_for_claim(self, claim_id: str) -> list[Evidence]:
        """Get all evidence items for a specific claim."""
        return self._store.get_evidence_for_claim(claim_id)

    def update_status(
        self,
        evidence_id: str,
        status: EvidenceStatus,
    ) -> bool:
        """Update the status of an evidence item."""
        return self._store.update_status(evidence_id, status)

    def count(
        self,
        *,
        project_id: str | None = None,
        source_type: EvidenceSourceType | None = None,
        status: EvidenceStatus | None = None,
    ) -> int:
        """Count evidence items with optional filters."""
        return self._store.count(
            project_id=project_id,
            source_type=source_type,
            status=status,
        )


# ── Search Adapter ──────────────────────────────────────────────────────


class UMRSSearchAdapter:
    """Adapter that wraps the UMRS search_api subsystem.

    This adapter provides the same interface as the existing
    UnifiedSearch but delegates all operations to the UMRS
    search_api subsystem.

    Architecture: UMRS owns search. R4 orchestrates.
    """

    def __init__(self, search: UnifiedSearch | None = None) -> None:
        self._search = search or UnifiedSearch()

    def search(
        self,
        query: str,
        *,
        scope: SearchScope = SearchScope.ALL,
        depth: SearchDepth = SearchDepth.MODERATE,
        project_id: str | None = None,
        limit: int = 20,
    ) -> SearchResults:
        """Perform a unified search."""
        return self._search.search(
            query,
            scope=scope,
            depth=depth,
            project_id=project_id,
            limit=limit,
        )

    def search_catalog(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search only the artifact catalog."""
        return self._search.search_catalog(
            query,
            project_id=project_id,
            limit=limit,
        )

    def search_knowledge_graph(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search only the knowledge graph."""
        return self._search.search_knowledge_graph(
            query,
            project_id=project_id,
            limit=limit,
        )

    def search_embeddings(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search only the embeddings."""
        return self._search.search_embeddings(
            query,
            project_id=project_id,
            limit=limit,
        )

    def search_rag(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search only the RAG indexes."""
        return self._search.search_rag(
            query,
            project_id=project_id,
            limit=limit,
        )


# ── Embeddings Adapter ─────────────────────────────────────────────────


class UMRSEmbeddingsAdapter:
    """Adapter that wraps the UMRS embeddings_api subsystem.

    This adapter provides the same interface as the existing
    InMemoryEmbeddingStore but delegates all operations to the UMRS
    embeddings_api subsystem.

    Architecture: UMRS owns embeddings. R4 orchestrates.
    """

    def __init__(self, store: InMemoryEmbeddingStore | None = None) -> None:
        # Default backend via satellite factory (PG when available).
        self._store = store or _build_default_embedding_store()

    def create_embedding(
        self,
        *,
        project_id: str,
        object_id: str,
        text: str,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new embedding from text."""
        return self._store.create_embedding(
            project_id=project_id,
            object_id=object_id,
            text=text,
            model=model,
            metadata=metadata,
        )

    def get_embedding(self, embedding_id: str) -> Any | None:
        """Get an embedding by ID."""
        return self._store.get_embedding(embedding_id)

    def search_similar(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 10,
        project_id: str | None = None,
    ) -> list[Any]:
        """Search for embeddings similar to the query embedding."""
        return self._store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            project_id=project_id,
        )

    def update_embedding(
        self,
        embedding_id: str,
        *,
        text: str | None = None,
        model: str | None = None,
    ) -> bool:
        """Update an embedding's text and/or model."""
        return self._store.update_embedding(
            embedding_id,
            text=text,
            model=model,
        )

    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding."""
        return self._store.delete_embedding(embedding_id)

    def list_embeddings(
        self,
        *,
        project_id: str | None = None,
        status: Any | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """List embeddings with optional filters."""
        return self._store.list_embeddings(
            project_id=project_id,
            status=status,
            limit=limit,
        )

    def count(
        self,
        *,
        project_id: str | None = None,
        status: Any | None = None,
    ) -> int:
        """Count embeddings with optional filters."""
        return self._store.count(
            project_id=project_id,
            status=status,
        )
