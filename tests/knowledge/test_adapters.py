"""Tests for the UMRS API adapters (Phase 3).

These tests verify that the adapters correctly delegate to the UMRS
subsystems and provide the same interface as the existing in-memory
implementations.
"""

from __future__ import annotations

from umrs.embeddings_api import InMemoryEmbeddingStore
from umrs.evidence import Evidence, EvidenceSourceType, InMemoryEvidenceStore
from umrs.experiment_registry import ExperimentStatus, InMemoryExperimentRegistry
from umrs.knowledge.adapters import (
    UMRSEmbeddingsAdapter,
    UMRSEvidenceAdapter,
    UMRSKnowledgeGraphAdapter,
    UMRSKnowledgeModelAdapter,
    UMRSScientificExperimentRegistryAdapter,
    UMRSScientificMemoryAdapter,
    UMRSSearchAdapter,
)
from umrs.knowledge.canonical_graph_view import in_memory_view
from umrs.knowledge.graph_types import RelationType
from umrs.knowledge_model import InMemoryKnowledgeModelStore, KnowledgeObjectType
from umrs.scientific_memory import MemoryEntry, MemoryEntryType
from umrs.search_api import SearchScope

# ── Scientific Memory Adapter Tests ──────────────────────────────────────


def test_scientific_memory_adapter_init():
    """Test UMRSScientificMemoryAdapter initialization."""
    adapter = _sm_adapter()
    assert adapter is not None


def test_scientific_memory_adapter_add_and_get():
    """Test adding and getting entries through the adapter."""
    adapter = _sm_adapter()
    entry = MemoryEntry(
        project_id="proj-001",
        entry_type=MemoryEntryType.GENERATED_CODE,
        content="print('hello')",
    )
    entry_id = adapter.add(entry)

    assert entry_id == entry.entry_id
    retrieved = adapter.get(entry_id)
    assert retrieved is not None
    assert retrieved.content == "print('hello')"


def test_scientific_memory_adapter_list_entries():
    """Test listing entries through the adapter."""
    adapter = _sm_adapter()
    adapter.add(MemoryEntry(entry_id="n1", project_id="p1", entry_type=MemoryEntryType.NOTE))
    adapter.add(MemoryEntry(entry_id="n2", project_id="p1", entry_type=MemoryEntryType.NOTE))
    adapter.add(MemoryEntry(entry_id="n3", project_id="p2", entry_type=MemoryEntryType.NOTE))

    all_entries = adapter.list_entries()
    assert len(all_entries) == 3

    proj1_entries = adapter.list_entries(project_id="p1")
    assert len(proj1_entries) == 2


def test_scientific_memory_adapter_search():
    """Test searching entries through the adapter."""
    adapter = _sm_adapter()
    adapter.add(MemoryEntry(entry_id="n1", project_id="p1", content="machine learning", tags=["ml"]))
    adapter.add(MemoryEntry(entry_id="n2", project_id="p1", content="quantum computing", tags=["quantum"]))

    results = adapter.search("machine learning")
    assert len(results) == 1
    assert results[0].entry_id == "n1"


def test_scientific_memory_adapter_count():
    """Test counting entries through the adapter."""
    adapter = _sm_adapter()
    adapter.add(MemoryEntry(entry_id="n1", project_id="p1", entry_type=MemoryEntryType.NOTE))
    adapter.add(MemoryEntry(entry_id="n2", project_id="p1", entry_type=MemoryEntryType.NOTE))
    adapter.add(MemoryEntry(entry_id="n3", project_id="p2", entry_type=MemoryEntryType.NOTE))

    assert adapter.count() == 3
    assert adapter.count(project_id="p1") == 2


def test_scientific_memory_adapter_get_timeline():
    """Test getting timeline through the adapter."""
    adapter = _sm_adapter()
    adapter.add(MemoryEntry(entry_id="n1", project_id="p1", content="First"))
    adapter.add(MemoryEntry(entry_id="n2", project_id="p1", content="Second"))

    timeline = adapter.get_timeline("p1")
    assert len(timeline) == 2


# ── Knowledge Model Adapter Tests ────────────────────────────────────────


def test_knowledge_model_adapter_init():
    """Test UMRSKnowledgeModelAdapter initialization."""
    adapter = _km_adapter()
    assert adapter is not None


def test_knowledge_model_adapter_create_and_get():
    """Test creating and getting objects through the adapter."""
    adapter = _km_adapter()
    obj_id = adapter.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "My Paper"},
    )

    obj = adapter.get_object(obj_id)
    assert obj is not None
    assert obj.content == {"title": "My Paper"}


def test_knowledge_model_adapter_list_objects():
    """Test listing objects through the adapter."""
    adapter = _km_adapter()
    adapter.create_object(project_id="p1", object_type=KnowledgeObjectType.PAPER)
    adapter.create_object(project_id="p1", object_type=KnowledgeObjectType.CODE)
    adapter.create_object(project_id="p2", object_type=KnowledgeObjectType.PAPER)

    all_objs = adapter.list_objects()
    assert len(all_objs) == 3

    proj1_objs = adapter.list_objects(project_id="p1")
    assert len(proj1_objs) == 2


def test_knowledge_model_adapter_search_objects():
    """Test searching objects through the adapter."""
    adapter = _km_adapter()
    adapter.create_object(
        project_id="p1",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Machine Learning Paper"},
    )
    adapter.create_object(
        project_id="p1",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Quantum Computing Paper"},
    )

    results = adapter.search_objects("machine learning")
    assert len(results) == 1




def _sm_adapter():
    from umrs.scientific_memory import InMemoryScientificMemoryStore

    return UMRSScientificMemoryAdapter(store=InMemoryScientificMemoryStore())


def _km_adapter():
    return UMRSKnowledgeModelAdapter(store=InMemoryKnowledgeModelStore())


def _exp_adapter():
    return UMRSScientificExperimentRegistryAdapter(registry=InMemoryExperimentRegistry())


def _ev_adapter():
    return UMRSEvidenceAdapter(store=InMemoryEvidenceStore())


def _emb_adapter():
    return UMRSEmbeddingsAdapter(store=InMemoryEmbeddingStore())


# ── Knowledge Graph Adapter Tests ────────────────────────────────────────


def test_knowledge_graph_adapter_init():
    """Test UMRSKnowledgeGraphAdapter initialization."""
    adapter = UMRSKnowledgeGraphAdapter(graph=in_memory_view())
    assert adapter is not None


def test_knowledge_graph_adapter_add_node_and_edge():
    """Test adding nodes and edges through the adapter."""
    adapter = UMRSKnowledgeGraphAdapter(graph=in_memory_view())
    adapter.add_node(node_id="a", node_type="paper")
    adapter.add_node(node_id="b", node_type="code")
    edge_id = adapter.add_edge(
        source_id="a",
        target_id="b",
        relation_type=RelationType.IMPLEMENTS,
    )

    assert edge_id != ""

    node = adapter.get_node("a")
    assert node is not None
    assert node.node_type == "paper"


def test_knowledge_graph_adapter_get_neighbors():
    """Test getting neighbors through the adapter."""
    adapter = UMRSKnowledgeGraphAdapter(graph=in_memory_view())
    adapter.add_edge(source_id="a", target_id="b", relation_type=RelationType.RELATED_TO)
    adapter.add_edge(source_id="a", target_id="c", relation_type=RelationType.RELATED_TO)

    neighbors = adapter.get_neighbors("a")
    assert set(neighbors) == {"b", "c"}


def test_knowledge_graph_adapter_find_path():
    """Test finding path through the adapter."""
    adapter = UMRSKnowledgeGraphAdapter(graph=in_memory_view())
    adapter.add_edge(source_id="a", target_id="b", relation_type=RelationType.RELATED_TO)
    adapter.add_edge(source_id="b", target_id="c", relation_type=RelationType.RELATED_TO)

    path = adapter.find_path("a", "c")
    assert path is not None
    assert path[0] == "a"
    assert path[-1] == "c"


def test_knowledge_graph_adapter_load_from_linker_graph():
    """Adapter should mirror linker graph nodes and edges."""
    from umrs.knowledge.link import (
        KnowledgeEdge as LinkerEdge,
    )
    from umrs.knowledge.link import (
        KnowledgeGraph as LinkerGraph,
    )
    from umrs.knowledge.link import (
        KnowledgeNode as LinkerNode,
    )
    from umrs.knowledge.link import (
        RelationType as LinkerRelationType,
    )

    linker_graph = LinkerGraph()
    linker_graph.add_node(LinkerNode(node_id="paper-1", node_type="paper", title="Paper"))
    linker_graph.add_node(LinkerNode(node_id="code-1", node_type="code", title="Code"))
    linker_graph.add_edge(
        LinkerEdge(
            source_id="code-1",
            target_id="paper-1",
            relation_type=LinkerRelationType.IMPLEMENTS,
        )
    )

    adapter = UMRSKnowledgeGraphAdapter(graph=in_memory_view())
    adapter.load_from_linker_graph(linker_graph)

    assert adapter.get_node("paper-1") is not None
    assert adapter.get_node("code-1") is not None
    assert adapter.get_related("code-1", RelationType.IMPLEMENTS) == ["paper-1"]


# ── Experiment Registry Adapter Tests ────────────────────────────────────


def test_experiment_registry_adapter_init():
    """Test UMRSScientificExperimentRegistryAdapter initialization."""
    adapter = _exp_adapter()
    assert adapter is not None


def test_experiment_registry_adapter_create_and_get():
    """Test creating and getting experiments through the adapter."""
    adapter = _exp_adapter()
    exp_id = adapter.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
        protocol={"steps": ["step1"]},
        parameters={"lr": 0.001},
    )

    exp = adapter.get_experiment(exp_id)
    assert exp is not None
    assert exp.project_id == "proj-001"
    assert exp.hypothesis_id == "hyp-001"


def test_experiment_registry_adapter_update_status():
    """Test updating experiment status through the adapter."""
    adapter = _exp_adapter()
    exp_id = adapter.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )

    # PENDING -> DESIGNED
    assert adapter.update_experiment_status(exp_id, ExperimentStatus.DESIGNED) is True
    assert adapter.get_experiment(exp_id).status == ExperimentStatus.DESIGNED


def test_experiment_registry_adapter_list_experiments():
    """Test listing experiments through the adapter."""
    adapter = _exp_adapter()
    adapter.create_experiment(project_id="p1", hypothesis_id="h1")
    adapter.create_experiment(project_id="p1", hypothesis_id="h2")
    adapter.create_experiment(project_id="p2", hypothesis_id="h3")

    all_exps = adapter.list_experiments()
    assert len(all_exps) == 3

    proj1_exps = adapter.list_experiments(project_id="p1")
    assert len(proj1_exps) == 2


# ── Evidence Adapter Tests ──────────────────────────────────────────────


def test_evidence_adapter_init():
    """Test UMRSEvidenceAdapter initialization."""
    adapter = _ev_adapter()
    assert adapter is not None


def test_evidence_adapter_add_and_get():
    """Test adding and getting evidence through the adapter."""
    adapter = _ev_adapter()
    evd = Evidence(
        project_id="proj-001",
        claim_id="claim-001",
        source_type=EvidenceSourceType.UPLOADED_FILE,
        content="Evidence content",
    )
    evd_id = adapter.add_evidence(evd)

    retrieved = adapter.get_evidence(evd_id)
    assert retrieved is not None
    assert retrieved.content == "Evidence content"


def test_evidence_adapter_list_evidence():
    """Test listing evidence through the adapter."""
    adapter = _ev_adapter()
    adapter.add_evidence(Evidence(evidence_id="evd-001", project_id="p1", source_type=EvidenceSourceType.UPLOADED_FILE, claim_id="c1", relevance_score=0.8))
    adapter.add_evidence(Evidence(evidence_id="evd-002", project_id="p1", source_type=EvidenceSourceType.EXTERNAL_PAPER, claim_id="c1", relevance_score=0.9))
    adapter.add_evidence(Evidence(evidence_id="evd-003", project_id="p2", source_type=EvidenceSourceType.UPLOADED_FILE, claim_id="c2", relevance_score=0.7))

    all_evd = adapter.list_evidence()
    assert len(all_evd) == 3

    proj1_evd = adapter.list_evidence(project_id="p1")
    assert len(proj1_evd) == 2


def test_evidence_adapter_get_evidence_for_claim():
    """Test getting evidence for a claim through the adapter."""
    adapter = _ev_adapter()
    adapter.add_evidence(Evidence(evidence_id="evd-001", project_id="p1", claim_id="claim-001", relevance_score=0.8))
    adapter.add_evidence(Evidence(evidence_id="evd-002", project_id="p1", claim_id="claim-001", relevance_score=0.9))
    adapter.add_evidence(Evidence(evidence_id="evd-003", project_id="p1", claim_id="claim-002", relevance_score=0.7))

    claim1_evd = adapter.get_evidence_for_claim("claim-001")
    assert len(claim1_evd) == 2


# ── Search Adapter Tests ────────────────────────────────────────────────


def test_search_adapter_init():
    """Test UMRSSearchAdapter initialization."""
    adapter = UMRSSearchAdapter()
    assert adapter is not None


def test_search_adapter_search():
    """Test searching through the adapter."""
    adapter = UMRSSearchAdapter()
    results = adapter.search("test query", scope=SearchScope.ALL)

    assert results is not None
    assert results.query == "test query"


# ── Embeddings Adapter Tests ────────────────────────────────────────────


def test_embeddings_adapter_init():
    """Test UMRSEmbeddingsAdapter initialization."""
    adapter = _emb_adapter()
    assert adapter is not None


def test_embeddings_adapter_create_and_get():
    """Test creating and getting embeddings through the adapter."""
    adapter = _emb_adapter()
    emb_id = adapter.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="This is a test",
    )

    emb = adapter.get_embedding(emb_id)
    assert emb is not None
    assert emb.text == "This is a test"
    assert len(emb.vector) > 0


def test_embeddings_adapter_list_embeddings():
    """Test listing embeddings through the adapter."""
    adapter = _emb_adapter()
    adapter.create_embedding(project_id="p1", object_id="o1", text="test1")
    adapter.create_embedding(project_id="p1", object_id="o2", text="test2")
    adapter.create_embedding(project_id="p2", object_id="o3", text="test3")

    all_embs = adapter.list_embeddings()
    assert len(all_embs) == 3

    proj1_embs = adapter.list_embeddings(project_id="p1")
    assert len(proj1_embs) == 2


def test_embeddings_adapter_count():
    """Test counting embeddings through the adapter."""
    adapter = _emb_adapter()
    adapter.create_embedding(project_id="p1", object_id="o1", text="test1")
    adapter.create_embedding(project_id="p1", object_id="o2", text="test2")
    adapter.create_embedding(project_id="p2", object_id="o3", text="test3")

    assert adapter.count() == 3
    assert adapter.count(project_id="p1") == 2
