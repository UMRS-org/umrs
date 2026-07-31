"""Factory-conformance tests for the PG-backed research satellite stores.

These exercise the *selection* path (`build_default_store(db_pool=None)`)
without needing a live Postgres: the factory must still return an in-memory
implementation exposing the exact same helper set, so callers that swap in a
pool later don't need to change call sites.
"""

from __future__ import annotations


def test_embeddings_store_factory_memory():
    from umrs.embeddings_api import build_default_store

    store = build_default_store()
    assert type(store).__name__ in {"InMemoryEmbeddingStore", "PgEmbeddingStore"}
    for name in ("create_embedding", "search_similar", "count"):
        assert callable(getattr(store, name, None)), f"missing {name}"


def test_knowledge_graph_factory_memory():
    from umrs.knowledge_graph_api import build_default_graph

    graph = build_default_graph()
    assert type(graph).__name__ in {
        "InMemoryKnowledgeGraph",
        "PgKnowledgeGraphStore",
        "PgKnowledgeGraph",
    }
    for name in ("add_node", "get_node", "get_neighbors"):
        assert callable(getattr(graph, name, None)), f"missing {name}"


def test_knowledge_model_factory_memory():
    from umrs.knowledge_model import build_default_store

    store = build_default_store()
    assert type(store).__name__ in {
        "InMemoryKnowledgeModelStore",
        "PgKnowledgeModelStore",
    }
    for name in (
        "create_object",
        "get_object",
        "update_object",
        "delete_object",
        "list_objects",
        "search_objects",
    ):
        assert callable(getattr(store, name, None)), f"missing {name}"


def test_tenants_factory_memory():
    from umrs.tenants import build_default_manager

    manager = build_default_manager()
    assert type(manager).__name__ in {"InMemoryTenantManager", "TenantManager", "PgTenantManager"}
    for name in ("get", "provision", "deprovision", "list_tenants"):
        assert callable(getattr(manager, name, None)), f"missing {name}"


def test_evidence_store_create_and_get():
    from umrs.evidence import Evidence, build_default_store

    store = build_default_store()
    evidence = Evidence(project_id="p1", claim_id="c1", content="x")
    eid = store.add_evidence(evidence)
    assert store.get_evidence(eid) is not None
    assert store.get_evidence("does-not-exist") is None


def test_experiment_registry_create_and_list():
    from umrs.experiment_registry import build_default_store

    store = build_default_store()
    experiment_id = store.create_experiment(project_id="p1", hypothesis_id="h1")
    assert store.get_experiment(experiment_id) is not None
    assert isinstance(store.list_experiments(project_id="p1"), list)


def test_scientific_memory_add_and_search():
    from umrs.scientific_memory import MemoryEntry, build_default_store

    store = build_default_store()
    store.add(
        MemoryEntry(
            project_id="p1",
            content="observation about 0.95",
            summary="t",
        )
    )
    results = store.search("0.95", project_id="p1")
    assert isinstance(results, list) and len(results) >= 1
