"""Backend conformance tests for batch-2 research satellites (K1 batch 2).

Restored post ADR-026 wave C with the sat_kg trio excised: embeddings
(pgvector), knowledge model, tenants are live faces; the `sat_kg_*` graph
tables + facade are tombstoned/deleted (migrations 0089 wave C).

Every behavioral test runs against the in-memory reference and the Postgres
implementation (migration 0060). Postgres tests are skipped when no
pgvector-enabled database is reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from umrs import embeddings_api as emb_mod
from umrs import knowledge_model as km_mod
from umrs import tenants as tenants_mod

_MIGRATION = (
    Path(__file__).resolve().parents[1] / "src/umrs/migrations/0060_research_satellites_batch2.sql"
)

_TABLES = (
    "sat_embeddings, sat_knowledge_objects, "
    "sat_knowledge_object_versions, sat_knowledge_tombstones, sat_tenants"
)


def _pg_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            try:
                cur.execute(_MIGRATION.read_text())
            except Exception as exc:
                pytest.skip(f"migration 0060 needs pgvector: {exc}")
            cur.execute(f"TRUNCATE {_TABLES} CASCADE")
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture(params=["memory", "pg"])
def emb_store(request):
    if request.param == "memory":
        return emb_mod.InMemoryEmbeddingStore()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.embeddings_api.pg_store import PgEmbeddingStore

    return PgEmbeddingStore(pool)


@pytest.fixture(params=["memory", "pg"])
def km_store(request):
    if request.param == "memory":
        return km_mod.InMemoryKnowledgeModelStore()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.knowledge_model.pg_store import PgKnowledgeModelStore

    return PgKnowledgeModelStore(pool)


@pytest.fixture(params=["memory", "pg"])
def tenant_mgr(request):
    if request.param == "memory":
        return tenants_mod.TenantManager()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.tenants.pg_store import PgTenantManager

    return PgTenantManager(pool)


# ── Embeddings ────────────────────────────────────────────────────────


def test_emb_create_get_round_trip(emb_store):
    emb_id = emb_store.create_embedding(
        project_id="proj-1",
        object_id="obj-1",
        text="vLLM offline inference",
        metadata={"src": "test"},
    )
    loaded = emb_store.get_embedding(emb_id)
    assert loaded is not None
    assert loaded.project_id == "proj-1"
    assert loaded.object_id == "obj-1"
    assert loaded.metadata.get("src") == "test"
    assert len(loaded.vector) == len(emb_mod.hash_based_vector(""))
    assert emb_store.get_embedding("missing") is None


def test_emb_search_similar_orders_and_filters(emb_store):
    kept = emb_store.create_embedding(
        project_id="proj-1", object_id="a", text="slurm gpu scheduling"
    )
    emb_store.create_embedding(
        project_id="proj-2", object_id="b", text="completely unrelated topic"
    )
    query = emb_store.get_embedding(kept).vector
    hits = emb_store.search_similar(query_embedding=query, top_k=5)
    assert hits
    assert hits[0].embedding_id == kept
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    # Project filter narrows results
    proj1 = emb_store.search_similar(query_embedding=query, project_id="proj-1")
    assert all(h.object_id == "a" for h in proj1)


def test_emb_update_delete_and_count(emb_store):
    emb_id = emb_store.create_embedding(project_id="proj-1", object_id="o", text="original")
    assert emb_store.list_embeddings(project_id="proj-1") != []
    if hasattr(emb_store, "update_embedding"):
        assert emb_store.update_embedding(emb_id, text="rewritten") is True
        assert emb_store.get_embedding(emb_id).text == "rewritten"
    assert emb_store.count(project_id="proj-1") == 1
    assert emb_store.delete_embedding(emb_id) is True
    assert emb_store.get_embedding(emb_id) is None
    assert emb_store.delete_embedding(emb_id) is False


# ── Knowledge model ───────────────────────────────────────────────────


def test_km_create_get_versions(km_store):
    obj_id = km_store.create_object(
        project_id="proj-1",
        object_type=km_mod.KnowledgeObjectType.HYPOTHESIS,
        content={"text": "GNMT v2 improves BLEU"},
        created_by="tester",
    )
    obj = km_store.get_object(obj_id)
    assert obj is not None
    assert obj.object_type == km_mod.KnowledgeObjectType.HYPOTHESIS
    assert obj.content["text"].startswith("GNMT")
    assert km_store.get_object("missing") is None

    assert km_store.update_object(obj_id, content={"text": "revised claim"}) is True
    obj = km_store.get_object(obj_id)
    assert obj.content["text"] == "revised claim"
    if hasattr(km_store, "get_versions"):
        versions = km_store.get_versions(obj_id)
        assert len(versions) >= 2


def test_km_immutable_rejected(km_store):
    obj_id = km_store.create_object(
        project_id="proj-1",
        object_type=km_mod.KnowledgeObjectType.OBSERVATION,
        content={"v": 1},
        is_immutable=True,
    )
    with pytest.raises(km_mod.ImmutableObjectError):
        km_store.update_object(obj_id, content={"v": 2})


def test_km_list_search_delete(km_store):
    km_store.create_object(
        project_id="proj-1",
        object_type=km_mod.KnowledgeObjectType.PAPER,
        content={"title": "attention is all you need"},
    )
    km_store.create_object(
        project_id="proj-2",
        object_type=km_mod.KnowledgeObjectType.DATASET,
        content={"title": "wmt14"},
    )
    proj1 = km_store.list_objects(project_id="proj-1")
    assert len(proj1) == 1
    hits = km_store.search_objects(query="attention", project_id="proj-1")
    assert len(hits) == 1


def test_km_delete_object_tombstones(km_store):
    obj_id = km_store.create_object(
        project_id="proj-1",
        object_type=km_mod.KnowledgeObjectType.FINDING,
        content={"f": 1},
    )
    deleted = km_store.delete_object(obj_id, deleted_by="tester", deletion_reason="bogus")
    assert deleted is True
    # Soft delete: object remains but is marked DELETED (both backends),
    # and re-delete is an idempotent tombstone overwrite (returns True).
    ghost = km_store.get_object(obj_id)
    assert ghost is not None
    assert ghost.status == km_mod.KnowledgeObjectStatus.DELETED
    assert km_store.delete_object(obj_id, deleted_by="t", deletion_reason="again") is True
    # Missing objects still report False.
    assert km_store.delete_object("nonexistent") is False


# ── Tenants ───────────────────────────────────────────────────────────


def test_tenant_provision_get_list(tenant_mgr):
    spec = tenant_mgr.provision(project_id="proj-alpha", project_name="Alpha", owner_id="alice")
    assert spec.tenant_id
    assert spec.schema_name == "tenant_proj_alpha"
    loaded = tenant_mgr.get(spec.tenant_id)
    assert loaded is not None
    assert loaded.owner_id == "alice"
    by_project = tenant_mgr.get_by_project("proj-alpha")
    assert by_project is not None and by_project.tenant_id == spec.tenant_id
    owned = tenant_mgr.list_tenants(owner_id="alice")
    assert len(owned) == 1


def test_tenant_suspend_deprovision(tenant_mgr):
    spec = tenant_mgr.provision(
        project_id="proj-beta", project_name="Beta", owner_id="bob", tier="free"
    )
    assert tenant_mgr.suspend(spec.tenant_id) is True
    assert tenant_mgr.get(spec.tenant_id).status == tenants_mod.TenantStatus.SUSPENDED
    assert tenant_mgr.deprovision(spec.tenant_id) is True
    assert tenant_mgr.get(spec.tenant_id) is None
    assert tenant_mgr.deprovision(spec.tenant_id) is False


def test_tenant_limits_and_schema_sql(tenant_mgr):
    spec = tenant_mgr.provision(
        project_id="proj-gamma", project_name="Gamma", owner_id="carol", tier="free"
    )
    assert spec.resource_limits.get("max_documents", 0) > 0
    # check_limits returns True while usage below quota
    assert tenant_mgr.check_limits(spec.tenant_id, "documents", 1) is True
    sql = tenant_mgr.get_schema_sql(spec.tenant_id)
    if sql is not None:
        assert "tenant_proj_gamma" in sql
