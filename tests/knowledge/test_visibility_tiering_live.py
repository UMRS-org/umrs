"""ADR-050 visibility tiers — live Postgres behaviour (R-02, 2026-09-13).

Runs only against a CI-shaped database (``POSTGRES_HOST/PORT/USER/PASSWORD/DB``
as the umrs CI job exports them, or ``UMRS_TEST_DB_URL``); skips cleanly
otherwise. Applies pending migrations (idempotent), seeds three tiers × two
projects into ``knowledge_nodes`` (+ 0075 embeddings) AND ``catalog_objects``,
then asserts for every read lane:

- caller A sees A's project rows and A's system rows;
- caller A sees B's SYSTEM rows and nothing else of B;
- nobody ever sees an ambient row, not even its owner.

Lanes: rag (``_RAGSearchBackend``), graph (``_GraphSearchBackend`` over the
project default graph), embeddings table-reader + similarity
(``_EmbeddingsSearchBackend``), catalog search (``CatalogMetaStore.search``).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState

PROJECT_A = "vis-live-a"
PROJECT_B = "vis-live-b"
PROJECTS = (PROJECT_A, PROJECT_B)
TIERS = ("project", "system", "ambient")
MODEL = "vis-live-probe-model"


def ci_shape_db_url() -> str | None:
    """DSN of the CI-shaped database, or None when the env is absent."""
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "test")
    password = os.environ.get("POSTGRES_PASSWORD", "test")
    db = os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


pytestmark = pytest.mark.skipif(
    ci_shape_db_url() is None,
    reason="POSTGRES_* env not set — CI-shaped Postgres absent (live tiering pins skipped)",
)


def _node_id(project: str, tier: str) -> str:
    return f"vis-live::{project}::{tier}"


def _s3_key(project: str, tier: str) -> str:
    return f"{project}/uploads/tiering-{tier}.csv"


def _cleanup(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM knowledge_node_embeddings WHERE node_id LIKE 'vis-live::%%'")
        cur.execute("DELETE FROM knowledge_nodes WHERE project_id = ANY(%s)", (list(PROJECTS),))
        cur.execute("DELETE FROM catalog_objects WHERE project_id = ANY(%s)", (list(PROJECTS),))
    conn.commit()


_NODE_SQL = (
    "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, "
    "content, content_hash, graph_id, visibility, search_text) "
    "VALUES (%s, %s, %s, 'concept', %s::jsonb, %s, '', %s, %s)"
)


def _seed_node(cur, node_id: str, project: str, tier: str, chunk: bool) -> None:
    """One row per lane family: the graph lane reads non-chunk nodes; the rag
    lane reads chunk rows (content.kind = 'chunk') through their search_text
    (0120, ADR-093 C6), so a concept node alone is not in that lane."""
    content = (
        '{"kind": "chunk", "text": "tiering probe row"}'
        if chunk
        else '{"text": "tiering probe row"}'
    )
    search_text = "tiering probe row" if chunk else None
    label = f"tiering {tier} {project}"
    cur.execute(_NODE_SQL, (node_id, project, label, content, f"hash-{node_id}", tier, search_text))


def _seed_nodes(conn) -> None:
    with conn.cursor() as cur:
        for i, project in enumerate(PROJECTS):
            for tier in TIERS:
                node_id = _node_id(project, tier)
                _seed_node(cur, node_id, project, tier, chunk=False)
                _seed_node(cur, f"{node_id}::chunk", project, tier, chunk=True)
                # Distinct vectors per project so the similarity lane has a
                # deterministic order; the gate — not the ranking — is under test.
                vector = "[1,0,0]" if i == 0 else "[0,1,0]"
                cur.execute(
                    "INSERT INTO knowledge_node_embeddings "
                    "(node_id, model_name, dimension, embedding, content_hash) "
                    "VALUES (%s, %s, 3, %s::vector, %s)",
                    (node_id, MODEL, vector, f"hash-{node_id}"),
                )
    conn.commit()


def _seed_catalog(pool) -> None:
    from umrs.data_catalog.pg_store import CatalogMetaStore

    store = CatalogMetaStore(pool)
    for project in PROJECTS:
        for tier in TIERS:
            store.upsert(
                CatalogObjectMeta(
                    s3_key=_s3_key(project, tier),
                    object_etag="etag-vis",
                    state=CatalogState.READY,
                    owner="vis-live-user",
                    acl_scope=project,
                    updated_at="2026-09-13T00:00:00",
                    summary=f"tiering probe {tier} dataset",
                    deterministic_summary=f"tiering probe {tier} dataset",
                    detected_modality="structured",
                    tags=["tiering"],
                ),
                project,
            )
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            for project in PROJECTS:
                for tier in TIERS:
                    cur.execute(
                        "UPDATE catalog_objects SET visibility = %s "
                        "WHERE project_id = %s AND s3_key = %s",
                        (tier, project, _s3_key(project, tier)),
                    )
        conn.commit()
    finally:
        pool.putconn(conn)


@pytest.fixture(scope="module")
def pool():
    from psycopg2.pool import ThreadedConnectionPool
    from umrs.migrations import apply_migrations

    pg_pool = ThreadedConnectionPool(1, 4, ci_shape_db_url())
    conn = pg_pool.getconn()
    try:
        apply_migrations(conn)  # idempotent: applies only what is pending (0093 included)
        _cleanup(conn)
        _seed_nodes(conn)
    finally:
        pg_pool.putconn(conn)
    _seed_catalog(pg_pool)
    yield pg_pool
    conn = pg_pool.getconn()
    try:
        _cleanup(conn)
    finally:
        pg_pool.putconn(conn)
        pg_pool.closeall()


def _expected_visible(caller: str) -> set[tuple[str, str]]:
    """(project, tier) pairs the ADR-050 rule lets ``caller`` see."""
    other = PROJECT_B if caller == PROJECT_A else PROJECT_A
    return {(caller, "project"), (caller, "system"), (other, "system")}


def _pairs_from_node_ids(node_ids: list[str]) -> set[tuple[str, str]]:
    pairs = set()
    for node_id in node_ids:
        _prefix, project, tier = node_id.split("::")[:3]  # chunk rows add "::chunk"
        pairs.add((project, tier))
    return pairs


def _assert_family_rule(seen: set[tuple[str, str]], caller: str, lane: str) -> None:
    assert seen == _expected_visible(caller), f"{lane} lane for {caller}: {sorted(seen)}"
    assert not any(tier == "ambient" for _p, tier in seen), f"{lane}: ambient row served"


# ── schema truth (the R-02 defect) ──────────────────────────────────────────


def test_catalog_objects_carries_visibility_column(pool):
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_default FROM information_schema.columns "
                "WHERE table_name = 'catalog_objects' AND column_name = 'visibility'"
            )
            row = cur.fetchone()
            cur.execute(
                "SELECT 1 FROM pg_constraint WHERE conname = 'catalog_objects_visibility_check'"
            )
            check = cur.fetchone()
    finally:
        pool.putconn(conn)
    assert row is not None and "'project'" in str(row[0])
    assert check is not None


# ── rag lane ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("caller", PROJECTS)
def test_rag_lane_family_rule(pool, caller):
    from umrs.api.knowledge_query import _RAGSearchBackend

    hits = _RAGSearchBackend(pool).search_rag("tiering", project_id=caller, limit=50)
    _assert_family_rule(_pairs_from_node_ids([h["source_id"] for h in hits]), caller, "rag")


# ── graph lane (project default graph + legacy '' rows) ─────────────────────


@pytest.mark.parametrize("caller", PROJECTS)
def test_graph_lane_family_rule(pool, caller):
    from umrs.api.knowledge_query import _graph_backend_for

    backend, reason, selected, _excluded = _graph_backend_for(pool, caller, None, None)
    assert backend is not None, reason
    assert selected, "default graph must resolve for the caller project"
    hits = backend.search_knowledge_graph("tiering", project_id=caller, limit=50)
    _assert_family_rule(_pairs_from_node_ids([h["source_id"] for h in hits]), caller, "graph")


# ── embeddings lanes (table-reader + similarity) ────────────────────────────


@pytest.mark.parametrize("caller", PROJECTS)
def test_embeddings_table_reader_family_rule(pool, caller):
    from umrs.api.knowledge_query import _EmbeddingsSearchBackend

    hits = _EmbeddingsSearchBackend(pool).search_embeddings(MODEL, project_id=caller, limit=50)
    assert all(h["metadata"]["lane"] == "table-reader" for h in hits)
    _assert_family_rule(
        _pairs_from_node_ids([h["source_id"] for h in hits]), caller, "embeddings/table-reader"
    )


@pytest.mark.parametrize("caller", PROJECTS)
def test_embeddings_similarity_family_rule(pool, caller):
    from umrs.api.knowledge_query import _EmbeddingsSearchBackend

    hits = _EmbeddingsSearchBackend(pool).search_embeddings(
        "ignored",
        project_id=caller,
        limit=50,
        query_embedding=[1.0, 0.0, 0.0],
        model_name=MODEL,  # a query vector names its space (ADR-083)
    )
    assert all(h["metadata"]["lane"] == "similarity" for h in hits)
    _assert_family_rule(
        _pairs_from_node_ids([h["source_id"] for h in hits]), caller, "embeddings/similarity"
    )


# ── catalog lane ────────────────────────────────────────────────────────────


def _catalog_pairs(hits: list[Any]) -> set[tuple[str, str]]:
    pairs = set()
    for hit in hits:
        project, _uploads, name = hit.meta.s3_key.split("/")
        pairs.add((project, name.removeprefix("tiering-").removesuffix(".csv")))
    return pairs


@pytest.mark.parametrize("caller", PROJECTS)
def test_catalog_search_family_rule(pool, caller):
    from umrs.data_catalog.pg_store import CatalogMetaStore

    hits = CatalogMetaStore(pool).search(caller, "tiering", limit=50)
    _assert_family_rule(_catalog_pairs(hits), caller, "catalog")


@pytest.mark.parametrize("caller", PROJECTS)
def test_catalog_search_without_query_text_is_gated_too(pool, caller):
    """The no-token path (structured filters only) rides the same predicate."""
    from umrs.data_catalog.pg_store import CatalogMetaStore

    hits = CatalogMetaStore(pool).search(caller, "", limit=50, tag="tiering")
    _assert_family_rule(_catalog_pairs(hits), caller, "catalog/no-query")
