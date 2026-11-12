"""ADR-116 review fixes on a real Postgres: the reads that must not see
superseded IRs or entity-resolution label vectors.

Runs when ``UMRS_TEST_DB_URL`` (or the CI job's ``POSTGRES_*``) names a
migrated database, and skips cleanly otherwise. Every row a test writes
belongs to a project id unique to that test and is deleted afterwards.

Pinned against the old code: ``KnowledgeStore.list_irs`` returned the
superseded IRs of a re-extracted artifact; the corpus-attrition report and
the catalogue's project vector count counted ``<embedder>#label`` vectors.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from umrs.knowledge.store import KnowledgeNode, KnowledgeStore


def _pg_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    host = os.environ.get("POSTGRES_HOST")
    if explicit or not host:
        return explicit or None
    user = os.environ.get("POSTGRES_USER", "test")
    secret = os.environ.get("POSTGRES_PASSWORD", "test")
    port, name = os.environ.get("POSTGRES_PORT", "5432"), os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{secret}@{host}:{port}/{name}"


pytestmark = pytest.mark.skipif(
    not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*"
)


async def _init_codecs(conn) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )


@pytest.fixture
async def db():
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(dsn=_pg_url(), min_size=1, max_size=2, init=_init_codecs)
    project = f"st-it-{uuid.uuid4().hex[:10]}"
    try:
        yield KnowledgeStore(pool), pool, project
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM knowledge_nodes WHERE project_id = $1", project)
            await conn.execute(
                "DELETE FROM knowledge_ir WHERE artifact_id IN "
                "(SELECT artifact_id FROM knowledge_artifacts WHERE project_id = $1)",
                project,
            )
            await conn.execute("DELETE FROM knowledge_artifacts WHERE project_id = $1", project)
        await pool.close()


async def _artifact(pool, project: str) -> str:
    aid = f"art-{uuid.uuid4().hex[:12]}"
    await pool.execute(
        "INSERT INTO knowledge_artifacts (artifact_id, content_hash, kind, project_id) "
        "VALUES ($1, $2, 'document', $3)",
        aid,
        f"h-{aid}",
        project,
    )
    return aid


async def test_list_irs_reads_current_irs_unless_asked_for_all(db):
    store, pool, project = db
    aid = await _artifact(pool, project)
    await store.insert_ir("ir-old-" + aid, aid, "document", {"v": 1}, "h1", version=1)
    await store.insert_ir("ir-new-" + aid, aid, "document", {"v": 2}, "h2", version=2)
    await pool.execute(
        "UPDATE knowledge_ir SET status = 'superseded' WHERE ir_id = $1", "ir-old-" + aid
    )

    current = await store.list_irs(project)
    typed = await store.list_irs(project, "document")
    every = await store.list_irs(project, status=None)

    assert [ir.ir_id for ir in current] == ["ir-new-" + aid]
    assert [ir.ir_id for ir in typed] == ["ir-new-" + aid]
    assert sorted(ir.ir_id for ir in every) == ["ir-new-" + aid, "ir-old-" + aid]
    assert await store.list_irs(project, "figure") == []


async def _node_with_vectors(store: KnowledgeStore, project: str) -> None:
    """One chunk vector and three label vectors on the project's nodes."""
    for i in range(3):
        node = KnowledgeNode(
            node_id=f"node-sem-n{i}-{project}",
            project_id=project,
            label=f"N{i}",
            node_type="entity",
            content={"artifact_id": "art-a", "semantic_type": "material"},
            content_hash=f"h{i}",
        )
        await store.upsert_node(node)
        await store.upsert_node_embedding(node.node_id, "/m/e5#label", [1.0, float(i)])
    await store.upsert_node_embedding(f"node-sem-n0-{project}", "/m/e5", [1.0, 0.0, 0.0])


async def test_the_attrition_report_leaves_label_vectors_out(db):
    from umrs.api.knowledge_attrition import _report_for_project

    store, pool, project = db
    await _node_with_vectors(store, project)
    report = await _report_for_project(pool, project)
    assert report["rows_total"] == 1


class _SyncPool:
    """The psycopg2 getconn/putconn shape the catalogue bridge takes."""

    def __init__(self, url: str) -> None:
        psycopg2 = pytest.importorskip("psycopg2")
        self._conn = psycopg2.connect(url)

    def getconn(self):
        return self._conn

    def putconn(self, _conn) -> None:
        return None

    def close(self) -> None:
        self._conn.close()


async def test_the_catalogue_vector_count_leaves_label_vectors_out(db):
    from umrs.data_catalog.links import count_vector_rows_for_project

    store, _pool, project = db
    await _node_with_vectors(store, project)
    sync = _SyncPool(_pg_url())
    try:
        assert count_vector_rows_for_project(sync, project) == 1
    finally:
        sync.close()
