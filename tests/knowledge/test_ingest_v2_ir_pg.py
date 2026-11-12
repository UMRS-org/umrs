"""ADR-116 D-2 on real Postgres: a new extractor version re-extracts.

``insert_ir`` is write-once and ``knowledge_ir`` is ``UNIQUE (artifact_id,
version)`` with every row at version 1, so before ADR-116 a bumped extractor
version could neither insert its IR (a unique violation read as "duplicate")
nor pass the already-understood probe (any current IR matched).

Review fixes pinned here: an attempt that died after the new IR's insert is
rebuilt on retry (the file node of the older IR does not count), and the
unreviewed semantic rows the older IR wrote are dropped after the rewrite.

Runs when ``UMRS_TEST_DB_URL`` (or the CI job's ``POSTGRES_*``) names a
migrated database, and skips cleanly otherwise.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge.ir import build_knowledge_ir
from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode, KnowledgeStore


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

PROJECT = "p-ing"


def _ir(artifact_id: str, version: str):
    return build_knowledge_ir(
        project_id=PROJECT,
        artifact_id=artifact_id,
        artifact_type="document",
        content=b"zirconia note",
        extracted={},
        extractor_version=version,
    )


async def _insert_old_ir(store: KnowledgeStore, aid: str) -> Any:
    old = _ir(aid, "1.1.0")
    await store.insert_ir(
        ir_id=old.ir_id,
        artifact_id=aid,
        ir_type="document",
        content=old.to_dict(),
        content_hash=old.content_hash,
    )
    return old


async def _steps(pool: Any, aid: str) -> dict[str, Any]:
    """What each step returned, for one comparison in the test."""
    store = KnowledgeStore(pool)
    new = _ir(aid, ka._KNOWLEDGE_EXTRACTOR_VERSION)
    result = ka.KnowledgeUnderstandResult()
    seen: dict[str, Any] = {
        "old_inserted": bool(await _insert_old_ir(store, aid)),
        # an IR from the older extractor does not count as understood
        "probe_before": await ka._existing_current_ir(store, aid),
        "new_inserted": await ka._store_ir(store, new, aid, "document", result),
        "probe_after": await ka._existing_current_ir(store, aid) == new.ir_id,
        # a retry under the same extractor is idempotent
        "retry_inserted": await ka._store_ir(store, new, aid, "document", result),
        "failed": result.failed_items,
    }
    rows = await pool.fetch(
        "SELECT ir_id, version, status FROM knowledge_ir WHERE artifact_id = $1 ORDER BY version",
        aid,
    )
    seen["rows"] = [(r["ir_id"] == new.ir_id, r["version"], r["status"]) for r in rows]
    return seen


def _node(aid: str, node_id: str, ir_id: str, **kw: Any) -> KnowledgeNode:
    content = {"artifact_id": aid, **kw.pop("content", {})}
    return KnowledgeNode(
        node_id=node_id,
        project_id=PROJECT,
        label=kw.pop("label", node_id),
        node_type=kw.pop("node_type", "entity"),
        content=content,
        content_hash=node_id,
        source_ir=ir_id,
        **kw,
    )


def _mentions(aid: str, target: str) -> KnowledgeEdge:
    return KnowledgeEdge(
        edge_id=f"edge-merge-{aid}-{target}",
        project_id=PROJECT,
        source_id=f"node-file-{aid}",
        target_id=target,
        relation_type="mentions",
        content={"semantic": True, "artifact_id": aid},
        status="proposed",
    )


async def _seed_old_extraction(store: KnowledgeStore, aid: str, old_ir: str) -> dict[str, str]:
    """The 1.1.0 run's rows: file node, a stale entity, a reviewed one, an
    established one, a recurring one, each with its mentions edge."""
    ids = {k: f"node-sem-{k}-{aid}" for k in ("stale", "reviewed", "established", "kept")}
    await store.upsert_node(_node(aid, f"node-file-{aid}", old_ir, node_type="file"))
    await store.upsert_node(_node(aid, ids["stale"], old_ir, status="proposed"))
    review = {"review": {"established_by": "alice"}}
    await store.upsert_node(_node(aid, ids["reviewed"], old_ir, status="proposed", content=review))
    await store.upsert_node(_node(aid, ids["established"], old_ir, status="established"))
    await store.upsert_node(_node(aid, ids["kept"], old_ir, status="proposed"))
    for node_id in ids.values():
        await store.upsert_edge(_mentions(aid, node_id))
    return ids


async def _reextract(pool: Any, aid: str) -> dict[str, Any]:
    store = KnowledgeStore(pool)
    old = await _insert_old_ir(store, aid)
    ids = await _seed_old_extraction(store, aid, old.ir_id)
    new = _ir(aid, ka._KNOWLEDGE_EXTRACTOR_VERSION)
    await ka._store_ir(store, new, aid, "document", ka.KnowledgeUnderstandResult())
    job = SimpleNamespace(job_id="j", project_id=PROJECT, spec={"graph_id": ""})
    # the attempt died here: the new IR is current, the graph is the old one
    seen: dict[str, Any] = {
        "probe_new_ir": await ka._graph_written(store, job, aid, new.ir_id),
    }
    await store.upsert_node(_node(aid, f"node-file-{aid}", new.ir_id, node_type="file"))
    await store.upsert_node(_node(aid, ids["kept"], new.ir_id, status="proposed"))
    new_edges = [_mentions(aid, ids["kept"])]
    await store.upsert_edge(new_edges[0])
    seen["probe_after_write"] = await ka._graph_written(store, job, aid, new.ir_id)
    await ka._drop_superseded_rows(store, (PROJECT, "", aid, new.ir_id), new_edges)
    nodes = await pool.fetch(
        "SELECT node_id FROM knowledge_nodes WHERE content->>'artifact_id' = $1", aid
    )
    edges = await pool.fetch(
        "SELECT target_id FROM knowledge_edges WHERE content->>'artifact_id' = $1", aid
    )
    short = {v: k for k, v in ids.items()} | {f"node-file-{aid}": "file"}
    seen["nodes"] = sorted(short[r["node_id"]] for r in nodes)
    seen["edges"] = sorted(short[r["target_id"]] for r in edges)
    return seen


async def _with_artifact(url: str, steps: Any) -> dict[str, Any]:
    import asyncpg

    pool = await asyncpg.create_pool(url, min_size=1, max_size=2, init=ka._init_json_codecs)
    aid = f"art-ing-{uuid.uuid4().hex[:12]}"
    try:
        await pool.execute(
            "INSERT INTO knowledge_artifacts (artifact_id, content_hash, kind, project_id) "
            "VALUES ($1, $2, 'document', $3)",
            aid,
            f"h-{aid}",
            PROJECT,
        )
        return await steps(pool, aid)
    finally:
        await pool.execute("DELETE FROM knowledge_nodes WHERE content->>'artifact_id' = $1", aid)
        await pool.execute("DELETE FROM knowledge_artifacts WHERE artifact_id = $1", aid)
        await pool.close()


def test_a_new_extractor_version_writes_a_new_current_ir():
    assert asyncio.run(_with_artifact(_pg_url(), _steps)) == {
        "old_inserted": True,
        "probe_before": None,
        "new_inserted": True,
        "probe_after": True,
        "retry_inserted": False,
        "failed": [],
        "rows": [(False, 1, "superseded"), (True, 2, "current")],
    }


def test_a_reextraction_that_died_is_rebuilt_and_drops_the_old_unreviewed_rows():
    assert asyncio.run(_with_artifact(_pg_url(), _reextract)) == {
        # the old file node does not make the new IR "already understood"
        "probe_new_ir": False,
        "probe_after_write": True,
        # the stale proposed entity and its edge are gone; the reviewed one
        # and established ones, and the one this extraction wrote again stay
        "nodes": ["established", "file", "kept", "reviewed"],
        "edges": ["established", "kept", "reviewed"],
    }
