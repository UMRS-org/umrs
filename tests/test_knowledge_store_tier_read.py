"""D-35 read side (2026-11-24): list_nodes/list_edges carry the stored tier.

Migration 0106 gave nodes and edges ``status``/``confidence`` and the upserts
write them, but the ``list_nodes``/``list_edges`` SELECTs did not read them:
every row came back with the dataclass default (``established``, 0.0), so a
proposed loop finding read as established on ``/graph``, ``/graph-pg`` and
every consumer of the store reads.

The hermetic half drives the store over a pool double that returns only the
columns the SELECT names; the Postgres half runs when ``UMRS_TEST_DB_URL``
(or the CI job's ``POSTGRES_*``) names a migrated database.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode, KnowledgeStore

_STORED_NODE = {
    "node_id": "n-1",
    "project_id": "p-tier",
    "label": "finding",
    "node_type": "concept",
    "content": {},
    "content_hash": "h",
    "source_ir": None,
    "created_at": None,
    "updated_at": None,
    "graph_id": "",
    "status": "proposed",
    "confidence": 0.4,
}
_STORED_EDGE = {
    "edge_id": "e-1",
    "project_id": "p-tier",
    "source_id": "n-1",
    "target_id": "n-2",
    "relation_type": "supports",
    "content": {},
    "created_at": None,
    "graph_id": "",
    "status": "proposed",
    "confidence": 0.3,
}


def _selected_columns(sql: str) -> set[str]:
    head = sql.split("FROM", 1)[0].replace("SELECT", "", 1)
    return {c.strip() for c in re.split(r"[,\s]+", head) if c.strip()}


class _Conn:
    """Returns the stored row cut down to the columns the SELECT names."""

    async def fetch(self, sql: str, *_params: Any) -> list[dict[str, Any]]:
        stored = _STORED_NODE if "knowledge_nodes" in sql else _STORED_EDGE
        wanted = _selected_columns(sql)
        return [{k: v for k, v in stored.items() if k in wanted}]

    async def __aenter__(self) -> _Conn:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _Pool:
    def acquire(self) -> _Conn:
        return _Conn()


def test_list_nodes_reads_the_stored_tier():
    nodes = asyncio.run(KnowledgeStore(_Pool()).list_nodes("p-tier"))
    assert [(n.status, n.confidence) for n in nodes] == [("proposed", 0.4)]


def test_list_edges_reads_the_stored_tier():
    edges = asyncio.run(KnowledgeStore(_Pool()).list_edges("p-tier"))
    assert [(e.status, e.confidence) for e in edges] == [("proposed", 0.3)]


def test_studio_graph_payloads_carry_the_tier():
    from umrs.api.knowledge import _durable_edge_payload, _durable_node_payload

    node = _durable_node_payload(KnowledgeNode(**_STORED_NODE))
    edge = _durable_edge_payload(KnowledgeEdge(**_STORED_EDGE))
    assert (node["status"], node["confidence"]) == ("proposed", 0.4)
    assert (edge["status"], edge["confidence"]) == ("proposed", 0.3)


# ── real Postgres ────────────────────────────────────────────────────────────


def _pg_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    host = os.environ.get("POSTGRES_HOST")
    if explicit or not host:
        return explicit or None
    user = os.environ.get("POSTGRES_USER", "test")
    secret = os.environ.get("POSTGRES_PASSWORD", "test")
    port = os.environ.get("POSTGRES_PORT", "5432")
    name = os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{secret}@{host}:{port}/{name}"


async def _codecs(conn: Any) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )


async def _write_and_read(url: str, project: str) -> SimpleNamespace:
    import asyncpg

    pool = await asyncpg.create_pool(url, min_size=1, max_size=2, init=_codecs)
    store = KnowledgeStore(pool)
    try:
        for node_id, status in (("n-prop", "proposed"), ("n-est", "established")):
            await store.upsert_node(
                KnowledgeNode(
                    node_id=f"{node_id}-{project}",
                    project_id=project,
                    label=node_id,
                    node_type="concept",
                    status=status,
                    confidence=0.25,
                )
            )
        await store.upsert_edge(
            KnowledgeEdge(
                edge_id=f"e-prop-{project}",
                project_id=project,
                source_id=f"n-prop-{project}",
                target_id=f"n-est-{project}",
                relation_type="supports",
                status="proposed",
                confidence=0.5,
            )
        )
        return SimpleNamespace(
            nodes=await store.list_nodes(project),
            edges=await store.list_edges(project),
        )
    finally:
        await pool.execute("DELETE FROM knowledge_edges WHERE project_id = $1", project)
        await pool.execute("DELETE FROM knowledge_nodes WHERE project_id = $1", project)
        await pool.close()


@pytest.mark.skipif(not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*")
def test_a_proposed_row_reads_back_proposed_on_postgres():
    project = f"p-tier-{uuid.uuid4().hex[:8]}"
    seen = asyncio.run(_write_and_read(_pg_url() or "", project))
    tiers = {n.label: (n.status, round(n.confidence, 2)) for n in seen.nodes}
    assert tiers == {"n-prop": ("proposed", 0.25), "n-est": ("established", 0.25)}
    assert [(e.status, round(e.confidence, 2)) for e in seen.edges] == [("proposed", 0.5)]
