"""ADR-081 W2a term registry, Postgres side (``umrs.knowledge.term_registry``).

The hermetic half drives ``fetch_snapshot`` over a connection double; the
Postgres half runs when ``UMRS_TEST_DB_URL`` (or the CI job's
``POSTGRES_*``) names a migrated database (0146/0147 applied).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

import pytest
from umrs.knowledge import term_registry as reg
from umrs.knowledge.term_types import CORE_REF, Gap, RegistrySnapshot, Violation


class _Conn:
    """Answers the registry's reads from canned values."""

    def __init__(self, graph: dict[str, Any] | None, record: str | None, seeded: bool) -> None:
        self.graph, self.record, self.seeded = graph, record, seeded
        self.reads: list[str] = []

    async def fetchrow(self, sql: str, *_a: Any) -> dict[str, Any] | None:
        self.reads.append(sql)
        return self.graph

    async def fetchval(self, sql: str, *_a: Any) -> Any:
        self.reads.append(sql)
        return self.record if sql == reg._RECORD_SQL else self.seeded

    async def fetch(self, sql: str, *_a: Any) -> list[dict[str, Any]]:
        self.reads.append(sql)
        return []


def _snapshot(conn: _Conn) -> RegistrySnapshot | None:
    return asyncio.run(reg.fetch_snapshot(conn, project_id="p", graph_id="g"))


def test_unseeded_registry_gives_none():
    assert _snapshot(_Conn(None, None, seeded=False)) is None


def test_graph_without_ontology_resolves_to_core_in_warn_mode():
    graph = {"ontology_id": None, "ontology_version": None, "ontology_mode": "enforce"}
    snap = _snapshot(_Conn(graph, None, seeded=True))
    assert snap is not None
    assert (snap.ontology_ref, snap.mode, snap.record_id) == (CORE_REF, "warn", "bundled:r4-core@2")


def test_ontology_without_admitted_set_falls_back_to_core_warn():
    graph = {"ontology_id": "r4-default", "ontology_version": "1", "ontology_mode": "enforce"}
    snap = _snapshot(_Conn(graph, None, seeded=True))
    assert snap is not None and (snap.ontology_ref, snap.mode) == (CORE_REF, "warn")


def test_bound_ontology_keeps_the_graph_mode():
    graph = {"ontology_id": "demo", "ontology_version": "0.1", "ontology_mode": "enforce"}
    snap = _snapshot(_Conn(graph, "rec-1", seeded=False))
    assert snap is not None
    assert (snap.ontology_ref, snap.mode, snap.record_id) == ("demo@0.1", "enforce", "rec-1")


def test_snapshot_cache_reuses_a_read_including_none():
    conn = _Conn(None, None, seeded=False)
    cache = reg.SnapshotCache(ttl_s=60)

    async def twice() -> None:
        assert await cache.fetch(conn, project_id="p", graph_id="g") is None
        assert await cache.fetch(conn, project_id="p", graph_id="g") is None

    asyncio.run(twice())
    assert len(conn.reads) == 2  # graph row + seeded probe, once


def test_snapshot_cache_expires():
    cache = reg.SnapshotCache(ttl_s=-1)
    cache.put(("p", "g"), None)
    assert cache.get(("p", "g")) == (False, None)


def test_record_batch_refuses_an_unknown_actor_before_writing():
    class _NoWrite:
        async def execute(self, *_a: Any) -> None:
            raise AssertionError("wrote")

    with pytest.raises(ValueError, match="loop"):
        asyncio.run(
            reg.record_batch(
                _NoWrite(),
                batch_id="b",
                graph_id="g",
                project_id="p",
                writer="w",
                origin_actor="loop",
                ontology_ref=CORE_REF,
                mode="warn",
                counts={},
                violations=(),
            )
        )


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


needs_pg = pytest.mark.skipif(
    not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*"
)


async def _seed_record(conn: Any, tag: str, project: str | None) -> str:
    """One ontology record with two classes and a relation (unique ids)."""
    record = f"rec-{tag}-{project or 'global'}"
    await conn.execute(
        "INSERT INTO knowledge_ontologies (record_id, ontology_id, version, entity_types, "
        "relation_types, owner_id, project_id) VALUES ($1, $2, '1', '[]', '[]', 'u', $3)",
        record,
        f"demo-{tag}",
        project,
    )
    return record


async def _seed_terms(conn: Any, tag: str, record: str) -> None:
    a, b, r = f"{tag}:A", f"{tag}:B", f"{tag}:r"
    for term_id, kind, bucket, parents in (
        (a, "class", "entity", []),
        (b, "class", "file", [a]),
        (r, "relation", None, []),
    ):
        await conn.execute(
            "INSERT INTO knowledge_terms (term_id, iri, kind, label, bucket, parents) "
            "VALUES ($1, $2, $3, $1, $4, $5) ON CONFLICT DO NOTHING",
            term_id,
            f"http://{tag}/{term_id}",
            kind,
            bucket,
            parents,
        )
        await conn.execute(
            "INSERT INTO knowledge_ontology_terms (record_id, term_id) VALUES ($1, $2)",
            record,
            term_id,
        )
    for anc, desc, depth in ((a, a, 0), (b, b, 0), (a, b, 1), (r, r, 0)):
        await conn.execute(
            "INSERT INTO knowledge_term_closure VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            anc,
            desc,
            depth,
        )


async def _bind_graph(conn: Any, graph: str, project: str, tag: str, mode: str) -> None:
    await conn.execute(
        "INSERT INTO knowledge_graphs (graph_id, name, project_id, ontology_id, "
        "ontology_version, ontology_mode) VALUES ($1, $1, $2, $3, '1', $4)",
        graph,
        project,
        f"demo-{tag}",
        mode,
    )


async def _registry_round_trip(url: str, tag: str) -> dict[str, Any]:
    import asyncpg

    conn = await asyncpg.connect(url)
    tx = conn.transaction()
    await tx.start()
    try:
        record = await _seed_record(conn, tag, f"p-{tag}")
        await _seed_terms(conn, tag, record)
        await _bind_graph(conn, f"g-{tag}", f"p-{tag}", tag, "enforce")
        own = await reg.fetch_snapshot(conn, project_id=f"p-{tag}", graph_id=f"g-{tag}")
        other = await reg.fetch_snapshot(conn, project_id=f"q-{tag}", graph_id=f"g-{tag}")
        await reg.record_gaps(
            conn,
            graph_id=f"g-{tag}",
            writer="w",
            gaps=[Gap("attribute", "colour", "n1"), Gap("attribute", "colour", "n2")],
        )
        await reg.record_gaps(
            conn, graph_id=f"g-{tag}", writer="w", gaps=[Gap("attribute", "colour")]
        )
        gap_n = await conn.fetchval(
            "SELECT n FROM knowledge_vocab_gaps WHERE graph_id = $1 AND value = 'colour'",
            f"g-{tag}",
        )
        await reg.record_batch(
            conn,
            batch_id=f"b-{tag}",
            graph_id=f"g-{tag}",
            project_id=f"p-{tag}",
            writer="w",
            origin_actor="agent",
            ontology_ref=f"demo-{tag}@1",
            mode="enforce",
            counts={"nodes": 2},
            violations=[Violation("class_unknown", "n1", "x:Y")],
        )
        batch = await conn.fetchrow(
            "SELECT origin_actor, counts, violations FROM knowledge_write_batches WHERE batch_id = $1",
            f"b-{tag}",
        )
        version = await conn.fetchval(
            "SELECT graph_version FROM knowledge_graphs WHERE graph_id = $1", f"g-{tag}"
        )
        return {
            "own": own,
            "other": other,
            "gap_n": gap_n,
            "batch": dict(batch),
            "version": version,
        }
    finally:
        await tx.rollback()
        await conn.close()


@pytest.fixture(scope="module")
def round_trip() -> dict[str, Any]:
    if not _pg_url():
        pytest.skip("needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*")
    tag = f"t{uuid.uuid4().hex[:8]}"
    seen = asyncio.run(_registry_round_trip(_pg_url() or "", tag))
    return {**seen, "tag": tag}


@needs_pg
def test_bound_graph_reads_its_record_and_mode(round_trip):
    own, tag = round_trip["own"], round_trip["tag"]
    assert (own.ontology_ref, own.mode) == (f"demo-{tag}@1", "enforce")
    assert sorted(t.term_id for t in own.terms) == [f"{tag}:A", f"{tag}:B", f"{tag}:r"]
    assert (f"{tag}:A", f"{tag}:B", 1) in own.closure


@needs_pg
def test_term_rows_round_trip(round_trip):
    tag = round_trip["tag"]
    by_id = {t.term_id: t for t in round_trip["own"].terms}
    assert (by_id[f"{tag}:B"].bucket, by_id[f"{tag}:B"].parents) == ("file", (f"{tag}:A",))


@needs_pg
def test_another_project_never_reads_this_projects_record(round_trip):
    other = round_trip["other"]
    assert other is None or (other.ontology_ref, other.mode) == (CORE_REF, "warn")


@needs_pg
def test_gaps_accumulate(round_trip):
    assert round_trip["gap_n"] == 3


@needs_pg
def test_batch_row_and_graph_version(round_trip):
    batch = round_trip["batch"]
    assert batch["origin_actor"] == "agent"
    assert json.loads(batch["counts"]) == {"nodes": 2}
    assert json.loads(batch["violations"]) == [
        {"code": "class_unknown", "subject_id": "n1", "detail": "x:Y"}
    ]
    assert round_trip["version"] == 1
