"""K1.2 (knowledge-sharing plan, 2026-10-24): artifact contributions (0129).

On the CI-shape Postgres (migrated ``umrs_test``); skipped without one.
Before K1.2 there was no contribution table, ``/k2/artifacts`` listed every
project artifact to every member (another member's personal-graph files
included) and ignored its ``kind``/``status`` filters, and a second uploader
of the same bytes left no trace (per-project deduplication).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "umrs"
    / "migrations"
    / "0129_knowledge_artifact_contributions.sql"
)
_SECRET = "k1-contributions-secret"


def _db_url() -> str | None:
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


pytestmark = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


async def _codecs(conn: Any) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )


class _Acquired:
    async def __aenter__(self) -> Any:
        import asyncpg

        self.conn = await asyncpg.connect(_db_url())
        await _codecs(self.conn)
        return self.conn

    async def __aexit__(self, *_exc: object) -> None:
        await self.conn.close()


class _LoopFreePool:
    """An asyncpg "pool" that opens one connection per acquire: safe across
    the event loops asyncio.run and the TestClient portal each bring."""

    def acquire(self) -> _Acquired:
        return _Acquired()


@pytest.fixture(scope="module")
def sync_pool():
    from psycopg2.pool import ThreadedConnectionPool

    pool = ThreadedConnectionPool(1, 4, _db_url())
    yield pool
    pool.closeall()


@pytest.fixture()
def world(sync_pool, monkeypatch):
    """A fresh project with a personal graph for alice and a named project
    graph; artifacts written through the ingest chain's own insert site."""
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge.graph_registry import GraphRegistryStore, default_graph_id

    url = _db_url()
    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: url)
    project = f"k1c-{uuid.uuid4().hex[:8]}"
    registry = GraphRegistryStore(sync_pool)
    personal = registry.create_graph(
        f"alice-notes-{project}", owner_id="alice", project_id=None, kind="personal"
    ).graph_id
    shared = registry.create_graph(
        f"methods-{project}", owner_id="bob", project_id=project, kind="project"
    ).graph_id
    w = SimpleNamespace(
        project=project, personal=personal, shared=shared, default=default_graph_id(project)
    )
    yield w
    asyncio.run(_cleanup(project, [personal, shared]))


async def _cleanup(project: str, graphs: list[str]) -> None:
    async with _Acquired() as conn:
        await conn.execute("DELETE FROM knowledge_artifacts WHERE project_id = $1", project)
        await conn.execute("DELETE FROM knowledge_graphs WHERE graph_id = ANY($1)", graphs)
        if await conn.fetchval("SELECT to_regclass('jobs') IS NOT NULL"):
            await conn.execute("DELETE FROM jobs WHERE project_id = $1", project)


def _ingest(w: Any, *, user: str, graph: str, content: str, kind: str = "document") -> str:
    """One artifact through ``_store_artifacts_in_db_sync`` (the ingest chain's
    insert site) as *user* into *graph*; returns the stored id."""
    import hashlib

    from umrs.jobs import knowledge_adapters as ka

    job = SimpleNamespace(user_id=user, spec={"graph_id": graph, "origin_actor": "user"})
    desc = {
        "artifact_id": f"art-{uuid.uuid4().hex[:12]}",
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "kind": kind,
        "source_ref": f"knowledge/{w.project}/{content}",
        "metadata": {"filename": f"{content}.md"},
    }
    stored = ka._store_artifacts_in_db_sync(
        w.project, [desc], contributor=ka._ingest_contributor(job, w.project)
    )
    assert not stored.failed, stored.failed
    return desc["artifact_id"]


def _client(user: str, project: str, sync_pool: Any, monkeypatch: Any) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import knowledge_v2
    from umrs.api.files import get_project_id

    from r4_contracts.handshake_hmac import sign_claims

    async def _pool(_request: Any) -> Any:
        return _LoopFreePool()

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", _SECRET)
    monkeypatch.setattr("umrs.api._asyncpg_pool.knowledge_asyncpg_pool", _pool)
    app = FastAPI()
    app.state.umrs_db_pool = sync_pool
    app.include_router(knowledge_v2.router)
    app.dependency_overrides[get_project_id] = lambda: project
    return TestClient(
        app,
        raise_server_exceptions=False,
        headers={
            "X-User-ID": user,
            "X-Verified-Project-ID": project,
            "X-Verified-Is-Admin": "false",
            "X-Verified-HMAC": sign_claims(_SECRET, user, project, False),
        },
    )


def _listed(client: Any, **params: Any) -> dict[str, dict[str, Any]]:
    resp = client.get("/umrs/k2/artifacts", params=params)
    assert resp.status_code == 200, resp.text
    return {a["artifact_id"]: a for a in resp.json()["artifacts"]}


def _who(artifact: dict[str, Any]) -> set[tuple[str, str]]:
    return {(c["user_id"], c["graph_id"]) for c in artifact["contributions"]}


# ── migration ───────────────────────────────────────────────────────────────


def test_migration_is_idempotent_and_keyed_per_contributor() -> None:
    async def _run() -> list[str]:
        async with _Acquired() as conn:
            sql = _MIGRATION.read_text()
            await conn.execute(sql)
            await conn.execute(sql)
            rows = await conn.fetch(
                "SELECT a.attname FROM pg_index i JOIN pg_attribute a "
                "ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                "WHERE i.indrelid = 'knowledge_artifact_contributions'::regclass "
                "AND i.indisprimary"
            )
            return sorted(str(r["attname"]) for r in rows)

    assert asyncio.run(_run()) == ["artifact_id", "graph_id", "user_id"]


# ── attribution through the insert site ─────────────────────────────────────


def test_upload_lists_its_contributor(world, sync_pool, monkeypatch) -> None:
    aid = _ingest(world, user="alice", graph=world.default, content="alpha")
    listed = _listed(_client("bob", world.project, sync_pool, monkeypatch))
    assert _who(listed[aid]) == {("alice", world.default)}
    assert listed[aid]["contributions"][0]["origin_actor"] == "user"


def test_second_uploader_of_the_same_bytes_is_recorded(world, sync_pool, monkeypatch) -> None:
    first = _ingest(world, user="alice", graph=world.default, content="same")
    second = _ingest(world, user="bob", graph=world.shared, content="same")
    assert second == first  # deduplicated to the stored artifact
    listed = _listed(_client("alice", world.project, sync_pool, monkeypatch))
    assert _who(listed[first]) == {("alice", world.default), ("bob", world.shared)}


def test_personal_graph_upload_is_hidden_from_another_member(
    world, sync_pool, monkeypatch
) -> None:
    private = _ingest(world, user="alice", graph=world.personal, content="diary")
    public = _ingest(world, user="alice", graph=world.default, content="paper")
    for_bob = _listed(_client("bob", world.project, sync_pool, monkeypatch))
    assert public in for_bob and private not in for_bob
    for_alice = _listed(_client("alice", world.project, sync_pool, monkeypatch))
    assert private in for_alice


def test_a_shared_copy_never_reveals_the_personal_contribution(
    world, sync_pool, monkeypatch
) -> None:
    aid = _ingest(world, user="alice", graph=world.personal, content="both")
    _ingest(world, user="bob", graph=world.shared, content="both")
    listed = _listed(_client("bob", world.project, sync_pool, monkeypatch))
    assert _who(listed[aid]) == {("bob", world.shared)}


def test_readable_artifact_ids_applies_the_same_rule(world, sync_pool, monkeypatch) -> None:
    from umrs.api._artifact_scope import hidden_artifact_graphs
    from umrs.knowledge.contributions import readable_artifact_ids

    private = _ingest(world, user="alice", graph=world.personal, content="p1")
    public = _ingest(world, user="alice", graph=world.default, content="p2")
    client = _client("bob", world.project, sync_pool, monkeypatch)
    request = SimpleNamespace(headers=client.headers, app=SimpleNamespace(state=SimpleNamespace()))
    hidden = hidden_artifact_graphs(request, sync_pool, world.project)
    assert world.personal in hidden
    ids = asyncio.run(readable_artifact_ids(_LoopFreePool(), world.project, hidden))
    assert public in ids and private not in ids


# ── filters ─────────────────────────────────────────────────────────────────


def test_created_by_and_graph_filters_exclude_others(world, sync_pool, monkeypatch) -> None:
    mine = _ingest(world, user="alice", graph=world.default, content="mine")
    theirs = _ingest(world, user="bob", graph=world.shared, content="theirs")
    client = _client("alice", world.project, sync_pool, monkeypatch)
    assert set(_listed(client, created_by="alice")) == {mine}
    assert set(_listed(client, graph_id=world.shared)) == {theirs}


def test_kind_and_since_filters_work(world, sync_pool, monkeypatch) -> None:
    doc = _ingest(world, user="alice", graph=world.default, content="doc")
    code = _ingest(world, user="alice", graph=world.default, content="code", kind="code")
    client = _client("alice", world.project, sync_pool, monkeypatch)
    assert set(_listed(client, kind="code")) == {code}
    assert set(_listed(client)) == {doc, code}
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert _listed(client, since=future) == {}


def test_status_filter_is_honoured(world, sync_pool, monkeypatch) -> None:
    aid = _ingest(world, user="alice", graph=world.default, content="retired")

    async def _archive() -> None:
        async with _Acquired() as conn:
            await conn.execute(
                "UPDATE knowledge_artifacts SET status = 'archived' WHERE artifact_id = $1", aid
            )

    asyncio.run(_archive())
    client = _client("alice", world.project, sync_pool, monkeypatch)
    assert aid not in _listed(client)
    assert aid in _listed(client, status="archived")


# ── backfill ────────────────────────────────────────────────────────────────


def _seed_history(world: Any, sync_pool: Any) -> dict[str, str]:
    """Pre-0129 history: artifacts with no contribution, one ingest job each
    for alice (personal graph) and bob (no graph), one untraced artifact."""
    from umrs.jobs.schema import CREATE_ATTEMPTS_TABLE, CREATE_JOBS_TABLE

    ids = {k: f"art-{k}-{uuid.uuid4().hex[:8]}" for k in ("alice", "bob", "orphan")}
    conn = sync_pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(CREATE_JOBS_TABLE)
            cur.execute(CREATE_ATTEMPTS_TABLE)
            for key, aid in ids.items():
                cur.execute(
                    "INSERT INTO knowledge_artifacts (artifact_id, content_hash, kind, "
                    "project_id, source_ref, status, metadata) VALUES "
                    "(%s, %s, 'document', %s, %s, 'active', %s::jsonb)",
                    (aid, uuid.uuid4().hex, world.project, f"k/{key}.md", json.dumps({})),
                )
            for user, graph in (("alice", world.personal), ("bob", "")):
                job = f"job-{uuid.uuid4().hex[:12]}"
                spec = {"graph_id": graph} if graph else {}
                cur.execute(
                    "INSERT INTO jobs (job_id, project_id, user_id, job_type, spec, "
                    "idempotency_key) VALUES (%s, %s, %s, 'knowledge_ingest', %s::jsonb, %s)",
                    (job, world.project, user, json.dumps(spec), job),
                )
                cur.execute(
                    "INSERT INTO job_attempts (attempt_id, job_id, attempt_no, backend, "
                    "metrics) VALUES (%s, %s, 1, 'compose', %s::jsonb)",
                    (f"att-{job}", job, json.dumps({"artifacts": [{"artifact_id": ids[user]}]})),
                )
    finally:
        sync_pool.putconn(conn)
    return ids


def _plan(sync_pool: Any, project: str, *, apply: bool = False) -> tuple[Any, int]:
    from umrs.knowledge.contribution_backfill import apply_backfill, plan_backfill

    conn = sync_pool.getconn()
    try:
        plan = plan_backfill(conn, [project])
        return plan, (apply_backfill(conn, plan) if apply else 0)
    finally:
        sync_pool.putconn(conn)


def test_backfill_dry_run_traces_jobs_and_counts_the_untraced(world, sync_pool) -> None:
    ids = _seed_history(world, sync_pool)
    plan, written = _plan(sync_pool, world.project)
    assert {(c.artifact_id, c.graph_id, c.user_id) for c in plan.traced} == {
        (ids["alice"], world.personal, "alice"),
        (ids["bob"], world.default, "bob"),
    }
    assert [(c.artifact_id, c.user_id, c.origin_actor) for c in plan.untraced] == [
        (ids["orphan"], "", "unknown")
    ]
    assert (plan.summary()["untraced_placeholders"], written) == (1, 0)


def test_backfill_apply_is_idempotent_and_keeps_personal_items_hidden(
    world, sync_pool, monkeypatch
) -> None:
    ids = _seed_history(world, sync_pool)
    _plan_before, written = _plan(sync_pool, world.project, apply=True)
    assert written == 3
    again, _ = _plan(sync_pool, world.project)
    assert (again.traced, again.untraced, again.already) == ([], [], 3)
    for_bob = _listed(_client("bob", world.project, sync_pool, monkeypatch))
    assert ids["alice"] not in for_bob  # backfilled into alice's personal graph
    assert {ids["bob"], ids["orphan"]} <= set(for_bob)


def test_an_attributed_row_replaces_the_placeholder(world, sync_pool, monkeypatch) -> None:
    from umrs.knowledge.contributions import Contribution, record_contribution

    aid = _ingest(world, user="alice", graph=world.default, content="placeheld")

    async def _run() -> None:
        async with _Acquired() as conn:
            await conn.execute(
                "DELETE FROM knowledge_artifact_contributions WHERE artifact_id = $1", aid
            )
            await record_contribution(conn, Contribution(aid, world.default, world.project))
            await record_contribution(
                conn, Contribution(aid, world.default, world.project, "carol", "user")
            )

    asyncio.run(_run())
    listed = _listed(_client("bob", world.project, sync_pool, monkeypatch))
    assert _who(listed[aid]) == {("carol", world.default)}


def test_backfill_cli_is_a_dry_run_by_default(world, sync_pool, capsys) -> None:
    from umrs.knowledge.contribution_backfill import main

    _seed_history(world, sync_pool)
    assert main(["--db-url", _db_url() or "", "--project", world.project]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["mode"], report["traced"], report["untraced_placeholders"]) == (
        "dry-run",
        2,
        1,
    )
    plan, _ = _plan(sync_pool, world.project)
    assert len(plan.traced) == 2  # nothing was written
