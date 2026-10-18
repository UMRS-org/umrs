"""H-10.1: module-global per-project state in the sidecar (2026-10-18).

Hardening plan H-10.1 (Appendix A §A.3); the decision per item is recorded in
``docs/development/dormant-modules.md`` ("Module-global per-project state").
Each block pins the defect it removes:

1. **M3.5 explorer, retired.** ``api/knowledge.py`` served module-global
   controllers over ONE graph (project "default") to every caller: a project
   B caller read its node counts and neighbours, and ``POST /correction``
   wrote edges into it with a client-asserted ``corrected_by``.
2. **Verification chat, retired.** ``POST /umrs/verification/chat`` reasoned
   over the same graph whatever the caller's project; its controller had no
   session store, so ``/sessions`` always answered ``[]``.
3. **Audit trace, project-keyed.** ``GET /umrs/ops/audit/trace`` filtered
   the newest ``limit`` events of every project by project afterwards, so a
   project's own events vanished behind newer events of other projects.
4. **Skill registry, DB-backed.** Nothing called ``set_skill_store``: the
   REST layer ran the in-memory registry, registrations died with the
   process, and the D-27 proposals the knowledge jobs write into ``skills``
   never reached ``/umrs/skills/pending``. ``SkillStore.search`` also
   selected one column too few for its row mapper.
5. **Store DSN chain.** The checkpoint, trace and egress-audit singletons
   read ``UMRS_DB_URL`` / ``DATABASE_URL`` only; umrs-front sets the
   component vars (``DB_HOST``, ``UMRS_JOBS_DB``, ...), so they ran in memory.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h10-module-state-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _headers("alice", "proj-a")
BOB = _headers("bob", "proj-b")


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


def _client(*routers: Any) -> TestClient:
    app = FastAPI()
    for router in routers:
        app.include_router(router)
    return TestClient(app)


# ── 1/2. M3.5 explorer and verification chat ──────────────────────────────

_SEEDED = ("h10-secret-a", "h10-secret-b")


def _drop_seeded(graph: Any) -> None:
    for node_id in _SEEDED:
        graph._nodes.pop(node_id, None)
        graph._outgoing.pop(node_id, None)
        graph._incoming.pop(node_id, None)
    graph._edges = {
        key: edge
        for key, edge in graph._edges.items()
        if edge.source_id not in _SEEDED and edge.target_id not in _SEEDED
    }


@pytest.fixture
def default_project_graph():
    """Project "default"'s nodes in the module-global explorer graph.

    Only the pre-H-10.1 module has that graph; once retired there is nothing
    to seed and every request below must miss.
    """
    from umrs.api import knowledge
    from umrs.knowledge.graph_types import RelationType

    graph = getattr(knowledge, "_m35_knowledge_graph", None)
    if graph is None:
        yield None
        return
    graph.add_node(node_id="h10-secret-a", node_type="paper", title="default secret")
    graph.add_node(node_id="h10-secret-b", node_type="paper", title="default other")
    graph.add_edge(
        source_id="h10-secret-a",
        target_id="h10-secret-b",
        relation_type=RelationType.RELATED_TO,
    )
    yield graph
    _drop_seeded(graph)


def test_project_b_reads_no_shared_explorer_counts(strict, default_project_graph):
    from umrs.api import knowledge

    resp = _client(knowledge.router).get("/umrs/knowledge/dashboard", headers=BOB)
    assert resp.status_code == 404, resp.text
    assert "graph_node_count" not in resp.text


def test_project_b_reads_no_shared_explorer_neighbours(strict, default_project_graph):
    from umrs.api import knowledge

    resp = _client(knowledge.router).get(
        "/umrs/knowledge/graph/h10-secret-a/neighbors", headers=BOB
    )
    assert resp.status_code == 404, resp.text
    assert "h10-secret-b" not in resp.text


def test_project_b_cannot_write_the_shared_explorer_graph(strict, default_project_graph):
    """A forged ``corrected_by`` and a write into project "default"."""
    from umrs.api import knowledge

    resp = _client(knowledge.router).post(
        "/umrs/knowledge/correction",
        headers=BOB,
        json={
            "correction_type": "relates",
            "corrected_by": "alice",
            "reason": "h10",
            "affected_object_ids": [],
            "details": {"source_id": "h10-secret-a", "target_id": "h10-bob-node"},
        },
    )
    assert resp.status_code in (404, 405), resp.text
    assert "alice" not in resp.text


def test_verification_chat_does_not_reason_over_the_shared_graph():
    """The chat route went first (H-10.1); its module went with the last
    findings stub (follow-up, 2026-10-18)."""
    import importlib.util

    assert importlib.util.find_spec("umrs.api.verification") is None


def _mounted(router: Any) -> set[tuple[str, str]]:
    return {(method, route.path) for route in router.routes for method in route.methods}


_RETIRED = {
    ("GET", "/umrs/knowledge/dashboard"),
    ("GET", "/umrs/knowledge/objects"),
    ("GET", "/umrs/knowledge/objects/{object_id}"),
    ("GET", "/umrs/knowledge/graph/{object_id}/neighbors"),
    ("POST", "/umrs/knowledge/correction"),
    ("POST", "/umrs/verification/chat"),
    ("GET", "/umrs/verification/sessions"),
    ("GET", "/umrs/verification/sessions/{session_id}"),
}
_KEPT = {
    ("GET", "/umrs/knowledge/readiness"),
    ("GET", "/umrs/knowledge/graph"),
}


def test_explorer_and_chat_routes_are_retired_and_the_rest_stay():
    from umrs.api import knowledge

    mounted = _mounted(knowledge.router)
    assert not _RETIRED & mounted
    assert _KEPT.issubset(mounted)


def test_no_module_global_explorer_state_remains():
    from umrs.api import knowledge

    leftovers = [
        name
        for module in (knowledge,)
        for name in vars(module)
        if name.startswith("_m35_")
        or name in {"_dashboard_controller", "_correction_controller"}
        or name == "_verification_controller"
    ]
    assert leftovers == []


# ── 3. audit trace ────────────────────────────────────────────────────────


def _emit_own_then_foreign(audit_events: Any) -> None:
    audit_events.emit_audit("h10.own", actor="bob", project_id="proj-b")
    for _ in range(60):
        audit_events.emit_audit("h10.noise", actor="alice", project_id="proj-a")


def test_audit_trace_buffer_keeps_own_events_behind_newer_foreign_ones(strict, monkeypatch):
    from umrs.api import audit_events, ops

    monkeypatch.setattr(audit_events, "_audit_store", None)
    monkeypatch.setattr(audit_events, "_audit_buffer", [])
    _emit_own_then_foreign(audit_events)

    resp = _client(ops.router).get("/umrs/ops/audit/trace", headers=BOB, params={"limit": 50})
    assert resp.status_code == 200, resp.text
    assert [e["event"] for e in resp.json()["events"]] == ["h10.own"]


class _ProjectAwareStore:
    """Honours the filters like ``AuditStore.query`` (newest first)."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.rows.append(record)

    def query(self, *, limit: int, project_id: str = "", **_filters: Any) -> list:
        newest = list(reversed(self.rows))
        return [r for r in newest if not project_id or r["project_id"] == project_id][:limit]


def test_audit_trace_store_query_is_scoped_before_the_limit(strict, monkeypatch):
    from umrs.api import audit_events, ops

    store = _ProjectAwareStore()
    monkeypatch.setattr(audit_events, "_audit_store", store)
    monkeypatch.setattr(audit_events, "_audit_buffer", [])
    _emit_own_then_foreign(audit_events)

    resp = _client(ops.router).get("/umrs/ops/audit/trace", headers=BOB, params={"limit": 50})
    assert resp.status_code == 200, resp.text
    assert [e["event"] for e in resp.json()["events"]] == ["h10.own"]


# ── 4. skill registry ─────────────────────────────────────────────────────

# Every init ``_init_deep_stores`` ran before H-10.1; stubbed so a test boots
# only the skill registry (the others would keep the fake pool globally).
_OTHER_INITS = (
    "_init_job_store",
    "_init_catalog_meta_store",
    "_init_model_store",
    "_init_mcp_store",
    "_init_audit_store",
    "_init_governance_store",
    "_init_research_sessions_store",
    "_init_companion_trust_store",
    "_init_usage_pool",
    "_init_loop_plan_host",
)


def _boot_deep_stores(monkeypatch, pool: Any) -> None:
    from umrs.server import http_sidecar

    for name in _OTHER_INITS:
        monkeypatch.setattr(http_sidecar, name, lambda *_a, **_k: None)
    http_sidecar._init_deep_stores(SimpleNamespace(state=SimpleNamespace()), pool)


def test_sidecar_boot_backs_the_skill_registry_with_the_pool(monkeypatch):
    from umrs.api import skills

    monkeypatch.setattr(skills, "_skill_store", None)
    pool = object()  # boot never borrows a connection
    _boot_deep_stores(monkeypatch, pool)

    registry = skills._get_registry()
    assert registry is not skills._registry
    assert getattr(registry, "_pool", None) is pool


@pytest.fixture
def skills_pool(monkeypatch):
    """A psycopg2 pool on the migrated CI-shaped database, else skip."""
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn

    dsn, _local = _resolve_knowledge_dsn()
    if not dsn:
        pytest.skip("needs Postgres (CI shape: POSTGRES_* or DATABASE_URL)")
    psycopg2 = pytest.importorskip("psycopg2")
    from psycopg2.pool import ThreadedConnectionPool

    try:
        pool = ThreadedConnectionPool(1, 4, dsn)
    except psycopg2.Error as exc:
        pytest.skip(f"Postgres unreachable: {exc}")
    _purge_h10_skills(pool)
    from umrs.api import skills

    monkeypatch.setattr(skills, "_skill_store", None)
    yield pool
    _purge_h10_skills(pool)
    pool.closeall()


def _sql(pool: Any, statement: str, params: tuple = ()) -> None:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(statement, params)
        conn.commit()
    finally:
        pool.putconn(conn)


def _purge_h10_skills(pool: Any) -> None:
    _sql(pool, "DELETE FROM skills WHERE name LIKE %s", ("h10-%",))


def _insert_loop_proposal(pool: Any, name: str, project_id: str) -> None:
    """The row ``jobs/knowledge_adapters._compile_skill_proposals`` writes."""
    _sql(
        pool,
        "INSERT INTO skills (skill_id, name, version, scope, owner_id, project_id,"
        " skill_md, manifest, review_state, enabled)"
        " VALUES (%s, %s, '1.0.0', 'project', 'compile-knowledge-skills', %s,"
        " %s, %s::jsonb, %s::jsonb, false)",
        (
            f"skl-{name}",
            name,
            project_id,
            f"# {name}",
            json.dumps({"_compiled_skill": True, "status": "pending_review", "name": name}),
            '{"status": "pending_review"}',
        ),
    )


def _names(resp: Any) -> list[str]:
    assert resp.status_code == 200, resp.text
    return [skill["manifest"]["name"] for skill in resp.json()]


def test_loop_proposal_reaches_the_review_plane(strict, monkeypatch, skills_pool):
    from umrs.api import skills

    _insert_loop_proposal(skills_pool, "h10-fit-peaks", "proj-a")
    _boot_deep_stores(monkeypatch, skills_pool)
    client = _client(skills.router)

    assert "h10-fit-peaks" in _names(client.get("/umrs/skills/pending", headers=ALICE))
    assert "h10-fit-peaks" not in _names(client.get("/umrs/skills/pending", headers=BOB))
    approved = client.post(
        "/umrs/skills/skl-h10-fit-peaks/review",
        headers=ALICE,
        json={"decision": "approved", "test_receipt": {"exit_code": 0, "tests_present": True}},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["enabled"] is True


def test_registered_skill_survives_a_restart(strict, monkeypatch, skills_pool):
    from umrs.api import skills

    _boot_deep_stores(monkeypatch, skills_pool)
    created = _client(skills.router).post(
        "/umrs/skills",
        headers=ALICE,
        json={"manifest": {"name": "h10-user-skill", "version": "1.0.0"}, "scope": "user"},
    )
    assert created.status_code == 201, created.text

    monkeypatch.setattr(skills, "_registry", skills.SkillRegistry())  # new process
    _boot_deep_stores(monkeypatch, skills_pool)
    client = _client(skills.router)
    assert "h10-user-skill" in _names(client.get("/umrs/skills", headers=ALICE))
    assert "h10-user-skill" not in _names(client.get("/umrs/skills", headers=BOB))


def test_skill_store_search_maps_its_rows(skills_pool):
    """``search`` selected 11 columns for a 12-column row mapper."""
    from umrs.skills import SkillScope
    from umrs.skills.store import SkillStore

    conn = skills_pool.getconn()
    try:
        store = SkillStore(conn)
        store.register_skill(name="h10-searchable", scope=SkillScope.USER, owner_id="alice")
        conn.commit()
        hits = store.search("h10-search")
    finally:
        skills_pool.putconn(conn)
    assert [skill.manifest.name for skill in hits] == ["h10-searchable"]


def test_skill_search_route_reads_the_store(strict, monkeypatch, skills_pool):
    from umrs.api import skills

    _boot_deep_stores(monkeypatch, skills_pool)
    client = _client(skills.router)
    client.post(
        "/umrs/skills",
        headers=ALICE,
        json={"manifest": {"name": "h10-searchable", "version": "1.0.0"}, "scope": "user"},
    )
    hits = client.get("/umrs/skills/search", headers=ALICE, params={"q": "h10-search"})
    assert _names(hits) == ["h10-searchable"]


# ── 5. store DSN chain ────────────────────────────────────────────────────

_FRONT_COMPONENTS = {
    "DB_HOST": "db.h10.invalid",
    "DB_PORT": "5432",
    "UMRS_JOBS_DB": "umrs_db",
    "DB_SUPERUSER": "umrs",
    "DB_SUPERUSER_PASSWORD": "pw",
}
_EXPLICIT_DSN_VARS = ("UMRS_DB_URL", "DATABASE_URL", "UMRS_DATA_CATALOG_DB")
_OTHER_COMPONENTS = ("POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")


@pytest.fixture
def front_env(monkeypatch) -> str:
    """The umrs-front container shape: component vars only, no DSN var."""
    for var in _EXPLICIT_DSN_VARS + _OTHER_COMPONENTS:
        monkeypatch.delenv(var, raising=False)
    for var, value in _FRONT_COMPONENTS.items():
        monkeypatch.setenv(var, value)
    return "postgresql://umrs:pw@db.h10.invalid:5432/umrs_db"


def test_checkpoint_store_uses_the_front_dsn(front_env, monkeypatch):
    from umrs.api import loop

    monkeypatch.setattr(loop, "_checkpoint_store_singleton", None)
    store = loop._checkpoint_store()
    assert store.db_enabled
    assert store._db_url == front_env


def test_trace_store_uses_the_front_dsn(front_env, monkeypatch):
    from umrs.api import traces

    monkeypatch.setattr(traces, "_trace_store", None)
    store = traces.get_trace_store()
    assert store.db_enabled
    assert store._db_url == front_env


def test_egress_audit_store_uses_the_front_dsn(front_env):
    from umrs.api import egress_audit

    store = egress_audit._EgressAuditStore()
    assert store.db_enabled
    assert store._db_url == front_env


def test_unconfigured_stores_stay_in_memory(front_env, monkeypatch):
    from umrs.api._asyncpg_pool import sync_store_dsn

    for var in _FRONT_COMPONENTS:
        monkeypatch.delenv(var, raising=False)
    assert sync_store_dsn() is None


class _DownCheckpointStore:
    def save_checkpoint(self, *_args: Any, **_kwargs: Any) -> None:
        raise ConnectionError("checkpoint database unreachable")


def test_iteration_checkpoint_failure_never_fails_the_iteration(monkeypatch):
    """With the store on Postgres a database error must stay a warning."""
    from umrs.knowledge import loop_hosting
    from umrs.knowledge.loop import LoopIteration, ResearchLoop

    wiring = SimpleNamespace(checkpoint_store=_DownCheckpointStore)
    monkeypatch.setattr(loop_hosting, "_wiring", lambda: wiring)
    loop = ResearchLoop(loop_id="loop-h10", project_id="proj-a", max_iterations=2)
    iteration = LoopIteration(iteration_id="it-1", iteration_number=1, status="completed")

    loop_hosting._checkpoint_iteration(loop, iteration, job_id="job-h10", attempt_id="att-1")
