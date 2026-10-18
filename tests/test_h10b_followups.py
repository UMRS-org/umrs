"""H-10.1 follow-ups (2026-10-18): verification findings, pipelines, skills key.

Decisions: ``docs/development/dormant-modules.md`` ("Module-global
per-project state", the 2026-10-18 follow-up rows). Each block pins the
defect it removes:

1. **Verification findings routes, retired.** ``set_store`` had no caller, so
   ``POST /umrs/verification/chat/resolve``, ``GET /findings`` and
   ``PATCH /findings/{key}`` answered 503. Wiring the store would not have
   helped: the TUI's resolve names report ``"manual"``, which no
   ``verification_reports`` row has (foreign-key violation, a 500), and the
   two findings routes were admin-only, projectless copies of
   ``/umrs/knowledge/findings``. They answered 410 with that pointer (umrs
   d4a8ad85); once the TUI's resolve key was retired (same day, second
   follow-up) nothing called them, so the module, its mount and the
   ``verification`` lane family are deleted.
2. **Pipeline definitions, retired.** ``_pipeline_store`` held every
   project's definitions in process memory, lost at restart, and nothing at
   ingest or search time read them (chunking and embedding come from
   ``UMRS_RAG_CHUNK_MAX_TOKENS`` / ``UMRS_RAG_EMBEDDING_MODEL``). The CRUD
   answered 410 like rag-profiles; once ``r4 rag ingest`` and the TUI
   ``/pipeline`` CRUD were retired, the routes, the ``pipelines`` lane family
   and the ``/gw/pipelines`` edge locations went. The component registry
   stays.
3. **Skills key.** ``UNIQUE(scope, owner_id, name)`` had no project, so a
   user's project skill registered under the same name in a second project
   overwrote the first, the loop's proposals for a second project were
   dropped, and a bundle import counted another project's stub as present.
   Migration 0123 keys project rows by project.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h10b-followups-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _headers("alice", "proj-a")


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


def _client(router: Any) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ── 1. verification findings routes: deleted ──────────────────────────────

_VERIFICATION_ROUTES = (
    ("POST", "/umrs/verification/chat/resolve"),
    ("GET", "/umrs/verification/findings"),
    ("PATCH", "/umrs/verification/findings/{finding_key}"),
)


def _mounted(router: Any) -> set[tuple[str, str]]:
    return {(method, route.path) for route in router.routes for method in route.methods}


def test_verification_router_module_is_deleted():
    assert importlib.util.find_spec("umrs.api.verification") is None


def test_verification_family_has_no_lane():
    from umrs.shared.lanes import FAMILY_LANES

    assert "verification" not in FAMILY_LANES


def test_per_project_findings_lane_stays():
    """The replacement the 410 pointed to: GET/PATCH /umrs/knowledge/findings."""
    from umrs.api import knowledge

    mounted = _mounted(knowledge.router)
    assert ("GET", "/umrs/knowledge/findings") in mounted
    assert ("PATCH", "/umrs/knowledge/findings/{finding_key}") in mounted
    assert not {path for _m, path in mounted if path.startswith("/umrs/verification")}


# ── 2. pipeline definitions: deleted ──────────────────────────────────────

_PIPELINE_CALLS = (
    ("POST", "/umrs/pipelines", {"name": "h10b-pipe"}),
    ("GET", "/umrs/pipelines", None),
    ("GET", "/umrs/pipelines/pipe-1", None),
    ("PATCH", "/umrs/pipelines/pipe-1", {"name": "renamed"}),
    ("DELETE", "/umrs/pipelines/pipe-1", None),
)


@pytest.mark.parametrize(("method", "path", "body"), _PIPELINE_CALLS)
def test_pipeline_crud_routes_are_deleted(strict, method, path, body):
    from umrs.api import pipelines

    resp = _client(pipelines.router).request(method, path, json=body, headers=ALICE)
    assert resp.status_code == 404, resp.text


def test_pipelines_family_has_no_lane():
    from umrs.api import pipelines
    from umrs.shared.lanes import FAMILY_LANES

    assert "pipelines" not in FAMILY_LANES
    assert not {path for _m, path in _mounted(pipelines.router) if path.startswith("/umrs/pipelines")}


def test_no_module_global_pipeline_store_remains():
    from umrs.api import pipelines

    leftovers = {
        "PipelineStore",
        "_pipeline_store",
        "get_pipeline_store",
        "set_pipeline_store",
        "_PIPELINES_GONE",
    } & set(vars(pipelines))
    assert leftovers == set()


@pytest.mark.parametrize("kind", ["chunkers", "embedders", "rerankers"])
def test_component_registry_stays(kind):
    from umrs.api import pipelines

    resp = _client(pipelines.router).get(f"/umrs/components/{kind}")
    assert resp.status_code == 200
    assert resp.json()


# ── 3. skills key: in-memory registry (no database) ───────────────────────


def _manifest(name: str) -> Any:
    from r4_contracts.skills import SkillManifest

    return SkillManifest.from_dict({"name": name, "version": "1.0.0"})


def test_memory_registry_keeps_a_project_skill_per_project():
    from umrs.skills import SkillRegistry, SkillScope

    registry = SkillRegistry()
    for project in ("proj-a", "proj-b"):
        registry.register(
            _manifest("h10b-fit"),
            f"# {project}",
            scope=SkillScope.PROJECT,
            owner_id="alice",
            project_id=project,
        )
    rows = registry.list_skills(scope=SkillScope.PROJECT, owner_id="alice")
    assert sorted(skill.project_id for skill in rows) == ["proj-a", "proj-b"]


def test_memory_registry_user_and_system_ids_are_unchanged():
    """Only project-scope ids gain the project; stored ids stay valid."""
    import hashlib

    from umrs.skills import RegisteredSkill, SkillScope

    for scope in (SkillScope.USER, SkillScope.SYSTEM):
        skill = RegisteredSkill(manifest=_manifest("h10b-x"), scope=scope, owner_id="u")
        expected = hashlib.sha256(f"{scope.value}:u:h10b-x".encode()).hexdigest()[:16]
        assert skill.skill_id == expected


# ── 3. skills key: PostgreSQL (CI shape: POSTGRES_* or DATABASE_URL) ──────


def _knowledge_dsn() -> str:
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn

    dsn, _local = _resolve_knowledge_dsn()
    if not dsn:
        pytest.skip("needs Postgres (CI shape: POSTGRES_* or DATABASE_URL)")
    return dsn


@pytest.fixture
def pg_conn():
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        conn = psycopg2.connect(_knowledge_dsn())
    except psycopg2.Error as exc:
        pytest.skip(f"Postgres unreachable: {exc}")
    _purge(conn)
    yield conn
    conn.rollback()
    _purge(conn)
    conn.close()


def _purge(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM skills WHERE name LIKE %s", ("h10b-%",))
    conn.commit()


def _rows(conn: Any, name: str) -> list[tuple[str, str | None, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT scope, project_id, skill_md FROM skills WHERE name = %s"
            " ORDER BY project_id NULLS FIRST",
            (name,),
        )
        return list(cur.fetchall())


def _register(conn: Any, name: str, scope: Any, project: str | None, body: str) -> Any:
    from umrs.skills.store import SkillStore

    skill = SkillStore(conn).register_skill(
        name=name,
        scope=scope,
        owner_id="alice",
        project_id=project,
        skill_md=body,
    )
    conn.commit()
    return skill


def test_project_skill_in_a_second_project_keeps_the_first(pg_conn):
    from umrs.skills import SkillScope

    first = _register(pg_conn, "h10b-fit", SkillScope.PROJECT, "proj-a", "# a")
    second = _register(pg_conn, "h10b-fit", SkillScope.PROJECT, "proj-b", "# b")

    assert first.skill_id != second.skill_id
    assert _rows(pg_conn, "h10b-fit") == [
        ("project", "proj-a", "# a"),
        ("project", "proj-b", "# b"),
    ]


def test_project_skill_reregistered_in_its_project_updates_in_place(pg_conn):
    from umrs.skills import SkillScope

    first = _register(pg_conn, "h10b-fit", SkillScope.PROJECT, "proj-a", "# v1")
    again = _register(pg_conn, "h10b-fit", SkillScope.PROJECT, "proj-a", "# v2")

    assert again.skill_id == first.skill_id
    assert _rows(pg_conn, "h10b-fit") == [("project", "proj-a", "# v2")]


def test_user_skill_stays_one_row_across_projects(pg_conn):
    from umrs.skills import SkillScope

    _register(pg_conn, "h10b-mine", SkillScope.USER, None, "# v1")
    _register(pg_conn, "h10b-mine", SkillScope.USER, None, "# v2")
    assert _rows(pg_conn, "h10b-mine") == [("user", None, "# v2")]


def test_pre_0123_project_row_is_updated_not_duplicated(pg_conn):
    """A row stored under the old id formula still takes the upsert."""
    from umrs.skills import SkillScope

    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO skills (skill_id, name, scope, owner_id, project_id, skill_md)"
            " VALUES ('skl-h10b-legacy', 'h10b-legacy', 'project', 'alice', 'proj-a', '# old')"
        )
    pg_conn.commit()
    skill = _register(pg_conn, "h10b-legacy", SkillScope.PROJECT, "proj-a", "# new")

    assert skill.skill_id == "skl-h10b-legacy"
    assert _rows(pg_conn, "h10b-legacy") == [("project", "proj-a", "# new")]


def test_get_skill_by_name_reads_the_named_project(pg_conn):
    from umrs.skills import SkillScope
    from umrs.skills.store import SkillStore

    _register(pg_conn, "h10b-fit", SkillScope.PROJECT, "proj-a", "# a")
    store = SkillStore(pg_conn)
    found = store.get_skill_by_name("h10b-fit", SkillScope.PROJECT, "alice", project_id="proj-a")
    assert found is not None and found.project_id == "proj-a"
    assert (
        store.get_skill_by_name("h10b-fit", SkillScope.PROJECT, "alice", project_id="proj-b")
        is None
    )


def _project_ref_bundle_import(conn: Any, project: str) -> tuple[int, int]:
    from umrs.knowledge.bundle_import import _import_skill_refs
    from umrs.skills.store import SkillStore

    store = SkillStore(conn)
    refs = [{"ref": "project:h10b-lit", "version_pin": "1.0.0", "scope": "project"}]
    counts = _import_skill_refs(store, refs, project)
    conn.commit()
    return counts


def test_bundle_import_registers_a_stub_per_target_project(pg_conn):
    assert _project_ref_bundle_import(pg_conn, "proj-a") == (1, 0)
    assert _project_ref_bundle_import(pg_conn, "proj-b") == (1, 0)
    assert _project_ref_bundle_import(pg_conn, "proj-b") == (0, 1)
    assert [row[1] for row in _rows(pg_conn, "h10b-lit")] == ["proj-a", "proj-b"]


class _AsyncpgProposalPool:
    """``fetch`` serves IR rows; ``fetchval`` runs the real INSERT."""

    def __init__(self, conn: Any, rows: list[dict[str, Any]]) -> None:
        self._conn = conn
        self._rows = rows

    async def fetch(self, _query: str, *_args: Any) -> list[dict[str, Any]]:
        return self._rows

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._conn.fetchval(query, *args)


def _ir_row(project: str) -> dict[str, Any]:
    return {
        "ir_id": f"ir-h10b-{project}",
        "artifact_id": f"art-h10b-{project}",
        "content": {
            "objective": "h10b-calibrate the stage",
            "methodology": "zero the stage, record the z-offset",
        },
        "content_hash": f"h-{project}",
        "ir_type": "document",
        "project_id": project,
    }


async def _compile_for(conn: Any, project: str) -> int:
    from types import SimpleNamespace

    from umrs.jobs.knowledge_adapters import _compile_skill_proposals

    store = SimpleNamespace(_pool=_AsyncpgProposalPool(conn, [_ir_row(project)]))
    return await _compile_skill_proposals(store, project)


async def test_loop_proposal_of_a_second_project_is_registered(pg_conn):
    asyncpg = pytest.importorskip("asyncpg")
    conn = await asyncpg.connect(_knowledge_dsn())
    try:
        assert await _compile_for(conn, "proj-a") == 1
        assert await _compile_for(conn, "proj-b") == 1
        assert await _compile_for(conn, "proj-b") == 0  # idempotent
    finally:
        await conn.close()
    projects = [row[1] for row in _rows(pg_conn, "h10b-calibrate the stage")]
    assert projects == ["proj-a", "proj-b"]


# ── 3. migration 0123 ─────────────────────────────────────────────────────


def test_0123_follows_0122_and_names_the_writers_conflict_key():
    from umrs import migrations

    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0123") == 1
    assert versions.index("0123") == versions.index("0122") + 1
    sql = dict(migrations.get_migration_files())["0123"].read_text()
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    assert "ON skills (scope, owner_id, project_key, name);" in code
    assert "CONCURRENTLY" not in code.upper()


def test_writers_use_the_0123_conflict_key():
    import inspect

    from umrs.jobs import knowledge_adapters
    from umrs.skills import store

    old_target = "ON CONFLICT (scope, owner_id, name)"
    assert old_target not in inspect.getsource(store)
    assert old_target not in inspect.getsource(knowledge_adapters)


def test_0123_reapplies_cleanly(pg_conn):
    """The file is idempotent on a migrated database (second raw run)."""
    from umrs import migrations

    sql = dict(migrations.get_migration_files())["0123"].read_text()
    with pg_conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(sql)
        cur.execute(
            "SELECT count(*) FROM pg_constraint WHERE conrelid = 'skills'::regclass"
            " AND contype = 'u'"
        )
        assert cur.fetchone()[0] == 0
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'skills'"
            " AND indexname = 'uq_skills_scope_owner_project_name'"
        )
        assert "UNIQUE" in cur.fetchone()[0]
    pg_conn.rollback()
