"""H-1 item 5 + Appendix A.2 (hardening plan, 2026-10-17): the lineage lane.

- ``GET /umrs/knowledge/lineage/{artifact_id}``: ``?project_id`` was required
  but never reached the SQL, and the route took no identity, so any caller
  read any project's provenance chain.
- ``GET /umrs/a2l/artifacts``: ``?project_id`` honoured with no identity.
- ``POST /umrs/a2l/resolve``: content hash -> A2L mapping across all projects.
- ``POST /umrs/a2l/derive``: inserted a link for any knowledge artifact id
  (and any A2L artifact id) without an ownership check.
- ``POST /umrs/a2l/ledger/decisions``: the job was checked, ``body.artifact_id``
  was not (a decision could point at another project's artifact).

Strict mode with gateway-signed headers throughout; stores and pools are fakes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SECRET = "h1-lineage-secret"
ROOT_A = "11111111-1111-1111-1111-111111111111"  # proj-a's knowledge artifact
ROOT_B = "22222222-2222-2222-2222-222222222222"  # proj-b's knowledge artifact


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    from r4_contracts.handshake_hmac import sign_claims

    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _headers("alice", "proj-a")


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


class _FakePool:
    """A knowledge-store pool that answers like a database WITHOUT the
    project filter unless the query binds the owning project: link rows come
    back for any query that does not name a project; a knowledge artifact or
    ledger artifact lookup returns its owning project."""

    def __init__(self) -> None:
        self.knowledge = {ROOT_A: "proj-a", ROOT_B: "proj-b", "biz-b": "proj-b"}
        self.ledger = {"a2l-b": "proj-b", "a2l-a": "proj-a"}
        self.projects = {"proj-a", "proj-b"}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    @asynccontextmanager
    async def acquire(self):
        yield self

    def _named_project(self, args: tuple[Any, ...]) -> str | None:
        return next((a for a in args if isinstance(a, str) and a in self.projects), None)

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, args))
        named = self._named_project(args)
        if named not in (None, "proj-b"):
            return []
        return [
            {
                "artifact_id": ROOT_B,
                "knowledge_artifact_id": ROOT_B,
                "a2l_artifact_id": "a2l-b",
                "a2l_id": "a2l-b",
                "a2l_hash": "h-b",
                "edge_kind": "derived",
                "derived_at": "2026-10-01T00:00:00",
                "synced_at": "2026-10-01T00:00:00",
                "source": "auto",
            }
        ]

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((sql, args))
        if "INSERT INTO knowledge_a2l_links" in sql:
            return {"id": 7}
        if "FROM a2l_artifacts" in sql:
            owner = self.ledger.get(args[0])
            return {"project_id": owner} if owner else None
        if "FROM knowledge_artifacts" in sql and "a2l_artifact_refs" not in sql:
            return self._knowledge_row(args)
        named = self._named_project(args)
        if named not in (None, "proj-b"):
            return None
        return {"a2l_id": "a2l-b", "artifact_id": ROOT_B, "synced_at": "2026-10-01"}

    def _knowledge_row(self, args: tuple[Any, ...]) -> dict[str, Any] | None:
        owner = self.knowledge.get(args[0])
        named = self._named_project(args[1:])
        if owner is None or (named is not None and named != owner):
            return None
        return {"id": args[0], "project_id": owner}

    def sql_with(self, needle: str) -> list[tuple[str, tuple[Any, ...]]]:
        return [c for c in self.calls if needle in c[0]]


@pytest.fixture
def pool(monkeypatch):
    fake = _FakePool()
    monkeypatch.setattr("umrs.knowledge.store.get_store", lambda: SimpleNamespace(_pool=fake))
    return fake


def _client(router) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def lineage(pool):
    from umrs.api import knowledge_lineage

    return _client(knowledge_lineage.router)


@pytest.fixture
def a2l(pool):
    from umrs.api import a2l

    return _client(a2l.router)


# ── knowledge lineage ────────────────────────────────────────────────────────


def test_lineage_without_identity_is_refused_in_strict(lineage):
    resp = lineage.get(f"/umrs/knowledge/lineage/{ROOT_B}?project_id=proj-b")
    assert resp.status_code == 401


def test_lineage_for_another_project_is_refused(lineage):
    resp = lineage.get(f"/umrs/knowledge/lineage/{ROOT_B}?project_id=proj-b", headers=ALICE)
    assert resp.status_code == 403


def test_another_projects_artifact_has_no_lineage_in_mine(lineage, pool):
    """Naming my own project must not open a foreign artifact's chain."""
    resp = lineage.get(f"/umrs/knowledge/lineage/{ROOT_B}?project_id=proj-a", headers=ALICE)
    assert resp.status_code == 200
    assert resp.json()["ancestors"] == []
    assert resp.json()["descendants"] == []


def test_lineage_query_binds_the_project(lineage, pool):
    resp = lineage.get(f"/umrs/knowledge/lineage/{ROOT_A}", headers=ALICE)
    assert resp.status_code == 200
    assert pool.calls, "the lineage route never queried the store"
    assert all("proj-a" in args for _sql, args in pool.calls)


def test_own_lineage_is_returned(lineage, pool):
    bob = _headers("bob", "proj-b")
    resp = lineage.get(f"/umrs/knowledge/lineage/{ROOT_B}", headers=bob)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ancestors"][0]["artifact_id"] == "a2l-b"
    assert body["ancestors"][0]["created_at"].startswith("2026-10-01")


ROOT_A2 = "33333333-3333-3333-3333-333333333333"  # proj-a's second artifact


def _link_row(knowledge_id: str, a2l_id: str) -> dict[str, Any]:
    return {
        "knowledge_artifact_id": knowledge_id,
        "a2l_artifact_id": a2l_id,
        "a2l_id": a2l_id,
        "edge_kind": "derived",
        "derived_at": "2026-10-01T00:00:00",
    }


class _SharedA2lPool(_FakePool):
    """ROOT_A (proj-a) derives from ``a2l-shared``; ROOT_A2 (proj-a) and
    ROOT_B (proj-b) link to the same A2L id. The descendants query answers
    both unless it binds proj-a, like a database without the project join."""

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, args))
        if "ANY($1::text[])" in sql:
            rows = [_link_row(ROOT_A2, "a2l-shared"), _link_row(ROOT_B, "a2l-shared")]
            return rows[:1] if "proj-a" in args else rows
        if ROOT_A in args and "proj-a" in args:
            return [_link_row(ROOT_A, "a2l-shared")]
        return []


def test_descendants_stay_in_the_callers_project(monkeypatch):
    """Item 5's leak: another project's artifact linked to the same A2L id
    must not appear among my artifact's descendants."""
    from umrs.api import knowledge_lineage

    fake = _SharedA2lPool()
    monkeypatch.setattr("umrs.knowledge.store.get_store", lambda: SimpleNamespace(_pool=fake))
    resp = _client(knowledge_lineage.router).get(f"/umrs/knowledge/lineage/{ROOT_A}", headers=ALICE)
    assert resp.status_code == 200
    assert [a["artifact_id"] for a in resp.json()["ancestors"]] == ["a2l-shared"]
    assert [d["artifact_id"] for d in resp.json()["descendants"]] == [ROOT_A2]
    ((_sql, args),) = fake.sql_with("ANY($1::text[])")
    assert "proj-a" in args


# ── a2l artifacts / resolve / derive ─────────────────────────────────────────


def test_a2l_artifacts_without_identity_is_refused(a2l):
    assert a2l.get("/umrs/a2l/artifacts?project_id=proj-b").status_code == 401


def test_a2l_artifacts_for_another_project_is_refused(a2l):
    resp = a2l.get("/umrs/a2l/artifacts?project_id=proj-b", headers=ALICE)
    assert resp.status_code == 403


def test_a2l_artifacts_default_to_the_callers_project(a2l, pool):
    resp = a2l.get("/umrs/a2l/artifacts", headers=ALICE)
    assert resp.status_code == 200
    assert pool.calls[-1][1][0] == "proj-a"


def test_resolve_without_identity_is_refused(a2l):
    assert a2l.post("/umrs/a2l/resolve", json={"content_hash": "h-b"}).status_code == 401


def test_resolve_does_not_map_another_projects_hash(a2l, pool):
    resp = a2l.post("/umrs/a2l/resolve", json={"content_hash": "h-b"}, headers=ALICE)
    assert resp.status_code == 200
    assert resp.json()["a2l_id"] == ""
    assert resp.json()["artifact_id"] == ""


def test_resolve_in_another_named_project_is_refused(a2l):
    body = {"content_hash": "h-b", "project_id": "proj-b"}
    assert a2l.post("/umrs/a2l/resolve", json=body, headers=ALICE).status_code == 403


def test_resolve_without_a_project_in_mvp_names_the_missing_project(a2l, monkeypatch):
    """Dev (mvp) mode: an anonymous caller must now name the project; the old
    unscoped lookup across every project was the defect, so it is not kept."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    resp = a2l.post("/umrs/a2l/resolve", json={"content_hash": "h-b"})
    assert resp.status_code == 400
    assert "project_id is required" in resp.text


def test_resolve_in_a_named_project_in_mvp_still_works(a2l, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    body = {"content_hash": "h-b", "project_id": "proj-b"}
    resp = a2l.post("/umrs/a2l/resolve", json=body)
    assert resp.status_code == 200
    assert resp.json()["a2l_id"] == "a2l-b"


def _derive(client, knowledge_id: str, a2l_id: str = "a2l-new", headers=ALICE):
    body = {"knowledge_artifact_id": knowledge_id, "a2l_artifact_id": a2l_id}
    return client.post("/umrs/a2l/derive", json=body, headers=headers)


def test_derive_without_identity_is_refused(a2l, pool):
    assert _derive(a2l, ROOT_A, headers={}).status_code == 401
    assert not pool.sql_with("INSERT")


def test_derive_on_another_projects_artifact_is_not_found(a2l, pool):
    for foreign in (ROOT_B, "biz-b"):
        assert _derive(a2l, foreign).status_code == 404
    assert not pool.sql_with("INSERT")


def test_derive_to_another_projects_a2l_artifact_is_not_found(a2l, pool):
    assert _derive(a2l, ROOT_A, a2l_id="a2l-b").status_code == 404
    assert not pool.sql_with("INSERT")


def test_derive_on_own_artifact_creates_the_link(a2l, pool):
    resp = _derive(a2l, ROOT_A, a2l_id="a2l-a")
    assert resp.status_code == 200
    assert resp.json() == {"status": "created", "link_id": "7", "error": ""}
    ((_sql, args),) = pool.sql_with("INSERT INTO knowledge_a2l_links")
    assert args[:2] == (ROOT_A, "a2l-a")


# ── a2l ledger decisions ─────────────────────────────────────────────────────


@pytest.fixture
def ledger(monkeypatch):
    from umrs.a2l import ArtifactRecord
    from umrs.api import a2l_ledger

    mine = ArtifactRecord(id="l-mine", type="r", uri="s3://m", user_id="alice", project_id="proj-a")
    theirs = ArtifactRecord(
        id="l-theirs", type="r", uri="s3://t", user_id="bob", project_id="proj-b"
    )
    created: list[Any] = []

    class _Store:
        async def get_artifact(self, artifact_id):
            return {a.id: a for a in (mine, theirs)}.get(artifact_id)

        async def create_decision(self, req):
            created.append(req)
            return SimpleNamespace(
                id="d1",
                created_at="now",
                **{k: getattr(req, k) for k in ("decided_by", "decision", "job_id", "attempt_id")},
            )

    async def fake_store():
        return _Store()

    job = SimpleNamespace(job_id="j-a", project_id="proj-a", user_id="alice")
    monkeypatch.setattr(
        a2l_ledger,
        "get_optional_job_store",
        lambda: SimpleNamespace(get_job=lambda jid: job if jid == "j-a" else None),
    )
    app = FastAPI()
    app.include_router(a2l_ledger.router)
    app.dependency_overrides[a2l_ledger._get_store] = fake_store
    return TestClient(app, raise_server_exceptions=False), created


def _decide(client, artifact_id: str):
    body = {
        "decided_by": "alice",
        "decision": "approve",
        "job_id": "j-a",
        "artifact_id": artifact_id,
    }
    return client.post("/umrs/a2l/ledger/decisions", json=body, headers=ALICE)


def test_decision_on_another_projects_artifact_is_not_found(ledger):
    client, created = ledger
    assert _decide(client, "l-theirs").status_code == 404
    assert _decide(client, "l-missing").status_code == 404
    assert created == []


def test_decision_on_own_artifact_is_recorded(ledger):
    client, created = ledger
    assert _decide(client, "l-mine").status_code == 201
    assert len(created) == 1
