"""H-1 (hardening plan, 2026-10-17), item 5 for the registry routers.

These routers took no identity at all:

- /umrs/experiment-registry: create in any project, list/history of any
  project, and get/status/results on any experiment id.
- /umrs/egress/audit: anyone could write audit rows (forged trail) and a GET
  without filters returned every project's egress decisions.
- /umrs/reflective-memory: platform-global outcome/routing/heuristic memory;
  anyone could read it, delete entries, or deactivate every heuristic.

Contract after the fix (strict mode, gateway-signed identity):

- experiment registry: writes only into projects the caller may write; reads
  only projects the caller may read (no filter = the caller's own project);
  another project's experiment id reads as not found.
- egress audit: POST is admin-only (the operator sink; no in-repo writer since
  the Python gateway was removed, EgressGuard writes JSONL locally); GET is the
  caller's own trail in the caller's project, admins see all.
- reflective memory: admin-only (global data across projects; the inspector
  cannot filter heuristics by project).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-registries-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _headers("alice", "proj-a")
ADMIN = _headers("root", "ops", admin=True)


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


# ── experiment registry ──────────────────────────────────────────────────────


@pytest.fixture
def experiments():
    from umrs.api import experiment_registry as api
    from umrs.experiment_registry import InMemoryExperimentRegistry

    registry = InMemoryExperimentRegistry()
    own = registry.create_experiment(project_id="proj-a", hypothesis_id="h-a")
    foreign = registry.create_experiment(project_id="proj-b", hypothesis_id="h-b")
    previous = api._registry_store
    api.set_registry(registry)
    app = FastAPI()
    app.include_router(api.router)
    yield TestClient(app), registry, own, foreign
    api.set_registry(previous)


def _create_body(project: str) -> dict:
    return {"project_id": project, "hypothesis_id": "h", "created_by": "mallory"}


def test_experiment_create_in_foreign_project_is_refused(experiments):
    client, registry, _, _ = experiments
    resp = client.post("/umrs/experiment-registry", json=_create_body("proj-b"), headers=ALICE)
    assert resp.status_code == 403
    assert len(registry.list_experiments(project_id="proj-b")) == 1


def test_experiment_create_in_own_project_records_the_verified_user(experiments):
    client, registry, _, _ = experiments
    resp = client.post("/umrs/experiment-registry", json=_create_body("proj-a"), headers=ALICE)
    assert resp.status_code == 200
    exp = registry.get_experiment(resp.json()["experiment_id"])
    assert exp.project_id == "proj-a"
    assert exp.created_by == "alice"


def test_experiment_routes_require_identity_in_strict_mode(experiments):
    client, _, own, _ = experiments
    assert client.get("/umrs/experiment-registry").status_code == 401
    assert client.get(f"/umrs/experiment-registry/{own}").status_code == 401


def test_experiment_list_checks_identity_before_parsing_the_status(experiments):
    # Review 2026-10-18: the status filter was parsed before the caller was
    # resolved, so an anonymous strict caller got a 500 (ValueError) instead
    # of 401 and the auth check never ran on that path.
    from umrs.api import experiment_registry as api

    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/umrs/experiment-registry", params={"status": "bogus"})
    assert resp.status_code == 401


def test_experiment_list_with_unknown_status_is_a_bad_request(experiments):
    from umrs.api import experiment_registry as api

    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/umrs/experiment-registry", params={"status": "bogus"}, headers=ALICE)
    assert resp.status_code == 400


def test_experiment_list_without_filter_is_the_callers_project(experiments):
    client, _, own, _ = experiments
    resp = client.get("/umrs/experiment-registry", headers=ALICE)
    assert resp.status_code == 200
    ids = [e["experiment_id"] for e in resp.json()["experiments"]]
    assert ids == [own]


def test_experiment_list_of_foreign_project_is_refused(experiments):
    client, _, _, _ = experiments
    resp = client.get("/umrs/experiment-registry", params={"project_id": "proj-b"}, headers=ALICE)
    assert resp.status_code == 403


def test_experiment_history_of_foreign_project_is_refused(experiments):
    client, _, _, _ = experiments
    resp = client.get("/umrs/experiment-registry/history/proj-b", headers=ALICE)
    assert resp.status_code == 403


def test_foreign_experiment_reads_as_not_found(experiments):
    client, _, _, foreign = experiments
    resp = client.get(f"/umrs/experiment-registry/{foreign}", headers=ALICE)
    assert resp.status_code == 404


def test_foreign_experiment_status_is_not_changed(experiments):
    client, registry, _, foreign = experiments
    resp = client.put(
        f"/umrs/experiment-registry/{foreign}/status",
        json={"status": "designed"},
        headers=ALICE,
    )
    assert resp.status_code == 404
    assert registry.get_experiment(foreign).status.value == "pending"


def test_foreign_experiment_gets_no_result(experiments):
    client, registry, _, foreign = experiments
    resp = client.post(
        f"/umrs/experiment-registry/{foreign}/results",
        json={"metrics": {"acc": 1.0}},
        headers=ALICE,
    )
    assert resp.status_code == 404
    assert registry.get_experiment(foreign).results == []


def test_own_experiment_status_and_results_still_work(experiments):
    client, registry, own, _ = experiments
    status = client.put(
        f"/umrs/experiment-registry/{own}/status",
        json={"status": "designed"},
        headers=ALICE,
    )
    assert status.status_code == 200
    result = client.post(
        f"/umrs/experiment-registry/{own}/results",
        json={"metrics": {"acc": 1.0}},
        headers=ALICE,
    )
    assert result.status_code == 200
    assert registry.get_experiment(own).results[0].created_by == "alice"


def test_admin_lists_every_project(experiments):
    client, _, own, foreign = experiments
    resp = client.get("/umrs/experiment-registry", headers=ADMIN)
    assert resp.status_code == 200
    ids = {e["experiment_id"] for e in resp.json()["experiments"]}
    assert ids == {own, foreign}


# ── egress audit ─────────────────────────────────────────────────────────────


def _audit_row(user: str, project: str, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "user_id": user,
        "project_id": project,
        "destination_url": "https://api.example.com/x",
        "destination_domain": "api.example.com",
        "action": "allowed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def egress(monkeypatch):
    from umrs.api import egress_audit as api

    # The memory store: since H-10.1 the DSN chain also reads DATABASE_URL and
    # the DB_*/POSTGRES_* component vars (_asyncpg_pool.sync_store_dsn).
    monkeypatch.setattr("umrs.api._asyncpg_pool.sync_store_dsn", lambda: None)
    monkeypatch.setattr(api, "_store", None)
    store = api.get_store()
    for user, project, rid in (
        ("alice", "proj-a", "r-alice-a"),
        ("bob", "proj-a", "r-bob-a"),
        ("carol", "proj-b", "r-carol-b"),
    ):
        store.insert(api.EgressAuditRecord(**_audit_row(user, project, rid)))
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app), store


def _ids(resp) -> set[str]:
    return {item["request_id"] for item in resp.json()["items"]}


def test_egress_audit_post_by_non_admin_is_refused(egress):
    client, store = egress
    resp = client.post(
        "/umrs/egress/audit",
        json=_audit_row("alice", "proj-a", "r-forged"),
        headers=ALICE,
    )
    assert resp.status_code == 403
    assert "r-forged" not in {r.request_id for r in store.query(limit=100)}


def test_egress_audit_post_without_identity_is_refused(egress):
    client, _ = egress
    resp = client.post("/umrs/egress/audit", json=_audit_row("alice", "proj-a", "r-anon"))
    assert resp.status_code == 401


def test_egress_audit_post_by_signed_admin_is_recorded(egress):
    client, store = egress
    resp = client.post(
        "/umrs/egress/audit",
        json=_audit_row("alice", "proj-a", "r-admin"),
        headers=ADMIN,
    )
    assert resp.status_code == 201
    assert "r-admin" in {r.request_id for r in store.query(limit=100)}


def test_egress_audit_list_without_filter_is_the_callers_own_trail(egress):
    client, _ = egress
    resp = client.get("/umrs/egress/audit", headers=ALICE)
    assert resp.status_code == 200
    assert _ids(resp) == {"r-alice-a"}


def test_egress_audit_list_of_foreign_project_is_refused(egress):
    client, _ = egress
    resp = client.get("/umrs/egress/audit", params={"project_id": "proj-b"}, headers=ALICE)
    assert resp.status_code == 403


def test_egress_audit_list_of_another_user_is_refused(egress):
    client, _ = egress
    resp = client.get("/umrs/egress/audit", params={"user_id": "bob"}, headers=ALICE)
    assert resp.status_code == 403


def test_egress_audit_admin_lists_everything(egress):
    client, _ = egress
    resp = client.get("/umrs/egress/audit", headers=ADMIN)
    assert resp.status_code == 200
    assert _ids(resp) == {"r-alice-a", "r-bob-a", "r-carol-b"}


# ── reflective memory ────────────────────────────────────────────────────────


class _FakeInspector:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def inspect(self, **_kw):
        self.calls.append("inspect")
        return SimpleNamespace(to_dict=lambda: {"entries": []})

    def delete_entry(self, entry_id: str, tier: str) -> bool:
        self.calls.append(f"delete:{tier}:{entry_id}")
        return True

    def reset_tier(self, tier: str) -> int:
        self.calls.append(f"reset:{tier}")
        return 3


@pytest.fixture
def reflective(monkeypatch):
    from umrs.api import reflective_memory as api

    fake = _FakeInspector()
    monkeypatch.setattr(api, "_inspector", fake)
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app), fake


def _reflective_calls(client: TestClient, headers: dict[str, str]) -> list[int]:
    return [
        client.get("/umrs/reflective-memory", headers=headers).status_code,
        client.delete("/umrs/reflective-memory/heuristic/h-1", headers=headers).status_code,
        client.post("/umrs/reflective-memory/heuristic/reset", headers=headers).status_code,
    ]


def test_reflective_memory_is_refused_to_non_admins(reflective):
    client, fake = reflective
    assert _reflective_calls(client, ALICE) == [403, 403, 403]
    assert fake.calls == []


def test_reflective_memory_requires_identity_in_strict_mode(reflective):
    client, fake = reflective
    assert _reflective_calls(client, {}) == [401, 401, 401]
    assert fake.calls == []


def test_reflective_memory_rejects_a_forged_admin_flag(reflective):
    client, fake = reflective
    forged = dict(ALICE, **{"X-Verified-Is-Admin": "true"})
    assert _reflective_calls(client, forged) == [401, 401, 401]
    assert fake.calls == []


def test_reflective_memory_serves_a_signed_admin(reflective):
    client, fake = reflective
    assert _reflective_calls(client, ADMIN) == [200, 200, 200]
    assert fake.calls == ["inspect", "delete:heuristic:h-1", "reset:heuristic"]


def test_reflective_memory_keeps_dev_mode_open(reflective, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    client, _ = reflective
    assert _reflective_calls(client, {}) == [200, 200, 200]
