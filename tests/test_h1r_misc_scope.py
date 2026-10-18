"""H-1 remaining gaps, misc lane (hardening plan Appendix A, 2026-10-18).

Each block names the defect it pins:

1. ``GET /umrs/jobs/stats`` took a ``?project_id`` with no identity and was
   shadowed by ``GET /umrs/jobs/{job_id}`` (registered first), so it was dead.
   It is deleted; ``/umrs/jobs/stats`` is answered by the job lookup.
2. ``POST /umrs/knowledge/external/search`` was anonymous server-side egress
   to the academic sources; it now needs the verified project.
3. ``POST /umrs/companion/workers``: an identified caller could register an
   ownerless worker (empty ``owner_user_id``); an empty owner now defaults to
   the verified caller and another owner is refused. A caller-supplied
   ``job_id`` was bound unchecked; it must now be the caller's own companion
   job in that project (a foreign one reads as not found).
4. ``POST /umrs/companion/approvals/{id}/decision``: a non-owner of an owned
   approval got 401 "approval ownership mismatch", which shows that it
   exists; it now reads as not found, like an unknown id.
5. ``PATCH /umrs/verification/findings/{key}`` proceeded when ``get_caller``
   returned ``None``. Strict mode never yields ``None`` (``resolve_caller``
   raises 401 on a missing or forged tuple); the anonymous dev (mvp) caller
   could still transition projectless findings, so the route now needs a
   verified admin in every mode.
   Since 2026-10-18 (H-10.1 follow-up) the route was retired (410), still
   needing a verified caller in every mode; later that day its last client
   (the TUI resolve key) was retired and the route deleted with its module.
   Transitions go through ``PATCH /umrs/knowledge/findings/{key}``, which
   needs a project the caller may write.
6. The experiment registry's status filter answered a plain 400 with no
   typed code; it is now ``experiments.bad_request``.
7. ``knowledge/a2l_bridge.py`` called ``/gw/a2l/{resolve,artifacts}`` with no
   identity (nginx sends those to the ledger, so the calls never reached the
   intended routes); its only importer died on ``umrs.settings`` before
   building it, so it was dormant and is deleted.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.errors import ERROR_CODE_HEADER, PlatformError
from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1r-misc-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _headers("alice", "proj-a")
# A forged tuple: admin flag set, a foreign project, a bogus signature.
FORGED = {
    "X-User-ID": "mallory",
    "X-Verified-Project-ID": "proj-b",
    "X-Verified-Is-Admin": "true",
    "X-Verified-HMAC": "0" * 64,
}


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


@pytest.fixture
def mvp(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.delenv("UMRS_HANDSHAKE_HMAC_SECRET", raising=False)


def _client(router, **state: Any) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    for key, value in state.items():
        setattr(app.state, key, value)
    return TestClient(app, raise_server_exceptions=False)


# ── 1. GET /umrs/jobs/stats ─────────────────────────────────────────────────


def test_jobs_stats_route_is_gone():
    from umrs.api import jobs

    paths = {getattr(route, "path", "") for route in jobs.router.routes}
    assert "/umrs/jobs/stats" not in paths
    assert not hasattr(jobs, "get_job_stats")


def test_jobs_stats_path_is_a_job_lookup_that_needs_identity(strict):
    from umrs.api import jobs

    client = _client(jobs.router)
    assert client.get("/umrs/jobs/stats", params={"project_id": "proj-b"}).status_code == 401
    assert client.get("/umrs/jobs/stats", headers=FORGED).status_code == 401


# ── 2. POST /umrs/knowledge/external/search ────────────────────────────────


@pytest.fixture
def searches(monkeypatch) -> list[str]:
    """Replace the outbound searcher; record each query that would egress."""
    from umrs.knowledge import external

    calls: list[str] = []

    class _Searcher:
        def search_all(self, query: str, **_kw: Any) -> list:
            calls.append(query)
            return []

    monkeypatch.setattr(external, "ExternalKnowledgeSearcher", _Searcher)
    return calls


def _search(headers: dict[str, str] | None = None):
    from umrs.api import knowledge

    return _client(knowledge.router).post(
        "/umrs/knowledge/external/search",
        json={"query": "attention", "sources": ["arxiv"]},
        headers=headers or {},
    )


def test_external_search_without_identity_is_refused_before_egress(strict, searches):
    assert _search().status_code == 401
    assert searches == []


def test_external_search_with_a_forged_tuple_is_refused(strict, searches):
    assert _search(FORGED).status_code == 401
    assert searches == []


def test_external_search_with_a_signed_caller_runs(strict, searches, caplog):
    caplog.set_level("INFO", logger="umrs.api.knowledge")
    resp = _search(ALICE)
    assert resp.status_code == 200
    assert resp.json()["sources"] == ["arxiv"]
    assert searches == ["attention"]
    # Each egress is attributable to the person and the project (review
    # 2026-10-18: the log named only the project).
    assert "user=alice project=proj-a" in caplog.text


def test_external_search_in_dev_mode_takes_the_legacy_project_header(mvp, searches):
    assert _search({"X-Project-ID": "proj-a"}).status_code == 200
    assert searches == ["attention"]


# ── 3. POST /umrs/companion/workers ────────────────────────────────────────


@pytest.fixture
def workers(monkeypatch) -> list[Any]:
    """Fake worker registry; records each spec registered."""
    from umrs.api import companion

    specs: list[Any] = []

    class _Manager:
        def __init__(self, _pool: Any) -> None:
            pass

        def register_active(self, spec: Any, *, job_id: str) -> dict:
            specs.append(spec)
            return {
                "registry_key": f"{spec.owner_user_id}@{spec.project_id}",
                "owner_user_id": spec.owner_user_id,
                "project_id": spec.project_id,
                "job_id": job_id,
                "state": "active",
            }

    monkeypatch.setattr(companion, "CompanionWorkerManager", _Manager)
    return specs


def _job(user: str, project: str, job_type: str = "companion_worker") -> Any:
    from umrs.jobs.models import JobType

    return SimpleNamespace(user_id=user, project_id=project, job_type=JobType(job_type))


JOBS = {
    "job-1": _job("alice", "proj-a"),
    "job-bob": _job("bob", "proj-a"),
    "job-other-project": _job("alice", "proj-b"),
    "job-other-type": _job("alice", "proj-a", "workflow_run"),
}


@pytest.fixture
def job_store(monkeypatch):
    """A fake job store holding ``JOBS``; an unknown id raises as the real one."""
    from umrs.api import jobs
    from umrs.jobs.store import JobNotFoundError

    class _Store:
        def get_job(self, job_id: str) -> Any:
            if job_id not in JOBS:
                raise JobNotFoundError(f"Job not found: {job_id}")
            return JOBS[job_id]

    monkeypatch.setattr(jobs, "_job_store", _Store())


def _register(owner: str, headers: dict[str, str], job_id: str | None = "job-1"):
    from umrs.api import companion

    submitted: list[tuple[str, str]] = []

    async def _submitter(owner: str, project: str) -> str:
        submitted.append((owner, project))
        return "job-sub"

    client = _client(companion.router, umrs_db_pool=object(), companion_job_submitter=_submitter)
    body = {"owner_user_id": owner, "project_id": "proj-a", "job_id": job_id}
    return client.post("/umrs/companion/workers", json=body, headers=headers), submitted


def test_empty_owner_defaults_to_the_verified_caller(strict, workers, job_store):
    resp, _ = _register("", ALICE)
    assert resp.status_code == 201
    assert resp.json()["owner_user_id"] == "alice"
    assert [spec.owner_user_id for spec in workers] == ["alice"]


def test_submitted_worker_job_runs_as_the_verified_caller(strict, workers, job_store):
    resp, submitted = _register("", ALICE, job_id=None)
    assert resp.status_code == 201
    assert submitted == [("alice", "proj-a")]


def test_another_owner_is_refused(strict, workers, job_store):
    resp, submitted = _register("bob", ALICE, job_id=None)
    assert resp.status_code == 401
    assert resp.headers[ERROR_CODE_HEADER] == "companion.unauthorized"
    assert workers == []
    assert submitted == []


def test_worker_registration_with_a_forged_tuple_is_refused(strict, workers, job_store):
    resp, _ = _register("", FORGED)
    assert resp.status_code == 401
    assert workers == []


def test_dev_caller_without_identity_keeps_the_declared_owner(mvp, workers):
    resp, _ = _register("dev-user", {})
    assert resp.status_code == 201
    assert [spec.owner_user_id for spec in workers] == ["dev-user"]


# The worker row's job id is what its turns run under (``turns.begin``) and
# what the supervisor exports as R4_COMPANION_JOB_ID, so a caller-supplied id
# must be the caller's own companion job (review 2026-10-18: it was bound
# unchecked, so a caller could bind another user's job).


@pytest.mark.parametrize("job_id", ["job-bob", "job-other-project", "job-other-type", "job-ghost"])
def test_a_job_id_that_is_not_the_callers_companion_job_reads_as_not_found(
    strict, workers, job_store, job_id
):
    resp, submitted = _register("", ALICE, job_id=job_id)
    assert resp.status_code == 404
    assert resp.headers[ERROR_CODE_HEADER] == "companion.not_found"
    assert workers == [] and submitted == []


def test_a_foreign_job_and_an_unknown_one_answer_alike(strict, workers, job_store):
    foreign, _ = _register("", ALICE, job_id="job-bob")
    unknown, _ = _register("", ALICE, job_id="job-ghost")
    assert foreign.json()["detail"].replace("job-bob", "X") == unknown.json()["detail"].replace(
        "job-ghost", "X"
    )


def test_a_caller_job_id_without_a_job_store_is_refused(strict, workers, monkeypatch):
    from umrs.api import jobs

    monkeypatch.setattr(jobs, "_job_store", None)
    resp, _ = _register("", ALICE, job_id="job-1")
    assert resp.status_code == 503
    assert resp.headers[ERROR_CODE_HEADER] == "companion.unavailable"
    assert workers == []


def test_own_companion_job_id_is_bound(strict, workers, job_store):
    resp, submitted = _register("", ALICE, job_id="job-1")
    assert resp.status_code == 201
    assert resp.json()["job_id"] == "job-1"
    assert submitted == []


# ── 4. POST /umrs/companion/approvals/{id}/decision ────────────────────────


@pytest.fixture
def inbox():
    from umrs.knowledge import tool_approvals

    tool_approvals.clear_all()
    tool_approvals.request_approval(
        "loop-a", 1, {"kind": "workflow_trigger"}, approval_id="ap-alice", user_id="alice"
    )
    yield tool_approvals
    tool_approvals.clear_all()


def _decide(approval_id: str, headers: dict[str, str]):
    from umrs.api import companion

    return _client(companion.router).post(
        f"/umrs/companion/approvals/{approval_id}/decision",
        json={"decision": "once"},
        headers=headers,
    )


def test_another_users_approval_reads_as_not_found(strict, inbox):
    foreign = _decide("ap-alice", _headers("bob", "proj-a"))
    unknown = _decide("ap-ghost", _headers("bob", "proj-a"))
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.headers[ERROR_CODE_HEADER] == "companion.not_found"
    assert "ownership" not in foreign.text
    assert inbox.get("ap-alice")["status"] == "pending"


def test_approval_decision_with_a_forged_tuple_is_refused(strict, inbox):
    assert _decide("ap-alice", FORGED).status_code == 401
    assert inbox.get("ap-alice")["status"] == "pending"


def test_owner_still_decides_own_approval(strict, inbox):
    resp = _decide("ap-alice", ALICE)
    assert resp.status_code == 200
    assert inbox.get("ap-alice")["status"] == "approved"


# ── 5. PATCH /umrs/verification/findings/{key} (deleted 2026-10-18) ──────


def test_projectless_finding_transition_route_is_deleted():
    """The verified-admin rule ended with the route: it is not mounted, and
    the module that served it is gone (H-10.1 follow-up, 2026-10-18)."""
    assert importlib.util.find_spec("umrs.api.verification") is None


def test_finding_transition_lives_on_the_per_project_lane():
    from umrs.api import knowledge

    paths = {(m, r.path) for r in knowledge.router.routes for m in r.methods}
    assert ("PATCH", "/umrs/knowledge/findings/{finding_key}") in paths


# ── 6. experiments status filter ───────────────────────────────────────────


def test_experiments_bad_request_is_a_registered_code():
    assert PlatformError.experiments("bad_request", "m").http_status == 400


def test_unknown_experiment_status_is_a_typed_bad_request(strict):
    from umrs.api import experiment_registry as api
    from umrs.experiment_registry import InMemoryExperimentRegistry

    previous = api._registry_store
    api.set_registry(InMemoryExperimentRegistry())
    try:
        resp = _client(api.router).get(
            "/umrs/experiment-registry", params={"status": "bogus"}, headers=ALICE
        )
    finally:
        api.set_registry(previous)
    assert resp.status_code == 400
    assert resp.headers[ERROR_CODE_HEADER] == "experiments.bad_request"
    assert "bogus" in resp.json()["detail"]


# ── 7. knowledge/a2l_bridge.py ─────────────────────────────────────────────


def test_dormant_a2l_bridge_is_removed():
    assert importlib.util.find_spec("umrs.knowledge.a2l_bridge") is None


def test_ingest_a2l_hook_stays_best_effort():
    # The ingest hook imported the bridge inside a blanket try; with the
    # module gone it still returns quietly and makes no call.
    from umrs.jobs.knowledge_adapters import _bridge_a2l_ref

    assert _bridge_a2l_ref("art-1", "hash-1", "proj-a") is None
