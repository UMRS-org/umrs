"""H-1 leftovers (2026-10-18): endpoint host paths and lookup-first oracles.

(a) ``POST /umrs/endpoints/{job_id}/upgrade`` handed ``request.model_path``
    to ``service.upgrade_endpoint`` unchecked: any path, any user's weights.
    It now gets the provision rule: site-wide prefixes for everyone, and for
    a non-admin a host path it names must lie under its own roots unless it
    is the storage_ref of a catalog entry it may provision.
(b) ``POST /umrs/jobs`` with ``job_type=endpoint_provision`` checked
    ``spec.mounts`` only against the site-wide prefixes and ``spec.model_path``
    not at all; the per-caller rule now applies to the parent and every
    fan-out child, with the provision route's recipe policy (env_vars,
    container_image, tier=system admin-only) and a model that must resolve.
(c) upgrade, ``PUT /{job_id}/access``, ``DELETE /{job_id}/deprovision`` and
    ``PUT /{job_id}/used`` looked the endpoint up (404) before the strict
    identity check (401), so an anonymous direct caller learned which job ids
    exist.

Strict mode, gateway-signed tuples (``sign_claims``) and forged tuples.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import endpoints as endpoints_api
from umrs.api.endpoints import get_registry
from umrs.api.endpoints import router as endpoints_router
from umrs.orchestrator import caller_roots
from umrs.orchestrator.endpoint_registry import EndpointRegistry

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1x-endpoint-paths-secret"
JOB = "job-001"


def _signed(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


FORGED = {
    "X-User-ID": "mallory",
    "X-Verified-Project-ID": "proj-a",
    "X-Verified-Is-Admin": "true",
    "X-Verified-HMAC": "bogus",
}
ALICE = _signed("alice", "proj-a")
ADMIN = _signed("ops", "proj-a", admin=True)


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.delenv("UMRS_ENDPOINT_CALLER_MOUNT_ROOTS", raising=False)
    monkeypatch.delenv("UMRS_ENDPOINT_ALLOWED_MOUNT_PREFIXES", raising=False)
    monkeypatch.delenv("UMRS_ENDPOINT_DENIED_MOUNT_PREFIXES", raising=False)


class _Entry:
    def __init__(self, **row):
        self._row = row

    def to_dict(self):
        return dict(self._row)


class _ModelStore:
    def __init__(self, *rows):
        self.rows = [_Entry(**row) for row in rows]

    def list_models(self, _ctx, limit=500):  # noqa: ARG002 — mirrors ModelStore
        return self.rows


@pytest.fixture
def models(monkeypatch):
    """The model catalog the upgrade route consults (empty unless filled)."""
    store = _ModelStore()
    monkeypatch.setattr("umrs.api.models.get_optional_model_store", lambda: store)
    return store


@pytest.fixture
def upgraded(monkeypatch, models):  # noqa: ARG001 — installs the empty catalog
    """The model paths service.upgrade_endpoint was handed."""
    seen: list[str] = []

    async def _upgrade(*, job_id, new_model_path, timeout_sec):  # noqa: ARG001
        seen.append(new_model_path)
        return True, "job-new", None

    service = SimpleNamespace(
        upgrade_endpoint=AsyncMock(side_effect=_upgrade),
        deprovision=AsyncMock(return_value=None),
    )
    monkeypatch.setattr("umrs.orchestrator.service.get_orchestration_service", lambda: service)
    return seen


@pytest.fixture
def client(upgraded) -> TestClient:  # noqa: ARG001 — installs the fake service
    registry = EndpointRegistry()
    registry.register(
        project_id="proj-a",
        model="model-x",
        job_id=JOB,
        host="compute-01",
        port=11434,
        backend="slurm",
    )
    registry.update_metadata(JOB, {"model": "model-x", "model_path": "/models/model-x"})
    app = FastAPI()
    app.include_router(endpoints_router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _upgrade(client: TestClient, path: str, headers: dict, job: str = JOB):
    return client.post(f"/umrs/endpoints/{job}/upgrade", json={"model_path": path}, headers=headers)


# ── (a) upgrade model_path ──────────────────────────────────────────────────


def test_upgrade_to_a_path_under_the_callers_project_root(client, upgraded):
    r = _upgrade(client, "/projects/proj-a/models/v2", ALICE)
    assert r.status_code == 202, r.text
    assert upgraded == ["/projects/proj-a/models/v2"]
    assert r.json()["new_model_path"] == "/projects/proj-a/models/v2"


def test_upgrade_to_the_callers_home(client, upgraded):
    assert _upgrade(client, "/home/alice/weights/v2", ALICE).status_code == 202
    assert upgraded == ["/home/alice/weights/v2"]


@pytest.mark.parametrize(
    "path",
    [
        "/home/bob/weights",
        "/projects/proj-b/models/v2",
        "/projects/proj-a/../proj-b/models",
        "/models/unregistered",
    ],
)
def test_upgrade_refuses_a_foreign_host_path(client, upgraded, path):
    r = _upgrade(client, path, ALICE)
    assert r.status_code == 403, r.text
    assert "UMRS_ENDPOINT_CALLER_MOUNT_ROOTS" in r.text
    assert upgraded == []


def test_upgrade_refuses_a_path_outside_the_site_prefixes_for_everyone(client, upgraded):
    for headers in (ALICE, ADMIN):
        assert _upgrade(client, "/etc/secrets", headers).status_code == 400
    assert upgraded == []


def test_admin_upgrades_to_any_site_path(client, upgraded):
    assert _upgrade(client, "/home/bob/weights", ADMIN).status_code == 202
    assert upgraded == ["/home/bob/weights"]


def test_upgrade_to_a_registered_catalog_path(client, upgraded, models):
    models.rows = [_Entry(name="model-x-v2", scope="system", storage_ref="/models/model-x-v2")]
    assert _upgrade(client, "/models/model-x-v2", ALICE).status_code == 202
    assert upgraded == ["/models/model-x-v2"]


def test_upgrade_refuses_another_users_private_catalog_path(client, upgraded, models):
    models.rows = [
        _Entry(name="bobs", scope="user", owner_id="bob", storage_ref="/models/bobs-private")
    ]
    r = _upgrade(client, "/models/bobs-private", ALICE)
    assert r.status_code == 403, r.text
    assert upgraded == []


def test_upgrade_with_a_failing_catalog_fails_closed(client, upgraded, monkeypatch):
    class _Down:
        def list_models(self, *_a, **_k):
            raise RuntimeError("db down")

    monkeypatch.setattr("umrs.api.models.get_optional_model_store", lambda: _Down())
    assert _upgrade(client, "/models/model-x-v2", ALICE).status_code == 403
    assert upgraded == []


def test_upgrade_refuses_a_forged_tuple(client, upgraded):
    assert _upgrade(client, "/home/mallory/w", FORGED).status_code == 401
    assert upgraded == []


# ── (c) identity before the lookup ─────────────────────────────────────────


def _mutations(job: str):
    return [
        ("POST", f"/umrs/endpoints/{job}/upgrade", {"json": {"model_path": "/models/x"}}),
        ("PUT", f"/umrs/endpoints/{job}/access", {"json": {"access": []}}),
        ("DELETE", f"/umrs/endpoints/{job}/deprovision", {}),
        ("PUT", f"/umrs/endpoints/{job}/used", {}),
    ]


_IDS = ["upgrade", "access", "deprovision", "used"]


@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_anonymous_strict_caller_cannot_tell_ids_apart(client, index):
    """Unknown and existing job ids both answer 401 (old code: 404 vs 401)."""
    codes = set()
    for job in (JOB, "job-does-not-exist"):
        method, path, kwargs = _mutations(job)[index]
        codes.add(client.request(method, path, **kwargs).status_code)
    assert codes == {401}


@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_forged_tuple_is_401_for_unknown_and_existing_ids(client, index):
    for job in (JOB, "job-does-not-exist"):
        method, path, kwargs = _mutations(job)[index]
        assert client.request(method, path, headers=FORGED, **kwargs).status_code == 401


@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_signed_caller_still_gets_404_for_an_unknown_id(client, index):
    method, path, kwargs = _mutations("job-does-not-exist")[index]
    assert client.request(method, path, headers=ALICE, **kwargs).status_code == 404


@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_other_projects_endpoint_reads_as_not_found(client, index):
    method, path, kwargs = _mutations(JOB)[index]
    headers = _signed("bob", "proj-b")
    assert client.request(method, path, headers=headers, **kwargs).status_code == 404


# ── (b) POST /umrs/jobs endpoint_provision ─────────────────────────────────


class _JobStore:
    _get_conn = None  # no catalog autopin

    def __init__(self):
        self.created: list[dict] = []

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return "job-1"

    create_fan_out_graph = create_job

    def enqueue_job(self, _job_id):
        pass


def _system_entry(name: str) -> dict:
    """Every name resolves to a system-scope entry unless a test says otherwise."""
    return {"name": name, "scope": "system", "storage_ref": f"/models/{name}"}


@pytest.fixture
def jobs(monkeypatch):
    from umrs.api.jobs import get_job_store
    from umrs.api.jobs import router as jobs_router

    monkeypatch.setattr(endpoints_api, "_resolve_registered_model_entry", _system_entry)
    store = _JobStore()
    app = FastAPI()
    app.include_router(jobs_router)
    app.dependency_overrides[get_job_store] = lambda: store
    return TestClient(app), store


def _provision_job(spec: dict, fan_out: bool = False) -> dict:
    body = {
        "project_id": "proj-a",
        "user_id": "alice",
        "job_type": "endpoint_provision",
        "spec": {"model": "model-x", **spec},
        "enqueue_immediately": False,
    }
    if fan_out:  # a clean parent; the child carries the paths
        body["spec"] = {"model": "model-x"}
        body["fan_out_jobs"] = [
            {"name": "c", "job_type": "endpoint_provision", "spec": {"model": "m", **spec}}
        ]
    return body


_FOREIGN_SPECS = [
    {"mounts": {"project_data": {"source": "/projects/proj-b/data"}}},
    {"mounts": {"rag_index": {"source": "/home/bob/index"}}},
    {"mounts": {"mcp_servers": [{"source": "/shared/users/bob/mcp"}]}},
    {"model_path": "/home/bob/weights"},
    {"model_path": "/models/unregistered"},
]


@pytest.mark.parametrize("fan_out", [False, True], ids=["parent", "fan-out-child"])
@pytest.mark.parametrize("spec", _FOREIGN_SPECS)
def test_job_endpoint_provision_refuses_foreign_host_paths(jobs, spec, fan_out):
    client, store = jobs
    r = client.post("/umrs/jobs", json=_provision_job(spec, fan_out), headers=ALICE)
    assert r.status_code == 403, r.text
    assert "UMRS_ENDPOINT_CALLER_MOUNT_ROOTS" in r.text
    assert store.created == []


def test_job_endpoint_provision_admits_the_callers_own_paths(jobs):
    client, store = jobs
    spec = {
        "model_path": "/projects/proj-a/models/m",
        "mounts": {
            "project_data": {"source": "/projects/proj-a/data"},
            "rag_index": {"source": "/home/alice/index"},
            "mcp_servers": [{"source": "/shared/users/alice/mcp"}],
        },
    }
    r = client.post("/umrs/jobs", json=_provision_job(spec), headers=ALICE)
    assert r.status_code == 201, r.text
    assert len(store.created) == 1


def test_job_endpoint_provision_admin_keeps_site_paths(jobs):
    client, store = jobs
    spec = {"model_path": "/home/bob/weights", "mounts": {"rag_index": {"source": "/home/bob/i"}}}
    assert client.post("/umrs/jobs", json=_provision_job(spec), headers=ADMIN).status_code == 201
    assert len(store.created) == 1


def test_job_endpoint_provision_model_path_needs_the_site_prefixes(jobs):
    client, store = jobs
    r = client.post("/umrs/jobs", json=_provision_job({"model_path": "/etc/x"}), headers=ALICE)
    assert r.status_code == 400, r.text
    assert store.created == []


def test_job_endpoint_provision_catalog_storage_ref_is_not_caller_named(jobs, monkeypatch):
    client, store = jobs
    entry = {"name": "model-x", "scope": "system", "storage_ref": "/models/model-x"}
    monkeypatch.setattr(endpoints_api, "_resolve_registered_model_entry", lambda _m: entry)
    body = _provision_job({"model_path": "/models/model-x"})
    assert client.post("/umrs/jobs", json=body, headers=ALICE).status_code == 201
    assert len(store.created) == 1


def test_job_endpoint_provision_refuses_another_users_private_model(jobs, monkeypatch):
    client, store = jobs
    entry = {"name": "model-x", "scope": "user", "owner_id": "bob", "storage_ref": "/m/x"}
    monkeypatch.setattr(endpoints_api, "_resolve_registered_model_entry", lambda _m: entry)
    assert client.post("/umrs/jobs", json=_provision_job({}), headers=ALICE).status_code == 403
    assert store.created == []


def test_job_endpoint_provision_refuses_a_forged_tuple(jobs):
    client, store = jobs
    assert client.post("/umrs/jobs", json=_provision_job({}), headers=FORGED).status_code == 401
    assert store.created == []


# Recipe fields (review 2026-10-18): the executor merges spec.env_vars over
# the mount env, and PROJECT_DATA_PATH / RAG_INDEX_PATH / MCP_SERVERS_PATH are
# host bind sources, so env_vars skipped every path rule; container_image and
# tier=system are admin-only on POST /umrs/endpoints/provision.
_RECIPE_SPECS = [
    {"env_vars": {"PROJECT_DATA_PATH": "/home/bob/x", "RIVAULT_EXTRA_PROFILES": "project-data"}},
    {"env_vars": {"HARMLESS": "1"}},
    {"container_image": "docker://evil/image:latest"},
    {"tier": "system"},
]


@pytest.mark.parametrize("fan_out", [False, True], ids=["parent", "fan-out-child"])
@pytest.mark.parametrize("spec", _RECIPE_SPECS, ids=["env-path", "env", "image", "system-tier"])
def test_job_endpoint_provision_refuses_admin_only_recipe_fields(jobs, spec, fan_out):
    client, store = jobs
    r = client.post("/umrs/jobs", json=_provision_job(spec, fan_out), headers=ALICE)
    assert r.status_code == 403, r.text
    assert "admin" in r.text.lower()
    assert store.created == []


@pytest.mark.parametrize("spec", _RECIPE_SPECS, ids=["env-path", "env", "image", "system-tier"])
def test_job_endpoint_provision_admin_keeps_recipe_fields(jobs, spec):
    client, store = jobs
    assert client.post("/umrs/jobs", json=_provision_job(spec), headers=ADMIN).status_code == 201
    assert len(store.created) == 1


def test_job_endpoint_provision_user_and_project_tiers_stay_open(jobs):
    client, store = jobs
    for tier in ("user", "project"):
        r = client.post("/umrs/jobs", json=_provision_job({"tier": tier}), headers=ALICE)
        assert r.status_code == 201, r.text
    assert len(store.created) == 2


# Model ACL fails closed (review 2026-10-18): with no entry resolved the
# check passed and the executor resolved the name on its own (JSON fallback
# included), so an unchecked private entry could run.


@pytest.mark.parametrize("fan_out", [False, True], ids=["parent", "fan-out-child"])
def test_job_endpoint_provision_refuses_an_unresolved_model(jobs, monkeypatch, fan_out):
    client, store = jobs
    monkeypatch.setattr(endpoints_api, "_resolve_registered_model_entry", lambda _m: None)
    r = client.post("/umrs/jobs", json=_provision_job({}, fan_out), headers=ALICE)
    assert r.status_code == 400, r.text
    assert "not in the model catalog" in r.text
    assert store.created == []


def test_job_endpoint_provision_without_a_model_checks_the_default_entry(jobs, monkeypatch):
    client, store = jobs
    asked: list[str] = []

    def _bobs(name):
        asked.append(name)
        return {"name": name, "scope": "user", "owner_id": "bob", "storage_ref": "/m/d"}

    monkeypatch.setattr(endpoints_api, "_resolve_registered_model_entry", _bobs)
    body = _provision_job({})
    body["spec"].pop("model")
    assert client.post("/umrs/jobs", json=body, headers=ALICE).status_code == 403
    assert asked == ["default"]
    assert store.created == []


def test_job_endpoint_provision_without_a_model_and_no_default_entry(jobs, monkeypatch):
    client, store = jobs
    monkeypatch.setattr(endpoints_api, "_resolve_registered_model_entry", lambda _m: None)
    body = _provision_job({})
    body["spec"].pop("model")
    assert client.post("/umrs/jobs", json=body, headers=ALICE).status_code == 400
    assert store.created == []


def test_job_endpoint_provision_admin_skips_the_catalog_lookup(jobs, monkeypatch):
    client, store = jobs
    monkeypatch.setattr(endpoints_api, "_resolve_registered_model_entry", lambda _m: None)
    assert client.post("/umrs/jobs", json=_provision_job({}), headers=ADMIN).status_code == 201
    assert len(store.created) == 1


def test_job_runs_under_the_signed_project_and_user_not_the_body(jobs):
    """POST /umrs/jobs stays IDENTITY-ONLY in the scope guard's ALLOWLIST:
    its project binding is the signed tuple, not the model-scope check."""
    client, store = jobs
    body = _provision_job({})
    body.update(project_id="proj-b", user_id="bob")
    assert client.post("/umrs/jobs", json=body, headers=ALICE).status_code == 201
    assert (store.created[0]["project_id"], store.created[0]["user_id"]) == ("proj-a", "alice")


# ── the shared module ──────────────────────────────────────────────────────


def test_raw_mount_sources_reads_the_normalize_mounts_shape():
    mounts = {
        "mcp_servers": [{"source": "/a"}, "junk", {"source": " "}],
        "project_data": {"source": "/b"},
        "rag_index": "junk",
    }
    assert caller_roots.raw_mount_sources(mounts) == ["/a", "/b"]
    assert caller_roots.raw_mount_sources(None) == []
    assert caller_roots.raw_mount_sources({"mcp_servers": "x"}) == []


def test_unsafe_placeholder_values_name_no_root():
    assert caller_roots.caller_mount_roots("..", "") == []
    assert caller_roots.foreign_sources(["/projects/x"], "a/b", "") == ["/projects/x"]
