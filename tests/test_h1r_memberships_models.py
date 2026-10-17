"""H-1 memberships lane (2026-10-18): the model registry trusts only a
gateway-signed memberships list, and the durable-job partition of
/stage and /{model_name}/download is checked against the verified caller.

Before: ``get_access_context`` took ``X-Verified-Project-Memberships`` as
written (no HMAC covers it), so the ACL (list, create, publish, stage,
download into project scope) honoured any project a direct sidecar caller
named; ``?caller_project_id`` picked the job partition unchecked.

After: the list counts only when ``X-Verified-Memberships-HMAC`` verifies
against the signed (user, project, admin) tuple; otherwise the caller has
its signed project only. mvp mode keeps the legacy trust.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import count

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.jobs import get_job_store
from umrs.api.models import get_model_store, router
from umrs.model_registry.models import ModelEntry, ModelScope, ModelType
from umrs.model_registry.store import ModelNotFoundError
from umrs.model_registry.store.acl import ACLEnforcer

from r4_contracts.handshake_hmac import sign_claims, sign_memberships

SECRET = "h1r-memberships-secret"


def _identity(user: str, project: str, admin: bool = False, hmac: str | None = None) -> dict:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": hmac or sign_claims(SECRET, user, project, admin),
    }


def _with_members(headers: dict, members: str, signed_for: str | None = None) -> dict:
    """Memberships header plus its HMAC (signed over ``signed_for`` when given)."""
    admin = headers["X-Verified-Is-Admin"] == "true"
    sig = sign_memberships(
        SECRET,
        headers["X-User-ID"],
        headers["X-Verified-Project-ID"],
        admin,
        members if signed_for is None else signed_for,
    )
    return {
        **headers,
        "X-Verified-Project-Memberships": members,
        "X-Verified-Memberships-HMAC": sig,
    }


ALICE = _identity("alice", "proj-a")
ALICE_SIGNED_B = _with_members(ALICE, "proj-a,proj-b")
ALICE_UNSIGNED_B = {**ALICE, "X-Verified-Project-Memberships": "proj-a,proj-b"}
ALICE_TAMPERED_B = _with_members(ALICE, "proj-a,proj-b", signed_for="proj-a")
# Forged tuple: bogus HMAC, admin flag true, a foreign project.
FORGED = {
    **_identity("alice", "proj-b", admin=True, hmac="0" * 64),
    "X-Verified-Project-Memberships": "proj-b",
    "X-Verified-Memberships-HMAC": "0" * 64,
}


class FakeModelStore:
    """Enough of ModelStore for these routes, with the real ACL rules."""

    def __init__(self) -> None:
        self.acl = ACLEnforcer()
        self._ids = count(1)
        self.models: dict[str, ModelEntry] = {}
        for model in (
            ModelEntry(model_id="m-sys", name="base", scope=ModelScope.SYSTEM),
            ModelEntry(model_id="m-a", name="a", scope=ModelScope.PROJECT, project_id="proj-a"),
            ModelEntry(model_id="m-b", name="b", scope=ModelScope.PROJECT, project_id="proj-b"),
            ModelEntry(model_id="m-u", name="mine", scope=ModelScope.USER, owner_id="alice"),
        ):
            self.models[model.model_id] = model

    def list_models_with_reasons(self, ctx, limit=100):
        rows = [m for m in self.models.values() if self.acl.can_read(m, ctx)]
        return [(m, self.acl.read_reason(m, ctx)) for m in rows][:limit]

    def get_model(self, model_id, ctx):
        model = self.models.get(model_id)
        if model is None or not self.acl.can_read(model, ctx):
            raise ModelNotFoundError(model_id)
        return model

    def get_model_by_name(self, name, scope, ctx):
        del ctx  # system models are readable by everyone
        for model in self.models.values():
            if model.name == name and model.scope == scope:
                return model
        raise ModelNotFoundError(name)

    def check_policy_access(self, model_name, project_id):
        del model_name, project_id
        return True, None

    def create_model(self, *, name, scope, ctx, project_id=None, owner_id=None, **kwargs):
        self.acl.enforce_create(scope, ctx, project_id)
        model = ModelEntry(
            model_id=f"m-new-{next(self._ids)}",
            name=name,
            scope=scope,
            owner_id=owner_id,
            project_id=project_id,
            storage_ref=kwargs.get("storage_ref") or "",
            model_type=kwargs.get("model_type") or ModelType.LLM,
        )
        self.models[model.model_id] = model
        return model

    def delete_model(self, model_id, ctx):
        del ctx
        self.models.pop(model_id, None)


class FakeJobStore:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return f"job-{len(self.created)}"

    def enqueue_job(self, job_id):
        del job_id


@pytest.fixture
def stores():
    return FakeModelStore(), FakeJobStore()


@pytest.fixture
def client(stores, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    store, jobs = stores
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_model_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: jobs
    return TestClient(app)


def _listed(client, headers) -> set[str]:
    resp = client.get("/umrs/models", headers=headers)
    assert resp.status_code == 200, resp.text
    return {m["model_id"] for m in resp.json()["models"]}


# ── the memberships list: signed counts, anything else is ignored ───────────


def test_signed_memberships_widen_the_model_list(client):
    assert _listed(client, ALICE_SIGNED_B) == {"m-sys", "m-a", "m-b", "m-u"}


@pytest.mark.parametrize(
    "headers", [ALICE_UNSIGNED_B, ALICE_TAMPERED_B], ids=["unsigned", "tampered"]
)
def test_unsigned_or_tampered_memberships_are_ignored(client, headers):
    assert _listed(client, headers) == {"m-sys", "m-a", "m-u"}
    assert client.get("/umrs/models/m-b", headers=headers).status_code == 404


def test_memberships_signed_for_another_user_are_ignored(client):
    bob_sig = sign_memberships(SECRET, "bob", "proj-a", False, "proj-a,proj-b")
    headers = {**ALICE_UNSIGNED_B, "X-Verified-Memberships-HMAC": bob_sig}
    assert _listed(client, headers) == {"m-sys", "m-a", "m-u"}


def test_forged_tuple_is_refused(client):
    assert client.get("/umrs/models", headers=FORGED).status_code == 401
    resp = client.post(
        "/umrs/models",
        headers=FORGED,
        json={"name": "x", "scope": "project", "project_id": "proj-b", "storage_ref": "/m/x"},
    )
    assert resp.status_code == 401


def _create_in_b(client, headers):
    return client.post(
        "/umrs/models",
        headers=headers,
        json={"name": "x", "scope": "project", "project_id": "proj-b", "storage_ref": "/m/x"},
    )


def test_create_in_a_member_project_needs_a_signed_list(client):
    assert _create_in_b(client, ALICE_UNSIGNED_B).status_code == 403
    assert _create_in_b(client, ALICE_TAMPERED_B).status_code == 403
    assert _create_in_b(client, ALICE_SIGNED_B).status_code == 201


def _publish_to_b(client, headers):
    return client.post(
        "/umrs/models/m-u/publish", headers=headers, json={"target_project_id": "proj-b"}
    )


def test_publish_to_a_member_project_needs_a_signed_list(client):
    assert _publish_to_b(client, ALICE_UNSIGNED_B).status_code == 403
    assert _publish_to_b(client, ALICE_TAMPERED_B).status_code == 403
    assert _publish_to_b(client, ALICE_SIGNED_B).status_code == 201


# ── /stage and /{model_name}/download ───────────────────────────────────────


def _stage(client, headers, scope="user", project_id=None, params=None):
    body = {"huggingface_id": "org/m", "name": "staged", "scope": scope}
    if project_id:
        body["project_id"] = project_id
    return client.post("/umrs/models/stage", headers=headers, params=params or {}, json=body)


def test_stage_into_a_member_project_needs_a_signed_list(client, stores):
    assert _stage(client, ALICE_UNSIGNED_B, "project", "proj-b").status_code == 403
    resp = _stage(client, ALICE_SIGNED_B, "project", "proj-b")
    assert resp.status_code == 202, resp.text
    assert stores[1].created[-1]["project_id"] == "proj-b"


def test_stage_refuses_a_personal_namespace_even_when_listed(client):
    headers = _with_members(ALICE, "proj-a,user:bob")
    assert _stage(client, headers, "project", "user:bob").status_code == 403


def test_stage_refuses_a_foreign_caller_project_id(client, stores):
    resp = _stage(client, ALICE, params={"caller_project_id": "victim"})
    assert resp.status_code == 403
    assert stores[1].created == []


def test_stage_job_lands_in_the_verified_project(client, stores):
    assert _stage(client, ALICE, params={"caller_project_id": "proj-a"}).status_code == 202
    assert _stage(client, ALICE).status_code == 202
    assert [job["project_id"] for job in stores[1].created] == ["proj-a", "proj-a"]


def _download(client, headers, target_scope="user", project_id=None, params=None):
    body = {"target_scope": target_scope, "custom_name": f"copy-{target_scope}"}
    if project_id:
        body["project_id"] = project_id
    return client.post(
        "/umrs/models/base/download", headers=headers, params=params or {}, json=body
    )


def test_download_into_a_member_project_needs_a_signed_list(client, stores):
    assert _download(client, ALICE_UNSIGNED_B, "project", "proj-b").status_code == 403
    resp = _download(client, ALICE_SIGNED_B, "project", "proj-b")
    assert resp.status_code == 202, resp.text
    assert stores[1].created[-1]["project_id"] == "proj-b"


def test_download_refuses_a_foreign_caller_project_id(client, stores):
    resp = _download(client, ALICE, params={"caller_project_id": "victim"})
    assert resp.status_code == 403
    assert stores[1].created == []


def test_admin_may_name_any_caller_project_id(client, stores):
    root = _identity("root", "ops", admin=True)
    assert _download(client, root, params={"caller_project_id": "elsewhere"}).status_code == 202
    assert stores[1].created[-1]["project_id"] == "elsewhere"


# ── mvp keeps the legacy trust ──────────────────────────────────────────────


def test_mvp_mode_keeps_the_unsigned_memberships_list(client, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    headers = {k: v for k, v in ALICE_UNSIGNED_B.items() if k != "X-Verified-HMAC"}
    assert "m-b" in _listed(client, headers)


def test_fake_store_rules_match_the_real_acl():
    """Sanity: the fake uses ACLEnforcer, so a member sees its project rows."""
    from umrs.model_registry.store import AccessContext

    store = FakeModelStore()
    ctx = AccessContext(user_id="alice", project_memberships={"proj-b"})
    assert store.get_model("m-b", ctx).project_id == "proj-b"
    with pytest.raises(ModelNotFoundError):
        store.get_model("m-b", replace(ctx, project_memberships={"proj-a"}))
