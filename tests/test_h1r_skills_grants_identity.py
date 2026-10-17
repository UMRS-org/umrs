"""H-1 round 2, skills-grants lane (hardening plan H-1, 2026-10-18).

- skills: ``_caller_identity`` resolved the caller with ``request_identity``
  but never checked the gateway HMAC, so a direct sidecar caller could claim
  any user, project or admin.
- grants: ``_caller`` read ``X-User-ID`` / ``X-Verified-Project-ID`` /
  ``X-Verified-Is-Admin`` raw on every route (create, list, mine, revoke).

In strict mode a forged tuple (bogus HMAC, admin flag true, foreign project)
is a 401. A grant revoke is for the grantor, an asset writer or a verified
admin; anyone else reads the grant as not found.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_grants_api import _FakePool

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1r-skills-grants-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _forged(user: str = "mallory", project: str = "proj-b") -> dict[str, str]:
    """Bogus HMAC, admin flag true, a project the caller does not belong to."""
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true",
        "X-Verified-HMAC": "0" * 64,
    }


def _admin_flag_flipped() -> dict[str, str]:
    """Mallory's own valid non-admin signature with the admin flag set."""
    headers = _headers("mallory", "proj-m")
    headers["X-Verified-Is-Admin"] = "true"
    return headers


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


# ── skills ───────────────────────────────────────────────────────────────────


@pytest.fixture
def skills_client():
    from umrs.api import skills
    from umrs.skills import SkillRegistry

    skills.set_skill_registry(SkillRegistry())
    app = FastAPI()
    app.include_router(skills.router)
    yield TestClient(app)
    skills.set_skill_registry(SkillRegistry())


def _register(client, headers, scope="project", name="b-secret", project=None):
    body = {"scope": scope, "manifest": {"name": name, "description": "d"}, "skill_md": "x"}
    if project is not None:
        body["project_id"] = project
    return client.post("/umrs/skills", json=body, headers=headers)


def test_forged_tuple_cannot_list_skills(skills_client):
    assert skills_client.get("/umrs/skills", headers=_forged()).status_code == 401


def test_forged_admin_cannot_register_a_system_skill(skills_client):
    assert _register(skills_client, _forged(), scope="system").status_code == 401


def test_forged_tuple_cannot_read_another_projects_skill(skills_client):
    assert _register(skills_client, _headers("bob", "proj-b")).status_code == 201
    assert (
        skills_client.get("/umrs/skills/b-secret", headers=_forged("mallory", "proj-b")).status_code
        == 401
    )


def test_flipped_admin_flag_cannot_delete_anothers_skill(skills_client):
    assert _register(skills_client, _headers("bob", "proj-b"), scope="user").status_code == 201
    resp = skills_client.delete("/umrs/skills/b-secret", headers=_admin_flag_flipped())
    assert resp.status_code == 401


def test_unsigned_identity_headers_are_refused(skills_client):
    unsigned = {"X-User-ID": "bob", "X-Verified-Project-ID": "proj-b"}
    assert skills_client.get("/umrs/skills", headers=unsigned).status_code == 401


def test_signed_caller_registers_and_another_project_reads_not_found(skills_client):
    assert _register(skills_client, _headers("bob", "proj-b")).status_code == 201
    assert (
        skills_client.get("/umrs/skills/b-secret", headers=_headers("bob", "proj-b")).status_code
        == 200
    )
    assert (
        skills_client.get("/umrs/skills/b-secret", headers=_headers("alice", "proj-a")).status_code
        == 404
    )


# ── grants ───────────────────────────────────────────────────────────────────


@pytest.fixture
def grants_client(monkeypatch):
    from umrs.api import grants

    def _writable(_pool, _asset_id, caller):
        return caller.user_id == "bob"  # bob owns every model/graph here

    monkeypatch.setattr(grants, "_model_writable", _writable)
    monkeypatch.setattr(grants, "_graph_writable", _writable)
    app = FastAPI()
    app.include_router(grants.router)
    app.state.umrs_db_pool = _FakePool()
    return TestClient(app)


def _grant(client, headers, grantee="carol"):
    body = {
        "asset_kind": "model",
        "asset_id": "mdl-b",
        "grantee_kind": "user",
        "grantee_id": grantee,
    }
    return client.post("/umrs/grants", json=body, headers=headers)


def _bobs_grant(client) -> str:
    resp = _grant(client, _headers("bob", "proj-b"))
    assert resp.status_code == 200, resp.text
    return resp.json()["grant"]["grant_id"]


def _tombstoned(client, grant_id: str) -> bool:
    return client.app.state.umrs_db_pool.rows[grant_id][7] is not None


def test_forged_tuple_cannot_read_received_grants(grants_client):
    assert grants_client.get("/umrs/grants/mine", headers=_forged("carol")).status_code == 401


def test_forged_admin_cannot_grant_anothers_asset(grants_client):
    assert _grant(grants_client, _forged(), grantee="mallory").status_code == 401
    assert grants_client.app.state.umrs_db_pool.rows == {}


def test_forged_admin_cannot_list_an_assets_grants(grants_client):
    _bobs_grant(grants_client)
    params = {"asset_kind": "model", "asset_id": "mdl-b"}
    assert grants_client.get("/umrs/grants", params=params, headers=_forged()).status_code == 401


def test_forged_admin_cannot_revoke_a_foreign_grant(grants_client):
    grant_id = _bobs_grant(grants_client)
    resp = grants_client.post(f"/umrs/grants/{grant_id}/revoke", headers=_forged())
    assert resp.status_code == 401
    assert not _tombstoned(grants_client, grant_id)


def test_flipped_admin_flag_cannot_revoke(grants_client):
    grant_id = _bobs_grant(grants_client)
    resp = grants_client.post(f"/umrs/grants/{grant_id}/revoke", headers=_admin_flag_flipped())
    assert resp.status_code == 401
    assert not _tombstoned(grants_client, grant_id)


def test_signed_stranger_reads_a_foreign_grant_as_not_found(grants_client):
    grant_id = _bobs_grant(grants_client)
    resp = grants_client.post(
        f"/umrs/grants/{grant_id}/revoke", headers=_headers("mallory", "proj-b")
    )
    assert resp.status_code == 404
    assert not _tombstoned(grants_client, grant_id)


def test_grantor_revokes_own_grant_after_losing_write_access(grants_client, monkeypatch):
    from umrs.api import grants

    grant_id = _bobs_grant(grants_client)
    monkeypatch.setattr(grants, "_model_writable", lambda *_: False)
    resp = grants_client.post(f"/umrs/grants/{grant_id}/revoke", headers=_headers("bob", "proj-b"))
    assert resp.status_code == 200
    assert _tombstoned(grants_client, grant_id)


def test_verified_admin_revokes_a_foreign_grant(grants_client):
    grant_id = _bobs_grant(grants_client)
    resp = grants_client.post(
        f"/umrs/grants/{grant_id}/revoke", headers=_headers("root", "ops", admin=True)
    )
    assert resp.status_code == 200
    assert _tombstoned(grants_client, grant_id)


def test_signed_grantee_lists_received_grants(grants_client, monkeypatch):
    from umrs.api import circles

    monkeypatch.setattr(circles, "circles_for_user", lambda *_: [])
    grant_id = _bobs_grant(grants_client)
    resp = grants_client.get("/umrs/grants/mine", headers=_headers("carol", "proj-c"))
    assert resp.status_code == 200
    assert [g["grant_id"] for g in resp.json()["grants"]] == [grant_id]


def test_bare_user_header_is_refused_in_strict(grants_client):
    """The old ``_caller`` took a lone ``X-User-ID`` with no verified project
    and no signature, in strict mode too."""
    resp = grants_client.get("/umrs/grants/mine", headers={"X-User-ID": "carol"})
    assert resp.status_code == 401


def test_unsigned_tuple_cannot_create_grant(grants_client):
    headers = {"X-User-ID": "bob", "X-Verified-Project-ID": "proj-b"}
    assert _grant(grants_client, headers).status_code == 401
    assert grants_client.app.state.umrs_db_pool.rows == {}


def test_mvp_admin_flag_without_verified_project_is_not_admin(grants_client, monkeypatch):
    """Deliberate mvp change: the old ``_caller`` read ``X-Verified-Is-Admin``
    even without ``X-Verified-Project-ID``. An admin now needs the verified
    tuple (the gateway always sends it); a bare flag reads the grant as not
    found."""
    grant_id = _bobs_grant(grants_client)
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    headers = {"X-User-ID": "root", "X-Verified-Is-Admin": "true"}
    resp = grants_client.post(f"/umrs/grants/{grant_id}/revoke", headers=headers)
    assert resp.status_code == 404
    assert not _tombstoned(grants_client, grant_id)
