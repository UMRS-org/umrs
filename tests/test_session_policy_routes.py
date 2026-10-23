"""ADR-099 D-3: admin policy routes and the project owner's layer.

Admin routes take the admin credential of the RS-0 session routes (an admin
JWT here); the owner routes take the verified caller (overridden here) and
answer only the project's owner or an admin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from tests.test_rs0_admin_sessions import SECRET, _bearer
from umrs.api import audit_events
from umrs.api.admin_session_policy import router as policy_router
from umrs.api.jobs import CallerIdentity, get_caller_identity
from umrs.api.projects import ProjectNotFound, _store_dep
from umrs.api.projects import router as projects_router
from umrs.session_policy import PolicyRule, PolicyRuleStore

ADMIN = "/umrs/admin/session-policy"
OWNER = "/umrs/projects/opossum/session-policy"
CODE = "x-platform-error-code"


class _Sink:
    def __init__(self):
        self.rows: list[dict] = []

    def write(self, record):
        self.rows.append(record)

    def named(self, event: str) -> list[dict]:
        return [r for r in self.rows if r["event"] == event]


class _Projects:
    rows = {"opossum": {"project_id": "opossum", "owner_id": "olive", "status": "active"}}
    members = {("opossum", "alice")}

    def get(self, project_id):
        if project_id not in self.rows:
            raise ProjectNotFound(project_id)
        return dict(self.rows[project_id])

    def circle_affiliated(self, project_id, user_id):
        return (project_id, user_id) in self.members


def _identity(request: Request) -> CallerIdentity:
    return CallerIdentity(
        user_id=request.headers.get("x-test-user", ""),
        project_id=request.headers.get("x-test-project", ""),
        is_admin=request.headers.get("x-test-admin") == "1",
    )


@pytest.fixture
def h(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    sink = _Sink()
    monkeypatch.setattr(audit_events, "_audit_store", sink)
    app = FastAPI()
    app.include_router(policy_router)
    app.include_router(projects_router)
    store = PolicyRuleStore()
    app.state.session_policy_store = store
    app.state.companion_venue_service = SimpleNamespace(site_name="rccs")
    app.dependency_overrides[_store_dep] = _Projects
    app.dependency_overrides[get_caller_identity] = _identity
    return SimpleNamespace(client=TestClient(app), store=store, sink=sink)


def _as(user: str, *, admin: bool = False) -> dict[str, str]:
    return {"x-test-user": user, "x-test-admin": "1" if admin else "0"}


class TestAdminAccess:
    def test_no_credential_is_401(self, h):
        r = h.client.get(ADMIN)
        assert (r.status_code, r.headers[CODE]) == (401, "companionvenue.unauthorized")

    def test_non_admin_is_403_on_every_route(self, h):
        user = _bearer("alice", ["user"])
        assert h.client.get(ADMIN, headers=user).status_code == 403
        assert h.client.get(f"{ADMIN}/effective?user=a", headers=user).status_code == 403
        put = h.client.put(f"{ADMIN}/site/_", json={"doc": {}}, headers=user)
        assert put.status_code == 403
        assert h.client.delete(f"{ADMIN}/site/_", headers=user).status_code == 403
        assert h.store.list() == []


class TestAdminRules:
    def test_put_list_effective_delete(self, h):
        doc = {"budget": {"tokens_per_day_user": 2000000}}
        r = h.client.put(
            f"{ADMIN}/user/alice", json={"doc": doc, "note": "pilot"}, headers=_bearer()
        )
        assert r.status_code == 200, r.text
        assert r.json() | {"updated_at": None} == {
            "venue": "rccs",
            "scope": "user",
            "subject": "alice",
            "layer": "admin",
            "doc": doc,
            "set_by": "root",
            "note": "pilot",
            "updated_at": None,
        }
        listed = h.client.get(ADMIN, headers=_bearer()).json()
        assert listed["site"] == "rccs" and len(listed["rules"]) == 1
        eff = h.client.get(f"{ADMIN}/effective?user=alice&project=p", headers=_bearer()).json()
        assert eff["budget"]["tokens_per_day_user"] == 2000000
        assert eff["origin"]["budget.tokens_per_day_user"] == "user:admin"
        gone = h.client.delete(f"{ADMIN}/user/alice", headers=_bearer())
        assert gone.json() == {"deleted": True}
        again = h.client.delete(f"{ADMIN}/user/alice", headers=_bearer())
        assert (again.status_code, again.headers[CODE]) == (404, "sessionpolicy.not_found")

    def test_site_subject_and_the_all_sites_venue(self, h):
        body = {"doc": {"sessions": {"concurrent_user": 8}}, "venue": ""}
        r = h.client.put(f"{ADMIN}/site/_", json=body, headers=_bearer())
        assert r.status_code == 200, r.text
        assert (r.json()["venue"], r.json()["subject"]) == ("", "")
        only_all = h.client.get(f"{ADMIN}?venue=", headers=_bearer()).json()["rules"]
        assert len(only_all) == 1
        eff = h.client.get(f"{ADMIN}/effective?user=bob&venue=bdrt", headers=_bearer()).json()
        assert eff["sessions"]["concurrent_user"] == 8

    @pytest.mark.parametrize(
        ("path", "doc"),
        [
            ("/user/alice", {"budget": {"rpm_user": -5}}),
            ("/user/alice", {"budget": {"tokens_per_day_project": 5}}),
            ("/project/_", {}),
            ("/site/alice", {}),
            ("/galaxy/x", {}),
            ("/site/_", {"resources": {"cpus": 4}}),
        ],
    )
    def test_bad_rules_are_invalid(self, h, path, doc):
        r = h.client.put(f"{ADMIN}{path}", json={"doc": doc}, headers=_bearer())
        assert (r.status_code, r.headers[CODE]) == (400, "sessionpolicy.invalid"), r.text
        assert h.store.list() == []

    def test_writes_are_audited_with_actor_key_and_doc(self, h):
        doc = {"web": {"allow": ["none", "allowlist"]}}
        h.client.put(f"{ADMIN}/project/opossum", json={"doc": doc}, headers=_bearer())
        h.client.delete(f"{ADMIN}/project/opossum", headers=_bearer())
        (put,) = h.sink.named("venue.policy.set")
        assert put["actor"] == "root" and put["project_id"] == "opossum"
        assert put["details"]["key"] == {
            "venue": "rccs",
            "scope": "project",
            "subject": "opossum",
            "layer": "admin",
        }
        assert put["details"]["doc"] == doc
        (dropped,) = h.sink.named("venue.policy.delete")
        assert dropped["details"]["doc"] is None


class TestOwnerLayer:
    def test_owner_reads_the_layer_and_the_limits(self, h):
        r = h.client.get(OWNER, headers=_as("olive"))
        assert r.status_code == 200, r.text
        assert r.json()["owner_rule"] is None
        assert r.json()["limits"]["sessions"]["concurrent_user"] == 12

    def test_a_member_who_is_not_the_owner_is_refused(self, h):
        r = h.client.get(OWNER, headers=_as("alice"))
        assert (r.status_code, r.headers[CODE]) == (403, "sessionpolicy.forbidden")
        put = h.client.put(OWNER, json={"doc": {}}, headers=_as("alice"))
        assert (put.status_code, put.headers[CODE]) == (403, "sessionpolicy.forbidden")
        assert h.store.list() == []

    def test_a_stranger_reads_not_found(self, h):
        r = h.client.get(OWNER, headers=_as("mallory"))
        assert (r.status_code, r.headers[CODE]) == (404, "sessionpolicy.not_found")
        ghost = h.client.get("/umrs/projects/ghost/session-policy", headers=_as("olive"))
        assert ghost.status_code == 404

    def test_owner_narrows_and_it_binds(self, h):
        doc = {"web": {"domains": [".arxiv.org"]}, "lifetime": {"idle_s": 600}}
        r = h.client.put(OWNER, json={"doc": doc, "note": "lab rule"}, headers=_as("olive"))
        assert r.status_code == 200, r.text
        assert (r.json()["layer"], r.json()["venue"]) == ("owner", "rccs")
        eff = h.client.get(
            f"{ADMIN}/effective?user=alice&project=opossum", headers=_bearer()
        ).json()
        assert eff["web"]["allow"] == ["none", "allowlist"]
        assert eff["lifetime"]["idle_s"] == 600
        assert h.client.get(OWNER, headers=_as("olive")).json()["owner_rule"] == doc
        assert h.sink.named("venue.policy.set")[0]["actor"] == "olive"

    def test_owner_widening_is_refused(self, h):
        h.store.put(PolicyRule("rccs", "project", "opossum", "admin", {"budget": {"rpm_user": 60}}))
        r = h.client.put(OWNER, json={"doc": {"budget": {"rpm_user": 100}}}, headers=_as("olive"))
        assert (r.status_code, r.headers[CODE]) == (403, "sessionpolicy.widening_refused")
        assert "project:admin" in r.json()["detail"]
        assert h.store.get("rccs", "project", "opossum", "owner") is None

    def test_an_admin_may_write_the_owner_layer(self, h):
        r = h.client.put(
            OWNER, json={"doc": {"budget": {"rpm_user": 5}}}, headers=_as("root", admin=True)
        )
        assert r.status_code == 200 and r.json()["set_by"] == "root"


class TestUnknownSite:
    """P2: with the site config unreadable ``this_site()`` is ``""``, the
    all-sites key (sites share umrs_db). Old code wrote every site's rule
    for a PUT without a venue, an owner PUT, and deleted the all-sites row."""

    @pytest.fixture
    def nameless(self, h):
        h.client.app.state.companion_venue_service = SimpleNamespace(site_name="")
        return h

    def test_an_admin_write_without_a_venue_is_refused(self, nameless):
        doc = {"sessions": {"concurrent_user": 1}}
        r = nameless.client.put(f"{ADMIN}/site/_", json={"doc": doc}, headers=_bearer())
        assert (r.status_code, r.headers[CODE]) == (503, "sessionpolicy.unavailable"), r.text
        assert "--all-sites" in r.json()["detail"] and "--venue" in r.json()["detail"]
        assert nameless.store.list() == []

    def test_an_explicit_venue_or_all_sites_still_works(self, nameless):
        doc = {"sessions": {"concurrent_user": 8}}
        for venue in ("rccs", ""):
            body = {"doc": doc, "venue": venue}
            r = nameless.client.put(f"{ADMIN}/site/_", json=body, headers=_bearer())
            assert r.status_code == 200 and r.json()["venue"] == venue, r.text

    def test_delete_and_effective_without_a_venue_are_refused(self, nameless):
        nameless.store.put(PolicyRule("", "site", "", "admin", {"budget": {"rpm_user": 6}}))
        gone = nameless.client.delete(f"{ADMIN}/site/_", headers=_bearer())
        assert gone.status_code == 503 and len(nameless.store.list()) == 1
        eff = nameless.client.get(f"{ADMIN}/effective?user=alice", headers=_bearer())
        assert eff.status_code == 503

    def test_the_owner_layer_is_refused_always(self, nameless):
        put = nameless.client.put(
            OWNER, json={"doc": {"budget": {"rpm_user": 5}}}, headers=_as("olive")
        )
        assert (put.status_code, put.headers[CODE]) == (503, "sessionpolicy.unavailable")
        assert nameless.store.list() == []
        assert nameless.client.get(OWNER, headers=_as("olive")).status_code == 503
