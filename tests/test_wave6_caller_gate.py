"""Wave-6: tenant-binding gate for previously unsealed routers.

F2-F6 closed a class of tenant-binding gaps on routers that never received
the wave-3/4 HMAC treatment: evidence, scientific-memory, research-sessions,
verification, and the ops audit trace. This suite pins:

- ``CallerIdentity`` scope/write binding semantics (admin vs. non-admin)
- ``resolve_caller`` strict-mode rejection of unsigned/absent identity
- evidence router: write foreign project 403, list forced to own project,
  row-level access 404-hidden for foreign tenants
- research-sessions router: same binding applied to sessions + loop links
- scientific-memory: ``created_by`` forgery impossible under verified caller
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims
from umrs.api._verified_headers import CallerIdentity, resolve_caller
from umrs.api import evidence as evidence_api
from umrs.api import research_sessions as sessions_api
from umrs.api import scientific_memory as sci_api
from umrs.evidence import build_default_store as build_evidence_store
from umrs.research_sessions.store import SessionStore
from umrs.scientific_memory import InMemoryScientificMemoryStore

SECRET = "wave6-secret"


@pytest.fixture(autouse=True)
def _strict_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    yield


def _signed(user: str = "alice", project: str = "proj-a", admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


# ── CallerIdentity class ─────────────────────────────────────────────────


class TestCallerIdentity:
    caller = CallerIdentity("alice", "proj-a", is_admin=False)
    admin = CallerIdentity("root", "system", is_admin=True)

    def test_scope_filter_none_becomes_own_project(self):
        assert self.caller.scope_filter(None) == "proj-a"

    def test_scope_filter_own_ok(self):
        assert self.caller.scope_filter("proj-a") == "proj-a"

    def test_scope_filter_foreign_403(self):
        with pytest.raises(HTTPException) as exc:
            self.caller.scope_filter("proj-b")
        assert exc.value.status_code == 403

    def test_scope_filter_admin_keeps_none(self):
        assert self.admin.scope_filter(None) is None
        assert self.admin.scope_filter("proj-b") == "proj-b"

    def test_bind_for_write_defaults_to_own(self):
        assert self.caller.bind_for_write(None) == "proj-a"
        assert self.caller.bind_for_write("proj-a") == "proj-a"

    def test_bind_for_write_foreign_403(self):
        with pytest.raises(HTTPException) as exc:
            self.caller.bind_for_write("proj-b")
        assert exc.value.status_code == 403

    def test_bind_for_write_admin(self):
        assert self.admin.bind_for_write("proj-b") == "proj-b"

    def test_assert_can_access_foreign_hides_404(self):
        with pytest.raises(HTTPException) as exc:
            self.caller.assert_can_access("proj-b")
        assert exc.value.status_code == 404

    def test_assert_can_access_admin_ok(self):
        self.admin.assert_can_access("proj-b")

    def test_assert_can_access_none_ok(self):
        self.caller.assert_can_access(None)


class TestResolveCaller:
    def test_strict_unsigned_401(self):
        with pytest.raises(HTTPException) as exc:
            resolve_caller("alice", "proj-a", None, None)
        assert exc.value.status_code == 401

    def test_strict_forged_hmac_401(self):
        forged = sign_claims("wrong-secret", "alice", "proj-a", False)
        with pytest.raises(HTTPException) as exc:
            resolve_caller("alice", "proj-a", "false", forged)
        assert exc.value.status_code == 401

    def test_strict_absent_identity_401(self):
        with pytest.raises(HTTPException) as exc:
            resolve_caller(None, None, None, None)
        assert exc.value.status_code == 401

    def test_strict_signed_returns_identity(self):
        good = sign_claims(SECRET, "alice", "proj-a", False)
        caller = resolve_caller("alice", "proj-a", "false", good)
        assert caller is not None
        assert caller.user_id == "alice"
        assert caller.project_id == "proj-a"
        assert caller.is_admin is False

    def test_strict_signed_admin_tuple_required(self):
        # HMAC signed with admin=False but header claims admin=True → reject.
        good_nonadmin = sign_claims(SECRET, "root", "system", False)
        with pytest.raises(HTTPException) as exc:
            resolve_caller("root", "system", "true", good_nonadmin)
        assert exc.value.status_code == 401

    def test_mvp_anonymous_is_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        assert resolve_caller(None, None, None, None) is None


# ── evidence router (F2) ─────────────────────────────────────────────────


def _evidence_client():
    store = build_evidence_store()
    evidence_api.set_store(store)
    app = FastAPI()
    app.include_router(evidence_api.router)
    return TestClient(app), store


class TestEvidenceTenantBinding:
    def test_create_foreign_project_403(self):
        c, _ = _evidence_client()
        r = c.post(
            "/umrs/evidence",
            headers=_signed("alice", "proj-a"),
            json={"project_id": "proj-b", "claim_id": "c1"},
        )
        assert r.status_code == 403

    def test_create_own_project_ok(self):
        c, _ = _evidence_client()
        r = c.post(
            "/umrs/evidence",
            headers=_signed("alice", "proj-a"),
            json={"project_id": "proj-a", "claim_id": "c1"},
        )
        assert r.status_code == 200

    def test_list_forces_own_project(self):
        c, store = _evidence_client()
        # Seed rows for both projects directly in the store.
        from umrs.evidence import Evidence

        store.add_evidence(Evidence(project_id="proj-a", claim_id="c"))
        store.add_evidence(Evidence(project_id="proj-b", claim_id="c"))
        r = c.get("/umrs/evidence", headers=_signed("alice", "proj-a"))
        assert r.status_code == 200
        rows = r.json()["evidence"]
        assert rows and all(e["project_id"] == "proj-a" for e in rows)

    def test_list_foreign_filter_403(self):
        c, _ = _evidence_client()
        r = c.get(
            "/umrs/evidence?project_id=proj-b",
            headers=_signed("alice", "proj-a"),
        )
        assert r.status_code == 403

    def test_admin_can_list_all(self):
        c, store = _evidence_client()
        from umrs.evidence import Evidence

        store.add_evidence(Evidence(project_id="proj-a", claim_id="c"))
        store.add_evidence(Evidence(project_id="proj-b", claim_id="c"))
        r = c.get("/umrs/evidence", headers=_signed("root", "system", admin=True))
        assert r.status_code == 200
        assert len(r.json()["evidence"]) == 2

    def test_get_foreign_hidden_404(self):
        c, store = _evidence_client()
        from umrs.evidence import Evidence

        eid = store.add_evidence(Evidence(project_id="proj-b", claim_id="c"))
        r = c.get(f"/umrs/evidence/{eid}", headers=_signed("alice", "proj-a"))
        assert r.status_code == 404

    def test_update_status_foreign_hidden_404(self):
        c, store = _evidence_client()
        from umrs.evidence import Evidence

        eid = store.add_evidence(Evidence(project_id="proj-b", claim_id="c"))
        r = c.put(
            f"/umrs/evidence/{eid}/status",
            headers=_signed("alice", "proj-a"),
            json={"status": "retracted"},
        )
        assert r.status_code == 404

    def test_anonymous_strict_401(self):
        c, _ = _evidence_client()
        r = c.get("/umrs/evidence")
        assert r.status_code == 401

    def test_unsigned_strict_401(self):
        c, _ = _evidence_client()
        r = c.get(
            "/umrs/evidence",
            headers={"X-User-ID": "mallory", "X-Verified-Project-ID": "proj-a"},
        )
        assert r.status_code == 401


# ── research sessions router (F5) ────────────────────────────────────────


class _InMemorySessionStore(SessionStore):
    def __init__(self):
        from umrs.research_sessions import ResearchSession, SessionLoopLink

        self._sessions: dict[str, ResearchSession] = {}
        self._links: list[SessionLoopLink] = []

    def create(self, session):
        self._sessions[session.session_id] = session
        return session.session_id

    def get(self, session_id):
        return self._sessions.get(session_id)

    def update(self, session_id, title=None, status=None, metadata=None):
        s = self._sessions.get(session_id)
        if s is None:
            return None
        if title is not None:
            s.title = title
        if status is not None:
            s.status = status
        return s

    def list(self, project_id=None, status=None):
        out = list(self._sessions.values())
        if project_id:
            out = [s for s in out if s.project_id == project_id]
        if status:
            out = [s for s in out if s.status == status]
        return out

    def attach_loop(self, link):
        self._links.append(link)

    def detach_loop(self, session_id, loop_id):
        self._links = [
            link
            for link in self._links
            if not (link.session_id == session_id and link.loop_id == loop_id)
        ]

    def list_loops(self, session_id):
        return [link for link in self._links if link.session_id == session_id]


def _sessions_client():
    store = _InMemorySessionStore()
    sessions_api.set_session_store(store)
    app = FastAPI()
    app.include_router(sessions_api.router)
    return TestClient(app), store


class TestResearchSessionsTenantBinding:
    def test_create_foreign_403(self):
        c, _ = _sessions_client()
        r = c.post(
            "/umrs/research/sessions",
            headers=_signed("alice", "proj-a"),
            json={"project_id": "proj-b", "research_goal": "steal"},
        )
        assert r.status_code == 403

    def test_list_only_own_project(self):
        c, store = _sessions_client()
        from umrs.research_sessions import ResearchSession

        store.create(ResearchSession(session_id="s-a", project_id="proj-a", title="A"))
        store.create(ResearchSession(session_id="s-b", project_id="proj-b", title="B"))
        r = c.get("/umrs/research/sessions", headers=_signed("alice", "proj-a"))
        assert r.status_code == 200
        titles = {s["title"] for s in r.json()["sessions"]}
        assert titles == {"A"}

    def test_get_foreign_hidden_404(self):
        c, store = _sessions_client()
        from umrs.research_sessions import ResearchSession

        store.create(ResearchSession(session_id="s-b", project_id="proj-b", title="B"))
        r = c.get("/umrs/research/sessions/s-b", headers=_signed("alice", "proj-a"))
        assert r.status_code == 404

    def test_patch_foreign_hidden_404(self):
        c, store = _sessions_client()
        from umrs.research_sessions import ResearchSession

        store.create(ResearchSession(session_id="s-b", project_id="proj-b", title="B"))
        r = c.patch(
            "/umrs/research/sessions/s-b",
            headers=_signed("alice", "proj-a"),
            json={"title": "tampered"},
        )
        assert r.status_code == 404

    def test_detach_loop_foreign_hidden_404(self):
        c, store = _sessions_client()
        from umrs.research_sessions import ResearchSession, SessionLoopLink

        store.create(ResearchSession(session_id="s-b", project_id="proj-b", title="B"))
        store.attach_loop(SessionLoopLink(session_id="s-b", loop_id="loop-1"))
        r = c.delete(
            "/umrs/research/sessions/s-b/loops/loop-1",
            headers=_signed("alice", "proj-a"),
        )
        assert r.status_code == 404


# ── scientific memory router (F4) ────────────────────────────────────────


def _sci_client():
    store = InMemoryScientificMemoryStore()
    sci_api.set_store(store)
    app = FastAPI()
    app.include_router(sci_api.router)
    return TestClient(app)


class TestScientificMemoryTenantBinding:
    def test_created_by_forced_to_caller(self):
        c = _sci_client()
        r = c.post(
            "/umrs/scientific-memory/entries",
            headers=_signed("alice", "proj-a"),
            json={
                "project_id": "proj-a",
                "content": "x",
                "created_by": "mallory",  # forgery attempt
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["created_by"] == "alice"

    def test_create_foreign_project_403(self):
        c = _sci_client()
        r = c.post(
            "/umrs/scientific-memory/entries",
            headers=_signed("alice", "proj-a"),
            json={"project_id": "proj-b", "content": "x"},
        )
        assert r.status_code == 403

    def test_list_forces_own_project(self):
        c = _sci_client()
        c.post(
            "/umrs/scientific-memory/entries",
            headers=_signed("alice", "proj-a"),
            json={"project_id": "proj-a", "content": "mine"},
        )
        c.post(
            "/umrs/scientific-memory/entries",
            headers=_signed("bob", "proj-b"),
            json={"project_id": "proj-b", "content": "theirs"},
        )
        r = c.get(
            "/umrs/scientific-memory/entries",
            headers=_signed("alice", "proj-a"),
        )
        assert r.status_code == 200
        bodies = [e["content"] for e in r.json()["entries"]]
        assert bodies == ["mine"]

    def test_timeline_foreign_403(self):
        c = _sci_client()
        r = c.get(
            "/umrs/scientific-memory/timeline?project_id=proj-b",
            headers=_signed("alice", "proj-a"),
        )
        assert r.status_code == 403
