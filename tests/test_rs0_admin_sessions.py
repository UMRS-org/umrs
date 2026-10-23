"""RS-0 C-1: admin session list / show / stop over the frontend venue service.

Same fakes as the owner-route tests (in-memory broker + registry, FakePodman
provisioner, snapshot pool, fake job store); the admin credential is an
HS256 JWT signed with a test secret, or a gateway-signed admin identity.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt as jose_jwt
from tests.fake_pg_pool import FakePool
from tests.test_companion_venue_routes import (
    READY_ENV,
    FakeJobStore,
    FakePodman,
    FakeSealed,
    _meta,
    _Pool,
    _run_lane,
)
from umrs.api import audit_events
from umrs.api.admin_sessions import router as admin_router
from umrs.api.companion_venue import router as venue_router
from umrs.jobs.secret_broker import SecretBroker
from umrs.shared.handshake import ENV_VAR as HMAC_ENV
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.frontend_podman import FrontendPodmanProvisioner
from umrs.venues.frontend_service import FrontendVenueService
from umrs.venues.venue_sessions import VenueSessionStore

from r4_contracts.handshake_hmac import sign_claims

SECRET = "test-jwt-secret-not-real"
ADMIN_URL = "/umrs/admin/sessions"


def _jwt(sub: str, roles: list[str], project: str = "ops") -> str:
    claims = {"sub": sub, "roles": roles, "project_id": project, "exp": int(time.time()) + 600}
    return jose_jwt.encode(claims, SECRET, algorithm="HS256")


def _bearer(sub: str = "root", roles: list[str] | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {_jwt(sub, roles or ['admin'])}"}


class _AuditSink:
    def __init__(self):
        self.rows: list[dict] = []

    def write(self, record):
        self.rows.append(record)

    def events(self, prefix: str = "venue.") -> list[str]:
        return [r["event"] for r in self.rows if r["event"].startswith(prefix)]


@pytest.fixture
def audit(monkeypatch):
    sink = _AuditSink()
    monkeypatch.setattr(audit_events, "_audit_store", sink)
    return sink


@pytest.fixture
def admin_h(monkeypatch, audit):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    pool = _Pool()
    broker = SecretBroker()
    podman = FakePodman()
    provisioner = FrontendPodmanProvisioner(
        broker, podman=podman, session_root=Path(tempfile.mkdtemp(prefix="admin-venue-"))
    )
    jobs = FakeJobStore()
    sealed = FakeSealed(
        {
            ("alice", "openai-main"): _meta("openai-main"),
            ("bob", "openai-main"): _meta("openai-main"),
        }
    )
    service = FrontendVenueService(
        config=FrontendVenueConfig.from_env(READY_ENV),
        broker=broker,
        store=VenueSessionStore(),
        provisioner=provisioner,
        pool=pool,
        sealed_store=sealed,
        job_store=jobs,
        site_name="rccs",
    )
    app = FastAPI()
    app.include_router(venue_router)
    app.include_router(admin_router)
    app.state.umrs_db_pool = pool
    app.state.companion_venue_service = service
    h = SimpleNamespace(
        client=TestClient(app), pool=pool, broker=broker, service=service, jobs=jobs, audit=audit
    )
    for owner, sid in (("alice", "sess-a"), ("bob", "sess-b")):
        pool.seed(owner, sid)
        r = h.client.post(
            f"/umrs/companion/sessions/{sid}/venue/start",
            json={"source_refs": ["openai-main"]},
            headers={"x-user-id": owner},
        )
        assert r.status_code == 200, r.text
    _run_lane(h)
    for job in jobs.jobs.values():
        job.project_id = "proj-" + job.spec["owner"]
    service.store.update_runtime("sess-a", {"phase": "running", "partial": "secret draft"})
    return h


class TestAdminList:
    def test_admin_lists_every_users_sessions(self, admin_h):
        r = admin_h.client.get(ADMIN_URL, headers=_bearer())
        assert r.status_code == 200, r.text
        body = r.json()
        owners = sorted((s["session_id"], s["user_id"], s["state"]) for s in body["sessions"])
        assert owners == [("sess-a", "alice", "running"), ("sess-b", "bob", "running")]
        assert body["total"] == 2 and body["hosts"] == ["local"]
        assert all(s["host"] == "local" for s in body["sessions"])

    def test_filters_user_and_state(self, admin_h):
        admin_h.service.stop(owner="bob", session_id="sess-b")
        by_user = admin_h.client.get(ADMIN_URL, params={"user": "bob"}, headers=_bearer())
        assert [s["session_id"] for s in by_user.json()["sessions"]] == ["sess-b"]
        running = admin_h.client.get(ADMIN_URL, params={"state": "running"}, headers=_bearer())
        assert [s["session_id"] for s in running.json()["sessions"]] == ["sess-a"]

    def test_unknown_host_lists_nothing_and_bad_state_is_400(self, admin_h):
        other = admin_h.client.get(ADMIN_URL, params={"host": "cpu-01"}, headers=_bearer())
        assert other.status_code == 200 and other.json()["sessions"] == []
        bad = admin_h.client.get(ADMIN_URL, params={"state": "zombie"}, headers=_bearer())
        assert bad.status_code == 400
        assert bad.headers["x-platform-error-code"] == "companionvenue.bad_request"

    def test_list_hides_the_users_live_text(self, admin_h):
        rows = admin_h.client.get(ADMIN_URL, headers=_bearer()).json()["sessions"]
        alice = next(s for s in rows if s["user_id"] == "alice")
        assert alice["runtime"]["phase"] == "running"
        assert "partial" not in alice["runtime"]

    def test_list_is_audited(self, admin_h):
        admin_h.client.get(ADMIN_URL, params={"user": "bob"}, headers=_bearer("root"))
        row = [r for r in admin_h.audit.rows if r["event"] == "venue.admin.list"][-1]
        assert row["actor"] == "root" and row["scope"] == "system"
        assert row["details"]["filters"]["user"] == "bob" and row["details"]["count"] == 1


class TestAdminAuth:
    def test_no_credential_is_401(self, admin_h):
        r = admin_h.client.get(ADMIN_URL)
        assert r.status_code == 401
        assert r.headers["x-platform-error-code"] == "companionvenue.unauthorized"

    def test_non_admin_jwt_is_403_and_audited(self, admin_h):
        r = admin_h.client.get(ADMIN_URL, headers=_bearer("mallory", ["user"]))
        assert r.status_code == 403
        assert r.headers["x-platform-error-code"] == "companionvenue.forbidden"
        denied = [r for r in admin_h.audit.rows if r["event"] == "venue.admin.denied"]
        assert denied and denied[-1]["actor"] == "mallory" and denied[-1]["outcome"] == "denied"

    def test_non_admin_cannot_stop_another_users_session(self, admin_h):
        r = admin_h.client.post(f"{ADMIN_URL}/sess-a/stop", headers=_bearer("mallory", ["user"]))
        assert r.status_code == 403
        assert admin_h.service.store.get("sess-a").state == "running"

    def test_forged_jwt_is_401(self, admin_h):
        forged = jose_jwt.encode(
            {"sub": "x", "roles": ["admin"], "project_id": "p", "exp": int(time.time()) + 60},
            "some-other-secret",
            algorithm="HS256",
        )
        r = admin_h.client.get(ADMIN_URL, headers={"Authorization": f"Bearer {forged}"})
        assert r.status_code == 401

    def test_owner_identity_header_alone_is_not_admin(self, admin_h):
        r = admin_h.client.get(ADMIN_URL, headers={"x-user-id": "alice"})
        assert r.status_code == 401

    def test_gateway_signed_admin_passes_and_unsigned_flag_fails(self, admin_h, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv(HMAC_ENV, "hmac-test-secret")
        signed = {
            "x-user-id": "ops-admin",
            "x-verified-project-id": "ops",
            "x-verified-is-admin": "true",
            "x-verified-hmac": sign_claims("hmac-test-secret", "ops-admin", "ops", True),
        }
        assert admin_h.client.get(ADMIN_URL, headers=signed).status_code == 200
        forged = {
            **signed,
            "x-verified-hmac": sign_claims("hmac-test-secret", "ops-admin", "ops", False),
        }
        assert admin_h.client.get(ADMIN_URL, headers=forged).status_code == 401
        user = {
            "x-user-id": "alice",
            "x-verified-project-id": "ops",
            "x-verified-is-admin": "false",
            "x-verified-hmac": sign_claims("hmac-test-secret", "alice", "ops", False),
        }
        assert admin_h.client.get(ADMIN_URL, headers=user).status_code == 403


class TestDisabledAdmin:
    """ADR-097 Amendment D-8 (2026-10-23): an admin JWT outlived its owner's
    disable for its whole lifetime; a disabled account's credential is now
    a typed 403 on the admin routes, whatever its roles say."""

    @pytest.fixture
    def users(self, tmp_path, monkeypatch):
        from umrs.admin.user_store import UserStore

        monkeypatch.setenv("UMRS_USER_FILES_DIR", str(tmp_path))
        store = UserStore(tmp_path)
        store.create("root", role="admin")
        store.create("ops-admin", role="admin")
        return store

    def test_a_disabled_admins_jwt_is_403_and_audited(self, admin_h, users):
        assert admin_h.client.get(ADMIN_URL, headers=_bearer("root")).status_code == 200
        users.set_disabled("root", True)
        r = admin_h.client.get(ADMIN_URL, headers=_bearer("root"))
        assert r.status_code == 403
        assert r.headers["x-platform-error-code"] == "companionvenue.forbidden"
        assert "disabled" in r.json()["detail"]
        denied = [row for row in admin_h.audit.rows if row["event"] == "venue.admin.denied"]
        assert denied[-1]["actor"] == "root"
        assert denied[-1]["details"]["reason"] == "account_disabled"

    def test_a_disabled_admin_cannot_stop_a_session(self, admin_h, users):
        users.set_disabled("root", True)
        r = admin_h.client.post(f"{ADMIN_URL}/sess-a/stop", headers=_bearer("root"))
        assert r.status_code == 403
        assert admin_h.service.store.get("sess-a").state == "running"

    def test_re_enabling_restores_the_admin(self, admin_h, users):
        users.set_disabled("root", True)
        users.set_disabled("root", False)
        assert admin_h.client.get(ADMIN_URL, headers=_bearer("root")).status_code == 200

    def test_a_disabled_gateway_signed_admin_is_403(self, admin_h, users, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv(HMAC_ENV, "hmac-test-secret")
        signed = {
            "x-user-id": "ops-admin",
            "x-verified-project-id": "ops",
            "x-verified-is-admin": "true",
            "x-verified-hmac": sign_claims("hmac-test-secret", "ops-admin", "ops", True),
        }
        users.set_disabled("ops-admin", True)
        assert admin_h.client.get(ADMIN_URL, headers=signed).status_code == 403

    @pytest.mark.usefixtures("users")
    def test_an_unreadable_record_keeps_the_admin(self, admin_h, monkeypatch):
        from umrs.admin import account_cascade

        def _boom(user: str) -> None:
            raise RuntimeError("users table unreachable")

        monkeypatch.setattr(account_cascade, "load_record", _boom)
        assert admin_h.client.get(ADMIN_URL, headers=_bearer("root")).status_code == 200


class TestDisabledAdminReviewFixes:
    """ADR-097 Amendment D-8 review fixes (2026-10-23): an admin whose JWT
    ``sub`` is an IdP subject, and a JSON-lane disable while the ``users``
    row cannot be read, are refused too."""

    @pytest.fixture
    def users(self, tmp_path, monkeypatch):
        from umrs.admin.user_store import UserStore

        monkeypatch.setenv("UMRS_USER_FILES_DIR", str(tmp_path))
        store = UserStore(tmp_path)
        store.create("root", role="admin")
        store.create("ops-lead", role="admin", oidc_subject="241723")
        return store

    def test_a_disabled_idp_subject_admin_is_403(self, admin_h, users):
        assert admin_h.client.get(ADMIN_URL, headers=_bearer("241723")).status_code == 200
        users.set_disabled("ops-lead", True)
        r = admin_h.client.get(ADMIN_URL, headers=_bearer("241723"))
        assert r.status_code == 403
        assert "disabled" in r.json()["detail"]

    def test_a_json_disable_holds_while_postgres_is_down(self, admin_h, users, monkeypatch):
        from unittest.mock import MagicMock

        from umrs.admin import user_store

        users.set_disabled("root", True)
        pool = MagicMock()
        pool.getconn.side_effect = RuntimeError("users table unreachable")
        monkeypatch.setattr(user_store, "_PG_POOL", pool)
        assert admin_h.client.get(ADMIN_URL, headers=_bearer("root")).status_code == 403

    @pytest.mark.usefixtures("users")
    def test_postgres_down_without_a_json_disable_keeps_the_admin(self, admin_h, monkeypatch):
        from unittest.mock import MagicMock

        from umrs.admin import user_store

        pool = MagicMock()
        pool.getconn.side_effect = RuntimeError("users table unreachable")
        monkeypatch.setattr(user_store, "_PG_POOL", pool)
        assert admin_h.client.get(ADMIN_URL, headers=_bearer("root")).status_code == 200


class TestAdminShowAndStop:
    def test_show_any_owner_with_project_and_lease(self, admin_h):
        r = admin_h.client.get(f"{ADMIN_URL}/sess-b", headers=_bearer())
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["user_id"], body["project"], body["state"]) == ("bob", "proj-bob", "running")
        assert body["lease"]["is_valid"] is True and body["job"]["job_id"]
        row = [r for r in admin_h.audit.rows if r["event"] == "venue.admin.show"][-1]
        assert row["details"]["owner"] == "bob" and row["project_id"] == "proj-bob"

    def test_show_unknown_is_404(self, admin_h):
        r = admin_h.client.get(f"{ADMIN_URL}/nope", headers=_bearer())
        assert r.status_code == 404
        assert r.headers["x-platform-error-code"] == "companionvenue.not_found"

    @staticmethod
    def _admin_stop(admin_h) -> dict:
        r = admin_h.client.post(
            f"{ADMIN_URL}/sess-a/stop", json={"note": "policy breach"}, headers=_bearer("root")
        )
        assert r.status_code == 200, r.text
        return r.json()

    def test_admin_stops_another_users_session(self, admin_h):
        body = self._admin_stop(admin_h)
        assert (body["state"], body["user_id"], body["project"]) == (
            "stopped",
            "alice",
            "proj-alice",
        )
        assert "admin root: policy breach" in body["reason"]
        # bob's session is untouched
        assert admin_h.service.store.get("sess-b").state == "running"

    def test_admin_stop_revokes_the_lease_and_settles_the_job(self, admin_h):
        self._admin_stop(admin_h)
        leases = admin_h.broker.list_leases(scope="inference")
        assert [lease.is_valid for lease in leases if lease.session_id == "sess-a"] == [False]
        reasons = [reason for _job, reason in admin_h.jobs.cancelled]
        assert reasons == ["stopped by admin root: policy breach"]

    def test_admin_stop_writes_one_stop_row_with_cause_admin(self, admin_h):
        self._admin_stop(admin_h)
        (stop,) = [r for r in admin_h.audit.rows if r["event"] == "venue.session.stop"]
        got = (stop["actor"], stop["details"]["cause"], stop["details"]["owner"])
        assert got == ("root", "admin", "alice")

    def test_admin_stop_is_idempotent_and_unknown_is_404(self, admin_h):
        admin_h.client.post(f"{ADMIN_URL}/sess-a/stop", headers=_bearer())
        again = admin_h.client.post(f"{ADMIN_URL}/sess-a/stop", headers=_bearer())
        assert (again.status_code, again.json()["state"]) == (200, "stopped")
        assert admin_h.audit.events("venue.session.stop") == ["venue.session.stop"]
        assert admin_h.client.post(f"{ADMIN_URL}/ghost/stop", headers=_bearer()).status_code == 404


class TestAdminListStore:
    def test_memory_rows_newest_first_with_filters(self):
        store = VenueSessionStore()
        for sid, owner in (("s1", "alice"), ("s2", "bob")):
            store.upsert_provisioning(
                session_id=sid,
                user_id=owner,
                subnet="172.30.0.0/24",
                source_names=(),
                source_ids=(),
                default_source_id="",
                default_source_name="",
                web_policy="public",
            )
        assert [r.session_id for r, _ in store.list_for_admin()] == ["s2", "s1"]
        assert [r.session_id for r, _ in store.list_for_admin(user_id="alice")] == ["s1"]
        assert store.list_for_admin(project="p") == []

    def test_pg_query_joins_the_session_job_for_the_project(self):
        from tests.test_venue_sessions_store import _pg_row

        pool = FakePool(script=[[(*_pg_row(), "proj-1")]])
        rows = VenueSessionStore(pool).list_for_admin(user_id="alice", project="proj-1", limit=5)
        assert [(r.session_id, p) for r, p in rows] == [("s1", "proj-1")]
        sql, params = pool.record[-1]
        assert "LEFT JOIN jobs j ON j.job_id = s.job_id" in sql
        assert "ORDER BY s.started_at DESC" in sql
        assert params == ("alice", "alice", "", "", "proj-1", "proj-1", 5)
