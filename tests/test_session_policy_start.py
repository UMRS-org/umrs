"""ADR-099 D-1, D-2: the resolved session policy at venue start.

Same fakes as the owner-route tests (``tests/test_companion_venue_routes``):
a start without a web choice is refused while the choice is required, the
legacy default applies with the flag off, the resolved policy lands on the
row and in the job spec, the quota hook sees a ``StartCheck`` before
anything is reserved, admin views carry the policy, and the preview route
answers what a start may choose.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from tests.test_companion_venue_routes import (
    READY_ENV,
    FakeJobStore,
    FakePodman,
    FakeSealed,
    _meta,
    _Pool,
)
from tests.test_rs0_admin_sessions import SECRET, _bearer
from umrs.api import audit_events
from umrs.api.admin_sessions import router as admin_router
from umrs.api.companion_venue import router as venue_router
from umrs.jobs.secret_broker import SecretBroker
from umrs.session_policy import PolicyError, PolicyRule, PolicyRuleStore
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.frontend_podman import FrontendPodmanProvisioner
from umrs.venues.frontend_service import FrontendVenueService
from umrs.venues.start_checks import StartCheck
from umrs.venues.venue_sessions import VenueSessionStore

H = {"x-user-id": "alice"}
BASE = "/umrs/companion/sessions/sess-1/venue"
CODE = "x-platform-error-code"


class _Sink:
    def __init__(self):
        self.rows: list[dict] = []

    def write(self, record):
        self.rows.append(record)

    def named(self, event: str) -> list[dict]:
        return [r for r in self.rows if r["event"] == event]


class _BrokenStore(PolicyRuleStore):
    def applicable(self, **_kw):
        raise ConnectionError("db down")


def _harness(monkeypatch, *, env=None, store=None, quota_check=None):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    sink = _Sink()
    monkeypatch.setattr(audit_events, "_audit_store", sink)
    pool = _Pool()
    broker = SecretBroker()
    service = FrontendVenueService(
        config=FrontendVenueConfig.from_env(READY_ENV),
        broker=broker,
        store=VenueSessionStore(),
        provisioner=FrontendPodmanProvisioner(
            broker,
            podman=FakePodman(),
            session_root=Path(tempfile.mkdtemp(prefix="policy-venue-")),
        ),
        pool=pool,
        sealed_store=FakeSealed(
            {
                ("alice", "openai-main"): _meta("openai-main"),
                ("bob", "openai-main"): _meta("openai-main"),
            }
        ),
        job_store=FakeJobStore(),
        site_name="rccs",
        quota_check=quota_check,
        policy_store=store if store is not None else PolicyRuleStore(),
        policy_env=env if env is not None else {},
    )
    app = FastAPI()
    app.include_router(venue_router)
    app.include_router(admin_router)
    app.state.umrs_db_pool = pool
    app.state.companion_venue_service = service
    pool.seed("alice", "sess-1")
    return SimpleNamespace(
        client=TestClient(app), service=service, jobs=service.job_store, sink=sink
    )


@pytest.fixture
def h(monkeypatch):
    return _harness(monkeypatch)


def _start(h, headers=H, **body):
    payload = {"source_refs": ["openai-main"], **body}
    return h.client.post(f"{BASE}/start", json=payload, headers=headers)


class TestTheChoiceAtStart:
    def test_start_without_a_choice_is_refused_and_reserves_nothing(self, h):
        r = _start(h)
        assert r.status_code == 400, r.text
        assert r.headers[CODE] == "sessionpolicy.web_choice_required"
        assert "none | allowlist | public" in r.json()["detail"]
        assert h.jobs.created == []
        assert h.service.store.get("sess-1") is None

    def test_flag_off_restores_the_legacy_default(self, monkeypatch):
        h = _harness(monkeypatch, env={"UMRS_SESSION_WEB_CHOICE_REQUIRED": "false"})
        r = _start(h)
        assert r.status_code == 200, r.text
        assert r.json()["web_policy"] == "public"
        assert r.json()["policy"]["origin"]["web.mode"] == "default"

    def test_the_policy_is_stored_on_the_row_and_the_spec(self, h):
        r = _start(h, web_policy="allowlist", web_allow_domains=[".arxiv.org"])
        assert r.status_code == 200, r.text
        row = h.service.store.get("sess-1")
        assert row.web_policy == "allowlist"
        assert row.policy["web"] == {"mode": "allowlist", "domains": [".arxiv.org"]}
        assert row.policy["sessions"]["concurrent_user"] == 12
        assert row.policy["origin"]["web.mode"] == "choice"
        spec = h.jobs.created[0]["spec"]
        assert spec["policy"] == row.policy
        assert (spec["web_policy"], spec["web_allow_domains"]) == ("allowlist", [".arxiv.org"])
        status = h.client.get(BASE, headers=H).json()
        assert status["policy"] == row.policy

    def test_a_choice_wider_than_a_cap_is_refused(self, monkeypatch):
        store = PolicyRuleStore()
        store.put(PolicyRule("rccs", "user", "alice", "admin", {"web": {"allow": ["none"]}}))
        h = _harness(monkeypatch, store=store)
        r = _start(h, web_policy="public")
        assert r.status_code == 403, r.text
        assert r.headers[CODE] == "sessionpolicy.widening_refused"
        assert "user:admin" in r.json()["detail"]
        assert h.jobs.created == []

    def test_allowlist_without_domains_is_invalid(self, h):
        r = _start(h, web_policy="allowlist")
        assert (r.status_code, r.headers[CODE]) == (400, "sessionpolicy.invalid")

    def test_unreadable_rules_fail_closed(self, monkeypatch):
        h = _harness(monkeypatch, store=_BrokenStore())
        r = _start(h, web_policy="none")
        assert r.status_code == 503, r.text
        assert r.headers[CODE] == "sessionpolicy.unavailable"
        assert h.jobs.created == []

    def test_the_start_audit_names_the_choice(self, h):
        _start(h, web_policy="none")
        (row,) = h.sink.named("venue.session.start")
        assert row["details"]["policy_web"] == {"mode": "none", "domains": []}
        assert row["details"]["policy_origin"] == {"web.mode": "choice", "web.domains": "choice"}

    def test_a_restart_by_the_session_job_keeps_the_policy(self, h):
        _start(h, web_policy="none")
        row = h.service.store.get("sess-1")
        h.service.store.upsert_provisioning(
            session_id="sess-1",
            user_id="alice",
            subnet=row.subnet,
            source_names=row.source_names,
            source_ids=row.source_ids,
            default_source_id=row.default_source_id,
            default_source_name=row.default_source_name,
            web_policy="none",
            job_id=row.job_id,
        )
        assert h.service.store.get("sess-1").policy == row.policy


class TestTheStartCheck:
    def test_the_hook_gets_a_start_check_with_the_policy(self, monkeypatch):
        seen: list[StartCheck] = []
        h = _harness(monkeypatch, quota_check=seen.append)
        assert _start(h, web_policy="none", project="").status_code == 200
        (check,) = seen
        assert (check.owner, check.venue, check.session_id) == ("alice", "rccs", "sess-1")
        assert check.policy["web"]["mode"] == "none"

    def test_a_refusing_hook_reserves_nothing(self, monkeypatch):
        def refuse(_check):
            raise HTTPException(status_code=429, detail="cap", headers={CODE: "quota.x"})

        h = _harness(monkeypatch, quota_check=refuse)
        r = _start(h, web_policy="none")
        assert r.status_code == 429
        assert h.jobs.created == []
        assert h.service.store.get("sess-1") is None


class TestPreview:
    def test_preview_before_and_after_a_start(self, monkeypatch):
        store = PolicyRuleStore()
        store.put(
            PolicyRule(
                "rccs",
                "site",
                "",
                "admin",
                {"web": {"domains": [".arxiv.org"], "suggested_domains": [".arxiv.org"]}},
            )
        )
        h = _harness(monkeypatch, store=store)
        r = h.client.get(f"{BASE}/policy", headers=H)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choice_required"] is True and body["current"] is None
        assert body["limits"]["web"]["allow"] == ["none", "allowlist"]
        assert body["limits"]["web"]["suggested_domains"] == [".arxiv.org"]
        _start(h, web_policy="allowlist", web_allow_domains=["export.arxiv.org"])
        current = h.client.get(f"{BASE}/policy", headers=H).json()["current"]
        assert current["web"] == {"mode": "allowlist", "domains": ["export.arxiv.org"]}

    def test_another_users_preview_never_shows_this_sessions_policy(self, h):
        _start(h, web_policy="none")
        r = h.client.get(f"{BASE}/policy", headers={"x-user-id": "bob"})
        assert r.status_code == 200 and r.json()["current"] is None

    def test_preview_needs_an_identity(self, h):
        assert h.client.get(f"{BASE}/policy").status_code == 401

    def test_preview_fails_closed(self, monkeypatch):
        h = _harness(monkeypatch, store=_BrokenStore())
        r = h.client.get(f"{BASE}/policy", headers=H)
        assert (r.status_code, r.headers[CODE]) == (503, "sessionpolicy.unavailable")


class TestAdminViews:
    def test_admin_show_and_list_carry_the_policy(self, h):
        _start(h, web_policy="allowlist", web_allow_domains=["pypi.org"])
        shown = h.client.get("/umrs/admin/sessions/sess-1", headers=_bearer()).json()
        assert shown["policy"]["web"] == {"mode": "allowlist", "domains": ["pypi.org"]}
        assert shown["policy"]["origin"]["lifetime.idle_s"] == "default"
        listed = h.client.get("/umrs/admin/sessions", headers=_bearer()).json()["sessions"]
        assert listed[0]["policy"]["web"]["mode"] == "allowlist"


def test_plan_run_refusals_are_typed_sessionpolicy_errors():
    from umrs.api.plan_run_venue import _refused

    with pytest.raises(HTTPException) as err:
        _refused(PolicyError("web_choice_required", "choose"), "start")
    assert err.value.status_code == 400
    assert err.value.headers[CODE] == "sessionpolicy.web_choice_required"
