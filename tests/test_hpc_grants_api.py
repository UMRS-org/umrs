"""The grant and usage routes (ADR-112 waves C/D; plan §6 C/D.0 C4, D4,
K8, K15, K18).

On the B1 route app (a scripted gate, real transport keys under tmp):
storing and cancelling are the person's own act (``person_required``); a
gate older than 3 is ``gate_version``; the live ``grant-check`` decides;
a cancel the gate refuses changes nothing, an unreachable gate still ends
the grant here (``gate_cancel: pending``, ``relayed: false``); usage reads
only the caller's rows; the project report hides other members' HPC
identifiers from an owner who is not an admin.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from grant_fixtures import SIG, encoded, grant_body
from test_hpc_accounts_api import ALICE, SITE, _code, _hello, _register
from test_hpc_accounts_api import env as _accounts_env  # noqa: F401 — fixture
from umrs.hpc_accounts import grant_service
from umrs.hpc_accounts.usage_ledger import (
    InMemoryUsageLedger,
    LedgerUnit,
    Resources,
    set_usage_ledger,
)

IMAGE = {"ref": "umrs-step:1", "image_id": "a" * 64, "archive_sha256": "b" * 64}
AGENT = {**ALICE, "x-r4-actor": "agent"}


def _check_answer(state: str = "active", jobs: int = 0) -> tuple[int, bytes, bytes]:
    data = {"grant_id": "x", "state": state, "usage": {"jobs": jobs, "open": 0}}
    return 0, json.dumps(data).encode(), b""


@pytest.fixture
def api(request, monkeypatch):
    client, store, gate, state, keys = request.getfixturevalue("_accounts_env")
    gate.answers["hello"] = _hello(gate_version="3")
    gate.answers["grant-check"] = _check_answer()
    gate.answers["grant-cancel"] = (0, b'{"ok": true, "already": false}', b"")
    monkeypatch.setattr(grant_service, "digests_lookup", lambda *_a, **_k: lambda n, v: ("1" * 64, "2" * 64) if (n, v) == ("wf", "1.0.0") else None)
    monkeypatch.setattr(grant_service, "site_images", lambda _site: frozenset({tuple(IMAGE.values())}))
    _register(client)
    assert client.post("/umrs/sites/me/hpc/links", json={"project_id": "p1", "slurm_account": "grp-a"}, headers=ALICE).status_code == 200
    return client, store, gate


def _body(**over: Any) -> dict[str, Any]:
    return grant_body(site=SITE, **over)


def _post(client, body: dict[str, Any], headers=ALICE):
    return client.post("/umrs/sites/me/hpc/grants", json={"payload": encoded(body), "signature": SIG}, headers=headers)


def _cancel(client, grant_id: str, headers=ALICE):
    control = {"op": "grant-cancel", "grant_id": grant_id}
    import base64

    payload = base64.b64encode(json.dumps(control).encode()).decode()
    return client.post(f"/umrs/sites/me/hpc/grants/{grant_id}/cancel", json={"payload": payload, "signature": SIG}, headers=headers)


def test_a_grant_is_stored_after_a_live_grant_check(api):
    client, _store, gate = api
    r = _post(client, _body())
    assert r.status_code == 200, r.text
    view = r.json()["grant"]
    assert (view["status"], view["covers_now"], view["stale"]) == ("active", True, [])
    assert SIG not in r.text and gate.calls[-1][-1] == "grant-check"
    assert [g["grant_id"] for g in client.get("/umrs/sites/me/hpc/grants", headers=ALICE).json()["grants"]] == [view["grant_id"]]


def test_storing_and_cancelling_are_the_persons_own_act(api):
    client, _store, gate = api
    assert _code(_post(client, _body(), AGENT)) == "hpcaccess.person_required"
    assert _code(_cancel(client, "a" * 32, AGENT)) == "hpcaccess.person_required"
    assert gate.calls[-1][-1] != "grant-check"


def test_a_gate_older_than_three_refuses_gate_version(api):
    client, _store, gate = api
    gate.answers["hello"] = _hello(gate_version="2")
    _register(client)
    r = _post(client, _body())
    assert (r.status_code, _code(r)) == (409, "hpcaccess.gate_version")
    assert "grants_need_gate_3" in r.text


def test_a_live_check_that_reports_cancelled_or_refuses_is_not_stored(api):
    client, _store, gate = api
    gate.answers["grant-check"] = _check_answer("cancelled")
    r = _post(client, _body())
    assert (r.status_code, _code(r)) == (422, "hpcaccess.grant_invalid") and "cancelled" in r.text
    gate.answers["grant-check"] = (4, b"", b'{"error": "bad_signature", "detail": "no"}')
    assert _code(_post(client, _body())) == "hpcaccess.gate_refused"
    assert client.get("/umrs/sites/me/hpc/grants", headers=ALICE).json()["grants"] == []


def test_a_refresh_mirrors_the_gates_usage(api):
    client, _store, gate = api
    _post(client, _body())
    gate.answers["grant-check"] = _check_answer(jobs=3)
    (view,) = client.get("/umrs/sites/me/hpc/grants?refresh=1", headers=ALICE).json()["grants"]
    assert view["usage"]["jobs"] == 3 and view["usage_at"]


def test_a_relayed_cancel_records_it_and_a_refused_one_changes_nothing(api):
    client, _store, gate = api
    gid = _post(client, _body()).json()["grant"]["grant_id"]
    gate.answers["grant-cancel"] = (4, b"", b'{"error": "replayed", "detail": "nonce"}')
    r = _cancel(client, gid)
    assert _code(r) == "hpcaccess.gate_refused"
    assert client.get("/umrs/sites/me/hpc/grants", headers=ALICE).json()["grants"][0]["status"] == "active"
    gate.answers["grant-cancel"] = (0, b'{"ok": true, "already": false}', b"")
    data = _cancel(client, gid).json()
    assert (data["relayed"], data["grant"]["status"], data["grant"]["gate_cancel"]) == (True, "cancelled", "recorded")
    assert data["result"]["ok"] is True


def test_an_unreachable_gate_still_ends_the_grant_here(api):
    """K8: a transport failure sets cancelled with gate_cancel pending."""
    client, _store, gate = api
    gid = _post(client, _body()).json()["grant"]["grant_id"]
    gate.answers["grant-cancel"] = (255, b"", b"ssh: connect to host: timed out")
    data = _cancel(client, gid).json()
    assert (data["relayed"], data["grant"]["status"], data["grant"]["gate_cancel"]) == (False, "cancelled", "pending")
    assert "result" not in data


def test_a_cancel_of_an_unknown_grant_or_another_grant_is_refused(api):
    client, _store, _gate = api
    assert (_cancel(client, "f" * 32).status_code, _code(_cancel(client, "f" * 32))) == (404, "hpcaccess.grant_not_found")
    gid = _post(client, _body()).json()["grant"]["grant_id"]
    import base64

    payload = base64.b64encode(json.dumps({"op": "grant-cancel", "grant_id": "e" * 32}).encode()).decode()
    r = client.post(f"/umrs/sites/me/hpc/grants/{gid}/cancel", json={"payload": payload, "signature": SIG}, headers=ALICE)
    assert _code(r) == "hpcaccess.gate_refused" and "not a grant-cancel of this grant" in r.text


def test_a_revoke_ends_the_grants(api):
    client, _store, _gate = api
    _post(client, _body())
    assert client.delete("/umrs/sites/me/hpc", headers=ALICE).status_code == 200
    (view,) = client.get("/umrs/sites/me/hpc/grants", headers=ALICE).json()["grants"]
    assert (view["status"], view["ended_reason"], view["covers_now"]) == ("cancelled", "account_revoked", False)


# ── usage (D4) ───────────────────────────────────────────────────────────────


def _unit(person: str, attempt: str, project: str = "p1", venue: str = SITE) -> LedgerUnit:
    return LedgerUnit(
        venue=venue, attempt_id=attempt, person=person, lane="staged_step", tag="t", project_id=project,
        site_login=f"{person}-login", access_mode="user_account", method="gated_key", grant_id="a" * 32,
    )


@pytest.fixture
def ledger():
    led = InMemoryUsageLedger(tables={"rows": [], "budgets": {"p1": 10.0}})
    led.pending(_unit("alice", "r1/s/1"), Resources(2, 1, 600))
    led.submitted(_unit("alice", "r1/s/1"), "11", 4)
    led.pending(_unit("bob", "r2/s/1"), Resources(1, 1, 1200))
    led.settled(_unit("bob", "r2/s/1"), {"state": "COMPLETED", "elapsed_s": 300, "alloc_cpus": 1, "alloc_gpus": 1}, "sacct")
    led.pending(_unit("alice", "r3/s/1", venue="other-front"), Resources(1, 0, 60))
    set_usage_ledger(led)
    return led


def test_my_usage_is_my_rows_at_this_venue_in_both_modes(env_usage, ledger):
    client = env_usage
    data = client.get("/umrs/sites/me/hpc/usage", headers=ALICE).json()
    assert data["venue"] == SITE and [r["person"] for r in data["rows"]] == ["alice"]
    assert data["totals"]["reserved_gpu_hours"] == pytest.approx(600 / 3600)
    assert client.get("/umrs/sites/me/hpc/usage").status_code == 401


@pytest.fixture
def env_usage(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import hpc_usage
    from umrs.orchestrator.site_config import SiteConfig

    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.setattr(hpc_usage, "_venue", lambda: SITE)
    app = FastAPI()
    app.include_router(hpc_usage.router)
    assert SiteConfig  # a service-account site serves it too: no mode check
    return TestClient(app)


def test_a_member_sees_own_rows_and_the_totals(ledger):
    from umrs.hpc_accounts.usage_reports import project_usage

    member = project_usage(ledger, "p1", caller="bob", role="member")
    assert [r["person"] for r in member["rows"]] == ["bob"]
    assert "members" not in member and member["budget"]["max_gpu_hours"] == 10.0


def test_an_owner_who_is_not_an_admin_sees_no_hpc_identifiers_of_others(ledger):
    """K15: no site_login, scheduler_job_id, tag, intent_id, grant_id or
    gate_log_seq on another member's rows."""
    from umrs.hpc_accounts.usage_ledger import PRIVATE_COLUMNS
    from umrs.hpc_accounts.usage_reports import project_usage

    owner = project_usage(ledger, "p1", caller="carol", role="owner")
    assert {m["person"] for m in owner["members"]} == {"alice", "bob"}
    shown = {key for row in owner["rows"] for key in row}
    assert not shown & set(PRIVATE_COLUMNS)


def test_an_admin_sees_every_column_and_can_filter_a_venue(ledger):
    from umrs.hpc_accounts.usage_reports import project_usage

    admin = project_usage(ledger, "p1", caller="root", role="admin")
    assert {r["site_login"] for r in admin["rows"]} == {"alice-login", "bob-login"}
    assert len(project_usage(ledger, "p1", caller="root", role="admin", venue=SITE)["rows"]) == 2


def test_the_project_route_reads_404_for_another_project(ledger, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import projects

    class _Store:
        def get(self, project_id: str) -> dict[str, Any]:
            if project_id != "p1":
                raise projects.ProjectNotFound(project_id)
            return {"project_id": "p1", "owner_id": "carol"}

        def circle_affiliated(self, *_a: Any) -> bool:
            return False

    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    app = FastAPI()
    app.include_router(projects.router)
    app.dependency_overrides[projects._store_dep] = lambda: _Store()
    client = TestClient(app)
    owner = client.get("/umrs/projects/p1/hpc-usage", headers={"X-User-ID": "carol", "X-Verified-Project-ID": "p9"})
    assert owner.status_code == 200, owner.text
    assert "members" in owner.json() and all("site_login" not in r for r in owner.json()["rows"] if r["person"] != "carol")
    member = client.get("/umrs/projects/p1/hpc-usage", headers={"X-User-ID": "bob", "X-Verified-Project-ID": "p1"})
    assert [r["person"] for r in member.json()["rows"]] == ["bob"]
    gone = client.get("/umrs/projects/p1/hpc-usage", headers={"X-User-ID": "dave", "X-Verified-Project-ID": "p2"})
    assert gone.status_code == 404
