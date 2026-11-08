"""Signed intents at the workflow routes (ADR-112 B2; plan §5 B2.0 §10.3).

A run whose HPC-lane identity is the person's own account starts only from
the person's signed intent. The proposal is the server's view of that run
(nothing runs); the trigger checks the intent is exactly the run it starts,
creates it as ``execution_id = run_id`` and keeps the payload and signature
for the gate. Without an intent: 428 ``signature_required``; a mismatched
or expired one: 422 ``intent_invalid``; a second trigger of one intent: 409
``intent_used``; an intent on a run that is not the person's: 422.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
from test_workflow_plans import CODE, SECRET, _h, wf  # noqa: F401 — fixture
from umrs.hpc_accounts.store import InMemorySiteAccountStore
from umrs.orchestrator.site_config import (
    HpcAccessConfig,
    ServiceAccountAccess,
    SiteConfig,
    UserAccountAccess,
)
from umrs.server import gated_step_lane as lane
from umrs.shared import hpc_user_work_access as access

from r4_contracts import hpc_gate
from r4_contracts.handshake_hmac import sign_claims

DOUBLE = str(Path(__file__).resolve().parent / "gated_gate_double.py")
IMAGE = "localhost/umrs-step:0123456789ab"
SIGNATURE = "-----BEGIN SSH SIGNATURE-----\nU1NIU0lH\n-----END SSH SIGNATURE-----\n"
_LANE_ENV = {
    "UMRS_SLURM_STEP_SUBMIT": "remote",
    "UMRS_SLURM_WORKDIR_ROOT": "/lvs0/x/umrs_steps",
    "UMRS_STEP_REMOTE_ENABLED": "1",
    "UMRS_STEP_HELPER": "/lvs0/x/umrs_step_helper.py",
    "UMRS_STEP_IMAGE": IMAGE,
}


def _site(tmp_path: Path) -> SiteConfig:
    site = SiteConfig(name="site-x", extra={"ssh_user": "op", "ssh_target": "login-0"})
    site.scheduler.hpc_access = HpcAccessConfig(
        modes_allowed=["service_account", "user_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(login="op", personal=True, allowed_projects=[]),
        user_account=UserAccountAccess(
            methods=["gated_key"],
            signing_domain="platform.example.org",
            profile={"image_archive_dir": "/shared/umrs/archives"},
        ),
    )
    digests = tmp_path / "image-digests.json"
    digests.write_text(
        json.dumps({IMAGE: {"x86_64": {"image_id": "a" * 64, "archive_sha256": "b" * 64}}})
    )
    se = site.step_executor
    se.partition, se.cpus, se.memory, se.gpus = "gpu", 8, "", 1
    se.time_limit, se.queue_timeout, se.image_digests = "02:00:00", "01:00:00", str(digests)
    return site


def _enrol(tables: dict, user: str = "alice", project: str = "proj-a") -> None:
    store = InMemorySiteAccountStore("site-x", tables)
    store.issue_key(user, "tk-000000000001", "SHA256:x")
    store.record_verified(
        user,
        "tk-000000000001",
        {
            "site_login": "alice_hpc", "verified_login": "alice_hpc",
            "verified_assocs": [{"account": "grp-a"}], "gate_version": "2",
            "gate_sha256": "0" * 64, "signer_fingerprints": [], "uv_required": False,
        },
    )
    store.upsert_link(user, project, "grp-a")


@pytest.fixture
def gated(wf, monkeypatch):  # noqa: F811 — the plan tests' fixture
    """The plan tests' app on a site that allows user accounts, ``alice``
    enrolled and linked for ``proj-a``, her gate a scripted double."""
    client, store, tmp_path = wf
    for name, value in _LANE_ENV.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "relax" / "m.py").write_text("class S1: pass\nclass S2: pass\n")
    site = _site(tmp_path)
    tables: dict = {"accounts": {}, "links": {}}
    _enrol(tables)
    monkeypatch.setattr(access, "current_site", lambda: site)
    monkeypatch.setattr(access, "user_is_admin", lambda _u: False)
    monkeypatch.setattr(access, "_store_factory", lambda v: InMemorySiteAccountStore(v, tables))
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    from umrs.hpc_accounts.site_profile import published

    hello = {"gate_sha256": hpc_gate.gate_sha256(), "site_profile_sha256": published(site)[2]}
    (gate_dir / "plan.json").write_text(json.dumps({"hello": hello}))
    lane.set_transport_factory(
        lambda _v, _u, _login: lane.GateTransport(
            lambda verb, args: [sys.executable, DOUBLE, str(gate_dir), verb, *args]
        )
    )
    yield client, store, tmp_path, gate_dir
    lane.set_transport_factory(None)


def _propose(client, **body) -> dict:
    r = client.post(
        "/umrs/workflows/relax/intent-proposal",
        json={"params": {"steps": 3}, **body},
        headers=_h(),
    )
    assert r.status_code == 200, r.text
    return r.json()


def _body(answer: dict) -> dict:
    return json.loads(base64.b64decode(answer["intent_b64"]))


def _b64(body: dict) -> str:
    return base64.b64encode(hpc_gate.canonical_bytes(body, json_values=True)).decode()


def _trigger(client, intent: str | None = None, params: dict | None = None, headers=None):
    payload: dict = {"parameters": {"steps": 3} if params is None else params}
    if intent is not None:
        payload.update(intent=intent, signature=SIGNATURE)
    return client.post("/umrs/workflows/relax/trigger", json=payload, headers=headers or _h())


# ── the proposal ─────────────────────────────────────────────────────────────


def test_the_proposal_is_a_valid_intent_for_exactly_this_run(gated):
    client, store, _tmp, _gate = gated
    answer = _propose(client)
    assert set(answer) == {"intent_b64", "intent_id", "site_profile_sha256", "summary"}
    payload = base64.b64decode(answer["intent_b64"])
    body = hpc_gate.validate_intent(hpc_gate.parse_strict(payload, json_values=True))
    assert answer["intent_id"] == hpc_gate.intent_id(payload)
    who = (body["site"], body["principal"], body["login"], body["account"], body["project"])
    assert who == ("site-x", "alice", "alice_hpc", "grp-a", "proj-a")
    assert body["params"] == {"steps": 3, "project_id": "proj-a"}
    assert body["expires"] - body["issued"] == 3600
    assert store.jobs == {}  # nothing runs


def test_the_proposed_steps_follow_the_graph_the_site_and_the_tree(gated):
    client, _store, _tmp, _gate = gated
    body = _body(_propose(client))
    assert [s["name"] for s in body["steps"]] == ["s1", "s2"]
    assert (body["steps"][1]["after"], body["steps"][1]["inputs"]["chained"]) == (["s1"], [])
    s1 = body["steps"][0]
    assert (s1["gpus"], s1["partition"], s1["time"]) == (1, "gpu", 1800)
    assert s1["image"] == {"ref": IMAGE, "image_id": "a" * 64, "archive_sha256": "b" * 64}


def test_the_proposal_pins_the_source_tree_and_the_run_window(gated):
    client, _store, _tmp, _gate = gated
    body = _body(_propose(client))
    assert body["src_manifest"]["entrypoints"] == {"s1": "m:S1", "s2": "m:S2"}
    assert [f["path"] for f in body["src_manifest"]["files"]] == ["m.py"]
    assert body["max_run_seconds"] == 1800 + 600 + 2 * 3600


def test_each_proposed_config_is_the_shared_expected_config(gated):
    client, _store, _tmp, _gate = gated
    body = _body(_propose(client))
    expected = hpc_gate.expected_step_config(
        {"name": "s1", "entrypoint": "m:S1"}, "s1", body["params"]
    )
    assert body["steps"][0]["config"] == expected


def test_the_proposal_needs_a_current_gate_and_site_profile(gated):
    client, _store, _tmp, gate_dir = gated
    (gate_dir / "plan.json").write_text(json.dumps({"hello": {"gate_sha256": "0" * 64}}))
    r = client.post("/umrs/workflows/relax/intent-proposal", json={"params": {"steps": 3}}, headers=_h())
    assert (r.status_code, r.headers[CODE]) == (409, "hpcaccess.gate_version")


def test_the_proposal_route_runs_in_the_threadpool():
    """Fixer finding 2026-11-08: the proposal's live ``hello`` (SSH) and its
    input hashing (up to minutes) block; an ``async def`` route would run
    them on the event loop and stall every other request on the server."""
    import inspect

    from umrs.api import workflows

    assert not inspect.iscoroutinefunction(workflows.propose_workflow_intent)


@pytest.mark.parametrize("code", [255, 5])
def test_an_unreachable_gate_is_gate_refused_unreachable_not_intent_invalid(gated, code):
    client, _store, _tmp, gate_dir = gated
    plan = json.loads((gate_dir / "plan.json").read_text())
    (gate_dir / "plan.json").write_text(json.dumps({**plan, "fail": {"hello": code}}))
    r = client.post("/umrs/workflows/relax/intent-proposal", json={"params": {"steps": 3}}, headers=_h())
    assert (r.status_code, r.headers[CODE]) == (403, "hpcaccess.gate_refused")
    assert "unreachable" in r.text and "try again later" in r.text


def test_an_input_that_cannot_be_read_while_hashing_is_intent_invalid_by_name():
    from umrs.hpc_accounts.intents import IntentInvalid
    from umrs.server.intent_proposal import _Hasher

    class _Gone:
        def get_object(self, **_kw):
            raise KeyError("NoSuchKey")

    with pytest.raises(IntentInvalid) as exc:
        _Hasher(_Gone(), deadline=float("inf")).digest("wf-runner", "data/x.bin")
    assert exc.value.reason == "input_unreadable" and "data/x.bin" in str(exc.value)


def test_the_proposal_refuses_a_run_that_is_not_the_persons(gated):
    client, _store, _tmp, _gate = gated
    r = client.post(
        "/umrs/workflows/relax/intent-proposal",
        json={"params": {"steps": 3}},
        headers=_h(project="proj-b"),  # no link: wave A rules, admins only
    )
    assert r.headers[CODE] == "hpcaccess.project_not_allowed"


# ── the trigger ──────────────────────────────────────────────────────────────


def test_a_run_under_the_persons_account_needs_a_signature(gated):
    client, store, _tmp, _gate = gated
    r = _trigger(client)
    assert (r.status_code, r.headers[CODE]) == (428, "hpcaccess.signature_required")
    assert store.jobs == {}


def test_a_signed_trigger_starts_the_intents_run_and_keeps_the_envelope(gated):
    client, store, _tmp, _gate = gated
    answer = _propose(client)
    r = _trigger(client, answer["intent_b64"])
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["execution_id"] == _body(answer)["run_id"]
    assert data["intent_id"] == answer["intent_id"]
    [job] = store.jobs.values()
    assert job.spec["intent"] == {
        "b64": answer["intent_b64"], "signature": SIGNATURE, "id": answer["intent_id"],
    }
    assert job.spec["params"] == _body(answer)["params"]


def test_a_second_trigger_of_one_intent_is_intent_used(gated):
    client, store, _tmp, _gate = gated
    answer = _propose(client)
    assert _trigger(client, answer["intent_b64"]).status_code == 200
    r = _trigger(client, answer["intent_b64"])
    assert (r.status_code, r.headers[CODE]) == (409, "hpcaccess.intent_used")
    assert len(store.jobs) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workflow", "other"),
        ("version", "9.9"),
        ("project", "proj-b"),
        ("site", "site-y"),
        ("principal", "mallory"),
        ("login", "someone_else"),
        ("account", "grp-z"),
    ],
)
def test_an_intent_for_another_run_is_intent_invalid(gated, field, value):
    client, store, _tmp, _gate = gated
    body = {**_body(_propose(client)), field: value}
    r = _trigger(client, _b64(body))
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.intent_invalid")
    assert store.jobs == {}


def test_a_run_id_the_server_did_not_issue_is_refused(gated):
    """Fixer finding 2026-11-08: the run's execution id is the intent's
    ``run_id``; were it the caller's choice, one enrolled person could queue
    a run under another's execution id and take over its record and jobs.
    The proposal derives it from a server-only key; the trigger recomputes."""
    client, store, _tmp, _gate = gated
    first = _propose(client)
    assert _trigger(client, first["intent_b64"]).status_code == 200
    squat = {**_body(_propose(client)), "run_id": _body(first)["run_id"]}
    r = _trigger(client, _b64(squat))
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.intent_invalid")
    assert "run_id_not_issued" in r.text and len(store.jobs) == 1
    other = {**_body(_propose(client)), "run_id": "wf-0123456789ab"}
    assert "run_id_not_issued" in _trigger(client, _b64(other)).text


def test_the_issued_run_id_binds_principal_nonce_and_time(monkeypatch):
    from umrs.hpc_accounts.intents import issued_run_id

    monkeypatch.setenv("UMRS_SECRET_KEY", "k" * 32)
    base = issued_run_id("alice", "a" * 32, 1000)
    assert base == issued_run_id("alice", "a" * 32, 1000)
    others = {issued_run_id("bob", "a" * 32, 1000), issued_run_id("alice", "b" * 32, 1000),
              issued_run_id("alice", "a" * 32, 1001)}  # fmt: skip
    assert base not in others and len(others) == 3
    monkeypatch.setenv("UMRS_SECRET_KEY", "j" * 32)
    assert issued_run_id("alice", "a" * 32, 1000) != base


def test_an_expired_intent_is_intent_invalid(gated):
    client, store, _tmp, _gate = gated
    body = _body(_propose(client))
    body.update(issued=1000, expires=2000)
    r = _trigger(client, _b64(body))
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.intent_invalid")
    assert "expired" in r.text


def test_trigger_params_that_differ_from_the_intent_are_refused(gated):
    client, store, _tmp, _gate = gated
    answer = _propose(client)
    r = _trigger(client, answer["intent_b64"], params={"steps": 4})
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.intent_invalid")
    assert "params" in r.text and store.jobs == {}


def test_a_non_canonical_payload_is_intent_invalid(gated):
    client, _store, _tmp, _gate = gated
    body = _body(_propose(client))
    loose = base64.b64encode(json.dumps(body, indent=1).encode()).decode()
    r = _trigger(client, loose)
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.intent_invalid")


def test_an_intent_on_a_run_that_is_not_the_persons_is_refused(gated, monkeypatch):
    client, store, _tmp, _gate = gated
    answer = _propose(client)
    monkeypatch.setattr(access, "user_is_admin", lambda _u: True)  # wave A admits an admin
    monkeypatch.setattr(access, "_store_factory", lambda v: InMemorySiteAccountStore(v))
    r = _trigger(client, answer["intent_b64"])
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.intent_invalid")
    assert store.jobs == {}


def test_a_service_account_run_is_unchanged(gated, monkeypatch):
    client, store, _tmp, _gate = gated
    monkeypatch.setattr(access, "user_is_admin", lambda _u: True)
    monkeypatch.setattr(access, "_store_factory", lambda v: InMemorySiteAccountStore(v))
    r = _trigger(client)
    assert r.status_code == 200, r.text
    [job] = store.jobs.values()
    assert "intent" not in job.spec


# ── plan, commit and the ZMQ handler ─────────────────────────────────────────


def test_the_plan_says_the_run_needs_an_intent_and_commit_refuses(gated):
    client, store, _tmp, _gate = gated
    plan = client.post("/umrs/workflows/relax/plan", json={"parameters": {"steps": 3}}, headers=_h())
    assert plan.status_code == 200 and plan.json()["requires_intent"] is True
    body = {"signed": plan.json()["signed"], "digest": plan.json()["digest"]}
    r = client.post("/umrs/workflows/relax/commit", json=body, headers=_h())
    assert (r.status_code, r.headers[CODE]) == (428, "hpcaccess.signature_required")
    assert store.jobs == {}


def test_a_signed_trigger_runs_in_a_project_that_requires_a_plan(gated):
    """Fixer amendment 2026-11-08 (plan §5 B2.0 §10.3): a plan cannot carry
    an intent, so ``require_plan`` would refuse every gated run on both
    paths; the person's signature over the exact run is the approval."""
    client, store, _tmp, _gate = gated
    store.db_pool.policies["proj-a"] = (None, True)
    answer = _propose(client)
    r = _trigger(client, answer["intent_b64"])
    assert r.status_code == 200, r.text
    assert len(store.jobs) == 1
    unsigned = _trigger(client)
    assert (unsigned.status_code, unsigned.headers[CODE]) == (403, "workflows.plan_required")


def _zmq(store, extra: dict) -> dict:
    from umrs.server.job_tracking_handler import DurableJobCommandBackend
    from umrs.server.workflow_registry_handler import _handle_run_workflow

    payload = json.dumps(
        {"workflow_name": "relax", "params": {"project_id": "proj-a", "steps": 3},
         "_identity_verified": True, **extra}
    )
    raw = _handle_run_workflow(
        "alice", payload, DurableJobCommandBackend(store), trusted_identity=True
    )
    return json.loads(raw)


def test_the_zmq_run_needs_the_same_signature(gated):
    client, store, _tmp, _gate = gated
    refused = _zmq(store, {})
    assert (refused["status"], refused["code"]) == ("error", "hpcaccess.signature_required")
    answer = _propose(client)
    ok = _zmq(store, {"intent": answer["intent_b64"], "signature": SIGNATURE})
    assert ok["status"] == "ok" and ok["execution_id"] == _body(answer)["run_id"]


def test_the_runner_refuses_a_gated_run_without_its_intent(gated):
    from umrs.server import workflow_runner as wr

    assert "signature_required" in wr._hpc_identity_refusal("alice", "proj-a", False)
    assert wr._hpc_identity_refusal("alice", "proj-a", True) == ""


def test_the_handler_error_maps_to_the_typed_status(gated):
    from fastapi import HTTPException
    from umrs.api.workflows import _parse_command_response

    raw = json.dumps({"status": "error", "code": "hpcaccess.intent_used", "message": "m"})
    with pytest.raises(HTTPException) as info:
        _parse_command_response(raw, action="trigger")
    assert info.value.status_code == 409


def test_a_verified_header_tuple_signs(gated):
    # The helper headers the plan tests use carry a real HMAC (strict mode).
    assert _h()["X-Verified-HMAC"] == sign_claims(SECRET, "alice", "proj-a", False)
