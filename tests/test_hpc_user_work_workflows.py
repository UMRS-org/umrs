"""ADR-112 A.3.4: a run on the HPC lane is user work (wave A).

The staged step lane runs under the site's login, so a run that would put
uploaded code there (or built-in steps that ask for the slurm lane) is
refused, typed ``hpcaccess.*``, unless the site's ``hpc_access`` admits the
caller's **verified** project (or the caller's user record is an admin's):

- synchronously, before the run is accepted: the HTTP trigger, the ZMQ
  ``RUN_WORKFLOW`` handler and the plan quote;
- again in the runner (defence in depth), from the verified project the
  handler threads through, never the ``params.project_id`` claim.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_workflow_plans import CODE, SECRET, _h, wf  # noqa: F401 — fixture
from umrs.orchestrator.site_config import HpcAccessConfig, ServiceAccountAccess
from umrs.server import step_trust
from umrs.server import workflow_runner as wr
from umrs.server.workflow_execution_registry import ExecutionRecord
from umrs.shared import hpc_user_work_access as access

from r4_contracts.handshake_hmac import sign_claims

LISTED = "proj-listed"
NOT_ALLOWED = "hpcaccess.project_not_allowed"
_LANE_ENV = {
    "UMRS_SLURM_STEP_SUBMIT": "remote",
    "UMRS_SLURM_WORKDIR_ROOT": "/lvs0/x/umrs_steps",
    "UMRS_STEP_REMOTE_ENABLED": "1",
    "UMRS_STEP_HELPER": "/lvs0/x/umrs_step_helper.py",
    "UMRS_STEP_IMAGE": "localhost/umrs-step:0123456789ab",
}


def _site(allowed=(LISTED,)):
    hpc = HpcAccessConfig(
        modes_allowed=["service_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(
            login="operator", personal=True, allowed_projects=list(allowed)
        ),
    )
    return SimpleNamespace(
        name="cluster-a", extra={"ssh_user": "operator"}, scheduler=SimpleNamespace(hpc_access=hpc)
    )


@pytest.fixture
def hpc(monkeypatch, wf):  # noqa: F811 — the plan tests' fixture
    """The plan tests' app with the HPC lane open, ``relax``'s steps
    uploaded code, and a site that lists only ``proj-listed``."""
    client, store, tmp_path = wf
    for name, value in _LANE_ENV.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "relax" / "m.py").write_text("")  # m:S1, m:S2 are uploaded code
    monkeypatch.setattr(access, "current_site", lambda: _site())
    # The one admin source is the user record: only "ops" is an admin there.
    monkeypatch.setattr(access, "user_is_admin", lambda user: user == "ops")
    return client, store, tmp_path


def _admin_headers(user="ops", project="proj-a"):
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, True),
    }


def _trigger(client, headers):
    return client.post("/umrs/workflows/relax/trigger", json={"parameters": {}}, headers=headers)


# ── HTTP trigger ──────────────────────────────────────────────────────────


def test_trigger_refuses_an_unlisted_project_with_the_typed_code(hpc):
    client, store, _ = hpc
    r = _trigger(client, _h())
    assert (r.status_code, r.headers[CODE]) == (403, NOT_ALLOWED)
    assert "proj-a" in r.json()["detail"]
    assert store.jobs == {}


def test_trigger_admits_a_listed_project_and_an_admin(hpc):
    client, store, _ = hpc
    assert _trigger(client, _h(project=LISTED)).status_code == 200
    assert _trigger(client, _admin_headers()).status_code == 200
    assert len(store.jobs) == 2


def test_an_admitted_admin_run_is_not_refused_by_the_runner(hpc):
    """Review 2026-11-08: admission read the token's admin flag, the runner
    the user record, so an admitted run of an unlisted project then failed
    in the runner. Both now read the record."""
    client, store, _ = hpc
    assert _trigger(client, _admin_headers()).status_code == 200
    assert wr._hpc_identity_refusal("ops", "proj-a") == ""


def test_a_token_admin_without_an_admin_record_is_refused_at_the_door(hpc):
    """The reverse half: a token-only admin is refused synchronously (typed),
    not accepted and then failed by the runner."""
    client, store, _ = hpc
    r = _trigger(client, _admin_headers(user="token-admin"))
    assert (r.status_code, r.headers[CODE]) == (403, NOT_ALLOWED)
    assert store.jobs == {}


def test_trigger_on_an_undeclared_site_is_mode_undeclared(hpc, monkeypatch):
    client, store, _ = hpc
    monkeypatch.setattr(access, "current_site", lambda: None)
    r = _trigger(client, _h(project=LISTED))
    assert (r.status_code, r.headers[CODE]) == (403, "hpcaccess.mode_undeclared")


def test_a_run_without_uploaded_code_is_not_checked(hpc):
    client, store, tmp_path = hpc
    (tmp_path / "relax" / "m.py").unlink()  # platform code only
    assert _trigger(client, _h()).status_code == 200


def test_a_builtin_run_on_the_slurm_lane_is_checked(hpc):
    """Review 2026-11-08: built-in steps that ask for the slurm lane run
    under the site's login too; the old check looked only at uploaded code."""
    client, store, tmp_path = hpc
    (tmp_path / "relax" / "m.py").unlink()  # platform code only
    path = tmp_path / "relax" / "workflow.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "execution_mode": "slurm"}))
    r = _trigger(client, _h())
    assert (r.status_code, r.headers[CODE]) == (403, NOT_ALLOWED)
    assert store.jobs == {}
    assert _trigger(client, _h(project=LISTED)).status_code == 200


def test_a_closed_hpc_lane_is_not_this_check(hpc, monkeypatch):
    client, _store, _ = hpc
    monkeypatch.setattr(step_trust, "hpc_lane_available", lambda: False)
    r = _trigger(client, _h())
    assert r.headers.get(CODE) != NOT_ALLOWED


# ── plan quote ────────────────────────────────────────────────────────────


def test_plan_refuses_an_unlisted_project(hpc):
    client, _store, _ = hpc
    r = client.post("/umrs/workflows/relax/plan", json={"parameters": {"steps": 1}}, headers=_h())
    assert (r.status_code, r.headers[CODE]) == (403, NOT_ALLOWED)


def test_plan_quotes_a_listed_project(hpc):
    client, _store, _ = hpc
    r = client.post(
        "/umrs/workflows/relax/plan",
        json={"parameters": {"steps": 1}},
        headers=_h(project=LISTED),
    )
    assert r.status_code == 200, r.text
    assert r.json()["signed"]["lane"] == "slurm"


# ── ZMQ RUN_WORKFLOW ──────────────────────────────────────────────────────


def _zmq(store, project, *, verified, admin=False, user="alice"):
    from umrs.server.job_tracking_handler import DurableJobCommandBackend
    from umrs.server.workflow_registry_handler import _handle_run_workflow

    payload = json.dumps(
        {
            "workflow_name": "relax",
            "params": {"project_id": project},
            "_identity_verified": verified,
        }
    )
    reply = _handle_run_workflow(
        user,
        payload,
        DurableJobCommandBackend(store),
        trusted_identity=verified,
        caller_is_admin=admin,
    )
    return json.loads(reply)


def test_zmq_refuses_an_unlisted_verified_project_typed(hpc):
    _client, store, _ = hpc
    reply = _zmq(store, "proj-a", verified=True)
    assert (reply["status"], reply["code"]) == ("error", NOT_ALLOWED)
    assert store.jobs == {}


def test_zmq_never_admits_an_unverified_claim_of_a_listed_project(hpc):
    _client, store, _ = hpc
    reply = _zmq(store, LISTED, verified=False)
    assert (reply["status"], reply["code"]) == ("error", NOT_ALLOWED)
    assert store.jobs == {}


def test_zmq_admits_a_verified_listed_project_and_an_admin(hpc):
    _client, store, _ = hpc
    assert _zmq(store, LISTED, verified=True)["status"] == "ok"
    # "ops" is an admin by its user record (the one admin source); a session
    # admin flag alone does not admit (the runner would refuse it later).
    assert _zmq(store, "proj-a", verified=True, admin=True)["code"] == NOT_ALLOWED
    assert _zmq(store, "proj-a", verified=True, admin=True, user="ops")["status"] == "ok"
    assert len(store.jobs) == 2


# ── runner: defence in depth ──────────────────────────────────────────────


@pytest.fixture
def runner_lane(monkeypatch):
    monkeypatch.setattr(step_trust, "hpc_lane_available", lambda: True)
    monkeypatch.setattr(access, "current_site", lambda: _site())
    monkeypatch.setattr(access, "user_is_admin", lambda _user: False)
    built: list = []
    monkeypatch.setattr(wr, "build_step_executor", lambda **kw: built.append(kw))
    return built


def _upload(tmp_path):
    (tmp_path / "lanemod.py").write_text("raise SystemExit('never imported')\n")
    return str(tmp_path)


_UPLOADED = {"execution_mode": "slurm", "steps": [{"name": "s1", "entrypoint": "lanemod:S"}]}


def test_runner_refuses_an_unlisted_verified_project_despite_a_listed_claim(tmp_path, runner_lane):
    record = ExecutionRecord(
        execution_id="e-claim",
        workflow_name="w",
        user_id="alice",
        params={"project_id": LISTED},  # the claim names a listed project
    )
    wr.run_workflow_record(
        record,
        source_path=_upload(tmp_path),
        workflow_json=_UPLOADED,
        project_id=LISTED,
        user_id="alice",
        verified_project="proj-a",
    )
    assert record.status == "failed"
    assert NOT_ALLOWED in (record.error or "")
    assert runner_lane == []  # no executor was built


def test_runner_refuses_when_no_project_was_verified(tmp_path, runner_lane):
    record = ExecutionRecord(execution_id="e-none", workflow_name="w", user_id="alice")
    wr.run_workflow_record(
        record, source_path=_upload(tmp_path), workflow_json=_UPLOADED, project_id=LISTED
    )
    assert record.status == "failed" and NOT_ALLOWED in (record.error or "")


def test_runner_keeps_the_no_lane_text_for_a_closed_lane(tmp_path, runner_lane, monkeypatch):
    monkeypatch.setattr(step_trust, "hpc_lane_available", lambda: False)
    monkeypatch.delenv("UMRS_STEP_UPLOADED_CODE", raising=False)
    monkeypatch.setattr(step_trust, "_load_site_env", lambda: None)
    record = ExecutionRecord(execution_id="e-closed", workflow_name="w", user_id="alice")
    wr.run_workflow_record(
        record, source_path=_upload(tmp_path), workflow_json=_UPLOADED, verified_project=LISTED
    )
    assert record.error == wr._NO_HPC_LANE


def test_executor_name_admits_a_listed_verified_project(tmp_path, runner_lane):
    steps = _UPLOADED["steps"]
    source = _upload(tmp_path)
    allowed = wr._workflow_executor_name(
        _UPLOADED, steps, source, injected=False, run_identity=("alice", LISTED)
    )
    assert allowed == ("slurm", "")
    refused = wr._workflow_executor_name(
        _UPLOADED, steps, source, injected=False, run_identity=("alice", "proj-a")
    )
    assert refused[0] is None and NOT_ALLOWED in refused[1]


_BUILTIN_SLURM = {
    "execution_mode": "slurm",
    "steps": [{"name": "s1", "entrypoint": "umrs.steps.builtin:Echo"}],
}


def test_runner_refuses_a_builtin_slurm_run_of_an_unlisted_project(runner_lane):
    """Review 2026-11-08: a workflow of built-in steps with
    ``execution_mode: slurm`` reached the site's login for any project."""
    steps = _BUILTIN_SLURM["steps"]
    refused = wr._workflow_executor_name(
        _BUILTIN_SLURM, steps, None, injected=False, run_identity=("mallory", "unlisted")
    )
    assert refused[0] is None and NOT_ALLOWED in refused[1]
    allowed = wr._workflow_executor_name(
        _BUILTIN_SLURM, steps, None, injected=False, run_identity=("alice", LISTED)
    )
    assert allowed == ("slurm", "")


def test_the_legacy_uploaded_lane_is_checked_when_it_is_the_hpc_lane(
    tmp_path, runner_lane, monkeypatch
):
    """Review 2026-11-08: with ``step_executor.uploaded_code:
    legacy_in_process`` the manifest's lane (possibly slurm) was returned
    without the identity check."""
    monkeypatch.setattr(wr, "uploaded_code_lane", lambda _w: wr.LEGACY_IN_PROCESS)
    monkeypatch.setattr(wr, "_prepare_source_path", lambda _p: None)
    steps = _UPLOADED["steps"]
    source = _upload(tmp_path)
    refused = wr._workflow_executor_name(
        _UPLOADED, steps, source, injected=False, run_identity=("mallory", "unlisted")
    )
    assert refused[0] is None and NOT_ALLOWED in refused[1]
    allowed = wr._workflow_executor_name(
        _UPLOADED, steps, source, injected=False, run_identity=("alice", LISTED)
    )
    assert allowed == ("slurm", "")


def test_runner_leaves_an_in_process_builtin_run_alone(runner_lane, monkeypatch):
    monkeypatch.delenv("UMRS_STEP_EXECUTOR", raising=False)
    workflow = {"steps": _BUILTIN_SLURM["steps"]}
    got = wr._workflow_executor_name(
        workflow, workflow["steps"], None, injected=False, run_identity=("mallory", None)
    )
    assert got == (None, "")


def test_the_handler_threads_the_verified_project_not_the_claim(hpc, monkeypatch):
    from umrs.server import workflow_execution_registry as registry_mod
    from umrs.server import workflow_registry_handler as rh

    seen: list = []
    monkeypatch.setattr(
        wr, "run_workflow_record", lambda _record, **kw: seen.append(kw["verified_project"])
    )
    monkeypatch.setattr(
        registry_mod.WorkflowExecutionRegistry,
        "launch",
        lambda _self, record, runner: runner(record),
    )
    claim = {"project_id": LISTED}
    rh._launch_workflow_from_catalog(workflow_name="relax", client_id="alice", params=claim)
    rh._launch_workflow_from_catalog(
        workflow_name="relax", client_id="alice", params=claim, caller_project="proj-b"
    )
    rh._launch_workflow_from_catalog(
        workflow_name="relax", client_id="alice", params=claim, identity_verified=True
    )
    assert seen == [None, "proj-b", LISTED]


# ── W4 ingest triggers (review 2026-11-08) ────────────────────────────────


@pytest.fixture
def ingest(hpc, monkeypatch):
    """``relax`` subscribes to ingest events; runs are captured, not run.
    alice (the workflow's owner) belongs to proj-a and proj-listed."""
    from umrs.server import workflow_execution_registry as registry_mod

    _client, _store, tmp_path = hpc
    path = tmp_path / "relax" / "workflow.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "triggers": [{"on": "ingest", "bucket": "up"}]}))
    _repin_digests()  # the files of the registered version were edited in place
    members = {"alice": {"proj-a", LISTED}}
    monkeypatch.setattr(
        access, "user_in_project", lambda user, project: project in members.get(user, ())
    )
    seen: list = []
    monkeypatch.setattr(
        wr, "run_workflow_record", lambda _record, **kw: seen.append(kw["verified_project"])
    )
    monkeypatch.setattr(
        registry_mod.WorkflowExecutionRegistry,
        "launch",
        lambda _self, record, runner: runner(record),
    )
    return seen, members


def _repin_digests():
    """ADR-113 D-4: a run refuses files that changed since registration; a
    fixture that edits a registered tree records the new digest."""
    from umrs.server import workflow_registry_handler as rh
    from umrs.server.workflow_versions import tree_digest

    catalog = rh._load_catalog()
    for entry in catalog["workflows"]:
        entry["source_sha256"] = tree_digest(entry["source_path"])
    rh._save_catalog(catalog)


def _ingest_run(payload, *, event_project):
    from umrs.server.workflow_registry_handler import run_ingest_workflow

    return json.loads(run_ingest_workflow("alice", json.dumps(payload), event_project=event_project))


def _dispatch(project):
    from umrs.server import workflow_triggers

    return workflow_triggers.dispatch_catalog_event(
        project_id=project, bucket="up", s3_key="a.bin", owner="bob"
    )


def test_an_ingest_event_in_a_listed_project_runs_uploaded_code(ingest):
    """Before: ingest runs had no verified project, so every allow-listed
    upload's staged run was refused (and only logged)."""
    seen, _members = ingest
    ids = _dispatch(LISTED)
    assert len(ids) == 1 and ids[0].startswith("wf-")
    assert seen == [LISTED]  # the runner checks the same project


def test_an_ingest_event_in_an_unlisted_project_is_refused_typed(ingest):
    seen, _members = ingest
    reply = _ingest_run(
        {"workflow_name": "relax", "params": {"project_id": "proj-a"}}, event_project="proj-a"
    )
    assert (reply["status"], reply["code"]) == ("error", NOT_ALLOWED)
    assert _dispatch("proj-a") == []
    assert seen == []


def test_an_owner_outside_the_event_project_gains_nothing_from_it(ingest):
    """The event's project counts only when the workflow's owner belongs to
    it: subscribing to a listed project's uploads admits no outsider."""
    seen, members = ingest
    members["alice"] = {"proj-a"}
    assert _dispatch(LISTED) == []
    assert seen == []


def test_the_ingest_project_grants_no_verified_provenance(ingest):
    """The event project is for the HPC lane rule only: a payload's
    ``_identity_verified`` claim is dropped, so it admits nothing."""
    seen, _members = ingest
    payload = {
        "workflow_name": "relax",
        "params": {"project_id": LISTED},
        "_identity_verified": True,
    }
    assert _ingest_run(payload, event_project=None)["code"] == NOT_ALLOWED
    assert seen == []
