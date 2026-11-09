"""Runs under a standing grant at the trigger (ADR-112 wave C; plan §6
C/D.0 C4 "Admission", K1, K11, K14, K19).

On the B2 trigger app (``gated``: alice enrolled and linked for ``proj-a``
on a site that allows user accounts): a run without an intent uses the one
grant that covers it, or the one it names; none or several is 428
``signature_required`` naming the candidates; an intent with a ``grant_id``
is ``intent_invalid`` (``both``); a grant id on a run that is not the
person's is ``grant_invalid``. The run spec is the B2 intent body, unsigned,
under the grant's window, and the gate's ``grant_covers`` pre-check holds.
"""

from __future__ import annotations

import base64
import json
import secrets
import time
from typing import Any

import pytest
from test_gated_trigger import IMAGE, SIGNATURE, _propose
from test_gated_trigger import gated as _trigger_app  # noqa: F401 — fixture
from test_workflow_plans import CODE, _h, wf  # noqa: F401 — fixture
from umrs.hpc_accounts import grants as g
from umrs.hpc_accounts.grants import SignedGrant
from umrs.hpc_accounts.usage_ledger import InMemoryUsageLedger, set_usage_ledger
from umrs.shared import hpc_user_work_access as access

from r4_contracts import hpc_gate


@pytest.fixture
def gated(request):
    """The B2 trigger app (``test_gated_trigger.gated``)."""
    return request.getfixturevalue("_trigger_app")


GRANT_IMAGE = {"ref": IMAGE, "image_id": "a" * 64, "archive_sha256": "b" * 64}


@pytest.fixture(autouse=True)
def _ledger():
    ledger = InMemoryUsageLedger()
    set_usage_ledger(ledger)
    yield ledger
    set_usage_ledger(None)


def _relax() -> tuple[str, dict[str, Any], dict[str, Any]]:
    from umrs.server.workflow_registry_handler import resolve_workflow_version

    entry = resolve_workflow_version("relax", "alice").entry
    with open(f"{entry['source_path']}/workflow.json", encoding="utf-8") as handle:
        return entry["source_path"], json.load(handle), entry


def _grant(**over: Any) -> dict[str, Any]:
    source, workflow_json, entry = _relax()
    src_sha, steps_sha = g.server_digests(source, workflow_json)
    now = int(time.time())
    body: dict[str, Any] = {
        "v": 1, "type": "grant", "grant_id": secrets.token_hex(16), "site": "site-x",
        "login": "alice_hpc", "principal": "alice", "accounts": ["grp-a"], "projects": ["proj-a"],
        "workflows": [{"name": "relax", "version": str(entry["version"]), "src_manifest_sha256": src_sha,
                       "steps_sha256": steps_sha, "params": ["steps"]}],
        "images": [dict(GRANT_IMAGE)],
        "param_schema": {"type": "object", "additionalProperties": False, "required": [],
                         "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 10}}},
        "inputs": {"mode": "any", "listed": []},
        "caps": {"max_jobs": 10, "max_gpu_hours": 5, "max_core_hours": 0, "max_concurrent": 2},
        "not_before": now, "expires": now + 7200, "issued": now, "delegable": False,
        "nonce": secrets.token_hex(16),
    }
    body.update(over)
    return hpc_gate.validate_grant(body)


def _store_grant(body: dict[str, Any]) -> dict[str, Any]:
    store = access.site_account_store("site-x")
    grant = SignedGrant(hpc_gate.canonical_bytes(body), body, SIGNATURE)
    return g.grant_store_for(store).insert(g.row_from_grant("site-x", "alice", grant, {}))


def _trigger(client, **extra: Any):
    payload = {"parameters": {"steps": 3}, **extra}
    return client.post("/umrs/workflows/relax/trigger", json=payload, headers=_h())


def _spec(job: Any) -> dict[str, Any]:
    return json.loads(base64.b64decode(job.spec["intent"]["run_spec_b64"]))


def test_the_one_covering_grant_starts_the_run_with_a_composed_run_spec(gated):
    client, store, _tmp, _gate = gated
    row = _store_grant(_grant())
    r = _trigger(client)
    assert r.status_code == 200, r.text
    data = r.json()
    [job] = store.jobs.values()
    spec = _spec(job)
    assert (data["grant_id"], data["execution_id"]) == (row["grant_id"], spec["run_id"])
    assert data["run_spec_id"] == job.spec["intent"]["run_spec_id"]
    assert hpc_gate.validate_intent(spec)["type"] == "intent"
    assert (spec["login"], spec["account"], spec["project"]) == ("alice_hpc", "grp-a", "proj-a")
    assert spec["params"] == {"steps": 3, "project_id": "proj-a"}
    assert spec["max_run_seconds"] == 7200  # K19: min(RUN_MAX, expires - issued)
    skeleton = json.loads(base64.b64decode(job.spec["intent"]["workflow_steps_b64"]))
    assert hpc_gate.grant_covers(_grant_of(row), spec, skeleton) == ""


def _grant_of(row: dict[str, Any]) -> dict[str, Any]:
    return g.signed_body(row)


def test_a_named_grant_is_used_even_beside_another_that_covers(gated):
    client, store, _tmp, _gate = gated
    _store_grant(_grant())
    second = _store_grant(_grant())
    r = _trigger(client, grant_id=second["grant_id"])
    assert r.status_code == 200, r.text
    assert r.json()["grant_id"] == second["grant_id"]


def test_two_covering_grants_without_a_name_are_428_naming_both(gated):
    client, store, _tmp, _gate = gated
    ids = {_store_grant(_grant())["grant_id"], _store_grant(_grant())["grant_id"]}
    r = _trigger(client)
    assert (r.status_code, r.headers[CODE]) == (428, "hpcaccess.signature_required")
    assert all(i in r.text for i in ids) and store.jobs == {}


def test_no_covering_grant_is_428(gated):
    client, store, _tmp, _gate = gated
    _store_grant(_grant(projects=["proj-b"]))  # lists another project only
    r = _trigger(client)
    assert (r.status_code, r.headers[CODE]) == (428, "hpcaccess.signature_required")
    assert store.jobs == {}


def test_an_unknown_grant_id_is_grant_not_found(gated):
    client, store, _tmp, _gate = gated
    r = _trigger(client, grant_id="f" * 32)
    assert (r.status_code, r.headers[CODE]) == (404, "hpcaccess.grant_not_found")


def test_a_cancelled_grant_is_grant_invalid_cancelled(gated):
    client, store, _tmp, _gate = gated
    row = _store_grant(_grant())
    g.grant_store_for(access.site_account_store("site-x")).set_status("alice", row["grant_id"], "cancelled", "cancelled")
    r = _trigger(client, grant_id=row["grant_id"])
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.grant_invalid")
    assert "cancelled" in r.text and store.jobs == {}


def test_a_param_outside_the_grants_bounds_is_not_covered(gated):
    client, store, _tmp, _gate = gated
    row = _store_grant(_grant())
    r = client.post("/umrs/workflows/relax/trigger", json={"parameters": {"steps": 11}, "grant_id": row["grant_id"]}, headers=_h())
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.grant_invalid")
    assert "not_covered" in r.text and "params" in r.text and store.jobs == {}


def test_a_stale_named_grant_refuses_not_covered_with_the_reason(gated, monkeypatch):
    """K11: an explicit grant_id whose image the site no longer publishes."""
    client, store, _tmp, _gate = gated
    row = _store_grant(_grant(images=[{**GRANT_IMAGE, "image_id": "c" * 64}]))
    r = _trigger(client, grant_id=row["grant_id"])
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.grant_invalid")
    assert "image_unknown" in r.text


def test_an_intent_together_with_a_grant_id_is_both(gated):
    client, store, _tmp, _gate = gated
    answer = _propose(client)
    r = _trigger(client, intent=answer["intent_b64"], signature=SIGNATURE, grant_id="a" * 32)
    assert (r.status_code, r.headers[CODE]) == (422, "hpcaccess.intent_invalid")
    assert "both" in r.text and store.jobs == {}


def test_a_grant_id_on_a_service_account_run_is_not_covered(gated):
    from umrs.server.run_intent_admission import admit_run

    admission, refused = admit_run(None, {"grant_id": "a" * 32}, {}, {})
    assert admission is None and "hpcaccess.grant_invalid" in refused and "not_covered" in refused


def test_the_run_budget_is_checked_at_admission(gated, _ledger):
    """D3 (advisory): the whole run's estimate against the project's budget."""
    client, store, _tmp, _gate = gated
    _ledger.tables["budgets"]["proj-a"] = 0.4  # the run asks 1 GPU x 1800 s (step s1)
    _store_grant(_grant())
    r = _trigger(client)
    assert (r.status_code, r.headers[CODE]) == (403, "hpcaccess.budget_exceeded")
    assert store.jobs == {}


def test_the_run_window_is_the_grants_up_to_the_gates_limit():
    from umrs.server.grant_admission import run_max_seconds

    assert run_max_seconds({"issued": 0, "expires": 30 * 86400}) == 604800
    assert run_max_seconds({"issued": 0, "expires": 7200}) == 7200


def test_an_envelope_past_16_mib_is_too_large(monkeypatch):
    """K1: the server refuses at admission what the gate would refuse."""
    from umrs.server.grant_admission import ENVELOPE_MAX_BYTES, GrantRun, check_covers

    monkeypatch.setattr(hpc_gate, "grant_covers", lambda *_a: "")
    run = GrantRun("a" * 32, b"x" * (ENVELOPE_MAX_BYTES * 3 // 4), b"{}", {})
    row = {"payload_b64": base64.b64encode(b"{}").decode(), "signature": SIGNATURE}
    with pytest.raises(g.GrantInvalid) as info:
        check_covers(row, run)
    assert info.value.reason == "too_large"


# ── loops keep their grant (K12) ─────────────────────────────────────────────


class _JobStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> str:
        self.created.append(kwargs)
        return f"job-{len(self.created)}"

    def enqueue_job(self, _job_id: str) -> None:
        pass


def test_a_loop_records_its_grant_and_stops_when_it_ends(gated):
    """K12: the first iteration resolves the grant and the loop records it;
    once that grant is cancelled the loop stops with grant_ended, even
    though another grant covers the run."""
    from types import SimpleNamespace

    from umrs.knowledge.loop import AutonomousResearchLoop, GrantEnded, LoopStatus, _mark_failed
    from umrs.knowledge.umrs_workflow_port import UmrsWorkflowExecutionPort

    first = _store_grant(_grant())
    jobs = _JobStore()
    port = UmrsWorkflowExecutionPort(jobs)
    meta = {"owner_user_id": "alice", "owner_verified_project": "proj-a"}
    loop = SimpleNamespace(metadata=meta, project_id="proj-a")
    engine = AutonomousResearchLoop.__new__(AutonomousResearchLoop)
    engine._workflow_port = port
    execution_id = engine._trigger_workflow(loop, "relax", {"steps": 3, "project_id": "proj-a"})
    assert loop.metadata["grant_id"] == first["grant_id"] == port.grant_for(execution_id)
    spec = jobs.created[0]["spec"]
    assert spec["execution_id"] == execution_id and spec["intent"]["grant_id"] == first["grant_id"]
    _store_grant(_grant())  # another grant that covers the run
    store = g.grant_store_for(access.site_account_store("site-x"))
    store.set_status("alice", first["grant_id"], "cancelled", "cancelled")
    with pytest.raises(GrantEnded) as info:
        engine._trigger_workflow(loop, "relax", {"steps": 3, "project_id": "proj-a"})
    assert info.value.grant_id == first["grant_id"] and len(jobs.created) == 1
    state = SimpleNamespace(status=None, phase=None, metadata={}, completed_at=None)
    _mark_failed(state, info.value)  # type: ignore[arg-type]
    assert state.metadata["state_reason"] == f"grant_ended:{first['grant_id']}"
    assert state.status == LoopStatus.STOPPED


def test_the_plan_names_the_one_covering_grant(gated):
    """C4: ``/plan`` answers ``requires_intent`` and the covering grant (or
    null); a grant is not a per-run approval, so plan and commit still apply."""
    client, _store, _tmp, _gate = gated
    plan = client.post("/umrs/workflows/relax/plan", json={"parameters": {"steps": 3}}, headers=_h())
    assert (plan.json()["requires_intent"], plan.json()["grant_id"]) == (True, None)
    row = _store_grant(_grant())
    plan = client.post("/umrs/workflows/relax/plan", json={"parameters": {"steps": 3}}, headers=_h())
    assert plan.json()["grant_id"] == row["grant_id"]


def test_a_grant_run_rides_the_durable_spec_into_the_run_record():
    """0137: the record keeps the grant, the run spec and the skeleton; the
    status shows grant_id and run_spec_id only; the durable row round-trips."""
    from umrs.hpc_accounts.intents import intent_from_spec
    from umrs.server.workflow_execution_registry import WorkflowExecutionRegistry
    from umrs.workflow import pg_registry

    mapping = {"grant_id": "c" * 32, "run_spec_b64": "e30=", "run_spec_id": "d" * 64, "workflow_steps_b64": "e30="}
    assert intent_from_spec({"intent": mapping}) == mapping
    record = WorkflowExecutionRegistry().create("wf", "alice", execution_id="wf-0123456789ab", intent=mapping)
    assert (record.grant_id, record.run_spec_b64, record.intent_id) == ("c" * 32, "e30=", None)
    view = record.to_dict()
    assert (view["grant_id"], view["run_spec_id"]) == ("c" * 32, "d" * 64) and "e30=" not in str(view)
    row = {"record_json": view, **mapping}
    loaded = pg_registry._record_from_row(row)
    assert (loaded.run_spec_b64, loaded.workflow_steps_b64, loaded.grant_id) == ("e30=", "e30=", "c" * 32)
    assert pg_registry._COLUMNS.endswith("grant_id, run_spec_b64, run_spec_id, workflow_steps_b64")
