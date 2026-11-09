"""The gated lane for grant runs and its ledger write points (ADR-112 waves
C/D; plan §6 C/D.0 C2 envelope, C4 "Executor", K6, D2).

Driven over the B2 gate double (``gated_gate_double.py``):

- a grant run sends the C2 envelope (grant, run spec, skeleton) and keys the
  gate's per-run state by the run key (X2): ``lookup`` / ``mkwd`` / job name;
- ``grant_exhausted`` with detail ``concurrent`` is sent again on the same
  workdir after a backoff (K6); detail ``jobs`` fails at once;
- every step writes the usage ledger: pending, submitted (with the gate log
  head's ``n``), settled from the gate's ``usage``; a submit the gate refused
  is ``not_submitted``; a budget refusal at the pending write sends nothing.
"""

from __future__ import annotations

import base64
import json

import pytest
from step_lane_harness import FakeS3
from test_gated_step_lane import (
    DATA,
    OUT,
    REPLY,
    RUN,
    SRC_SHA,
    _calls,
    _gate_state,
    _intent,
    _plan,
    _run,
    _verbs,
)
from test_gated_step_lane import gate as _lane_gate  # noqa: F401 — fixture
from umrs.hpc_accounts.usage_ledger import BudgetExceeded, get_usage_ledger
from umrs.server import gated_step_lane as lane
from umrs.server.step_executor import StepExecutionError

GRANT_ID = "c" * 32
GRANT_B64 = base64.b64encode(b'{"grant": 1}').decode()
SKELETON_B64 = base64.b64encode(b'{"steps":[]}').decode()
SIG = "-----BEGIN SSH SIGNATURE-----\nAA==\n-----END SSH SIGNATURE-----\n"


@pytest.fixture
def gate(request):
    """The gate double, its state dir and the executor (``test_gated_step_lane.gate``)."""
    return request.getfixturevalue("_lane_gate")


def _bind_grant(executor, records: list, **intent_over) -> lane.GatedRun:
    body = {**_intent(SRC_SHA), "project": "p1", **intent_over}
    spec_b64 = base64.b64encode(json.dumps(body, sort_keys=True).encode()).decode()
    run = lane.GatedRun(
        venue="site-x", username="alice", intent_b64=spec_b64, signature="", intent_id="",
        kind="grant", key=lane.grant_run_key(GRANT_ID, RUN), grant_id=GRANT_ID,
        grant_payload_b64=GRANT_B64, grant_signature=SIG, workflow_steps_b64=SKELETON_B64,
        run_spec_id="d" * 64, max_concurrent=2,
    )  # fmt: skip
    executor.bind_run(RUN, on_job=lambda _step, job: records.append(dict(job)), gated=run)
    return run


def _s3() -> FakeS3:
    return FakeS3({("data", "in.txt"): DATA})


def test_a_grant_run_sends_the_grant_envelope_under_its_run_key(gate):
    state, src, executor = gate
    _plan(state, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    records: list = []
    run = _bind_grant(executor, records)
    _run(executor, src, _s3())
    lookup, mkwd = _calls(state)[:2]
    assert lookup == ["lookup", run.key, "s1"] and mkwd == ["mkwd", run.key, "s1"]
    [envelope] = _gate_state(state)["envelopes"]
    assert envelope == {
        "grant": {"payload": GRANT_B64, "signature": SIG},
        "run_spec": run.intent_b64,
        "workflow_steps": SKELETON_B64,
    }
    last = records[-1]
    assert (last["grant_id"], last["run_key"], last["intent_id"]) == (GRANT_ID, run.key, "")
    assert last["job_name"] == f"umrs-gated-{run.key[:16]}-s1"


def test_an_endpoint_credential_is_never_part_of_a_grant_envelope():
    run = lane.GatedRun("v", "u", "e30=", "", "", kind="grant", grant_id=GRANT_ID, grant_payload_b64=GRANT_B64)
    with pytest.raises(lane.GateFailure) as info:
        run.envelope({"api_key": "k"})
    assert info.value.name == "grant_mismatch"


def test_a_concurrent_refusal_is_retried_on_the_same_workdir(gate, monkeypatch):
    """K6: it comes before the nonce and the seal; the step then passes."""
    state, src, executor = gate
    slept: list[float] = []
    monkeypatch.setattr(lane, "_backoff_sleep", slept.append)
    _plan(state, reply=REPLY, out={f"{OUT}/res.txt": "x"},
          refuse_once={"submit": ["grant_exhausted", "concurrent: 2 open of 2"]})  # fmt: skip
    _bind_grant(executor, [])
    _run(executor, src, _s3())
    assert _verbs(state)[:5] == ["lookup", "mkwd", "stage-in", "submit", "submit"]
    assert _verbs(state).count("mkwd") == 1 and slept == [30.0]


def test_exhausted_jobs_fail_at_once(gate):
    state, src, executor = gate
    _plan(state, refuse={"submit": ["grant_exhausted", "jobs: 10 of 10"]})
    _bind_grant(executor, [])
    with pytest.raises(StepExecutionError) as info:
        _run(executor, src, _s3())
    assert info.value.message.startswith("hpcaccess.gate_refused: grant_exhausted")
    assert _verbs(state).count("submit") == 1


def test_each_step_is_ledgered_pending_submitted_settled(gate):
    state, src, executor = gate
    _plan(state, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    run = _bind_grant(executor, [])
    _run(executor, src, _s3())
    ledger = get_usage_ledger()
    rows = ledger.tables["rows"]
    assert [r["state"] for r in rows] == ["pending", "submitted", "COMPLETED"]
    first, submitted, settled = rows
    assert (first["alloc_cpus"], first["alloc_gpus"], first["elapsed_s"]) == (1, 0, 120)
    assert (submitted["gate_log_seq"], settled["source"]) == (7, "gate_usage")
    assert (settled["grant_id"], settled["tag"]) == (GRANT_ID, f"umrs-gated-{run.key[:16]}-s1")
    assert settled["attempt_id"] == submitted["attempt_id"] == first["attempt_id"]
    assert settled["attempt_id"].startswith(f"{RUN}/s1/1/")
    assert (settled["project_id"], settled["access_mode"], settled["method"]) == ("p1", "user_account", "gated_key")


def test_a_refused_submit_is_not_submitted_in_the_ledger(gate):
    state, src, executor = gate
    _plan(state, refuse={"submit": ["grant_cancelled", "cancelled"]})
    _bind_grant(executor, [])
    with pytest.raises(StepExecutionError):
        _run(executor, src, _s3())
    assert [r["state"] for r in get_usage_ledger().tables["rows"]] == ["pending", "not_submitted"]


def test_a_budget_refusal_at_the_pending_write_sends_nothing(gate):
    """K14: the step's pending write binds; no stage-in, no submit."""
    state, src, executor = gate
    get_usage_ledger().tables["budgets"]["p1"] = 0.01
    _bind_grant(executor, [], steps=[{**_intent(SRC_SHA)["steps"][0], "gpus": 1, "time": 3600}])
    with pytest.raises(StepExecutionError) as info:
        _run(executor, src, _s3())
    assert "hpcaccess.budget_exceeded" in info.value.message
    assert _verbs(state) == ["lookup"] and get_usage_ledger().tables["rows"] == []
    assert BudgetExceeded.code == "hpcaccess.budget_exceeded"


def test_an_intent_run_is_ledgered_with_its_intent_id(gate):
    from test_gated_step_lane import _bind

    state, src, executor = gate
    _plan(state, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    run = _bind(executor, SRC_SHA, [])
    _run(executor, src, _s3())
    (row,) = get_usage_ledger().current("site-x")
    assert (row["intent_id"], row["grant_id"], row["state"]) == (run.intent_id, "", "COMPLETED")
