"""The gated step lane (ADR-112 B2; plan §5 B2.0 §10.5 and §10.6).

A run under the person's own account reaches the cluster only through the
person's gate: these tests drive ``RemoteSlurmStepExecutor`` with an
injected gate transport (``gated_gate_double.py`` run as a local command, in
place of ``ssh <login>@<target> '<verb line>'``) and check the verbs in
order, the bytes staged, the envelope submitted, the recovery rules, the
typed refusals, the step record and the reaper's gated leg. The helper lane
is never touched on this path: its SSH is patched to fail the test.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from step_lane_harness import FakeS3
from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.pipeline_sdk.step import StepContext
from umrs.server import gated_step_lane as lane
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor
from umrs.server.step_executor import EXECUTOR_SOURCE_DIR_METADATA_KEY, StepExecutionError
from umrs.server.step_staging import STAGE_REQUEST_METADATA_KEY

DOUBLE = str(Path(__file__).resolve().parent / "gated_gate_double.py")
RUN = "wf-0123456789ab"
OUT = "wf-runner-output"
DATA = b"input bytes"
DATA_SHA = hashlib.sha256(DATA).hexdigest()
SRC = b"class Upper: pass\n"


def _intent(src_sha: str, entrypoint: str = "steps.m:Upper") -> dict:
    return {
        "login": "alice_hpc",
        "account": "grp-a",
        "run_id": RUN,
        "output_bucket": OUT,
        "src_manifest": {
            "files": [{"path": "steps/m.py", "sha256": src_sha, "size": len(SRC)}],
            "entrypoints": {"s1": entrypoint},
        },
        "steps": [
            {
                "name": "s1",
                "after": [],
                "partition": "gpu",
                "cpus": 1,
                "gpus": 0,
                "time": 120,
                "inputs": {
                    "objects": [
                        {
                            "mode": "listed",
                            "bucket": "data",
                            "key": "in.txt",
                            "sha256": DATA_SHA,
                            "size": len(DATA),
                        }
                    ],
                    "prefixes": [],
                    "chained": [],
                },
            }
        ],
    }


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """The gate double's state dir, the run's binding, and its executor."""
    state = tmp_path / "gate"
    state.mkdir()
    src = tmp_path / "src"
    (src / "steps").mkdir(parents=True)
    (src / "steps" / "m.py").write_bytes(SRC)
    lane.set_transport_factory(
        lambda _venue, _user, _login: lane.GateTransport(
            lambda verb, args: [sys.executable, DOUBLE, str(state), verb, *args]
        )
    )
    executor = RemoteSlurmStepExecutor(
        ssh_config=SSHControlConfig(ssh_target="login", ssh_user="op"),
        enabled=True,
        poll_interval=0,
    )

    def no_helper(*_a, **_k):
        raise AssertionError("the helper lane (the platform login) was used")

    monkeypatch.setattr(executor, "_ssh_run", no_helper)
    monkeypatch.setattr(executor, "_stream", no_helper)
    profile = {"partitions": {"gpu": {"exclusive": False, "max_cpus": 64, "max_gpus": 8}}}
    monkeypatch.setattr(
        "umrs.hpc_accounts.site_profile.published", lambda _site: (profile, b"", "sha")
    )
    yield state, src, executor
    lane.set_transport_factory(None)


def _plan(state: Path, **plan) -> None:
    (state / "plan.json").write_text(json.dumps(plan))


def _calls(state: Path) -> list[list[str]]:
    path = state / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["line"] for line in path.read_text().splitlines()]


def _verbs(state: Path) -> list[str]:
    return [line[0] for line in _calls(state)]


def _gate_state(state: Path) -> dict:
    return json.loads((state / "state.json").read_text())


def _bind(executor, src_sha: str, records: list, **intent_over) -> lane.GatedRun:
    body = {**_intent(src_sha), **intent_over}
    payload = json.dumps(body, sort_keys=True).encode()
    run = lane.GatedRun(
        venue="site-x",
        username="alice",
        intent_b64=base64.b64encode(payload).decode(),
        signature="-----BEGIN SSH SIGNATURE-----\nAA==\n-----END SSH SIGNATURE-----\n",
        intent_id=hashlib.sha256(payload).hexdigest(),
    )
    executor.bind_run(RUN, on_job=lambda _step, job: records.append(dict(job)), gated=run)
    return run


def _ctx(src: Path, s3: FakeS3, parents: dict | None = None) -> StepContext:
    return StepContext(
        s3=s3,
        input_bucket="data",
        input_keys=[],
        output_bucket=OUT,
        config={},
        metadata={
            STAGE_REQUEST_METADATA_KEY: {
                "owner": "alice",
                "step": "s1",
                "params": {},
                "stage_inputs": None,
            },
            EXECUTOR_SOURCE_DIR_METADATA_KEY: str(src),
        },
        run_id=RUN,
        parent_results=dict(parents or {}),
    )


def _run(executor, src: Path, s3: FakeS3, entrypoint: str = "steps.m:Upper"):
    return executor.execute(entrypoint=entrypoint, config={}, ctx=_ctx(src, s3))


SRC_SHA = hashlib.sha256(SRC).hexdigest()
REPLY = {
    "status": "ok",
    "result": {"output_keys": ["s3://wf-runner-output/res.txt"], "extra": {"slurm": "forged"}},
}


def test_a_gated_step_drives_the_gate_verbs_in_order(gate):
    state, src, executor = gate
    _plan(
        state,
        states=["PENDING", "RUNNING", "COMPLETED"],
        reply=REPLY,
        out={f"{OUT}/res.txt": "result"},
    )
    records: list = []
    run = _bind(executor, SRC_SHA, records)
    s3 = FakeS3({("data", "in.txt"): DATA})
    result = _run(executor, src, s3)
    assert _verbs(state) == [
        "lookup",
        "mkwd",
        "stage-in",
        "submit",
        "status",
        "status",
        "status",
        "read-reply",
        "collect-out",
        "usage",
        "rmwd",
    ]
    lookup, mkwd = _calls(state)[:2]
    assert lookup == ["lookup", run.intent_id, "s1"] and mkwd == ["mkwd", run.intent_id, "s1"]
    # The reply's key is the physical key the outputs were stored under.
    assert result.output_keys == [f"s3://{OUT}/runs/{RUN}/s1/res.txt"]
    assert s3.objects[(OUT, f"runs/{RUN}/s1/res.txt")] == b"result"
    # The step cannot forge the measured accounting: the gate's usage wins.
    assert result.extra["slurm"]["gpus"] == 2 and result.extra["slurm"]["elapsed_seconds"] == 120


def test_stage_in_carries_the_manifest_and_each_signed_blob_once(gate):
    state, src, executor = gate
    _plan(state, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    _bind(executor, SRC_SHA, [])
    _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    [wd] = _gate_state(state)["wds"].values()
    assert wd["members"] == [["src/steps/m.py", len(SRC)], [f"in/blobs/{DATA_SHA}", len(DATA)]]


def test_submit_sends_the_signed_payload_and_signature_and_nothing_else(gate):
    state, src, executor = gate
    _plan(state, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    run = _bind(executor, SRC_SHA, [])
    _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert _gate_state(state)["envelopes"] == [
        {"payload": run.intent_b64, "signature": run.signature}
    ]


def test_the_step_record_names_the_gated_transport_intent_and_log_head(gate):
    state, src, executor = gate
    _plan(state, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    records: list = []
    run = _bind(executor, SRC_SHA, records)
    _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    last = records[-1]
    assert last["transport"] == "gated:site-x:alice"
    assert (last["intent_id"], last["gate_log"]) == (run.intent_id, {"n": 7, "h": "ab" * 32})
    assert (last["access_mode"], last["site_login"], last["slurm_account"]) == (
        "user_account",
        "alice_hpc",
        "grp-a",
    )
    assert last["workdir"].startswith(run.intent_id[:16] + "-s1-") and last["cleaned"] is True


def test_a_gate_refusal_fails_the_step_by_name_and_is_not_retried(gate):
    state, src, executor = gate
    _plan(state, refuse={"submit": ["input_mismatch", "blob sizes differ"]})
    _bind(executor, SRC_SHA, [])
    with pytest.raises(StepExecutionError) as info:
        _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert info.value.message.startswith("hpcaccess.gate_refused: input_mismatch")
    assert _verbs(state).count("submit") == 1


def test_a_lost_submit_without_a_nonce_starts_over_with_a_fresh_workdir(gate):
    state, src, executor = gate
    _plan(state, fail={"submit": 255}, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    _bind(executor, SRC_SHA, [])
    _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert _verbs(state)[:8] == [
        "lookup",
        "mkwd",
        "stage-in",
        "submit",
        "lookup",
        "rmwd",
        "mkwd",
        "stage-in",
    ]
    assert _verbs(state).count("submit") == 2


def test_a_lost_submit_and_unreachable_lookup_leave_pending(gate, monkeypatch):
    from umrs.hpc_accounts.usage_ledger import get_usage_ledger

    state, src, executor = gate
    executor.unreachable_budget = 0
    original = lane.GateTransport.call
    lookups: list[str] = []
    submits: list[str] = []

    def uncertain(self, verb, *args, **kwargs):
        if verb == "submit":
            submits.append(verb)
            return subprocess.CompletedProcess([verb], 255, b"", b"connection lost")
        if verb == "lookup":
            lookups.append(verb)
            if len(lookups) > 1:
                return subprocess.CompletedProcess([verb], -1, b"", b"still unreachable")
        return original(self, verb, *args, **kwargs)

    monkeypatch.setattr(lane.GateTransport, "call", uncertain)
    _bind(executor, SRC_SHA, [])
    with pytest.raises(StepExecutionError):
        _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert len(submits) == 1 and len(lookups) == 2
    assert [r["state"] for r in get_usage_ledger().tables["rows"]] == ["pending"]
    assert "rmwd" not in _verbs(state)


def test_a_failed_stage_in_is_abandoned_for_a_fresh_workdir(gate):
    state, src, executor = gate
    _plan(state, fail={"stage-in": 255}, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    _bind(executor, SRC_SHA, [])
    _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert _verbs(state)[:5] == ["lookup", "mkwd", "stage-in", "rmwd", "mkwd"]


def test_an_earlier_attempts_job_is_adopted_never_submitted_twice(gate):
    state, src, executor = gate
    _plan(
        state,
        nonce={"wd": "w-1", "job_id": "777", "state": "RUNNING"},
        reply=REPLY,
        out={f"{OUT}/res.txt": "x"},
    )
    _bind(executor, SRC_SHA, [])
    _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert "submit" not in _verbs(state) and "mkwd" not in _verbs(state)
    assert ["status", "777"] in _calls(state)


def test_a_submit_the_gate_cannot_resolve_fails_submit_unknown(gate):
    state, src, executor = gate
    _plan(state, nonce={"wd": "w-1", "job_id": "", "state": "UNKNOWN"})
    _bind(executor, SRC_SHA, [])
    with pytest.raises(StepExecutionError) as info:
        _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert "hpcaccess.gate_refused: submit_unknown" in info.value.message
    assert "submit" not in _verbs(state)


def test_a_changed_entrypoint_sends_nothing(gate):
    state, src, executor = gate
    _bind(executor, SRC_SHA, [])
    with pytest.raises(StepExecutionError) as info:
        _run(executor, src, FakeS3({("data", "in.txt"): DATA}), entrypoint="steps.m:Other")
    assert "entrypoint_changed" in info.value.message and _calls(state) == []


def test_a_routed_endpoint_the_intent_did_not_sign_sends_nothing(gate):
    from umrs.server.step_executor import EXECUTOR_ENDPOINT_METADATA_KEY

    state, src, executor = gate
    _bind(executor, SRC_SHA, [])
    ctx = _ctx(src, FakeS3({("data", "in.txt"): DATA}))
    ctx.metadata[EXECUTOR_ENDPOINT_METADATA_KEY] = {
        "api_base": "http://10.0.0.5:8000/v1",
        "kind": "vllm",
    }
    with pytest.raises(StepExecutionError) as info:
        executor.execute(entrypoint="steps.m:Upper", config={}, ctx=ctx)
    assert "endpoint_changed" in info.value.message and _calls(state) == []


def test_the_platform_key_never_rides_as_an_endpoint_credential(gate, monkeypatch):
    from umrs.server.step_executor import EXECUTOR_ENDPOINT_METADATA_KEY

    state, src, executor = gate
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    _bind(
        executor,
        SRC_SHA,
        [],
        steps=[
            {
                **_intent(SRC_SHA)["steps"][0],
                "endpoint": {"host": "10.0.0.5", "port": 8000, "kind": "vllm"},
            }
        ],
    )
    ctx = _ctx(src, FakeS3({("data", "in.txt"): DATA}))
    ctx.metadata[EXECUTOR_ENDPOINT_METADATA_KEY] = {
        "api_base": "http://10.0.0.5:8000/v1",
        "kind": "vllm",
        "api_key": "sk-platform",
    }
    with pytest.raises(StepExecutionError) as info:
        executor.execute(entrypoint="steps.m:Upper", config={}, ctx=ctx)
    assert "endpoint_credential_not_scoped" in info.value.message and _calls(state) == []


def test_a_changed_input_fails_before_the_gate_sees_a_torn_blob(gate):
    state, src, executor = gate
    _bind(executor, SRC_SHA, [])
    with pytest.raises(StepExecutionError) as info:
        _run(executor, src, FakeS3({("data", "in.txt"): b"other bytes"}))
    assert "input_changed" in info.value.message
    assert "submit" not in _verbs(state)


def test_a_switched_off_lane_refuses_a_gated_step_before_any_call(gate):
    state, src, executor = gate
    executor.enabled = False
    _bind(executor, SRC_SHA, [])
    with pytest.raises(StepExecutionError, match="remote_enabled"):
        _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert _calls(state) == []


# ── the reaper's gated leg ───────────────────────────────────────────────────


def test_the_reaper_cancels_a_live_gated_job_through_the_persons_gate(gate):
    state, _src, executor = gate
    _plan(state, states=["RUNNING"])
    facts = {"transport": "gated:site-x:alice", "workdir": "w-9", "job_id": "991"}
    assert executor.reap_step(RUN, "s1", facts) is False  # cancelled, not yet ended
    assert _calls(state) == [["status", "991"], ["cancel", "991"]]


def test_the_reaper_removes_the_workdir_of_an_ended_gated_job(gate):
    state, _src, executor = gate
    _plan(state, states=["CANCELLED"])
    facts = {"transport": "gated:site-x:alice", "workdir": "w-9", "job_id": "991"}
    assert executor.reap_step(RUN, "s1", facts) is True
    assert _calls(state) == [["status", "991"], ["rmwd", "w-9"]]


def test_the_reaper_counts_a_workdir_the_gate_already_swept_as_done(gate):
    """Fixer finding 2026-11-08: the gate's daily sweep removes idle
    workdirs; ``rmwd`` then exits 3 (absent), which is done, not a failure
    retried on every boot."""
    state, _src, executor = gate
    _plan(state, states=["COMPLETED"], fail={"rmwd": 3})
    facts = {"transport": "gated:site-x:alice", "workdir": "w-9", "job_id": "991"}
    assert executor.reap_step(RUN, "s1", facts) is True
    assert _calls(state) == [["status", "991"], ["rmwd", "w-9"]]


def test_the_reaper_counts_a_job_the_gate_pruned_as_ended(gate):
    """``status`` refuses ``not_mine`` once the gate pruned the job's state
    (window + 7 days): nothing to cancel; the workdir is removed."""
    state, _src, executor = gate
    _plan(state, refuse={"status": ["not_mine", "job 991 was not submitted by this gate"]})
    facts = {"transport": "gated:site-x:alice", "workdir": "w-9", "job_id": "991"}
    assert executor.reap_step(RUN, "s1", facts) is True
    assert _calls(state) == [["status", "991"], ["rmwd", "w-9"]]


def test_the_reaper_leaves_a_live_gated_job_for_a_retry(gate):
    state, _src, executor = gate
    _plan(state, states=["RUNNING"])
    facts = {"transport": "gated:site-x:alice", "workdir": "w-9", "job_id": "991"}
    assert executor.reap_step(RUN, "s1", facts, for_retry=True) is False
    assert _calls(state) == [["status", "991"]]


def test_an_unreachable_gate_is_logged_and_left(gate, caplog):
    def gone(_venue, _user, _login):
        raise lane.GateFailure("not_enrolled", "no key", "setup")

    lane.set_transport_factory(gone)
    facts = {"transport": "gated:site-x:alice", "workdir": "w-9", "job_id": "991"}
    _state, _src, executor = gate
    assert executor.reap_step(RUN, "s1", facts) is False
    assert "gate unreachable" in caplog.text


def test_reaper_lane_filter_keeps_this_venues_gated_jobs(monkeypatch):
    from umrs.server import step_reaper

    monkeypatch.setattr(step_reaper, "_this_venue", lambda: "site-x")
    here = {"transport": "gated:site-x:alice", "workdir": "w-1", "job_id": "1"}
    there = {"transport": "gated:site-y:alice", "workdir": "w-2", "job_id": "2"}
    helper = {"transport": "helper", "workdir": "/lvs/root/run-s1-1-x", "job_id": "3"}
    assert step_reaper._in_lane(here, "/lvs/root") is True
    assert step_reaper._in_lane(there, "/lvs/root") is False
    assert step_reaper._in_lane(helper, "/lvs/root") is True
    assert step_reaper._in_lane(here, "") is True  # a gated-only reaper


def test_the_daily_sweep_asks_each_recent_active_person(gate, monkeypatch):
    from types import SimpleNamespace

    from umrs.hpc_accounts.store import InMemorySiteAccountStore
    from umrs.server import step_reaper
    from umrs.server.workflow_execution_registry import StepProgress

    state, _src, _executor = gate
    monkeypatch.setattr(step_reaper, "_this_venue", lambda: "site-x")
    store = InMemorySiteAccountStore("site-x")
    for user in ("alice", "bob"):
        store.issue_key(user, "tk-000000000001", "SHA256:x")
    store.record_verified(
        "alice",
        "tk-000000000001",
        {
            "site_login": "a",
            "verified_login": "a",
            "verified_assocs": [],
            "gate_version": "2",
            "gate_sha256": "",
            "signer_fingerprints": [],
            "uv_required": False,
        },
    )
    step = StepProgress(
        name="s1",
        extra={
            "slurm_jobs": {
                "w-1": {"transport": "gated:site-x:alice"},
                "w-2": {"transport": "gated:site-x:bob"},
            }
        },
    )
    record = SimpleNamespace(submitted_at=1000.0, steps={"s1": step})
    swept = step_reaper.sweep_gated_accounts([record], now=2000.0, store=store)
    assert list(swept) == ["alice"]  # bob's account is not active
    assert _calls(state) == [["sweep", "--older-than", str(step_reaper.GATED_SWEEP_AGE_SECONDS)]]


def test_a_gated_only_site_gets_a_reaper_that_never_sweeps_with_the_helper(monkeypatch):
    from types import SimpleNamespace

    from umrs.server import remote_slurm_step_executor as rse
    from umrs.server import step_reaper

    for name in ("UMRS_STEP_HELPER", "UMRS_SLURM_WORKDIR_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rse, "_site_ssh_config", lambda: SSHControlConfig(ssh_target="login"))
    monkeypatch.setattr(step_reaper, "_allows_gated_runs", lambda: True)
    lane_site = SimpleNamespace(remote_enabled=True, helper_path="", workdir_root="")
    monkeypatch.setattr(step_reaper, "_site_step_executor", lambda: lane_site)
    reaper = step_reaper._build_reaper()
    assert reaper is not None and (reaper.helper_path, step_reaper.lane_root(reaper)) == ("", "")
    assert reaper.sweep_workdirs() == {"removed": []}
    lane_site.remote_enabled = False  # not yet onboarded: no reaper at all
    assert step_reaper._build_reaper() is None


def test_transport_tags_round_trip():
    assert lane.parse_transport_tag(lane.transport_tag("site-x", "a.b")) == ("site-x", "a.b")
    assert lane.parse_transport_tag("helper") is None
    assert lane.parse_transport_tag("gated:site-x") is None


def test_feed_stage_in_dedupes_a_digest_required_twice(tmp_path):
    src = tmp_path / "src"
    (src / "steps").mkdir(parents=True)
    (src / "steps" / "m.py").write_bytes(SRC)
    intent = _intent(SRC_SHA)
    step = intent["steps"][0]
    step["inputs"]["objects"].append({**step["inputs"]["objects"][0], "key": "copy.txt"})
    sink = io.BytesIO()
    s3 = FakeS3({("data", "in.txt"): DATA, ("data", "copy.txt"): DATA})
    lane.feed_stage_in(sink, _ctx(src, s3), intent, step, str(src))
    sink.seek(0)
    with tarfile.open(fileobj=sink) as tar:
        assert tar.getnames() == ["src/steps/m.py", f"in/blobs/{DATA_SHA}"]


# ── the account cascade's gated leg (§10.6) ──────────────────────────────────


def _records_with(*jobs: dict) -> list:
    from types import SimpleNamespace

    step = SimpleNamespace(extra={"slurm_jobs": {str(i): job for i, job in enumerate(jobs)}})
    return [SimpleNamespace(steps={"s1": step})]


def test_the_cascade_cancels_the_persons_live_gated_jobs_at_this_venue(gate):
    state, _src, _executor = gate
    records = _records_with(
        {"transport": "gated:site-x:alice", "job_id": "991", "state": "RUNNING"},
        {"transport": "gated:site-x:alice", "job_id": "992", "state": "COMPLETED"},
        {"transport": "gated:site-x:alice", "job_id": "993", "state": "CANCELLED by 7"},
        {"transport": "gated:site-x:bob", "job_id": "994", "state": "PENDING"},
        {"transport": "gated:site-y:alice", "job_id": "995", "state": "PENDING"},
        {"transport": "helper", "job_id": "996", "state": "RUNNING"},
        {"transport": "gated:site-x:alice", "job_id": "", "state": ""},
    )
    assert lane.cancel_person_jobs("site-x", "alice", records) == 1
    assert _calls(state) == [["cancel", "991"]]


def test_the_cascade_leg_counts_what_the_gate_cancelled(gate):
    from umrs.admin import account_cascade as ac

    state, _src, _executor = gate
    records = _records_with(
        {"transport": "gated:site-x:alice", "job_id": "991", "state": "PENDING"}
    )
    result = ac.CascadeResult(user="alice", project="", reason="r", trigger="t", venue="site-x")
    ac.cancel_gated_jobs(result, records)
    assert result.gated_jobs_cancelled == 1 and _calls(state) == [["cancel", "991"]]
    assert result.details()["gated_jobs_cancelled"] == 1


def test_a_project_removal_cancels_no_gated_job(gate):
    from umrs.admin import account_cascade as ac

    state, _src, _executor = gate
    records = _records_with(
        {"transport": "gated:site-x:alice", "job_id": "991", "state": "PENDING"}
    )
    result = ac.CascadeResult(user="alice", project="p1", reason="r", trigger="t", venue="site-x")
    ac.cancel_gated_jobs(result, records)
    assert result.gated_jobs_cancelled == 0 and _calls(state) == []


def test_run_cascade_cancels_gated_jobs_before_it_destroys_the_keys(monkeypatch):
    from umrs.admin import account_cascade as ac

    order: list = []
    monkeypatch.setattr(ac, "cancel_gated_jobs", lambda _result: order.append("cancel"))
    monkeypatch.setattr(ac, "end_site_accounts", lambda _store, _result: order.append("keys"))

    class _Broker:
        venue = "site-x"

        def revoke_by_owner(self, *_a, **_k):
            return []

    ac.run_cascade(
        "alice",
        project=None,
        reason="r",
        actor="a",
        trigger="t",
        broker=_Broker(),
        site_accounts=object(),
    )
    assert order == ["cancel", "keys"]


def test_an_unreachable_gate_cancels_nothing_and_never_raises(gate, caplog):
    def gone(_venue, _user, _login):
        raise lane.GateFailure("not_enrolled", "no key", "setup")

    lane.set_transport_factory(gone)
    records = _records_with(
        {"transport": "gated:site-x:alice", "job_id": "991", "state": "RUNNING"}
    )
    assert lane.cancel_person_jobs("site-x", "alice", records) == 0
    assert "gate unreachable" in caplog.text


# ── one attempt per gated step (§10.5) ───────────────────────────────────────


class _Refusing:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **_k):
        self.calls += 1
        raise StepExecutionError("worker", "hpcaccess.gate_refused: image_mismatch: x")


@pytest.mark.parametrize(("intent_id", "attempts"), [("ab" * 32, 1), (None, 3)])
def test_a_gated_step_takes_one_attempt_whatever_its_retry_policy(intent_id, attempts):
    import threading
    from types import SimpleNamespace

    from umrs.server import workflow_runner as wr
    from umrs.server.workflow_execution_registry import ExecutionRecord

    executor = _Refusing()
    record = ExecutionRecord(
        execution_id=RUN, workflow_name="w", user_id="alice", intent_id=intent_id
    )
    result, error, used, _ms = wr._execute_with_retry(
        executor=executor,
        spec={"entrypoint": "m:S", "max_retries": 2, "retry_delay": 0},
        config={},
        ctx=None,
        record=record,
        step_name="s1",
        decision=SimpleNamespace(endpoint_id=""),
        routing_log={},
        state_lock=threading.Lock(),
    )
    assert (result, used, executor.calls) == (None, attempts, attempts)
    assert "image_mismatch" in error


# ── a stage-in stream the server could not finish (§10.5) ────────────────────


def test_a_stream_the_server_broke_off_is_a_fresh_workdir_not_a_refusal(gate, monkeypatch):
    state, src, executor = gate
    _plan(state, reply=REPLY, out={f"{OUT}/res.txt": "x"})
    _bind(executor, SRC_SHA, [])
    dropped: list = []
    original = lane.feed_stage_in

    def drop_once(sink, *args):
        if not dropped:
            dropped.append(True)
            raise OSError("the object store connection dropped")
        return original(sink, *args)

    monkeypatch.setattr(lane, "feed_stage_in", drop_once)
    _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert _verbs(state)[:6] == ["lookup", "mkwd", "stage-in", "rmwd", "mkwd", "stage-in"]


def test_a_gate_refusal_of_a_whole_stream_is_never_retried(gate):
    state, src, executor = gate
    _plan(state, refuse={"stage-in": ["digest_mismatch", "in/blobs/x"]})
    _bind(executor, SRC_SHA, [])
    with pytest.raises(StepExecutionError, match="gate_refused: digest_mismatch"):
        _run(executor, src, FakeS3({("data", "in.txt"): DATA}))
    assert _verbs(state).count("stage-in") == 1
