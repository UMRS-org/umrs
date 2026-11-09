"""End to end: standing grants and the usage ledger as ONE system (ADR-112
waves C and D; plan §6, the C/D.0 wire contract).

The integrator's harness. It extends the B2 world of
``test_gated_step_lane_e2e`` (the person's REAL gate installed by r4's own
install script into a temp HPC home, a fake sshd that runs the forced
command, fake Slurm and podman, the real step worker in JSON mode) with r4's
OWN grant flow: ``r4.hpc_grants.run_grant`` asks the server's
``grant-proposal`` route, reads the export, recomputes every digest, builds
the grant, signs it with a real ``ssh-keygen`` key made in ``tmp_path`` and
stores it through ``POST /grants`` (a live ``grant-check`` at the gate).
From then on the server starts runs with no further signature:

- two runs of the granted version pass; a third is refused by the gate
  when ``max_jobs`` is spent;
- an ungranted version and out-of-schema params are refused by the server's
  admission AND, when a server skips that check, by the gate;
- ``chained`` inputs pass only from the seed and from this grant's own
  gate-recorded outputs, never from another grant's;
- r4's ``grant-cancel`` stops every further submit;
- every submit is in the usage ledger: pending, submitted, then settled
  from the gate's ``usage``; a project budget below the reservation refuses
  ``hpcaccess.budget_exceeded`` before anything is sent;
- a service-account staged run (the platform login) is ledgered too;
- two concurrent submits under ``max_concurrent: 1`` admit exactly one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_gated_step_lane_e2e import (
    HEADERS,
    INPUT,
    PROJECT,
    SITE,
    USER,
    VERSION,
    WORKFLOW,
    R4Client,
    _error,
    _executor,
    _no_run,
    _ok,
    _script,
    _tools_ok,
)
from test_gated_step_lane_e2e import world as _e2e_world  # noqa: F401 — the B2 world fixture
from umrs.hpc_accounts import grant_service
from umrs.hpc_accounts.grants import grant_store_for, signed_body
from umrs.hpc_accounts.usage_ledger import LedgerUnit, Resources, get_usage_ledger
from umrs.server import workflow_runner as wr
from umrs.server.grant_admission import compose_run_spec, encode_run
from umrs.server.run_intent_admission import admit_run
from umrs.server.workflow_execution_registry import ExecutionRecord

hpc_gate = pytest.importorskip("r4_contracts.hpc_gate")
hpc_grants = pytest.importorskip("r4.hpc_grants")
hpc_intents = pytest.importorskip("r4.hpc_intents")

pytestmark = pytest.mark.skipif(
    not _tools_ok(), reason="needs /usr/bin/python3 >= 3.9, ssh-keygen, bash and sha256sum"
)

#: The granted workflow declares its one parameter (a grant carries no
#: parameter the workflow does not declare; ``maxLength`` bounds it).
PARAMETERS = {
    "type": "object",
    "required": ["input_key"],
    "properties": {"input_key": {"type": "string", "maxLength": 200, "description": "the input"}},
}
OTHER = b"another grant's seed"


def _produced(data: bytes) -> bytes:
    """What a run over *data* writes as its second step's output."""
    return data.upper() + b"!"


# ── the world: the B2 world, the workflow registered with a schema ──────────


def _finder(w: Any) -> Any:
    """``grant_service.find_version`` over the world's one registered version."""

    def find(name: str, version: Any, _user: str, **_kw: Any) -> Any:
        if name != WORKFLOW or version not in (None, VERSION):
            return None
        return grant_service.Version(WORKFLOW, VERSION, w.entry, str(w.source), w.workflow_json())

    return find


@pytest.fixture
def world(request, monkeypatch):
    """The B2 world; ``workflow.json`` declares ``input_key``; the server
    finds the version through ``grant_service.find_version``."""
    w = request.getfixturevalue("_e2e_world")
    workflow = w.workflow_json()
    workflow["parameters"] = PARAMETERS
    (w.source / "workflow.json").write_text(json.dumps(workflow, indent=2))
    monkeypatch.setattr(grant_service, "find_version", _finder(w))
    return w


class GrantClient(R4Client):
    """r4's ``Client`` grant routes, answered by the server's own routes."""

    def hpc_grant_proposal(self, body: dict[str, Any]) -> dict[str, Any]:
        return _ok(self.world.api.post("/umrs/sites/me/hpc/grant-proposal", json=body, headers=HEADERS))

    def hpc_grant_store(self, envelope: dict[str, Any]) -> dict[str, Any]:
        return _ok(self.world.api.post("/umrs/sites/me/hpc/grants", json=envelope, headers=HEADERS))

    def hpc_grant_cancel(self, grant_id: str, envelope: dict[str, Any]) -> dict[str, Any]:
        path = f"/umrs/sites/me/hpc/grants/{grant_id}/cancel"
        return _ok(self.world.api.post(path, json=envelope, headers=HEADERS))


def _seed_file(w: Any, data: bytes) -> Path:
    path = w.cluster.tmp / "seeds" / hashlib.sha256(data).hexdigest()[:16]
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path


def _sign_grant(w: Any, *, seed: bytes = INPUT, max_jobs: int = 4, max_concurrent: int | None = None) -> str:
    """``r4 hpc grant ... --inputs chained --listed-input <seed> --yes``: the
    whole C5 flow against this server; the grant's id."""
    request = hpc_grants.GrantRequest(
        site=SITE, project=PROJECT, workflows=((WORKFLOW, VERSION),), max_jobs=max_jobs,
        max_gpu_hours=0, days=1, max_concurrent=max_concurrent, inputs="chained",
        listed_paths=(_seed_file(w, seed),), yes=True,
    )  # fmt: skip
    shown: list[str] = []
    answer = hpc_grants.run_grant(GrantClient(w), request, confirm=lambda _q: False, out=shown.append)
    assert hpc_grants.data_rule({"mode": "chained", "listed": [seed]}) in "\n".join(shown)
    view = answer["grant"]
    assert (view["status"], view["covers_now"], view["stale"]) == ("active", True, [])
    return str(answer["grant_id"])


def _stored_body(w: Any, grant_id: str) -> dict[str, Any]:
    return signed_body(grant_store_for(w.store).get(USER, grant_id))


# ── the server's side: admission, then the runner through the gate ──────────


def _target(w: Any) -> Any:
    return SimpleNamespace(entry=w.entry, source_path=str(w.source), workflow_json=w.workflow_json())


def _run_ref(version: str = VERSION) -> dict[str, Any]:
    return {"workflow": WORKFLOW, "version": version, "project": PROJECT, "principal": USER}


def _admit(w: Any, params: dict[str, Any], grant_id: str = "", version: str = VERSION) -> tuple[Any, Any]:
    """The trigger's admission: ``(admission, None)`` or ``(None, error JSON)``."""
    payload = {"grant_id": grant_id} if grant_id else {}
    return admit_run(w.identity(), payload, params, _run_ref(version), _no_run, _target(w))


def _admitted(w: Any, key: str, grant_id: str = "") -> Any:
    admission, error = _admit(w, {"input_key": key}, grant_id)
    assert error is None, error
    assert admission.grant is not None
    return admission.grant


def _run(w: Any, gated: Any, **executor: Any) -> ExecutionRecord:
    """The runner drives every step of the admitted run through the gate."""
    record = ExecutionRecord(
        execution_id=gated.run_id, workflow_name=WORKFLOW, user_id=USER,
        params=dict(gated.body["params"]), **gated.spec(),
    )  # fmt: skip
    wr.run_workflow_record(
        record, source_path=str(w.source), workflow_json=w.workflow_json(),
        executor=_executor(**executor), project_id=PROJECT, user_id=USER, verified_project=PROJECT,
    )  # fmt: skip
    return record


def _unchecked(w: Any, grant_id: str, params: dict[str, Any], version: str = VERSION) -> Any:
    """A run spec composed as admission composes it, WITHOUT its coverage
    pre-check (a server that skipped it): the gate is the last word."""
    body = compose_run_spec(
        _target(w), w.identity(), _run_ref(version), {**params, "project_id": PROJECT},
        _stored_body(w, grant_id),
    )  # fmt: skip
    return encode_run(grant_id, body, w.workflow_json())


def _refused(w: Any, record: ExecutionRecord, *words: str) -> None:
    assert record.status == "failed"
    for word in words:
        assert word in _error(record), _error(record)


def _submits(w: Any, grant_id: str) -> list[str]:
    entries = w.cluster.log_entries()
    return [e["detail"] for e in entries if e["verb"] == "submit" and f"grant={grant_id}" in e["detail"]]


# ── the grant, its runs and its job cap ─────────────────────────────────────


def test_r4_signs_once_and_the_server_runs_the_granted_version_until_max_jobs(world, monkeypatch):
    signed: list[str] = []
    original = hpc_grants.sign

    def counting(payload: bytes, namespace: str, key: Any) -> str:
        signed.append(namespace)
        return original(payload, namespace, key)

    monkeypatch.setattr(hpc_grants, "sign", counting)
    grant_id = _sign_grant(world, max_jobs=4)
    first = _run(world, _admitted(world, "data/in.txt"))
    assert first.status == "succeeded", _error(first)
    second = _run(world, _admitted(world, "data/in.txt"))
    assert second.status == "succeeded", _error(second)
    third = _run(world, _admitted(world, "data/in.txt"))
    _refused(world, third, "gate_refused: grant_exhausted", "jobs")
    assert signed == [hpc_gate.namespace("grant", world.site.scheduler.hpc_access.user_account.signing_domain)]
    assert len(world.cluster.calls("sbatch")) == 4
    details = _submits(world, grant_id)
    assert len(details) == 4 and all("gpu_s=0 core_s=60" in d for d in details), details
    assert hpc_gate.verify_log(str(world.cluster.gate / "log" / "gate.jsonl"))


def test_the_witness_counts_the_grant_r4_signed_and_flags_nothing(world):
    grant_id = _sign_grant(world, max_jobs=4)
    assert _run(world, _admitted(world, "data/in.txt")).status == "succeeded"
    data = (world.cluster.gate / "log" / "gate.jsonl").read_bytes()
    report = hpc_intents.witness_site(SITE, data)
    assert report.chain_ok and not report.forked
    assert (report.unknown_grants, report.over_cap) == ([], [])
    assert {(s.grant_id, s.step) for s in report.grant_submits} == {(grant_id, "first"), (grant_id, "second")}


# ── what the grant does not cover: refused by the server and by the gate ────


def test_an_ungranted_version_is_refused_by_the_server_and_by_the_gate(world):
    grant_id = _sign_grant(world)
    _admission, error = _admit(world, {"input_key": "data/in.txt"}, version="2.0.0")
    assert "hpcaccess.signature_required" in error  # no grant covers 2.0.0: nothing chosen
    _admission, error = _admit(world, {"input_key": "data/in.txt"}, grant_id, version="2.0.0")
    assert "hpcaccess.grant_invalid" in error and "workflow" in error
    record = _run(world, _unchecked(world, grant_id, {"input_key": "data/in.txt"}, version="2.0.0"))
    _refused(world, record, "grant_mismatch", "is not granted")
    assert world.cluster.calls("sbatch") == []


def test_params_outside_the_granted_schema_are_refused_by_the_server_and_by_the_gate(world):
    grant_id = _sign_grant(world)
    _admission, error = _admit(world, {"input_key": "data/in.txt", "extra": 1})
    assert "hpcaccess.grant_invalid" in error and "extra" in error
    _admission, error = _admit(world, {"input_key": "k" * 201})
    assert "hpcaccess.grant_invalid" in error and "longer than 200" in error  # checked before any read
    _admission, error = _admit(world, {"input_key": "data/missing.txt"})
    assert "hpcaccess.grant_invalid" in error and "inputs:" in error  # typed, not a crash
    record = _run(world, _unchecked(world, grant_id, {"input_key": "data/in.txt", "extra": 1}))
    _refused(world, record, "grant_mismatch", "params: extra")
    assert world.cluster.calls("sbatch") == []


# ── chained inputs: the seed and this grant's own gate-recorded outputs ─────


def test_chained_inputs_pass_only_from_the_grants_own_gate_recorded_outputs(world):
    mine = _sign_grant(world, max_jobs=10)
    theirs = _sign_grant(world, seed=OTHER, max_jobs=10)
    _admission, error = _admit(world, {"input_key": "data/in.txt"})
    assert "hpcaccess.signature_required" in error and mine in error and theirs in error
    world.s3.objects[("wf-runner", "data/other.txt")] = OTHER
    assert _run(world, _admitted(world, "data/other.txt", theirs)).status == "succeeded"
    assert _run(world, _admitted(world, "data/in.txt", mine)).status == "succeeded"
    world.s3.objects[("wf-runner", "data/from-mine.txt")] = _produced(INPUT)
    own = _run(world, _admitted(world, "data/from-mine.txt", mine))
    assert own.status == "succeeded", _error(own)
    world.s3.objects[("wf-runner", "data/from-theirs.txt")] = _produced(OTHER)
    before = len(world.cluster.calls("sbatch"))
    foreign = _run(world, _admitted(world, "data/from-theirs.txt", mine))
    _refused(world, foreign, "grant_mismatch", "was not produced under this grant")
    assert len(world.cluster.calls("sbatch")) == before == 6


# ── grant-cancel ─────────────────────────────────────────────────────────────


def _cancel_log(w: Any, grant_id: str) -> list[tuple[str, bool]]:
    cancels = [e for e in w.cluster.log_entries() if e["verb"] == "grant-cancel"]
    return [(e["result"], f"grant={grant_id}" in e["detail"]) for e in cancels]


def test_grant_cancel_from_r4_stops_every_further_submit(world):
    grant_id = _sign_grant(world)
    waiting = _admitted(world, "data/in.txt")  # admitted before the cancel
    result = hpc_grants.cancel_grant(GrantClient(world), grant_id, None)
    assert (result.relayed, result.fork_note) == (True, None)
    view = result.answer["grant"]
    assert (view["status"], view["ended_reason"], view["gate_cancel"]) == ("cancelled", "cancelled", "recorded")
    _refused(world, _run(world, waiting), "gate_refused: grant_cancelled")
    assert world.cluster.calls("sbatch") == []
    _admission, error = _admit(world, {"input_key": "data/in.txt"}, grant_id)
    assert "hpcaccess.grant_invalid" in error
    assert "cancelled" in error
    _admission, error = _admit(world, {"input_key": "data/in.txt"})
    assert "hpcaccess.signature_required" in error
    assert _cancel_log(world, grant_id) == [("ok", True)]


# ── the usage ledger ─────────────────────────────────────────────────────────


def _history(attempt: str) -> list[tuple[str, str]]:
    rows = [
        r
        for r in get_usage_ledger().tables["rows"]
        if r["attempt_id"] == attempt or r["attempt_id"].startswith(f"{attempt}/")
    ]
    return [(r["state"], r["source"]) for r in sorted(rows, key=lambda r: r["ledger_id"])]


def test_every_grant_submit_is_ledgered_pending_submitted_then_settled_from_the_gate(world):
    grant_id = _sign_grant(world)
    gated = _admitted(world, "data/in.txt")
    record = _run(world, gated)
    assert record.status == "succeeded", _error(record)
    settled = [("pending", "estimate"), ("submitted", "estimate"), ("COMPLETED", "gate_usage")]
    assert [_history(f"{gated.run_id}/{step}/1") for step in ("first", "second")] == [settled, settled]
    jobs, identities, seqs = _ledger_view(get_usage_ledger().current(SITE, person=USER))
    assert jobs == _job_ids(record)
    assert identities == {(grant_id, "gated_key", "user_account", PROJECT, 7)}
    assert seqs == [int, int]


def _ledger_view(rows: list[dict[str, Any]]) -> tuple[set[str], set[tuple[Any, ...]], list[type]]:
    """(scheduler job ids, identity columns, ``gate_log_seq`` types) of current rows."""
    jobs = {r["scheduler_job_id"] for r in rows}
    return jobs, {_identity_view(r) for r in rows}, [type(r["gate_log_seq"]) for r in rows]


def _job_ids(record: ExecutionRecord) -> set[str]:
    return {str(job["job_id"]) for step in record.steps.values() for job in step.extra["slurm_jobs"].values()}


def _identity_view(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["grant_id"], row["method"], row["access_mode"], row["project_id"], row["elapsed_s"])


def test_the_recovery_sweep_resolves_lost_writes_through_the_real_gate(world, monkeypatch):
    """A server that died between ``submit`` and its ledger writes (the
    workdir kept, as an interrupted run leaves it): the sweep's default gate
    resolver finds each job by the run key (``lookup``), then settles it
    from the gate's ``usage``."""
    from datetime import datetime, timedelta, timezone

    from umrs.hpc_accounts.ledger_sweep import GateResolver, UsageLedgerSweep
    from umrs.server import gated_step_lane as lane

    _sign_grant(world)
    gated = _admitted(world, "data/in.txt")
    monkeypatch.setattr(lane.GatedLedger, "submitted", lambda *_a: None)
    monkeypatch.setattr(lane.GatedLedger, "settled", lambda *_a: None)
    assert _run(world, gated, keep_workdir=True).status == "succeeded"
    later = datetime.now(timezone.utc) + timedelta(minutes=16)
    sweep = UsageLedgerSweep(get_usage_ledger(), SITE, GateResolver(SITE), service=None, now=lambda: later)  # type: ignore[arg-type]
    assert sweep.run_once()["submitted"] == 2
    assert sweep.run_once()["settled"] == 2
    for step in ("first", "second"):
        assert _history(f"{gated.run_id}/{step}/1") == [
            ("pending", "estimate"), ("submitted", "estimate"), ("COMPLETED", "gate_usage"),
        ]  # fmt: skip


def _reserve_elsewhere(gpu_seconds: int) -> None:
    """Another member's step at another venue holds a GPU reservation on the
    project (the budget is lifetime and summed across venues)."""
    unit = LedgerUnit(
        venue="other-venue", attempt_id="wf-elsewhere/s/1", person="bob", lane="staged_step",
        tag="umrs-step-wf-elsewhere-s", project_id=PROJECT,
    )  # fmt: skip
    get_usage_ledger().pending(unit, Resources(1, 1, gpu_seconds))


def test_a_project_budget_below_the_reservation_refuses_before_anything_is_sent(world):
    _sign_grant(world)
    admitted = _admitted(world, "data/in.txt")  # the budget allowed it then
    _reserve_elsewhere(3600)
    get_usage_ledger().tables["budgets"][PROJECT] = 0.5  # 1800 GPU-seconds < the reservation
    before = list(world.cluster.verbs())
    record = _run(world, admitted)
    _refused(world, record, "hpcaccess.budget_exceeded")
    assert "submit" not in world.cluster.verbs()[len(before):] and world.cluster.calls("sbatch") == []
    assert _history(f"{admitted.run_id}/first/1") == []  # the refused pending wrote no row
    _admission, error = _admit(world, {"input_key": "data/in.txt"})
    assert "hpcaccess.budget_exceeded" in error  # the advisory check at admission


# ── the service-account staged lane writes the same ledger ──────────────────


def test_a_service_account_staged_run_writes_ledger_rows(tmp_path, monkeypatch):
    from step_lane_harness import make_cluster, write_step_tree

    cluster, tree = make_cluster(tmp_path, monkeypatch), write_step_tree(tmp_path / "bundle")
    record = ExecutionRecord(execution_id="wf-0123456789ab", workflow_name="lane", user_id="alice", params={"project_id": "p1"})
    workflow = {
        "execution_mode": "slurm",
        "steps": [
            {"name": "parent", "entrypoint": "steps.lane:Upper", "config": {"out": "a.txt"}},
            {"name": "child", "entrypoint": "steps.lane:Upper", "depends_on": ["parent"], "config": {"out": "b.txt"}},
        ],
    }  # fmt: skip
    wr.run_workflow_record(record, source_path=str(tree), workflow_json=workflow, executor=cluster.executor())
    assert record.status == "succeeded", record.error
    for step in ("parent", "child"):
        assert _history(f"wf-0123456789ab/{step}/1") == [
            ("pending", "estimate"), ("submitted", "estimate"), ("COMPLETED", "sacct"),
        ]  # fmt: skip
    rows = get_usage_ledger().current(get_usage_ledger().tables["rows"][0]["venue"])
    assert {(r["person"], r["project_id"], r["access_mode"], r["method"], r["grant_id"]) for r in rows} == {
        ("alice", "p1", "service_account", "platform", "")
    }
    assert {r["scheduler_job_id"] for r in rows} == set(cluster.jobs())


# ── max_concurrent under a race ──────────────────────────────────────────────

HELD_SBATCH = r"""#!__PYTHON__
import os, sys, time

STARTED, RELEASE, REAL = __STARTED__, __RELEASE__, __REAL__
open(STARTED, "a").close()
deadline = time.time() + 60
while not os.path.exists(RELEASE) and time.time() < deadline:
    time.sleep(0.05)
os.execv(REAL, [REAL, *sys.argv[1:]])
"""


def _hold_sbatch(w: Any) -> tuple[Path, Path]:
    """``sbatch`` waits (its job reserved, not yet queued) until released."""
    real = w.cluster.slurm_bin / "real"
    real.mkdir(mode=0o755)
    (w.cluster.slurm_bin / "sbatch").rename(real / "sbatch")
    started, release = w.cluster.tmp / "sbatch-started", w.cluster.tmp / "sbatch-release"
    _script(w.cluster.slurm_bin / "sbatch", HELD_SBATCH, STARTED=str(started), RELEASE=str(release), REAL=str(real / "sbatch"))
    return started, release


def _wait_for(path: Path, seconds: float = 60.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.05)


def test_two_concurrent_submits_under_max_concurrent_one_admit_exactly_one(world):
    grant_id = _sign_grant(world, max_concurrent=1)
    first, second = _admitted(world, "data/in.txt"), _admitted(world, "data/in.txt")
    started, release = _hold_sbatch(world)
    outcome: dict[str, ExecutionRecord] = {}
    racer = threading.Thread(target=lambda: outcome.update(first=_run(world, first)))
    racer.start()
    try:
        _wait_for(started)  # the first submit holds the grant's one slot at the gate
        outcome["second"] = _run(world, second, queue_timeout="00:00:02")
    finally:
        release.touch()
        racer.join(120)
    assert outcome["first"].status == "succeeded", _error(outcome["first"])
    _refused(world, outcome["second"], "gate_refused: grant_exhausted", "concurrent")
    assert len(world.cluster.calls("sbatch")) == 2  # the first run's two steps only
    assert {d.split(" run=")[1].split()[0] for d in _submits(world, grant_id)} == {first.run_id}
    assert _history(f"{second.run_id}/first/1") == [("pending", "estimate"), ("not_submitted", "gate_usage")]


def test_the_grant_envelope_is_what_the_gate_was_sent(world):
    """The C2 envelope: the stored grant bytes, the run spec, the skeleton."""
    grant_id = _sign_grant(world)
    gated = _admitted(world, "data/in.txt")
    row = grant_store_for(world.store).get(USER, grant_id)
    spec = gated.spec()
    assert base64.b64decode(spec["run_spec_b64"]) == gated.run_spec
    assert hpc_gate.grant_covers(signed_body(row), gated.body, json.loads(gated.workflow_steps)) == ""
    from umrs.server.intent_proposal import RUN_MAX_SECONDS

    assert gated.body["max_run_seconds"] == min(RUN_MAX_SECONDS, 86400)  # K19: the grant's window
