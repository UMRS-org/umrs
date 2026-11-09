"""End to end: runs under a standing grant through the person's REAL gate
(ADR-112 wave C; plan §6 C/D.0, the S harness of "Lanes, ownership and the
tests that bind this contract").

The B2 world (``test_gated_step_lane_e2e``: the gate installed by r4's own
install script into a temp HPC home, a fake sshd, fake Slurm, the real step
worker) with a grant the person signed with a real ``ssh-keygen`` key:

- the server stores it (``POST /grants``: the C4 checks and a live
  ``grant-check`` at the gate);
- a two-step chained run starts without any signature: the admission picks
  the grant, composes the run spec and the gate submits both steps under it;
- a second run reads the first run's output by digest (``chained`` mode);
- a third run exceeds ``max_jobs``: ``gate_refused: grant_exhausted``;
- an input that is neither listed nor produced under the grant is refused
  by the gate (``grant_mismatch``);
- each step is in the usage ledger: pending, submitted, then settled from
  the gate's ``usage`` with the grant's id.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from types import SimpleNamespace
from typing import Any

import pytest
from test_gated_step_lane_e2e import (
    ACCOUNT,
    DOMAIN,
    HEADERS,
    INPUT,
    LOGIN,
    PROJECT,
    SITE,
    USER,
    VERSION,
    WORKFLOW,
    _error,
    _executor,
    _no_run,
    _ok,
)
from test_gated_step_lane_e2e import world as _e2e_world  # noqa: F401 — the world fixture
from umrs.hpc_accounts import grant_service, grants
from umrs.hpc_accounts.usage_ledger import get_usage_ledger
from umrs.server import workflow_runner as wr
from umrs.server.run_intent_admission import admit_run
from umrs.server.workflow_execution_registry import ExecutionRecord

hpc_gate = pytest.importorskip("r4_contracts.hpc_gate")
hs = pytest.importorskip("r4.hpc_signing")

PRODUCED = INPUT.upper() + b"!"


@pytest.fixture
def world(request):
    """The B2 end-to-end world (``test_gated_step_lane_e2e.world``)."""
    return request.getfixturevalue("_e2e_world")  # the second step's output of a run over INPUT


def _schema() -> dict[str, Any]:
    prop = {"type": "string", "maxLength": 1024}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [],
        "properties": {"input_key": prop},
    }


def _grant_body(w: Any, max_jobs: int) -> dict[str, Any]:
    src_sha, steps_sha = grants.server_digests(str(w.source), w.workflow_json())
    image = {k: w.image[k] for k in ("ref", "image_id", "archive_sha256")}
    now = int(time.time())
    seed = {"sha256": hashlib.sha256(INPUT).hexdigest(), "size": len(INPUT)}
    return hpc_gate.validate_grant({
        "v": 1, "type": "grant", "grant_id": secrets.token_hex(16), "site": SITE, "login": LOGIN,
        "principal": USER, "accounts": [ACCOUNT], "projects": [PROJECT],
        "workflows": [{"name": WORKFLOW, "version": VERSION, "src_manifest_sha256": src_sha,
                       "steps_sha256": steps_sha, "params": ["input_key"]}],
        "images": [image], "param_schema": _schema(),
        "inputs": {"mode": "chained", "listed": [seed]},
        "caps": {"max_jobs": max_jobs, "max_gpu_hours": 0, "max_core_hours": 0, "max_concurrent": 2},
        "not_before": now, "expires": now + 7200, "issued": now, "delegable": False,
        "nonce": secrets.token_hex(16),
    })  # fmt: skip


def _store(w: Any, monkeypatch, max_jobs: int = 4) -> dict[str, Any]:
    """The person signs the grant; the server stores it (a live grant-check)."""
    digests = grants.server_digests(str(w.source), w.workflow_json())
    lookup = lambda name, version: digests if (name, version) == (WORKFLOW, VERSION) else None  # noqa: E731
    monkeypatch.setattr(grant_service, "digests_lookup", lambda *_a, **_k: lookup)
    body = _grant_body(w, max_jobs)
    payload = hpc_gate.canonical_bytes(body)
    signature = hs.sign(payload, hpc_gate.namespace("grant", DOMAIN), hs.key_path("ed25519"))
    envelope = {"payload": base64.b64encode(payload).decode(), "signature": signature}
    view = _ok(w.api.post("/umrs/sites/me/hpc/grants", json=envelope, headers=HEADERS))["grant"]
    assert (view["status"], view["covers_now"], view["usage"]["jobs"]) == ("active", True, 0)
    return body


def _admit(w: Any, input_key: str) -> Any:
    target = SimpleNamespace(
        entry=w.entry, source_path=str(w.source), workflow_json=w.workflow_json()
    )
    run = {"workflow": WORKFLOW, "version": VERSION, "project": PROJECT, "principal": USER}
    params = {"input_key": input_key, "project_id": PROJECT}
    admission, error = admit_run(w.identity(), {}, params, run, _no_run, target)
    assert error is None, error
    return admission.grant


def _run(w: Any, gated: Any) -> ExecutionRecord:
    spec = gated.spec()
    record = ExecutionRecord(
        execution_id=gated.run_id, workflow_name=WORKFLOW, user_id=USER,
        params=dict(gated.body["params"]), **spec,
    )  # fmt: skip
    wr.run_workflow_record(
        record, source_path=str(w.source), workflow_json=w.workflow_json(),
        executor=_executor(), project_id=PROJECT, user_id=USER, verified_project=PROJECT,
    )  # fmt: skip
    return record


def test_chained_runs_under_one_grant_until_its_jobs_are_spent(world, monkeypatch):
    grant = _store(world, monkeypatch, max_jobs=4)
    first = _admit(world, "data/in.txt")
    record = _run(world, first)
    assert record.status == "succeeded", _error(record)
    assert (
        world.s3.objects[("wf-runner-output", f"runs/{first.run_id}/second/second/out.txt")]
        == PRODUCED
    )
    # chained: an object holding bytes the grant's own run produced passes
    world.s3.objects[("wf-runner", "data/from-first.txt")] = PRODUCED
    second = _admit(world, "data/from-first.txt")
    record = _run(world, second)
    assert record.status == "succeeded", _error(record)
    third = _run(world, _admit(world, "data/in.txt"))
    assert third.status == "failed"
    assert "gate_refused: grant_exhausted" in _error(third), _error(third)
    assert len(world.cluster.calls("sbatch")) == 4
    views = _ok(world.api.get("/umrs/sites/me/hpc/grants?refresh=1", headers=HEADERS))["grants"]
    assert [(v["grant_id"], v["usage"]["jobs"]) for v in views] == [(grant["grant_id"], 4)]


def test_an_input_neither_listed_nor_produced_is_refused_by_the_gate(world, monkeypatch):
    _store(world, monkeypatch)
    world.s3.objects[("wf-runner", "data/other.txt")] = b"not produced under this grant"
    record = _run(world, _admit(world, "data/other.txt"))
    assert record.status == "failed"
    assert "grant_mismatch" in _error(record) and world.cluster.calls("sbatch") == []


def test_each_step_of_a_grant_run_is_ledgered_and_settled_from_the_gate(world, monkeypatch):
    grant = _store(world, monkeypatch)
    gated = _admit(world, "data/in.txt")
    record = _run(world, gated)
    assert record.status == "succeeded", _error(record)
    rows = sorted(get_usage_ledger().current(SITE, person=USER), key=_by_step)
    run_key = hpc_gate.grant_run_key(grant["grant_id"], gated.run_id)
    assert [_ledger_view(r) for r in rows] == [
        (
            "first",
            "COMPLETED",
            "gate_usage",
            grant["grant_id"],
            "",
            f"umrs-gated-{run_key[:16]}-first",
            7,
        ),
        (
            "second",
            "COMPLETED",
            "gate_usage",
            grant["grant_id"],
            "",
            f"umrs-gated-{run_key[:16]}-second",
            7,
        ),
    ]
    assert [(r["access_mode"], r["method"], bool(r["scheduler_job_id"])) for r in rows] == [
        ("user_account", "gated_key", True)
    ] * 2


def _by_step(row: dict[str, Any]) -> str:
    return str(row["step"])


def _ledger_view(row: dict[str, Any]) -> tuple[Any, ...]:
    keys = ("step", "state", "source", "grant_id", "intent_id", "tag", "elapsed_s")
    return tuple(row[k] for k in keys)


def test_a_cancelled_grant_is_refused_by_the_gate_and_the_server(world, monkeypatch):
    grant = _store(world, monkeypatch)
    gated = _admit(world, "data/in.txt")
    body = hpc_gate.grant_control_body(SITE, USER, grant["grant_id"])
    payload = hpc_gate.canonical_bytes(body)
    signature = hs.sign(payload, hpc_gate.namespace("control", DOMAIN), hs.key_path("ed25519"))
    answer = _ok(world.api.post(
        f"/umrs/sites/me/hpc/grants/{grant['grant_id']}/cancel",
        json={"payload": base64.b64encode(payload).decode(), "signature": signature}, headers=HEADERS,
    ))  # fmt: skip
    assert (answer["relayed"], answer["grant"]["status"], answer["grant"]["gate_cancel"]) == (
        True,
        "cancelled",
        "recorded",
    )
    record = _run(world, gated)  # admitted before the cancel: the gate refuses it
    assert record.status == "failed" and "grant_cancelled" in _error(record)
    target = SimpleNamespace(
        entry=world.entry, source_path=str(world.source), workflow_json=world.workflow_json()
    )
    run = {"workflow": WORKFLOW, "version": VERSION, "project": PROJECT, "principal": USER}
    _admission, error = admit_run(
        world.identity(),
        {"grant_id": grant["grant_id"]},
        {"input_key": "data/in.txt"},
        run,
        _no_run,
        target,
    )
    assert "hpcaccess.grant_invalid" in error and "cancelled" in error


def _near_expiry(gated: Any) -> Any:
    """The admitted run with a spec 100 s from its ``expires`` (as if the
    run waited in UMRS, throttled by its grant's concurrency)."""
    from umrs.server.grant_admission import encode_run

    now = int(time.time())
    body = {**gated.body, "issued": now - 3500, "expires": now + 100}
    return encode_run(gated.grant_id, body, _world_json(gated))


def _world_json(_gated: Any) -> dict[str, Any]:
    import json

    from test_gated_step_lane_e2e import WORKFLOW_JSON

    return json.loads(json.dumps(WORKFLOW_JSON))


def test_a_spec_close_to_its_expiry_is_composed_again_before_the_first_submit(world, monkeypatch):
    """K19: fresh issued, expires and nonce, the same run_id (so the same run
    key); the record keeps the new spec; the gate accepts it."""
    _store(world, monkeypatch)
    old = _near_expiry(_admit(world, "data/in.txt"))
    record = _run(world, old)
    assert record.status == "succeeded", _error(record)
    assert record.run_spec_id != old.run_spec_id
    fresh = hpc_gate.parse_strict(base64.b64decode(record.run_spec_b64), json_values=True)
    assert fresh["run_id"] == old.run_id and fresh["nonce"] != old.body["nonce"]
    assert (
        fresh["expires"] - fresh["issued"] == 3600 and fresh["issued"] >= old.body["issued"] + 3500
    )


def test_a_spec_is_never_composed_again_once_a_step_was_submitted(world, monkeypatch):
    from umrs.server import gated_step_lane as lane

    _store(world, monkeypatch)
    written: list[str] = []

    def run_of(gated: Any, fixed: bool) -> Any:
        b64 = base64.b64encode(gated.run_spec).decode()
        return lane.GatedRun(
            "v",
            "alice",
            b64,
            "",
            "",
            kind="grant",
            fixed=fixed,
            on_respec=lambda b, _i: written.append(b),
        )

    first = run_of(_near_expiry(_admit(world, "data/in.txt")), fixed=False)
    fresh = lane.fresh_run(first)
    assert fresh is not first and len(written) == 1  # the first step recomposes
    again = lane.fresh_run(first)
    assert (
        again.intent_b64 == fresh.intent_b64 and len(written) == 1
    )  # the next step does not recompose
    held = run_of(_near_expiry(_admit(world, "data/in.txt")), fixed=True)  # a step holds a job
    assert lane.fresh_run(held) is held and len(written) == 1


def test_parallel_root_bindings_share_one_recomposed_spec(world, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from umrs.server import gated_step_lane as lane

    _store(world, monkeypatch)
    old = _near_expiry(_admit(world, "data/in.txt"))
    record = ExecutionRecord(
        execution_id=old.run_id, workflow_name=WORKFLOW, user_id=USER,
        params=dict(old.body["params"]), **old.spec(),
    )  # fmt: skip
    written: list[tuple[str, str]] = []

    def write(b64: str, spec_id: str) -> None:
        written.append((b64, spec_id))
        record.update(run_spec_b64=b64, run_spec_id=spec_id)

    first = replace(lane.gated_run_for(record), on_respec=write)
    second = replace(lane.gated_run_for(record), on_respec=write)
    barrier = threading.Barrier(2)

    def fresh(binding: Any) -> Any:
        barrier.wait(timeout=10)
        return lane.fresh_run(binding)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(fresh, (first, second)))
    assert first.intent_b64 == second.intent_b64 == record.run_spec_b64
    assert first.run_spec_id == second.run_spec_id == record.run_spec_id
    assert written == [(record.run_spec_b64, record.run_spec_id)]
