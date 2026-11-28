"""Owner-side idempotent recommit of a grant-backed signed workflow plan.

The route fixture is the real owner-scoped workflow API over the in-memory
job-store fake. The gate double stays idle: a recommit must name the durable
run already accepted, never consume the grant or submit work again.
"""

from __future__ import annotations

import base64
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from test_gated_trigger import gated as _trigger_app  # noqa: F401 — fixture
from test_grant_admission import _grant, _store_grant
from test_workflow_plans import _commit, _plan, _register, wf  # noqa: F401
from umrs.hpc_accounts import grants as grant_records
from umrs.hpc_accounts.usage_ledger import (
    LedgerUnit,
    Resources,
    get_usage_ledger,
)
from umrs.jobs.models import JobStatus
from umrs.server import grant_admission
from umrs.server.job_tracking_handler import DurableJobCommandBackend
from umrs.shared import hpc_user_work_access as access


@pytest.fixture
def gated(request):
    client, store, tmp_path, gate_dir = request.getfixturevalue("_trigger_app")

    def by_idempotency_key(project_id: str, key: str):
        return next(
            (
                job
                for job in store.jobs.values()
                if job.project_id == project_id and job.idempotency_key == key
            ),
            None,
        )

    store.get_job_by_idempotency_key = by_idempotency_key
    return client, store, tmp_path, gate_dir


def _gate_log(gate_dir: Path) -> bytes:
    path = gate_dir / "calls.jsonl"
    return path.read_bytes() if path.exists() else b""


def _grants() -> list[dict[str, Any]]:
    store = grant_records.grant_store_for(access.site_account_store("site-x"))
    return store.list("alice")


def _run_identity(body: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        body.get(key)
        for key in ("execution_id", "durable_job_id", "grant_id", "run_spec_id", "plan_id")
    )


def _assert_existing(response: Any, first: dict[str, Any]) -> None:
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["existing_run"] is True
    assert _run_identity(body) == _run_identity(first)


def test_a_grant_plan_recommit_returns_its_one_existing_run(gated):
    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    grants_before, gate_before = _grants(), _gate_log(gate_dir)

    first_response = _commit(client, plan, auto=False)
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()
    assert "existing_run" not in first
    assert len(store.jobs) == 1

    second = _commit(client, plan, auto=False)
    third = _commit(client, plan, auto=False)
    _assert_existing(second, first)
    _assert_existing(third, first)
    assert len(store.jobs) == 1
    assert _grants() == grants_before
    assert _gate_log(gate_dir) == gate_before


def _assert_concurrent_identity(responses: list[Any]) -> None:
    bodies = [responses[0].json(), responses[1].json()]
    assert [responses[0].status_code, responses[1].status_code] == [200, 200]
    assert len({_run_identity(body) for body in bodies}) == 1
    assert sum(body.get("existing_run") is True for body in bodies) == 1
    assert sum("existing_run" not in body for body in bodies) == 1


def _assert_no_duplicate_side_effects(
    store: Any, grants_before: list[dict[str, Any]], gate_dir: Path, gate_before: bytes
) -> None:
    assert len(store.jobs) == 1
    assert _grants() == grants_before
    assert _gate_log(gate_dir) == gate_before


def test_concurrent_identical_grant_commits_share_the_durable_winner(gated, monkeypatch):
    """The store's unique-key analogue guards the insertion race. Grant
    composition is not a cross-request serial lock; this test claims only
    durable deduplication and absence of any second execution side effect.
    """
    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    original_admit = grant_admission.admit_grant
    original_create = store.create_job
    both_admitted = threading.Barrier(2)
    create_lock = threading.Lock()

    def admit_together(*args: Any, **kwargs: Any):
        both_admitted.wait(timeout=10)
        return original_admit(*args, **kwargs)

    def locked_create(*args: Any, **kwargs: Any):
        with create_lock:
            return original_create(*args, **kwargs)

    monkeypatch.setattr(grant_admission, "admit_grant", admit_together)
    monkeypatch.setattr(store, "create_job", locked_create)
    grants_before, gate_before = _grants(), _gate_log(gate_dir)

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _n: _commit(client, plan, auto=False), range(2)))

    _assert_concurrent_identity(responses)
    _assert_no_duplicate_side_effects(store, grants_before, gate_dir, gate_before)


def _mismatched_commit(client: Any, plan: dict[str, Any], mismatch: str):
    headers = {"user": "bob"} if mismatch == "principal" else {}
    if mismatch == "project":
        headers = {"project": "proj-b"}
    if mismatch == "signed_content":
        plan["signed"]["parameters"] = {"steps": 4}
    return _commit(client, plan, auto=False, **headers)


@pytest.mark.parametrize("mismatch", ["principal", "project", "signed_content"])
def test_a_grant_recommit_never_reveals_a_run_to_a_mismatch(gated, mismatch, tmp_path):
    client, store, _tmp, gate_dir = gated
    _register(tmp_path, scope="system")
    _store_grant(_grant())
    plan = _plan(client)
    first = _commit(client, plan)
    assert first.status_code == 200, first.text
    execution_id = first.json()["execution_id"]
    gate_before = _gate_log(gate_dir)

    response = _mismatched_commit(client, plan, mismatch)

    assert response.status_code != 200
    assert execution_id not in response.text
    assert "durable_job_id" not in response.text
    assert len(store.jobs) == 1
    assert _gate_log(gate_dir) == gate_before


@pytest.mark.parametrize("mutation", ["principal", "source", "run_spec", "execution"])
def test_a_grant_recommit_refuses_a_mutated_existing_pointer(gated, mutation):
    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    first = _commit(client, plan).json()
    job = store.jobs[first["durable_job_id"]]
    gate_before = _gate_log(gate_dir)

    if mutation == "principal":
        job.user_id = "bob"
    elif mutation == "source":
        job.spec["source_sha256"] = "0" * 64
    elif mutation == "run_spec":
        job.spec["intent"]["run_spec_id"] = "0" * 64
    else:
        job.spec["execution_id"] = "wf-000000000000"
    response = _commit(client, plan, auto=False)

    assert response.status_code != 200
    assert first["execution_id"] not in response.text
    assert "durable_job_id" not in response.text
    assert len(store.jobs) == 1
    assert _gate_log(gate_dir) == gate_before


@pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.CANCELLED])
def test_a_grant_recommit_returns_the_persisted_terminal_outcome(gated, status):
    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    first = _commit(client, plan).json()
    store.jobs[first["durable_job_id"]].status = status
    gate_before = _gate_log(gate_dir)

    response = _commit(client, plan, auto=False)

    _assert_existing(response, first)
    assert response.json()["durable_job"]["status"] == status.value
    assert len(store.jobs) == 1
    assert _gate_log(gate_dir) == gate_before


def test_an_uncommitted_planned_grant_job_is_not_reported_as_a_run(gated):
    client, store, _tmp, _gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    first = _commit(client, plan).json()
    store.jobs[first["durable_job_id"]].status = JobStatus.PLANNED

    response = _commit(client, plan, auto=False)

    assert response.status_code != 200
    assert first["execution_id"] not in response.text
    assert len(store.jobs) == 1


def _assert_recovered(response: Any, expected: dict[str, Any]) -> None:
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["existing_run"] is True
    for key, value in expected.items():
        assert body[key] == value


def test_a_lost_commit_response_recovers_the_durable_result_without_a_gate_submit(
    gated, monkeypatch
):
    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    original_submit = DurableJobCommandBackend.submit_workflow_run
    submit_calls = 0

    def lose_first_response(self: Any, client_id: str, payload: str) -> str:
        nonlocal submit_calls
        submit_calls += 1
        raw = original_submit(self, client_id, payload)
        if submit_calls == 1:
            raise ConnectionError("the response was lost after the durable commit")
        return raw

    monkeypatch.setattr(DurableJobCommandBackend, "submit_workflow_run", lose_first_response)
    gate_before = _gate_log(gate_dir)
    ambiguous = _commit(client, plan, auto=False)
    assert ambiguous.status_code != 200
    assert len(store.jobs) == 1
    [job] = store.jobs.values()
    expected = {
        "execution_id": job.spec["execution_id"],
        "durable_job_id": job.job_id,
        "grant_id": job.spec["intent"]["grant_id"],
        "run_spec_id": job.spec["intent"]["run_spec_id"],
        "plan_id": plan["plan_id"],
    }

    recovered = _commit(client, plan, auto=False)

    _assert_recovered(recovered, expected)
    assert len(store.jobs) == 1
    assert _gate_log(gate_dir) == gate_before
    assert _grants()[0]["grant_id"] == expected["grant_id"]


def _ledger_unit(first: dict[str, Any], suffix: str) -> LedgerUnit:
    execution_id = first["execution_id"]
    return LedgerUnit(
        venue="site-x",
        attempt_id=f"{execution_id}/s{suffix}/1",
        run_id=execution_id,
        step=f"s{suffix}",
        person="alice",
        lane="staged_step",
        tag=f"umrs-step-{execution_id}-s{suffix}",
        project_id="proj-a",
        site_login="alice_hpc",
        access_mode="user_account",
        method="gated_key",
        slurm_account="grp-a",
        grant_id=first["grant_id"],
    )


def _reserve_and_charge_the_first_run(first: dict[str, Any], job: Any) -> list[dict[str, Any]]:
    """Binding ledger writes, not a budget-check mock: one step still holds
    its maximum estimate and another has settled against real GPU-seconds."""
    from umrs.hpc_accounts.site_profile import published
    from umrs.server.grant_admission import run_estimate_seconds

    body = json.loads(base64.b64decode(job.spec["intent"]["run_spec_b64"]))
    seconds = run_estimate_seconds(body, published(access.current_site())[0])
    assert seconds == 1800
    ledger = get_usage_ledger()
    ledger.tables["budgets"]["proj-a"] = 0.75
    ledger.pending(_ledger_unit(first, "1"), Resources(8, 1, seconds))
    used_unit = _ledger_unit(first, "2")
    ledger.pending(used_unit, Resources(8, 1, 900))
    ledger.settled(
        used_unit,
        {"state": "COMPLETED", "alloc_cpus": 8, "alloc_gpus": 1, "elapsed_s": 900},
        "gate_usage",
    )
    totals = ledger.project_totals("proj-a")
    assert (totals["reserved_gpu_seconds"], totals["gpu_seconds"]) == (1800, 900)
    return copy.deepcopy(ledger.tables["rows"])


def _assert_unchanged_charges(
    store: Any,
    ledger_before: list[dict[str, Any]],
    grants_before: list[dict[str, Any]],
    gate_dir: Path,
    gate_before: bytes,
) -> None:
    assert len(store.jobs) == 1
    assert get_usage_ledger().tables["rows"] == ledger_before
    assert _grants() == grants_before
    assert _gate_log(gate_dir) == gate_before


def test_a_budget_only_second_admission_replays_after_pending_and_used_charges(gated):
    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    first_response = _commit(client, plan, auto=False)
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()
    [job] = store.jobs.values()
    ledger_before = _reserve_and_charge_the_first_run(first, job)
    grants_before, gate_before = _grants(), _gate_log(gate_dir)

    second = _commit(client, plan, auto=False)

    _assert_existing(second, first)
    assert second.json()["durable_job"]["status"] == JobStatus.QUEUED.value
    _assert_unchanged_charges(store, ledger_before, grants_before, gate_dir, gate_before)


@pytest.mark.parametrize("mismatch", ["plan", "project"])
def test_a_budget_only_replay_mismatch_keeps_its_original_denial(gated, mismatch):
    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    first_response = _commit(client, plan, auto=False)
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()
    [job] = store.jobs.values()
    ledger_before = _reserve_and_charge_the_first_run(first, job)
    grants_before, gate_before = _grants(), _gate_log(gate_dir)
    key = "steps" if mismatch == "plan" else "project_id"
    value = 4 if mismatch == "plan" else "proj-b"
    job.spec["params"][key] = value

    response = _commit(client, plan, auto=False)

    assert (response.status_code, response.headers["x-platform-error-code"]) == (
        403,
        "hpcaccess.budget_exceeded",
    )
    assert first["execution_id"] not in response.text
    assert "durable_job_id" not in response.text
    _assert_unchanged_charges(store, ledger_before, grants_before, gate_dir, gate_before)


def test_a_duplicate_waits_for_the_exact_planned_winner_to_enqueue(gated, monkeypatch):
    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    original_enqueue = store.enqueue_job
    original_get = store.get_job
    enqueue_entered, release_enqueue = threading.Event(), threading.Event()
    refreshed = threading.Event()
    reads = 0
    enqueue_calls = 0

    def paused_enqueue(job_id: str) -> None:
        nonlocal enqueue_calls
        enqueue_calls += 1
        enqueue_entered.set()
        assert release_enqueue.wait(timeout=5)
        original_enqueue(job_id)

    def watched_get(job_id: str):
        nonlocal reads
        job = original_get(job_id)
        if enqueue_entered.is_set() and not release_enqueue.is_set():
            reads += 1
            if reads >= 2 and job.status == JobStatus.PLANNED:
                refreshed.set()
        return job

    monkeypatch.setattr(store, "enqueue_job", paused_enqueue)
    monkeypatch.setattr(store, "get_job", watched_get)
    grants_before, gate_before = _grants(), _gate_log(gate_dir)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(_commit, client, plan, auto=False)
        assert enqueue_entered.wait(timeout=2)
        second_future = pool.submit(_commit, client, plan, auto=False)
        try:
            assert refreshed.wait(timeout=2), "the duplicate never reconciled PLANNED"
            assert not second_future.done()
        finally:
            release_enqueue.set()
        responses = [first_future.result(timeout=5), second_future.result(timeout=5)]

    _assert_concurrent_identity(responses)
    assert enqueue_calls == 1
    _assert_no_duplicate_side_effects(store, grants_before, gate_dir, gate_before)


def test_a_broken_planned_row_times_out_as_a_typed_conflict(gated, monkeypatch):
    from umrs.server import workflow_registry_handler as handler

    client, store, _tmp, gate_dir = gated
    _store_grant(_grant())
    plan = _plan(client)
    first_response = _commit(client, plan)
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()
    [job] = store.jobs.values()
    assert job.user_id == "alice"
    job.status = JobStatus.PLANNED
    monkeypatch.setattr(handler, "_GRANT_PLANNED_WAIT_SECONDS", 0, raising=False)
    grants_before, gate_before = _grants(), _gate_log(gate_dir)

    response = _commit(client, plan, auto=False)

    assert response.status_code == 409, response.text
    assert response.headers["x-platform-error-code"] == "hpcaccess.intent_used"
    assert first["execution_id"] not in response.text
    assert "durable_job_id" not in response.text
    _assert_no_duplicate_side_effects(store, grants_before, gate_dir, gate_before)


def _reregister_at(source: Path) -> None:
    from umrs.server.workflow_registry_handler import register_workflow

    definition = json.loads((source / "workflow.json").read_text())
    result = register_workflow(
        workflow_name="relax",
        version=str(definition["version"]),
        user_id="alice",
        source_path=str(source),
        workflow_json=definition,
    )
    assert result["status"] == "ok", result


def _reset_registration_at(source: Path) -> None:
    from umrs.server.workflow_registry_handler import unregister_workflow

    definition = (source / "workflow.json").read_bytes()
    uploaded_code = (source / "m.py").read_bytes()
    unregister_workflow("relax", user_id="alice")
    source.mkdir(exist_ok=True)
    (source / "workflow.json").write_bytes(definition)
    (source / "m.py").write_bytes(uploaded_code)
    _reregister_at(source)


def test_a_replay_after_a_catalog_move_returns_the_durable_source_path(gated):
    client, store, tmp, gate_dir = gated
    _reset_registration_at(tmp / "relax")
    _store_grant(_grant())
    plan = _plan(client)
    first_response = _commit(client, plan, auto=False)
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()
    moved = tmp / "moved-relax"
    (tmp / "relax").rename(moved)
    _reregister_at(moved)
    grants_before, gate_before = _grants(), _gate_log(gate_dir)

    second = _commit(client, plan, auto=False)

    _assert_existing(second, first)
    assert second.json()["source_path"] == first["source_path"]
    assert second.json()["source_path"] != str(moved)
    _assert_no_duplicate_side_effects(store, grants_before, gate_dir, gate_before)


def test_a_moved_catalog_replay_still_refuses_the_wrong_source_digest(gated):
    client, store, tmp, gate_dir = gated
    _reset_registration_at(tmp / "relax")
    _store_grant(_grant())
    plan = _plan(client)
    first_response = _commit(client, plan, auto=False)
    assert first_response.status_code == 200, first_response.text
    first = first_response.json()
    moved = tmp / "moved-relax"
    (tmp / "relax").rename(moved)
    _reregister_at(moved)
    [job] = store.jobs.values()
    job.spec["source_sha256"] = "0" * 64
    grants_before, gate_before = _grants(), _gate_log(gate_dir)

    response = _commit(client, plan, auto=False)

    assert response.status_code == 409, response.text
    assert response.headers["x-platform-error-code"] == "hpcaccess.intent_used"
    assert first["execution_id"] not in response.text
    assert "durable_job_id" not in response.text
    _assert_no_duplicate_side_effects(store, grants_before, gate_dir, gate_before)
