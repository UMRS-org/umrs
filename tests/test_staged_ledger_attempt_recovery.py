"""Staged-step ledger identity and job recovery regressions (ADR-112 D).

A fresh executor is a new dispatch even when it reuses a run id and the
Slurm integer attempt. Resuming or adopting a scheduler job must instead
reuse the ledger unit that already holds that job, so live settlement and
the recovery sweep cannot charge it twice.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from step_lane_harness import FakeS3
from test_remote_slurm_step_executor import (  # noqa: F401 — autouse fixture
    _no_bucket_registry,
)
from test_slurm_step_lifecycle import (
    _completed_earlier,
    _ctx,
    _tag,
)
from test_slurm_step_lifecycle import cluster as _lane_cluster  # noqa: F401 — fixture
from test_slurm_step_lifecycle import tree as _lane_tree  # noqa: F401 — fixture
from test_usage_ledger_sweep import TERMINAL, _Gate, _Sacct
from umrs.hpc_accounts.ledger_sweep import UsageLedgerSweep
from umrs.hpc_accounts.usage_ledger import Resources, get_usage_ledger
from umrs.server import gated_step_lane as gated
from umrs.server import remote_slurm_step_executor as rse
from umrs.server.step_ledger import (
    LEDGER_METADATA_KEY,
    LedgerRefusedStep,
    service_after,
    service_pending,
    service_unit,
)

TERMINAL_600 = {**TERMINAL, "elapsed": "00:10:00", "job_id": "907"}


@pytest.fixture
def cluster(request):
    return request.getfixturevalue("_lane_cluster")


@pytest.fixture
def tree(request):
    return request.getfixturevalue("_lane_tree")


def _owner() -> dict[str, Any]:
    return {LEDGER_METADATA_KEY: {"person": "alice", "project_id": "p1"}}


def _dispatches(cluster: Any, tree: Any) -> tuple[Any, Any, Any]:
    ctx = _ctx(tree, FakeS3(), **_owner())
    tag = _tag(cluster, tree, **_owner())
    first = cluster.executor()._begin(ctx, "s1", tag)
    second = cluster.executor()._begin(ctx, "s1", tag)
    return ctx, first, second


def _gated_ledger() -> Any:
    body = base64.b64encode(json.dumps({"project": "p1"}).encode()).decode()
    run = gated.GatedRun("site-x", "alice", body, "", "i" * 64)
    return gated.GatedLedger(run)


def _write_submitted(ctx: Any, first: Any, second: Any) -> None:
    service_pending(first, ctx, 1, 1, 60)
    first.job_id = "1001"
    service_after("submitted", first, ctx, "1001", None)
    service_pending(second, ctx, 1, 1, 60)
    second.job_id = "1002"
    service_after("submitted", second, ctx, "1002", None)


def test_fresh_executors_have_unique_ledger_ids_but_the_same_slurm_attempt(cluster, tree):
    _, first, second = _dispatches(cluster, tree)
    ctx = _ctx(tree, FakeS3(), **_owner())
    assert first.attempt == second.attempt == 1
    assert first.prefix == second.prefix
    assert service_unit(first, ctx).attempt_id != service_unit(second, ctx).attempt_id
    ledger = _gated_ledger()
    assert ledger.unit(first).attempt_id != ledger.unit(second).attempt_id


def test_a_fresh_dispatch_writes_its_own_pending_and_submitted_rows(cluster, tree):
    ctx, first, second = _dispatches(cluster, tree)
    _write_submitted(ctx, first, second)
    rows = get_usage_ledger().tables["rows"]
    assert [row["state"] for row in rows] == ["pending", "submitted"] * 2
    submitted_ids = {row["scheduler_job_id"] for row in rows if row["state"] == "submitted"}
    assert submitted_ids == {"1001", "1002"}


def test_a_second_dispatch_is_budget_checked_even_if_the_first_is_settled(cluster, tree):
    ctx, first, second = _dispatches(cluster, tree)
    usage = {
        "state": "COMPLETED",
        "elapsed_s": 60,
        "alloc_cpus": 1,
        "alloc_gpus": 1,
        "start": None,
        "end": None,
    }
    service_pending(first, ctx, 1, 1, 60)
    first.job_id = "1001"
    service_after("submitted", first, ctx, "1001", None)
    service_after("settled", first, ctx, usage, "sacct")
    get_usage_ledger().tables["budgets"]["p1"] = 60 / 3600
    with pytest.raises(LedgerRefusedStep, match="hpcaccess.budget_exceeded"):
        service_pending(second, ctx, 1, 1, 60)
    assert [row["state"] for row in get_usage_ledger().tables["rows"]] == [
        "pending",
        "submitted",
        "COMPLETED",
    ]


def _sweep_service(ledger: Any, venue: str) -> None:
    now = datetime.now(timezone.utc)
    UsageLedgerSweep(
        ledger,
        venue,
        gated=None,  # type: ignore[arg-type] — service rows only
        service=_Sacct(rows={"907": TERMINAL_600}),
        now=lambda: now + timedelta(minutes=20),
    ).run_once()


def _measurement(job_id: str, run: Any) -> dict[str, Any]:
    assert run is not None
    return {"job_id": job_id, "elapsed_seconds": 600, "gpus": 1, "gpu_seconds": 600}


@pytest.mark.parametrize(
    "states", [("COMPLETED",), ("RUNNING", "COMPLETED")], ids=["adopt", "resume"]
)
def test_service_recovery_settles_the_original_job_once(cluster, tree, monkeypatch, states):
    _completed_earlier(cluster, _tag(cluster, tree, **_owner()), job_id="907", states=states)
    ctx = _ctx(tree, FakeS3(), **_owner())
    executor = cluster.executor()
    original = executor._begin(ctx, "s1", _tag(cluster, tree, **_owner()))
    original_unit = service_unit(original, ctx)
    ledger = get_usage_ledger()
    ledger.pending(original_unit, Resources(1, 1, 600))
    ledger.submitted(original_unit, "907", None)
    monkeypatch.setattr(rse, "measure_slurm_accounting", _measurement)
    result = executor.execute(entrypoint="steps.lane:Upper", config={}, ctx=ctx)
    assert result.extra["slurm_job"]["job_id"] == "907"
    _sweep_service(ledger, original_unit.venue)
    rows = ledger.current(original_unit.venue)
    assert len(rows) == 1
    assert rows[0]["attempt_id"] == original_unit.attempt_id
    assert rows[0]["scheduler_job_id"] == "907"
    assert ledger.open_rows(original_unit.venue) == []
    assert ledger.project_totals("p1")["gpu_seconds"] == 600


def test_gated_adoption_settles_the_original_job_once(cluster, tree):
    _, first, second = _dispatches(cluster, tree)
    second.attempt = first.attempt + 1  # recovery on a long-lived executor
    first.job_id = second.job_id = "907"
    ledger = get_usage_ledger()
    first_writes = _gated_ledger()
    original_unit = first_writes.unit(first)
    ledger.pending(original_unit, Resources(1, 1, 600))
    first_writes.submitted(first, "907")
    # These are exactly GatedStepLane._adopt/write and _settle's ledger calls.
    second_writes = _gated_ledger()
    second_writes.submitted(second, "907")
    second_writes.settled(second, TERMINAL_600)
    now = datetime.now(timezone.utc)
    UsageLedgerSweep(
        ledger,
        original_unit.venue,
        gated=_Gate(usage={"907": TERMINAL_600}),
        service=None,  # type: ignore[arg-type] — gated rows only
        now=lambda: now + timedelta(minutes=20),
    ).run_once()
    rows = ledger.current(original_unit.venue)
    assert len(rows) == 1
    assert rows[0]["attempt_id"] == original_unit.attempt_id
    assert ledger.open_rows(original_unit.venue) == []
    assert ledger.project_totals("p1")["gpu_seconds"] == 600
