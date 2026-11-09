"""The HPC usage ledger (ADR-112 wave D; plan §6 C/D.0 D1-D3, K4, K5, K13,
K14). One contract, two stores: the in-memory ledger and Postgres (pg15,
skipped without ``DATABASE_URL``).

- writes go pending -> submitted -> settled, each idempotent, forward only;
- the budget is lifetime, summed over every venue, 0 is uncapped, open
  estimates count as reserved, ``unsettled`` rows count as used, and it is
  compared in integer GPU-seconds (a REAL 0.1 h admits exactly 360 s);
- the pending write is the binding check, also under concurrency;
- a pending write Postgres cannot take is ``ledger_unavailable``.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from hpc_ledger_pg import delete_ledger_rows, execute, ledger_pool_or_skip
from umrs.hpc_accounts import usage_ledger as ul
from umrs.hpc_accounts.usage_ledger import (
    BudgetExceeded,
    InMemoryUsageLedger,
    LedgerUnavailable,
    LedgerUnit,
    PgUsageLedger,
    Resources,
)


def _unit(venue: str, attempt: str = "wf-1/s1/1", project: str = "p1", **over: Any) -> LedgerUnit:
    fields = {
        "venue": venue, "attempt_id": attempt, "person": "alice", "lane": "staged_step",
        "tag": "umrs-step-wf-1-s1", "run_id": attempt.split("/")[0], "step": "s1", "project_id": project,
    }
    fields.update(over)
    return LedgerUnit(**fields)


USAGE = {"state": "COMPLETED", "elapsed_s": 100, "alloc_cpus": 4, "alloc_gpus": 2, "start": None, "end": None}


class _Pg:
    def __init__(self) -> None:
        self.pool = ledger_pool_or_skip()
        self.venues: list[str] = []
        self.projects: list[str] = []

    def ledger(self, budgets: dict[str, float]) -> PgUsageLedger:
        for project, hours in budgets.items():
            self.projects.append(project)
            execute(
                self.pool,
                "INSERT INTO project_budgets (project_id, max_gpu_hours) VALUES (%s, %s)"
                " ON CONFLICT (project_id) DO UPDATE SET max_gpu_hours = EXCLUDED.max_gpu_hours",
                (project, hours),
            )
        return PgUsageLedger(self.pool)

    def cleanup(self) -> None:
        delete_ledger_rows(self.pool, self.venues)
        execute(self.pool, "DELETE FROM project_budgets WHERE project_id = ANY(%s)", (self.projects,))


@pytest.fixture(params=["memory", "postgres"])
def make(request) -> Iterator[Any]:
    """``make(budgets) -> (ledger, venue_a, venue_b, project)``: two venues
    over one store, and a project unique to this test."""
    tag = uuid.uuid4().hex[:8]
    venues, project = [f"a-{tag}", f"b-{tag}"], f"p-{tag}"
    if request.param == "memory":
        def build(budget: float = 0.0) -> tuple[Any, str, str, str]:
            return InMemoryUsageLedger(tables={"rows": [], "budgets": {project: budget}}), *venues, project

        yield build
        return
    pg = _Pg()
    pg.venues.extend(venues)

    def build_pg(budget: float = 0.0) -> tuple[Any, str, str, str]:
        return pg.ledger({project: budget}), *venues, project

    try:
        yield build_pg
    finally:
        pg.cleanup()


def _states(ledger: Any, venue: str) -> list[str]:
    return [r["state"] for r in sorted(ledger.current(venue, limit=1000), key=lambda r: r["attempt_id"])]


def _all_rows(ledger: Any, venue: str, unit: LedgerUnit) -> int:
    if isinstance(ledger, InMemoryUsageLedger):
        return len([r for r in ledger.tables["rows"] if r["attempt_id"] == unit.attempt_id and r["venue"] == venue])
    rows = execute(ledger._pool, "SELECT count(*) FROM hpc_usage_ledger WHERE venue = %s AND attempt_id = %s", (venue, unit.attempt_id))
    return int(rows[0][0])


# ── the write order and K13a ─────────────────────────────────────────────────


def test_pending_then_submitted_then_settled_and_the_view_is_the_latest(make):
    ledger, venue, _b, project = make()
    unit = _unit(venue, project=project)
    ledger.pending(unit, Resources(4, 2, 600))
    ledger.submitted(unit, "123", 7)
    ledger.settled(unit, USAGE, "gate_usage")
    (row,) = ledger.current(venue)
    assert (row["state"], row["source"], row["scheduler_job_id"], row["gate_log_seq"]) == ("COMPLETED", "gate_usage", "123", 7)
    assert (row["alloc_gpus"], row["elapsed_s"], row["gpu_hours"]) == (2, 100, pytest.approx(200 / 3600))
    assert _all_rows(ledger, venue, unit) == 3


def test_every_write_is_idempotent_and_moves_forward_only(make):
    """K13a: pending twice, settled twice with the same values: one row each;
    pending after submitted writes nothing."""
    ledger, venue, _b, project = make()
    unit = _unit(venue, project=project)
    ledger.pending(unit, Resources(1, 1, 60))
    ledger.pending(unit, Resources(1, 1, 60))
    ledger.submitted(unit, "9", None)
    ledger.pending(unit, Resources(1, 1, 60))
    ledger.settled(unit, USAGE, "sacct")
    ledger.settled(unit, USAGE, "sacct")
    assert _all_rows(ledger, venue, unit) == 3
    ledger.settled(unit, {**USAGE, "elapsed_s": 50}, "sacct")  # a correction is a new row
    assert _all_rows(ledger, venue, unit) == 4
    assert ledger.current(venue)[0]["elapsed_s"] == 50


def test_post_submit_writes_use_the_job_holder_not_the_new_dispatch(make, monkeypatch):
    """Resuming someone else's dispatch cannot split one scheduler job into
    two usage units; a later duplicate does not change the first owner."""
    ledger, venue, _b, project = make()
    monkeypatch.setattr(ul, "_ledger", ledger)
    writer = getattr(ul, "write_job_after", None)
    assert writer is not None
    original = _unit(venue, project=project)
    adopted = _unit(venue, "wf-1/s1/new-dispatch", project=project)
    ledger.pending(original, Resources(4, 2, 600))
    ledger.submitted(original, "123", None)
    writer("submitted", adopted, "123", 9)
    writer("settled", adopted, "123", USAGE, "sacct")
    (row,) = ledger.current(venue)
    assert row["attempt_id"] == original.attempt_id
    assert row["source"] == "sacct" and ledger.project_totals(project)["gpu_seconds"] == 200
    ledger.submitted(adopted, "123", None)  # pre-existing duplicate association
    assert ledger.unit_for_scheduler_job(venue, "123") == original


def test_settling_an_unassociated_job_establishes_its_scheduler_id(make, monkeypatch):
    ledger, venue, _b, project = make()
    monkeypatch.setattr(ul, "_ledger", ledger)
    writer = getattr(ul, "write_job_after", None)
    assert writer is not None
    adopted = _unit(venue, "wf-1/s1/adopted", project=project)
    writer("settled", adopted, "999", USAGE, "sacct")
    (row,) = ledger.current(venue)
    assert (row["attempt_id"], row["scheduler_job_id"]) == (adopted.attempt_id, "999")
    assert (row["state"], row["source"]) == ("COMPLETED", "sacct")
    assert ledger.open_rows(venue) == []


def test_a_settlement_never_counts_fewer_gpus_or_cpus_than_the_estimate(make):
    """Review 2026-11-09: an accounting that does not track GPUs (or only
    typed ones) reports 0 GPUs; the row keeps the estimate's GPUs, and a
    later settlement from the sweep does not lower them again."""
    ledger, venue, _b, project = make(0.5)
    unit = _unit(venue, project=project)
    ledger.pending(unit, Resources(1, 1, 1800))
    ledger.submitted(unit, "77", None)
    measured = {**USAGE, "elapsed_s": 1800, "alloc_cpus": 8, "alloc_gpus": 0}
    ledger.settled(unit, measured, "gate_usage")
    ledger.settled(unit, measured, "sacct")
    row = ledger.current(venue)[0]
    assert (row["alloc_gpus"], row["alloc_cpus"], row["elapsed_s"]) == (1, 8, 1800)
    with pytest.raises(BudgetExceeded):
        ledger.pending(_unit(venue, "wf-1/s2/1", project=project), Resources(1, 1, 1800))


def test_a_settlement_counts_measured_gpu_overallocation(make):
    ledger, venue, _b, project = make()
    unit = _unit(venue, project=project)
    ledger.pending(unit, Resources(1, 1, 600))
    ledger.submitted(unit, "88", None)
    ledger.settled(
        unit,
        {**USAGE, "elapsed_s": 600, "alloc_cpus": 8, "alloc_gpus": 4},
        "sacct",
    )
    row = ledger.current(venue)[0]
    assert (row["alloc_gpus"], row["alloc_cpus"], row["elapsed_s"]) == (4, 8, 600)
    assert ledger.project_totals(project)["gpu_seconds"] == 2400


def test_not_submitted_carries_no_amount(make):
    ledger, venue, _b, project = make()
    unit = _unit(venue, project=project)
    ledger.pending(unit, Resources(2, 4, 3600))
    ledger.not_submitted(unit, "gate_usage")
    (row,) = ledger.current(venue)
    assert (row["state"], row["gpu_hours"]) == ("not_submitted", 0)
    assert ledger.project_totals(project)["gpu_seconds"] == 0


# ── the budget (D3) ──────────────────────────────────────────────────────────


def test_an_uncapped_project_never_refuses(make):
    ledger, venue, _b, project = make(0.0)
    ledger.pending(_unit(venue, project=project), Resources(1, 64, 604800))
    assert ledger.budget(project)["remaining_gpu_hours"] is None


def test_a_real_budget_of_a_tenth_hour_admits_exactly_360_gpu_seconds(make):
    """K13c: ``budget_s = round(0.1 x 3600)``; REAL 0.1 is not exactly 0.1."""
    ledger, venue, _b, project = make(0.1)
    ledger.pending(_unit(venue, "r/a/1", project=project), Resources(1, 1, 300))
    ledger.pending(_unit(venue, "r/b/1", project=project), Resources(1, 1, 60))
    with pytest.raises(BudgetExceeded, match="GPU-hours"):
        ledger.pending(_unit(venue, "r/c/1", project=project), Resources(1, 1, 1))


def test_the_budget_is_lifetime_across_venues_and_counts_estimates_as_reserved(make):
    ledger, venue_a, venue_b, project = make(1.0)
    first = _unit(venue_a, "r/a/1", project=project)
    ledger.pending(first, Resources(1, 1, 1800))  # reserved 1800 s at venue A
    ledger.settled(_unit(venue_b, "r/b/1", project=project), {**USAGE, "alloc_gpus": 1, "elapsed_s": 1200}, "sacct")
    view = ledger.budget(project)
    assert (view["reserved_gpu_hours"], view["used_gpu_hours"]) == (0.5, pytest.approx(1 / 3))
    with pytest.raises(BudgetExceeded):
        ledger.pending(_unit(venue_b, "r/c/1", project=project), Resources(1, 1, 1200))
    ledger.pending(_unit(venue_b, "r/d/1", project=project), Resources(1, 1, 600))


def test_an_unsettled_row_counts_as_used_at_its_estimate(make):
    """K5: a row past its limit becomes ``unsettled`` and counts as used."""
    ledger, venue, _b, project = make()
    unit = _unit(venue, project=project)
    ledger.pending(unit, Resources(2, 1, 600))
    ledger.submitted(unit, "5", None)
    ledger.mark_unsettled(unit)
    (row,) = ledger.current(venue)
    assert (row["state"], row["source"]) == ("unsettled", "estimate")
    sums = ledger.project_totals(project)
    assert (sums["gpu_seconds"], sums["reserved_gpu_seconds"]) == (600, 0)
    ledger.settled(unit, {**USAGE, "alloc_gpus": 1, "elapsed_s": 30}, "sacct")  # a late correction
    assert ledger.project_totals(project)["gpu_seconds"] == 30


def test_two_admissions_pass_and_the_second_steps_pending_refuses(make):
    """K14: the run-level check reserves nothing; the pending write binds."""
    ledger, venue, _b, project = make(1.0)
    ul.check_budget(ledger, project, 3600)
    ul.check_budget(ledger, project, 3600)
    ledger.pending(_unit(venue, "r1/s/1", project=project), Resources(1, 1, 3600))
    with pytest.raises(BudgetExceeded):
        ledger.pending(_unit(venue, "r2/s/1", project=project), Resources(1, 1, 3600))


def test_a_row_without_a_project_is_not_budget_checked(make):
    ledger, venue, _b, _project = make(0.01)
    ledger.pending(_unit(venue, project=""), Resources(1, 8, 86400))
    assert _states(ledger, venue) == ["pending"]


def test_concurrent_pendings_under_the_budget_lock_admit_exactly_one(make):
    """D3: the check and the insert run under one lock per project."""
    ledger, venue, _b, project = make(1.0)
    results: list[str] = []
    barrier = threading.Barrier(2)

    def one(name: str) -> None:
        barrier.wait()
        try:
            ledger.pending(_unit(venue, f"{name}/s/1", project=project), Resources(1, 1, 2400))
            results.append("ok")
        except BudgetExceeded:
            results.append("refused")

    threads = [threading.Thread(target=one, args=(n,)) for n in ("r1", "r2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["ok", "refused"]


# ── K4 and the PG failure path ───────────────────────────────────────────────


def test_the_estimate_follows_the_gates_rounding_and_the_whole_exclusive_node():
    assert ul.estimate(4, 1, 61) == Resources(4, 1, 120)
    node = {"exclusive": True, "max_cpus": 64, "max_gpus": 8}
    assert ul.estimate(4, 1, 600, node) == Resources(64, 8, 600)
    assert ul.estimate(4, 0, 600, node) == Resources(64, 0, 600)
    assert ul.estimate(4, 2, 600, {"exclusive": False, "max_cpus": 64, "max_gpus": 8}) == Resources(4, 2, 600)


def test_the_estimate_agrees_with_the_gates_grant_cost():
    from r4_contracts import hpc_gate

    cost = getattr(hpc_gate, "grant_cost", None)
    if cost is None:
        pytest.skip("the gate's grant_cost is not there yet")
    node = {"exclusive": True, "max_cpus": 64, "max_gpus": 8, "arch": "x86_64"}
    for step, part in (({"cpus": 4, "gpus": 1, "time": 61}, {"exclusive": False, "max_cpus": 64, "max_gpus": 8}), ({"cpus": 4, "gpus": 1, "time": 600}, node)):
        res = ul.estimate(step["cpus"], step["gpus"], step["time"], part)
        assert tuple(cost(step, part)) == (res.gpus * res.time_s, res.cpus * res.time_s)


class _BrokenPool:
    def getconn(self) -> Any:
        raise OSError("connection refused")

    def putconn(self, _conn: Any) -> None:  # pragma: no cover
        pass


def test_a_pending_write_postgres_cannot_take_is_ledger_unavailable():
    with pytest.raises(LedgerUnavailable):
        PgUsageLedger(_BrokenPool()).pending(_unit("v"), Resources(1, 0, 60))


def test_a_lane_pending_refusal_names_the_code_and_after_writes_never_raise(monkeypatch):
    monkeypatch.setattr(ul, "_ledger", PgUsageLedger(_BrokenPool()))
    with pytest.raises(ul.LedgerRefused, match="hpcaccess.ledger_unavailable"):
        ul.write_pending(_unit("v"), Resources(1, 0, 60))
    ul.write_after("submitted", _unit("v"), "1", None)  # logged, never raised


def test_usage_from_sacct_settles_only_terminal_measured_rows():
    row = {"state": "CANCELLED by 1", "elapsed": "00:10:00", "alloc_tres": "cpu=4,gres/gpu=2", "alloc_cpus": "4", "start": "2026-11-09T01:00:00", "end": ""}
    usage = ul.usage_from_sacct(row)
    assert usage is not None
    assert (usage["state"], usage["elapsed_s"], usage["alloc_gpus"], usage["alloc_cpus"]) == ("CANCELLED", 600, 2, 4)
    assert ul.usage_from_sacct({**row, "state": "RUNNING"}) is None
    assert ul.usage_from_sacct({**row, "elapsed": "garbage"}) is None


@pytest.mark.parametrize(
    ("tres", "expected"),
    [("cpu=4,gres/gpu:h100=2", 2), ("cpu=4,mem=4G", 0)],
)
def test_usage_from_sacct_accepts_typed_and_absent_gpu_tres(tres: str, expected: int):
    row = {
        "state": "COMPLETED",
        "elapsed": "00:10:00",
        "alloc_tres": tres,
        "alloc_cpus": "4",
    }
    usage = ul.usage_from_sacct(row)
    assert usage is not None
    assert usage["alloc_gpus"] == expected
