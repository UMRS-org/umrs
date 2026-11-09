"""The usage ledger's recovery sweep (ADR-112 wave D; plan §6 C/D.0 D2,
K5, K13b). Scripted resolvers stand in for the gate and the platform's
``sacct``; the clock is fixed.

- a pending gated row with a job becomes ``submitted``; with none (exit 3),
  ``not_submitted``; an ``UNKNOWN`` lookup or an unreachable gate leaves it;
- a pending platform row takes the one job named by its tag that no other
  row holds (a retried service step reuses the name, K13b), or the earliest
  of several; none after an hour is ``not_submitted``;
- submitted rows settle from the gate's ``usage`` and from ``sacct``; an
  unreachable gate leaves the estimate counting as reserved;
- a row still open 7 days past its time becomes ``unsettled`` and counts as
  used (K5).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from umrs.hpc_accounts.ledger_sweep import GateResolver, PlatformResolver, UsageLedgerSweep
from umrs.hpc_accounts.usage_ledger import InMemoryUsageLedger, LedgerUnit, Resources

V = "site-x"
T0 = datetime(2026, 11, 9, 1, 0, tzinfo=timezone.utc)
TERMINAL = {"state": "COMPLETED", "elapsed": "00:02:00", "alloc_tres": "cpu=2,gres/gpu=1", "alloc_cpus": "2", "start": "", "end": ""}


class _Gate:
    def __init__(self, lookup: tuple[str, str] = ("job", "55"), usage: dict | None = None, down: bool = False) -> None:
        self.answer, self.rows, self.down = lookup, usage or {}, down

    def lookup(self, _row: dict[str, Any]) -> tuple[str, str]:
        if self.down:
            raise OSError("unreachable")
        return self.answer

    def usage(self, _row: dict[str, Any], ids: list[str]) -> dict[str, dict[str, Any]]:
        if self.down:
            raise OSError("unreachable")
        return {i: {**self.rows[i], "job_id": i} for i in ids if i in self.rows}


class _Sacct:
    def __init__(self, matches: list[tuple[str, str]] | None = None, rows: dict | None = None) -> None:
        self.matches, self.rows = matches, rows or {}

    def match(self, _row: dict[str, Any], _since: datetime) -> list[tuple[str, str]] | None:
        return self.matches

    def sacct(self, ids: list[str]) -> dict[str, dict[str, Any]] | None:
        return {i: self.rows[i] for i in ids if i in self.rows}


def _unit(attempt: str, gated: bool = True) -> LedgerUnit:
    mode = ("user_account", "gated_key") if gated else ("service_account", "platform")
    return LedgerUnit(
        venue=V, attempt_id=attempt, person="alice", lane="staged_step", tag="umrs-step-wf-1-s1",
        access_mode=mode[0], method=mode[1], run_id=attempt.split("/")[0], step="s1",
        project_id="p1", intent_id="i" * 64 if gated else "",
    )  # fmt: skip


def _ledger_with(*units: LedgerUnit, at: datetime = T0) -> InMemoryUsageLedger:
    ledger = InMemoryUsageLedger()
    for unit in units:
        ledger.pending(unit, Resources(2, 1, 600))
    for row in ledger.tables["rows"]:
        row["recorded_at"] = at
    return ledger


def _sweep(ledger: Any, gate: Any = None, sacct: Any = None, after: timedelta = timedelta(minutes=20)) -> UsageLedgerSweep:
    return UsageLedgerSweep(ledger, V, gate or _Gate(), sacct or _Sacct([]), now=lambda: T0 + after)


def _state(ledger: Any) -> list[str]:
    return [r["state"] for r in sorted(ledger.current(V), key=lambda r: r["attempt_id"])]


def test_a_young_pending_row_is_left_alone():
    ledger = _ledger_with(_unit("wf-1/s1/1"))
    assert _sweep(ledger, after=timedelta(minutes=5)).run_once()["submitted"] == 0
    assert _state(ledger) == ["pending"]


def test_a_pending_gated_row_with_a_job_becomes_submitted():
    ledger = _ledger_with(_unit("wf-1/s1/1"))
    _sweep(ledger, _Gate(("job", "55"))).run_once()
    assert _state(ledger) == ["submitted"] and ledger.current(V)[0]["scheduler_job_id"] == "55"


def test_a_pending_gated_row_without_a_job_is_not_submitted():
    ledger = _ledger_with(_unit("wf-1/s1/1"))
    _sweep(ledger, _Gate(("absent", ""))).run_once()
    assert _state(ledger) == ["not_submitted"]


def test_an_unknown_lookup_or_an_unreachable_gate_leaves_the_row():
    ledger = _ledger_with(_unit("wf-1/s1/1"))
    _sweep(ledger, _Gate(("unknown", ""))).run_once()
    _sweep(ledger, _Gate(down=True)).run_once()
    assert _state(ledger) == ["pending"]


def test_a_pending_alias_of_a_job_another_unit_holds_is_not_submitted_again():
    """Crash recovery: the new dispatch associated the adopted job first, so
    the old pending estimate is an alias, not a second submission."""
    first, adopted = _unit("wf-1/s1/1"), _unit("wf-1/s1/new")
    ledger = _ledger_with(first, adopted)
    ledger.submitted(adopted, "55", None)
    counts = _sweep(ledger, _Gate(("job", "55"))).run_once()
    assert counts["not_submitted"] == 1 and counts["submitted"] == 0
    assert _state(ledger) == ["not_submitted", "submitted"]
    owners = [row["attempt_id"] for row in ledger.current(V) if row["scheduler_job_id"] == "55"]
    assert owners == [adopted.attempt_id]


def test_a_retried_service_steps_tag_resolves_to_the_job_on_no_other_row():
    """K13b: two jobs carry one name; attempt 1 holds 101; attempt 2 takes 102."""
    first, second = _unit("wf-1/s1/1", gated=False), _unit("wf-1/s1/2", gated=False)
    ledger = _ledger_with(first, second)
    ledger.submitted(first, "101", None)
    _sweep(ledger, sacct=_Sacct([("101", "2026-11-09T01:00:00"), ("102", "2026-11-09T01:02:00")])).run_once()
    assert [r["scheduler_job_id"] for r in sorted(ledger.current(V), key=lambda r: r["attempt_id"])] == ["101", "102"]


def test_the_earliest_of_several_free_jobs_is_taken():
    ledger = _ledger_with(_unit("wf-1/s1/1", gated=False))
    _sweep(ledger, sacct=_Sacct([("103", "2026-11-09T01:09:00"), ("102", "2026-11-09T01:01:00")])).run_once()
    assert ledger.current(V)[0]["scheduler_job_id"] == "102"


def test_no_match_after_an_hour_is_not_submitted_and_an_sacct_failure_leaves_it():
    ledger = _ledger_with(_unit("wf-1/s1/1", gated=False))
    _sweep(ledger, sacct=_Sacct([]), after=timedelta(minutes=30)).run_once()
    assert _state(ledger) == ["pending"]
    _sweep(ledger, sacct=_Sacct(None), after=timedelta(hours=2)).run_once()
    assert _state(ledger) == ["pending"]
    _sweep(ledger, sacct=_Sacct([]), after=timedelta(hours=2)).run_once()
    assert _state(ledger) == ["not_submitted"]


def test_submitted_rows_settle_from_the_gate_and_from_sacct():
    gated, service = _unit("wf-1/s1/1"), _unit("wf-2/s1/1", gated=False)
    ledger = _ledger_with(gated, service)
    ledger.submitted(gated, "55", 3)
    ledger.submitted(service, "66", None)
    counts = _sweep(ledger, _Gate(usage={"55": TERMINAL}), _Sacct(rows={"66": {**TERMINAL, "job_id": "66"}})).run_once()
    assert counts["settled"] == 2
    rows = {r["scheduler_job_id"]: r for r in ledger.current(V)}
    assert (rows["55"]["source"], rows["66"]["source"]) == ("gate_usage", "sacct")
    assert rows["55"]["elapsed_s"] == 120 and rows["55"]["alloc_gpus"] == 1


def test_an_unreachable_gate_leaves_the_estimate_reserved():
    unit = _unit("wf-1/s1/1")
    ledger = _ledger_with(unit)
    ledger.submitted(unit, "55", None)
    _sweep(ledger, _Gate(down=True)).run_once()
    assert _state(ledger) == ["submitted"]
    assert ledger.project_totals("p1")["reserved_gpu_seconds"] == 600


def test_a_row_open_seven_days_past_its_time_is_unsettled_and_counts_as_used():
    unit = _unit("wf-1/s1/1")
    ledger = _ledger_with(unit)
    ledger.submitted(unit, "55", None)
    ledger.tables["rows"][-1]["submitted_at"] = T0
    assert _sweep(ledger, _Gate(down=True), after=timedelta(days=7, minutes=11)).run_once()["unsettled"] == 1
    assert _state(ledger) == ["unsettled"]
    sums = ledger.project_totals("p1")
    assert (sums["gpu_seconds"], sums["reserved_gpu_seconds"]) == (600, 0)
    assert _sweep(ledger, after=timedelta(days=8)).run_once() == {"submitted": 0, "not_submitted": 0, "settled": 0, "unsettled": 0}


def test_the_platform_resolver_parses_sacct():
    calls: list[str] = []

    def run(command: str) -> tuple[int, str]:
        calls.append(command)
        if "--name" in command:
            return 0, "101|2026-11-09T01:00:00\n102|2026-11-09T01:01:00\n"
        return 0, "101|COMPLETED|00:01:00|cpu=1|1|2026-11-09T01:00:00|2026-11-09T01:01:00\n"

    resolver = PlatformResolver(run)
    assert resolver.match({"tag": "umrs-step-x"}, T0) == [("101", "2026-11-09T01:00:00"), ("102", "2026-11-09T01:01:00")]
    assert resolver.sacct(["101"])["101"]["state"] == "COMPLETED"
    assert "-S 2026-11-09T01:00:00 --name umrs-step-x" in calls[0]
    assert resolver.sacct.__self__ is resolver and PlatformResolver(lambda _c: (1, "")).match({"tag": "t"}, T0) is None


def test_the_gate_resolver_asks_lookup_by_the_runs_gate_key(monkeypatch):
    import subprocess

    from umrs.server import gated_step_lane as lane

    seen: list[tuple[str, ...]] = []

    class _T:
        def call(self, verb: str, *args: str, **_kw: Any) -> Any:
            seen.append((verb, *args))
            return subprocess.CompletedProcess([verb], 0, b'{"job_id": "77", "wd": "w"}', b"")

    monkeypatch.setattr(lane, "transport_for", lambda *_a: _T())
    row = {"person": "alice", "site_login": "a1", "grant_id": "c" * 32, "run_id": "wf-0123456789ab", "step": "s1"}
    assert GateResolver(V).lookup(row) == ("job", "77")
    from r4_contracts import hpc_gate

    assert seen == [("lookup", hpc_gate.grant_run_key("c" * 32, "wf-0123456789ab"), "s1")]
