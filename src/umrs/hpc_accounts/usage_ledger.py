"""The HPC usage ledger of user work (ADR-112 wave D; plan §6 C/D.0 D1-D3).

One row per state of one attempt (migration 0138, append-only). Every
user-work submit point writes three phases through :class:`UsageLedger`:

- ``pending(unit, estimate)`` before anything reaches the cluster: the
  requested resources (source ``estimate``). It is also the **binding**
  budget check (K14): for a project with ``project_budgets.max_gpu_hours >
  0``, ``used + reserved + this <= budget`` in integer GPU-seconds, under a
  per-project advisory lock, or :class:`BudgetExceeded`. A pending write
  that fails in Postgres is :class:`LedgerUnavailable` and the caller
  submits nothing;
- ``submitted(unit, scheduler_job_id, gate_log_seq)`` after the job id;
- ``settled(unit, usage, source)`` at a terminal Slurm state (source
  ``gate_usage`` or ``sacct``), or ``not_submitted(unit, source)``.

Every write is idempotent and moves forward only (K13a): it inserts a row
only when the attempt's current row differs, under a per-attempt lock;
``pending`` after ``submitted`` and a repeated settlement write nothing.

Amounts are integers (K13c): GPU-seconds = ``alloc_gpus x elapsed_s``,
core-seconds = ``alloc_cpus x elapsed_s``. An estimate's resources follow
the gate's ``grant_cost`` rule (K4): walltime rounded up to whole minutes,
the whole node on an ``exclusive`` partition (:func:`estimate`). Hour
columns are display values only and are never compared.

The budget is lifetime and summed over **all venues** (``project_budgets``
is keyed by project alone; the database is shared). ``used`` counts settled
rows and ``unsettled`` ones (at their estimate, K5); ``reserved`` counts
open estimates (``pending``/``submitted``).
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

LANES = ("staged_step", "hpc_script", "cluster_loop")
SOURCES = ("estimate", "gate_usage", "sacct")
ACCESS_MODES = ("service_account", "user_account")
METHODS = ("platform", "gated_key")
PENDING, SUBMITTED, NOT_SUBMITTED, UNSETTLED = (
    "pending",
    "submitted",
    "not_submitted",
    "unsettled",
)
OPEN_STATES = (PENDING, SUBMITTED)
ESTIMATE = "estimate"

#: The ledger's columns in the table's order (``ledger_id`` is the table's).
COLUMNS = (
    "venue",
    "attempt_id",
    "run_id",
    "step",
    "job_id",
    "scheduler_job_id",
    "tag",
    "person",
    "project_id",
    "site_login",
    "access_mode",
    "method",
    "slurm_account",
    "intent_id",
    "grant_id",
    "gate_log_seq",
    "lane",
    "submitted_at",
    "started_at",
    "ended_at",
    "state",
    "alloc_cpus",
    "alloc_gpus",
    "elapsed_s",
    "core_hours",
    "gpu_hours",
    "source",
)
#: Columns an owner who is not an admin never sees on another member's rows (K15).
PRIVATE_COLUMNS = (
    "site_login",
    "scheduler_job_id",
    "tag",
    "intent_id",
    "grant_id",
    "gate_log_seq",
)
_TIME_COLUMNS = ("submitted_at", "started_at", "ended_at", "recorded_at")
#: What makes two rows of one attempt "the same" (a write equal on these is dropped).
_COMPARED = (
    "state",
    "scheduler_job_id",
    "alloc_cpus",
    "alloc_gpus",
    "elapsed_s",
    "source",
)


class LedgerUnavailable(RuntimeError):
    """``hpcaccess.ledger_unavailable`` (503): a pending row could not be
    written, so nothing was submitted."""

    code = "hpcaccess.ledger_unavailable"


class BudgetExceeded(RuntimeError):
    """``hpcaccess.budget_exceeded`` (403): the project's GPU-hour budget
    would be exceeded."""

    code = "hpcaccess.budget_exceeded"

    def __init__(
        self,
        project: str,
        used_s: int,
        reserved_s: int,
        request_s: int,
        budget_s: int,
        *,
        detail: str | None = None,
    ) -> None:
        message = detail or (
            f"project {project}: used {_hours(used_s)} + reserved {_hours(reserved_s)} + "
            f"this {_hours(request_s)} > budget {_hours(budget_s)} GPU-hours"
        )
        super().__init__(message)
        self.project = project


class Resources(NamedTuple):
    """Requested (or measured) resources: CPUs, GPUs and seconds."""

    cpus: int
    gpus: int
    time_s: int


@dataclass(frozen=True)
class LedgerUnit:
    """The identity columns of one attempt (from its ``HpcIdentity``)."""

    venue: str
    attempt_id: str
    person: str
    lane: str
    tag: str
    access_mode: str = "service_account"
    method: str = "platform"
    run_id: str = ""
    step: str = ""
    job_id: str = ""
    project_id: str = ""
    site_login: str = ""
    slurm_account: str = ""
    intent_id: str = ""
    grant_id: str = ""


def unit_for(identity: Any, **fields_: Any) -> LedgerUnit:
    """A unit from an ``HpcIdentity`` (``to_metrics()`` shape) and the attempt's
    own fields: ``attempt_id``, ``person``, ``lane``, ``tag`` and the rest."""
    metrics = (
        identity.to_metrics()
        if hasattr(identity, "to_metrics")
        else dict(identity or {})
    )
    mode = str(metrics.get("access_mode") or "service_account")
    return LedgerUnit(
        venue=str(fields_.pop("venue", "") or metrics.get("site") or ""),
        access_mode=mode,
        method="gated_key" if mode == "user_account" else "platform",
        site_login=str(metrics.get("site_login") or ""),
        slurm_account=str(metrics.get("slurm_account") or ""),
        **fields_,
    )


def estimate(
    cpus: int, gpus: int, time_s: int, partition: dict[str, Any] | None = None
) -> Resources:
    """The canonical gate K4 cost as resources; zero-time keeps its request."""
    from r4_contracts import hpc_gate

    step = {"cpus": int(cpus), "gpus": int(gpus), "time": max(0, int(time_s))}
    wall = hpc_gate.grant_wall(step)
    if wall == 0:
        return Resources(step["cpus"], step["gpus"], 0)
    gpu_s, core_s = hpc_gate.grant_cost(step, partition or {"exclusive": False})
    return Resources(core_s // wall, gpu_s // wall, wall)


def requested_partition(profile: Any, *, cpus: int, gpus: int) -> dict[str, Any]:
    """Normalize a service/platform partition for the canonical K4 estimate.

    Site partition objects describe capacity differently from the gate's
    dictionaries. Missing profiles remain non-exclusive (legacy callers);
    an explicit exclusive profile must supply its real capacity.
    """
    if profile is None:
        return {"exclusive": False}
    if isinstance(profile, dict):

        def read(key: str, default: Any) -> Any:
            return profile.get(key, default)

        max_gpus = int(read("max_gpus", 0) or gpus)
    else:

        def read(key: str, default: Any) -> Any:
            return getattr(profile, key, default)

        from umrs.orchestrator.partition_choice import partition_gpus

        max_gpus = int(read("max_gpus", 0) or partition_gpus(profile) or gpus)
    exclusive = bool(read("exclusive", False))
    if not exclusive:
        return {"exclusive": False}
    return {
        "exclusive": True,
        "max_cpus": int(read("max_cpus", 0) or cpus),
        "max_gpus": max_gpus,
    }


def _hours(seconds: int) -> str:
    """Hours rounded up to two decimals, for display only."""
    return f"{math.ceil(int(seconds) * 100 / 3600) / 100:.2f}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def gpu_seconds(row: dict[str, Any]) -> int:
    return int(row.get("alloc_gpus") or 0) * int(row.get("elapsed_s") or 0)


def core_seconds(row: dict[str, Any]) -> int:
    return int(row.get("alloc_cpus") or 0) * int(row.get("elapsed_s") or 0)


def is_used(row: dict[str, Any]) -> bool:
    """Counts as used: a measured row, or ``unsettled`` (at its estimate)."""
    return row.get("source") != ESTIMATE or row.get("state") == UNSETTLED


def is_reserved(row: dict[str, Any]) -> bool:
    return row.get("source") == ESTIMATE and row.get("state") in OPEN_STATES


# ── the forward-only rule (K13a) ─────────────────────────────────────────────


def _rank(state: str) -> int:
    return {PENDING: 0, SUBMITTED: 1, UNSETTLED: 2}.get(state, 3)


def accepts(current: dict[str, Any] | None, new: dict[str, Any]) -> bool:
    """Whether *new* is written over the attempt's *current* row: a first
    row; a later state; or a final state that corrects another final one
    (a settlement with other values; the same values write nothing)."""
    if current is None:
        return True
    if all(current.get(k) == new.get(k) for k in _COMPARED):
        return False
    before, after = _rank(str(current.get("state"))), _rank(str(new.get("state")))
    return after > before or (after == before == 3)


def _refuse_pending_over_final(unit: LedgerUnit, current: dict[str, Any] | None) -> None:
    """A pending write over an attempt whose row is already final (settled,
    ``not_submitted`` or ``unsettled``) would submit a job no row records and
    no budget check sees, so it refuses and nothing is sent (review closure
    2026-11-12). A pending after ``submitted`` is still dropped (K13a)."""
    if current is not None and _rank(str(current.get("state"))) > _rank(SUBMITTED):
        raise LedgerUnavailable(
            f"the usage ledger cannot record attempt {unit.attempt_id}: it already "
            f"ended ({current.get('state')}); a new submit needs a new attempt"
        )


def _with_amounts(
    row: dict[str, Any], cpus: int, gpus: int, elapsed: int
) -> dict[str, Any]:
    row.update(alloc_cpus=int(cpus), alloc_gpus=int(gpus), elapsed_s=int(elapsed))
    row.update(
        core_hours=int(cpus) * int(elapsed) / 3600.0,
        gpu_hours=int(gpus) * int(elapsed) / 3600.0,
    )
    return row


def _base_row(unit: LedgerUnit, current: dict[str, Any] | None) -> dict[str, Any]:
    """The unit's identity over what the attempt's current row carries."""
    row = {k: (current or {}).get(k) for k in COLUMNS}
    row.update(_identity_over(unit, row))
    row.update({k: str(row.get(k) or "") for k in _TEXT_COLUMNS})
    cpus, gpus, elapsed = (
        int(row.get(k) or 0) for k in ("alloc_cpus", "alloc_gpus", "elapsed_s")
    )
    return _with_amounts(row, cpus, gpus, elapsed)


_TEXT_COLUMNS = (
    "job_id",
    "scheduler_job_id",
    "run_id",
    "step",
    "intent_id",
    "grant_id",
    "project_id",
    "site_login",
    "slurm_account",
)


def _identity_over(unit: LedgerUnit, row: dict[str, Any]) -> dict[str, Any]:
    """The unit's non-empty values, and its empty ones where the row has none."""
    return {k: v for k, v in asdict(unit).items() if v or row.get(k) in (None, "")}


def _pending_row(unit: LedgerUnit, current: Any, est: Resources) -> dict[str, Any]:
    row = _with_amounts(_base_row(unit, current), est.cpus, est.gpus, est.time_s)
    row.update(state=PENDING, source=ESTIMATE)
    return row


def _submitted_row(
    unit: LedgerUnit, current: Any, job: str, seq: int | None
) -> dict[str, Any]:
    row = _base_row(unit, current)
    row.update(state=SUBMITTED, source=ESTIMATE, scheduler_job_id=str(job or ""))
    row.update(gate_log_seq=seq, submitted_at=row.get("submitted_at") or _now())
    return row


def _usage_time(value: Any) -> Any:
    if isinstance(value, datetime) or value in (None, ""):
        return value or None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _settled_row(
    unit: LedgerUnit, current: Any, usage: dict[str, Any], source: str
) -> dict[str, Any]:
    """A measurement never counts fewer CPUs or GPUs than the attempt's
    current row (its estimate: an accounting that does not track GPUs, or
    tracks only typed ones, reports none; review 2026-11-09)."""
    row = _base_row(unit, current)
    cpus = max(int(usage.get("alloc_cpus") or 0), int(row["alloc_cpus"]))
    gpus = max(int(usage.get("alloc_gpus") or 0), int(row["alloc_gpus"]))
    _with_amounts(row, cpus, gpus, int(usage.get("elapsed_s") or 0))
    row.update(state=str(usage.get("state") or "UNKNOWN").upper(), source=source)
    row.update(
        started_at=_usage_time(usage.get("start")),
        ended_at=_usage_time(usage.get("end")),
    )
    return row


def _not_submitted_row(unit: LedgerUnit, current: Any, source: str) -> dict[str, Any]:
    row = _with_amounts(_base_row(unit, current), 0, 0, 0)
    row.update(state=NOT_SUBMITTED, source=source)
    return row


def _unsettled_row(current: dict[str, Any]) -> dict[str, Any]:
    row = {k: current.get(k) for k in COLUMNS}
    row.update(state=UNSETTLED, source=ESTIMATE)
    return row


def _check_source(source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"unknown ledger source {source!r}")
    return source


def budget_seconds(max_gpu_hours: Any) -> int:
    """``round(max_gpu_hours x 3600)``: the REAL column as integer seconds."""
    return round(float(max_gpu_hours or 0) * 3600)


def _check_budget(
    project: str,
    budget_h: Any,
    sums: tuple[int, int],
    request_s: int,
    *,
    estimate_complete: bool = True,
) -> None:
    budget_s = budget_seconds(budget_h)
    if not project or budget_s <= 0:
        return
    used_s, reserved_s = sums
    if not estimate_complete:
        raise BudgetExceeded(
            project,
            used_s,
            reserved_s,
            request_s,
            budget_s,
            detail=f"project {project}: the GPU-time estimate is unbounded; name valid --time and GPU resources",
        )
    if used_s + reserved_s + request_s > budget_s:
        raise BudgetExceeded(project, used_s, reserved_s, request_s, budget_s)


def check_budget(ledger: Any, project: str, request_s: int) -> None:
    """The advisory budget check of a whole run (admission, K14): it
    reserves nothing; each step's ``pending`` write is the binding one."""
    if not project:
        return
    sums = ledger.project_totals(project)
    pair = (int(sums["gpu_seconds"]), int(sums["reserved_gpu_seconds"]))
    _check_budget(project, ledger.max_gpu_hours(project), pair, int(request_s))


def _amount_totals(values: tuple[Any, ...]) -> dict[str, Any]:
    jobs, used_g, used_c, res_g, res_c = (int(value or 0) for value in values)
    return {
        "jobs": jobs,
        "gpu_hours": used_g / 3600.0,
        "core_hours": used_c / 3600.0,
        "reserved_gpu_hours": res_g / 3600.0,
        "reserved_core_hours": res_c / 3600.0,
        "gpu_seconds": used_g,
        "reserved_gpu_seconds": res_g,
    }


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """``{jobs, gpu_hours, core_hours, reserved_gpu_hours, reserved_core_hours}``
    of current rows, plus the integer seconds the budget compares."""
    used = [r for r in rows if is_used(r)]
    reserved = [r for r in rows if is_reserved(r)]
    return _amount_totals(
        (
            sum(1 for r in rows if r.get("scheduler_job_id")),
            sum(map(gpu_seconds, used)),
            sum(map(core_seconds, used)),
            sum(map(gpu_seconds, reserved)),
            sum(map(core_seconds, reserved)),
        )
    )


def budget_view(max_gpu_hours: Any, sums: dict[str, Any]) -> dict[str, Any]:
    """``{max_gpu_hours, used_gpu_hours, reserved_gpu_hours, remaining_gpu_hours}``;
    ``remaining`` is null when the project is uncapped (0)."""
    budget_s = budget_seconds(max_gpu_hours)
    used_s, res_s = (
        int(sums.get("gpu_seconds") or 0),
        int(sums.get("reserved_gpu_seconds") or 0),
    )
    remaining = None if budget_s <= 0 else max(0, budget_s - used_s - res_s) / 3600.0
    return {
        "max_gpu_hours": float(max_gpu_hours or 0),
        "used_gpu_hours": used_s / 3600.0,
        "reserved_gpu_hours": res_s / 3600.0,
        "remaining_gpu_hours": remaining,
    }


def row_view(
    row: dict[str, Any], *, private: bool = True, recorded: bool = False
) -> dict[str, Any]:
    """The report's row: every column (times as ISO text); without *private*
    the K15 columns are left out; ``recorded_at`` only with *recorded*."""
    keys = [k for k in ("ledger_id", *COLUMNS) if private or k not in PRIVATE_COLUMNS]
    view = {k: _iso(row.get(k)) for k in keys}
    if recorded:
        view["recorded_at"] = _iso(row.get("recorded_at"))
    return view


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


# ── the store ────────────────────────────────────────────────────────────────


class UsageLedger:
    """The ledger API (binding: plan §6 C/D.0 D2). Subclasses provide
    ``_locked`` (the per-attempt lock, plus the per-project budget lock),
    ``_current``, ``_insert``, ``_budget_sums`` and ``_rows``."""

    def pending(
        self, unit: LedgerUnit, estimate: Resources, *, estimate_complete: bool = True
    ) -> None:
        """Before submit: the estimate, and the binding budget check."""
        project = unit.project_id
        with self._locked(unit, budget=bool(project)) as tx:
            current = self._current(tx, unit)
            _refuse_pending_over_final(unit, current)
            row = _pending_row(unit, current, estimate)
            if not accepts(current, row):
                return
            budget_h, sums = self._budget_sums(tx, project)
            _check_budget(
                project,
                budget_h,
                sums,
                gpu_seconds(row),
                estimate_complete=estimate_complete,
            )
            self._insert(tx, row)

    def submitted(
        self, unit: LedgerUnit, scheduler_job_id: str, gate_log_seq: int | None
    ) -> None:
        self._advance(
            unit, lambda cur: _submitted_row(unit, cur, scheduler_job_id, gate_log_seq)
        )

    def settled(self, unit: LedgerUnit, usage: dict[str, Any], source: str) -> None:
        _check_source(source)
        self._advance(unit, lambda cur: _settled_row(unit, cur, usage, source))

    def not_submitted(self, unit: LedgerUnit, source: str) -> None:
        _check_source(source)
        self._advance(unit, lambda cur: _not_submitted_row(unit, cur, source))

    def mark_unsettled(self, unit: LedgerUnit) -> None:
        """K5: a row still open past its limit is charged at its estimate."""
        self._advance(unit, lambda cur: _unsettled_row(cur or {}) if cur else None)

    def _advance(self, unit: LedgerUnit, build: Any) -> None:
        with self._locked(unit, budget=False) as tx:
            current = self._current(tx, unit)
            row = build(current)
            if row is None:
                return
            if accepts(current, row):
                self._insert(tx, row)
            else:
                logger.debug(
                    "ledger: %s %s dropped (current %s)",
                    unit.attempt_id,
                    row.get("state"),
                    (current or {}).get("state"),
                )

    # reads

    def unit_for_scheduler_job(
        self, venue: str, scheduler_job_id: str
    ) -> LedgerUnit | None:
        """The first attempt ever associated with this venue's scheduler job.

        The append-only order matters when pre-fix duplicate associations
        exist: settlement stays with the original, estimate-bearing unit.
        """
        if not scheduler_job_id:
            return None
        row = self._scheduler_job_row(venue, scheduler_job_id)
        if row is None:
            return None
        names = LedgerUnit.__dataclass_fields__
        return LedgerUnit(**{name: str(row.get(name) or "") for name in names})

    def write_job_after(
        self, op: str, unit: LedgerUnit, scheduler_job_id: str, *args: Any
    ) -> None:
        """Atomically associate and update one venue's scheduler job.

        The scheduler-job lock covers both the canonical-owner lookup and all
        subsequent rows, so two first adopters cannot split one job's usage.
        """
        with self._scheduler_job_locked(unit.venue, scheduler_job_id):
            original = self.unit_for_scheduler_job(unit.venue, scheduler_job_id)
            target = original or unit
            if op == "submitted":
                self.submitted(target, scheduler_job_id, *args)
                return
            if op != "settled":
                raise ValueError(f"unsupported job-scoped ledger write {op!r}")
            if original is None:
                self.submitted(target, scheduler_job_id, None)
            self.settled(target, *args)

    def current(
        self,
        venue: str,
        *,
        person: str | None = None,
        project: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Current rows (each attempt's latest), newest first."""
        rows = [r for r in self._rows(venue) if _matches(r, person, project, since)]
        rows.sort(key=lambda r: int(r.get("ledger_id") or 0), reverse=True)
        return rows[: max(0, int(limit))]

    def open_rows(self, venue: str) -> list[dict[str, Any]]:
        """This venue's current rows still ``pending`` or ``submitted``."""
        return [r for r in self._rows(venue) if r.get("state") in OPEN_STATES]

    def current_totals(
        self,
        venue: str,
        *,
        person: str | None = None,
        project: str | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Totals over every matching current row, independent of a page limit."""
        return totals(
            [r for r in self._rows(venue) if _matches(r, person, project, since)]
        )

    def project_totals(self, project_id: str) -> dict[str, Any]:
        """Lifetime totals of *project_id* across every venue."""
        return totals(
            [r for r in self._rows(None) if r.get("project_id") == project_id]
        )

    def project_rows(self, project_id: str) -> list[dict[str, Any]]:
        rows = [r for r in self._rows(None) if r.get("project_id") == project_id]
        return sorted(rows, key=lambda r: int(r.get("ledger_id") or 0), reverse=True)

    def budget(self, project_id: str) -> dict[str, Any]:
        """The project's budget view (lifetime, across venues)."""
        return budget_view(
            self.max_gpu_hours(project_id), self.project_totals(project_id)
        )

    # subclass hooks
    def max_gpu_hours(self, project_id: str) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def _locked(self, unit: LedgerUnit, *, budget: bool) -> Any:  # pragma: no cover
        raise NotImplementedError

    @contextlib.contextmanager
    def _scheduler_job_locked(
        self, venue: str, scheduler_job_id: str
    ) -> Iterator[Any]:  # pragma: no cover - backend hook
        raise NotImplementedError

    def _current(
        self, tx: Any, unit: LedgerUnit
    ) -> dict[str, Any] | None:  # pragma: no cover
        raise NotImplementedError

    def _insert(self, tx: Any, row: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def _scheduler_job_row(
        self, venue: str, scheduler_job_id: str
    ) -> dict[str, Any] | None:  # pragma: no cover
        raise NotImplementedError

    def _budget_sums(
        self, tx: Any, project: str
    ) -> tuple[float, tuple[int, int]]:  # pragma: no cover
        raise NotImplementedError

    def _rows(self, venue: str | None) -> list[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError


def _matches(row: dict[str, Any], person: Any, project: Any, since: Any) -> bool:
    if person and row.get("person") != person:
        return False
    if project and row.get("project_id") != project:
        return False
    recorded = row.get("recorded_at")
    return not (since and isinstance(recorded, datetime) and recorded < since)


def _current_of(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The latest row per current-view key, in one pass over *rows*."""
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["venue"], row["run_id"], row["attempt_id"], row["step"])
        previous = latest.get(key)
        if previous is None or int(row["ledger_id"]) > int(previous["ledger_id"]):
            latest[key] = row
    return list(latest.values())


@dataclass
class InMemoryUsageLedger(UsageLedger):
    """The ledger over a list (no database). Pass one ``tables`` to several
    venues' ledgers to model the shared database; ``budgets`` maps a
    project to its ``max_gpu_hours``."""

    tables: dict[str, Any] = field(default_factory=lambda: {"rows": [], "budgets": {}})
    _lock: Any = field(default_factory=threading.RLock, repr=False)

    def max_gpu_hours(self, project_id: str) -> float:
        return float(self.tables["budgets"].get(project_id) or 0)

    @contextlib.contextmanager
    def _locked(self, unit: LedgerUnit, *, budget: bool) -> Iterator[None]:
        with self._lock:
            yield None

    @contextlib.contextmanager
    def _scheduler_job_locked(
        self, venue: str, scheduler_job_id: str
    ) -> Iterator[None]:
        with self._lock:
            yield None

    def _current(self, tx: Any, unit: LedgerUnit) -> dict[str, Any] | None:
        mine = [
            r
            for r in self.tables["rows"]
            if (r["venue"], r["attempt_id"]) == (unit.venue, unit.attempt_id)
        ]
        return dict(max(mine, key=lambda r: r["ledger_id"])) if mine else None

    def _insert(self, tx: Any, row: dict[str, Any]) -> None:
        rows = self.tables["rows"]
        rows.append({**row, "ledger_id": len(rows) + 1, "recorded_at": _now()})

    def _scheduler_job_row(
        self, venue: str, scheduler_job_id: str
    ) -> dict[str, Any] | None:
        with self._lock:
            return next(
                (
                    dict(row)
                    for row in self.tables["rows"]
                    if row["venue"] == venue
                    and row["scheduler_job_id"] == scheduler_job_id
                ),
                None,
            )

    def _budget_sums(self, tx: Any, project: str) -> tuple[float, tuple[int, int]]:
        if not project:
            return 0.0, (0, 0)
        sums = self.project_totals(project)
        return self.max_gpu_hours(project), (
            sums["gpu_seconds"],
            sums["reserved_gpu_seconds"],
        )

    def _rows(self, venue: str | None) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                dict(r)
                for r in self.tables["rows"]
                if venue is None or r["venue"] == venue
            ]
        return _current_of(rows)


# ── Postgres ─────────────────────────────────────────────────────────────────

_ROW_COLUMNS = ("ledger_id", *COLUMNS, "recorded_at")
_LATEST_FROM = (
    " FROM hpc_usage_ledger AS cur WHERE NOT EXISTS ("
    " SELECT 1 FROM hpc_usage_ledger AS newer"
    " WHERE newer.venue = cur.venue AND newer.run_id = cur.run_id"
    " AND newer.attempt_id = cur.attempt_id AND newer.step = cur.step"
    " AND newer.ledger_id > cur.ledger_id)"
)
_SELECT = (
    "SELECT " + ", ".join(f"cur.{column}" for column in _ROW_COLUMNS) + _LATEST_FROM
)
_USED = "(cur.source <> 'estimate' OR cur.state = 'unsettled')"
_RESERVED = "(cur.source = 'estimate' AND cur.state IN ('pending', 'submitted'))"
_TOTALS = (
    "SELECT COUNT(*) FILTER (WHERE cur.scheduler_job_id <> ''),"
    f" COALESCE(SUM(cur.alloc_gpus::bigint * cur.elapsed_s) FILTER (WHERE {_USED}), 0),"
    f" COALESCE(SUM(cur.alloc_cpus::bigint * cur.elapsed_s) FILTER (WHERE {_USED}), 0),"
    f" COALESCE(SUM(cur.alloc_gpus::bigint * cur.elapsed_s) FILTER (WHERE {_RESERVED}), 0),"
    f" COALESCE(SUM(cur.alloc_cpus::bigint * cur.elapsed_s) FILTER (WHERE {_RESERVED}), 0)"
    + _LATEST_FROM
)
_CURRENT_ONE = (
    f"SELECT {', '.join(_ROW_COLUMNS)} FROM hpc_usage_ledger"
    " WHERE venue = %s AND attempt_id = %s ORDER BY ledger_id DESC LIMIT 1"
)
_SCHEDULER_JOB_ONE = (
    f"SELECT {', '.join(_ROW_COLUMNS)} FROM hpc_usage_ledger"
    " WHERE venue = %s AND scheduler_job_id = %s ORDER BY ledger_id ASC LIMIT 1"
)
_INSERT = (
    f"INSERT INTO hpc_usage_ledger ({', '.join(COLUMNS)})"
    f" VALUES ({', '.join(['%s'] * len(COLUMNS))})"
)
_SUMS = (
    "SELECT"
    f" COALESCE(SUM(cur.alloc_gpus::bigint * cur.elapsed_s) FILTER (WHERE {_USED}), 0),"
    f" COALESCE(SUM(cur.alloc_gpus::bigint * cur.elapsed_s) FILTER (WHERE {_RESERVED}), 0)"
    + _LATEST_FROM
    + " AND cur.project_id = %s"
)
_BUDGET = "SELECT max_gpu_hours FROM project_budgets WHERE project_id = %s"


def _current_filters(
    venue: str | None,
    person: str | None,
    project: str | None,
    since: datetime | None,
    states: tuple[str, ...] = (),
) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("venue", venue),
        ("person", person),
        ("project_id", project),
    ):
        if value:
            clauses.append(f" AND cur.{column} = %s")
            params.append(value)
    if since is not None:
        clauses.append(" AND cur.recorded_at >= %s")
        params.append(since)
    if states:
        clauses.append(f" AND cur.state IN ({', '.join(['%s'] * len(states))})")
        params.extend(states)
    return "".join(clauses), tuple(params)


class PgUsageLedger(UsageLedger):
    """``hpc_usage_ledger`` over a psycopg2 pool (``getconn`` / ``putconn``)."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    @contextlib.contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextlib.contextmanager
    def _locked(self, unit: LedgerUnit, *, budget: bool) -> Iterator[Any]:
        """One transaction under ``hpc-budget:<project>`` (when *budget*),
        then ``hpc-ledger:<venue>:<attempt>`` advisory locks."""
        with self._conn() as conn:
            try:
                with conn.cursor() as cur:
                    if budget:
                        cur.execute(
                            "SELECT pg_advisory_xact_lock(hashtext(%s))",
                            ("hpc-budget:" + unit.project_id,),
                        )
                    key = f"hpc-ledger:{unit.venue}:{unit.attempt_id}"
                    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
                    yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @contextlib.contextmanager
    def _scheduler_job_locked(self, venue: str, scheduler_job_id: str) -> Iterator[Any]:
        """Serialize first association and later updates of one Slurm job."""
        key = f"hpc-ledger-job:{venue}:{scheduler_job_id}"
        with self._conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
                    yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _current(self, tx: Any, unit: LedgerUnit) -> dict[str, Any] | None:
        tx.execute(_CURRENT_ONE, (unit.venue, unit.attempt_id))
        row = tx.fetchone()
        return dict(zip(_ROW_COLUMNS, row)) if row else None

    def _insert(self, tx: Any, row: dict[str, Any]) -> None:
        tx.execute(_INSERT, tuple(row.get(k) for k in COLUMNS))

    def _scheduler_job_row(
        self, venue: str, scheduler_job_id: str
    ) -> dict[str, Any] | None:
        rows = self._query(_SCHEDULER_JOB_ONE, (venue, scheduler_job_id))
        return rows[0] if rows else None

    def _budget_sums(self, tx: Any, project: str) -> tuple[float, tuple[int, int]]:
        if not project:
            return 0.0, (0, 0)
        tx.execute(_BUDGET, (project,))
        found = tx.fetchone()
        tx.execute(_SUMS, (project,))
        used, reserved = tx.fetchone()
        return float(found[0]) if found else 0.0, (int(used), int(reserved))

    def pending(
        self, unit: LedgerUnit, estimate: Resources, *, estimate_complete: bool = True
    ) -> None:
        """D2: a failed pending write fails the submit before anything
        reaches the cluster (``hpcaccess.ledger_unavailable``)."""
        try:
            super().pending(unit, estimate, estimate_complete=estimate_complete)
        except (BudgetExceeded, LedgerUnavailable):
            raise
        except Exception as exc:
            raise LedgerUnavailable(
                f"the usage ledger is unavailable: {type(exc).__name__}"
            ) from exc

    def max_gpu_hours(self, project_id: str) -> float:
        with self._conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(_BUDGET, (project_id,))
                    found = cur.fetchone()
            finally:
                conn.rollback()
        return float(found[0]) if found else 0.0

    def _query_values(self, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        with self._conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return list(cur.fetchall())
            finally:
                conn.rollback()

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [dict(zip(_ROW_COLUMNS, row)) for row in self._query_values(sql, params)]

    def current(
        self,
        venue: str,
        *,
        person: str | None = None,
        project: str | None = None,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        filters, params = _current_filters(venue, person, project, since)
        sql = _SELECT + filters + " ORDER BY cur.ledger_id DESC LIMIT %s"
        return self._query(sql, (*params, max(0, int(limit))))

    def current_totals(
        self,
        venue: str,
        *,
        person: str | None = None,
        project: str | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        filters, params = _current_filters(venue, person, project, since)
        found = self._query_values(_TOTALS + filters, params)[0]
        return _amount_totals(found)

    def open_rows(self, venue: str) -> list[dict[str, Any]]:
        filters, params = _current_filters(venue, None, None, None, OPEN_STATES)
        return self._query(_SELECT + filters, params)

    def _rows(self, venue: str | None) -> list[dict[str, Any]]:
        filters, params = _current_filters(venue, None, None, None)
        return self._query(_SELECT + filters, params)

    def project_totals(self, project_id: str) -> dict[str, Any]:
        found = self._query_values(_TOTALS + " AND cur.project_id = %s", (project_id,))[
            0
        ]
        return _amount_totals(found)

    def project_rows(self, project_id: str) -> list[dict[str, Any]]:
        sql = _SELECT + " AND cur.project_id = %s ORDER BY cur.ledger_id DESC"
        return self._query(sql, (project_id,))


# ── the process's ledger ─────────────────────────────────────────────────────

_ledger: UsageLedger | None = None
_ledger_guard = threading.Lock()


def set_usage_ledger(ledger: UsageLedger | None) -> None:
    """Install the process's ledger (the server at startup; tests)."""
    global _ledger
    _ledger = ledger


def get_usage_ledger() -> UsageLedger:
    """The process's ledger: Postgres over the shared ``umrs_db`` pool when
    one is configured, else an in-memory one (a server without a database)."""
    global _ledger
    with _ledger_guard:
        if _ledger is None:
            _ledger = _default_ledger()
        return _ledger


def _default_ledger() -> UsageLedger:
    """A configured database that cannot be reached is
    :class:`LedgerUnavailable` (never a silent in-memory ledger)."""
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn
    from umrs.shared.hpc_user_work_access import _shared_pool

    if not _resolve_knowledge_dsn()[0]:
        return InMemoryUsageLedger()  # no database named: a server without one
    try:
        return PgUsageLedger(_shared_pool())
    except Exception as exc:
        raise LedgerUnavailable(
            f"the usage ledger is unavailable: {type(exc).__name__}"
        ) from exc


# ── what the submit points share (plan §6 C/D.0 D2) ──────────────────────────


class LedgerRefused(RuntimeError):
    """A submit the ledger stopped before anything reached the cluster:
    ``hpcaccess.budget_exceeded`` or ``hpcaccess.ledger_unavailable``.
    Never retried (K14: the run stops)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def write_pending(
    unit: LedgerUnit, est: Resources, *, estimate_complete: bool = True
) -> None:
    """The binding pending write; :class:`LedgerRefused` when it refuses."""
    try:
        get_usage_ledger().pending(unit, est, estimate_complete=estimate_complete)
    except (BudgetExceeded, LedgerUnavailable) as exc:
        raise LedgerRefused(exc.code, str(exc)) from exc


def write_after(op: str, unit: LedgerUnit, *args: Any) -> None:
    """A write after the submit (``submitted``, ``settled``,
    ``not_submitted``): a failure is logged and the sweep recovers it."""
    try:
        getattr(get_usage_ledger(), op)(unit, *args)
    except Exception:
        logger.warning(
            "ledger: %s of %s not written", op, unit.attempt_id, exc_info=True
        )


def write_job_after(
    op: str, unit: LedgerUnit, scheduler_job_id: str, *args: Any
) -> None:
    """Write job-scoped progress to the unit that first recorded that job.

    ``submitted`` takes ``(gate_log_seq,)`` and ``settled`` takes
    ``(usage, source)`` after *scheduler_job_id*. With no prior association
    (an adopted job predating the ledger), establish it before settlement so
    the final row cannot lose its scheduler id. As with :func:`write_after`,
    failures never fail the step; the sweep can retry them.
    """
    try:
        get_usage_ledger().write_job_after(op, unit, scheduler_job_id, *args)
    except Exception:
        logger.warning(
            "ledger: %s of %s not written", op, unit.attempt_id, exc_info=True
        )


def _terminal(state: str) -> str:
    from umrs.server.slurm_step_executor import TERMINAL_STATES

    word = str(state or "").split(" ")[0].strip().upper().rstrip("+")
    return word if word in TERMINAL_STATES else ""


def usage_from_sacct(row: dict[str, Any]) -> dict[str, Any] | None:
    """A settlement from one ``sacct`` row (the gate's ``usage`` row, or the
    platform's own): None unless the state is terminal and ``Elapsed``
    parses (never approximated)."""
    from umrs.shared.slurm_accounting import parse_alloc_tres_gpus, parse_slurm_elapsed

    state = _terminal(str(row.get("state") or ""))
    elapsed = parse_slurm_elapsed(str(row.get("elapsed") or ""))
    if not state or elapsed is None:
        return None
    cpus = str(row.get("alloc_cpus") or "0").strip()
    return {
        "state": state,
        "elapsed_s": int(elapsed),
        "alloc_cpus": int(cpus) if cpus.isdigit() else 0,
        "alloc_gpus": parse_alloc_tres_gpus(str(row.get("alloc_tres") or "")),
        "start": _sacct_time(row.get("start")),
        "end": _sacct_time(row.get("end")),
    }


def _sacct_time(value: Any) -> datetime | None:
    """A ``sacct`` time (site-local ISO without a zone: taken as UTC)."""
    try:
        parsed = datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


__all__ = [
    "ACCESS_MODES",
    "COLUMNS",
    "LANES",
    "METHODS",
    "OPEN_STATES",
    "PRIVATE_COLUMNS",
    "SOURCES",
    "BudgetExceeded",
    "InMemoryUsageLedger",
    "LedgerRefused",
    "LedgerUnavailable",
    "LedgerUnit",
    "PgUsageLedger",
    "Resources",
    "UsageLedger",
    "accepts",
    "budget_seconds",
    "budget_view",
    "check_budget",
    "estimate",
    "get_usage_ledger",
    "requested_partition",
    "row_view",
    "set_usage_ledger",
    "totals",
    "unit_for",
    "usage_from_sacct",
    "write_after",
    "write_job_after",
    "write_pending",
]
