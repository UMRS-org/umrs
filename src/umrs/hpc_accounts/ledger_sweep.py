"""The usage ledger's recovery sweep (ADR-112 wave D; plan §6 C/D.0 D2
"Recovery sweep", K5, K13b).

An in-server timer (every 300 s) over this venue's open rows only:

1. a row still ``pending`` after 15 minutes is resolved: a gated row by the
   gate's ``lookup`` (a job: ``submitted``; exit 3: ``not_submitted``;
   anything else leaves it); a platform row by ``sacct --name <tag>`` from
   ``recorded_at - 60 s``, among jobs no other row of the venue holds (one,
   or the earliest of several: ``submitted``; none after one hour:
   ``not_submitted``; an ``sacct`` failure leaves it);
2. a ``submitted`` row is settled at a terminal state: a gated row by the
   person's gate's ``usage`` (batches of at most 100 ids), a platform row by
   ``sacct`` on the platform login;
3. an unreachable gate or a revoked account leaves the estimate (it keeps
   counting as reserved);
4. a row still open 7 days after ``submitted_at`` (else ``recorded_at``)
   plus its requested time becomes ``unsettled`` at its estimate (K5).

The cluster calls are behind two resolvers so the sweep is testable; the
defaults reach the person's gate (:mod:`umrs.server.gated_step_lane`) and
the platform login.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from .usage_ledger import LedgerUnit, usage_from_sacct

logger = logging.getLogger(__name__)

PENDING_GRACE = timedelta(minutes=15)
NOT_SUBMITTED_AFTER = timedelta(hours=1)
UNSETTLED_AFTER = timedelta(days=7)
MATCH_SLACK = timedelta(seconds=60)
USAGE_BATCH = 100
INTERVAL_S = 300.0

_UNIT_FIELDS = tuple(f.name for f in fields(LedgerUnit))


class GatedResolver(Protocol):
    def lookup(self, row: dict[str, Any]) -> tuple[str, str]:
        """``("job", id)``, ``("absent", "")`` or ``("unknown", "")``."""

    def usage(
        self, row: dict[str, Any], job_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """``{job_id: usage row}`` from the person's gate; raises when unreachable."""


class ServiceResolver(Protocol):
    def match(
        self, row: dict[str, Any], since: datetime
    ) -> list[tuple[str, str]] | None:
        """``[(job_id, submit time)]`` named ``row["tag"]`` since *since*; None: sacct failed."""

    def sacct(self, job_ids: list[str]) -> dict[str, dict[str, Any]] | None:
        """``{job_id: sacct row}``; None: sacct failed."""


def unit_of(row: dict[str, Any]) -> LedgerUnit:
    return LedgerUnit(**{k: str(row.get(k) or "") for k in _UNIT_FIELDS})


def _aware(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def _gated(row: dict[str, Any]) -> bool:
    return row.get("method") == "gated_key"


def _limit(row: dict[str, Any]) -> datetime | None:
    """When an open row becomes ``unsettled`` (K5)."""
    start = _aware(row.get("submitted_at")) or _aware(row.get("recorded_at"))
    if start is None:
        return None
    return start + timedelta(seconds=int(row.get("elapsed_s") or 0)) + UNSETTLED_AFTER


@dataclass
class UsageLedgerSweep:
    """One venue's recovery sweep over *ledger* (see the module text)."""

    ledger: Any
    venue: str
    gated: GatedResolver
    service: ServiceResolver
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def run_once(self) -> dict[str, int]:
        """One pass; the count of rows it changed per outcome."""
        counts = {"submitted": 0, "not_submitted": 0, "settled": 0, "unsettled": 0}
        taken = self._taken_ids()  # one venue read per cycle, not per pending row
        rows = self.ledger.open_rows(self.venue)
        rows = self._expire(rows, counts)
        for row in [r for r in rows if r.get("state") == "pending"]:
            self._resolve_pending(row, counts, taken)
        self._settle([r for r in rows if r.get("state") == "submitted"], counts)
        return counts

    # K5

    def _expire(
        self, rows: list[dict[str, Any]], counts: dict[str, int]
    ) -> list[dict[str, Any]]:
        now, kept = self.now(), []
        for row in rows:
            limit = _limit(row)
            if limit is not None and now > limit:
                self.ledger.mark_unsettled(unit_of(row))
                counts["unsettled"] += 1
            else:
                kept.append(row)
        return kept

    # 1. pending

    def _old(self, row: dict[str, Any], age: timedelta) -> bool:
        recorded = _aware(row.get("recorded_at"))
        return recorded is not None and self.now() - recorded > age

    def _resolve_pending(
        self, row: dict[str, Any], counts: dict[str, int], taken: set[str]
    ) -> None:
        if not self._old(row, PENDING_GRACE):
            return
        outcome = (
            self._pending_gated(row, taken)
            if _gated(row)
            else self._pending_service(row, taken)
        )
        if outcome:
            counts[outcome] += 1

    def _pending_gated(self, row: dict[str, Any], taken: set[str]) -> str:
        try:
            kind, job_id = self.gated.lookup(row)
        except Exception:  # noqa: BLE001 - an unreachable gate leaves the row
            return ""
        if kind == "job":
            if job_id in taken:
                # The new dispatch already associated this adopted job. This
                # pending estimate is its crash-window alias, not another job.
                self.ledger.not_submitted(unit_of(row), "gate_usage")
                return "not_submitted"
            self.ledger.submitted(unit_of(row), job_id, None)
            taken.add(job_id)
            return "submitted"
        if kind == "absent":
            self.ledger.not_submitted(unit_of(row), "gate_usage")
            return "not_submitted"
        return ""

    def _taken_ids(self) -> set[str]:
        """Job ids already on a row of this venue (K13b)."""
        return {
            str(r.get("scheduler_job_id"))
            for r in self.ledger.current(self.venue, limit=1 << 30)
            if r.get("scheduler_job_id")
        }

    def _pending_service(self, row: dict[str, Any], taken: set[str]) -> str:
        since = (_aware(row.get("recorded_at")) or self.now()) - MATCH_SLACK
        found = self.service.match(row, since)
        if found is None:
            return ""
        free = sorted((submit, job) for job, submit in found if job not in taken)
        if free:
            self.ledger.submitted(unit_of(row), free[0][1], None)
            taken.add(free[0][1])
            return "submitted"
        if self._old(row, NOT_SUBMITTED_AFTER):
            self.ledger.not_submitted(unit_of(row), "sacct")
            return "not_submitted"
        return ""

    # 2. submitted

    def _settle(self, rows: list[dict[str, Any]], counts: dict[str, int]) -> None:
        for batch, gated in self._batches(rows):
            measured = self._measure(batch, gated)
            for row in batch:
                usage = usage_from_sacct(
                    measured.get(str(row.get("scheduler_job_id")), {})
                )
                if usage is not None:
                    job_id = str(row.get("scheduler_job_id") or "")
                    source = "gate_usage" if gated else "sacct"
                    self.ledger.write_job_after(
                        "settled", unit_of(row), job_id, usage, source
                    )
                    counts["settled"] += 1

    def _batches(
        self, rows: list[dict[str, Any]]
    ) -> list[tuple[list[dict[str, Any]], bool]]:
        """Gated rows per (person, login), platform rows together; each at
        most :data:`USAGE_BATCH` ids."""
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = (
                (True, row.get("person"), row.get("site_login"))
                if _gated(row)
                else (False,)
            )
            groups.setdefault(key, []).append(row)
        return [
            (group[i : i + USAGE_BATCH], bool(key[0]))
            for key, group in groups.items()
            for i in range(0, len(group), USAGE_BATCH)
        ]

    def _measure(
        self, batch: list[dict[str, Any]], gated: bool
    ) -> dict[str, dict[str, Any]]:
        ids = [str(r.get("scheduler_job_id")) for r in batch]
        try:
            found = (
                self.gated.usage(batch[0], ids) if gated else self.service.sacct(ids)
            )
        except Exception:  # noqa: BLE001 - unreachable gate or revoked account: the estimate stays
            logger.info(
                "ledger sweep: %d rows of %s not measured",
                len(ids),
                batch[0].get("person"),
            )
            return {}
        return found or {}


# ── the default resolvers ────────────────────────────────────────────────────


def _gate_key(row: dict[str, Any]) -> str:
    if row.get("grant_id"):
        from r4_contracts import hpc_gate

        return str(
            hpc_gate.grant_run_key(str(row["grant_id"]), str(row.get("run_id") or ""))
        )
    return str(row.get("intent_id") or "")


class GateResolver:
    """The person's own gate over their transport key."""

    def __init__(self, venue: str) -> None:
        self.venue = venue

    def _transport(self, row: dict[str, Any]) -> Any:
        from umrs.server.gated_step_lane import transport_for

        return transport_for(
            self.venue, str(row.get("person") or ""), str(row.get("site_login") or "")
        )

    def lookup(self, row: dict[str, Any]) -> tuple[str, str]:
        import json

        proc = self._transport(row).call(
            "lookup", _gate_key(row), str(row.get("step") or "")
        )
        if proc.returncode == 3:
            return "absent", ""
        if proc.returncode != 0:
            return "unknown", ""
        job_id = str(json.loads(proc.stdout or b"{}").get("job_id") or "")
        return ("job", job_id) if job_id else ("unknown", "")

    def usage(
        self, row: dict[str, Any], job_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        import json

        proc = self._transport(row).call("usage", *job_ids)
        if proc.returncode != 0:
            raise RuntimeError(f"gate usage failed (rc={proc.returncode})")
        jobs = json.loads(proc.stdout or b"{}").get("jobs") or []
        return {str(j.get("job_id")): j for j in jobs if isinstance(j, dict)}


class PlatformResolver:
    """``sacct`` on the platform login (the site's own SSH control master)."""

    def __init__(self, run: Callable[[str], tuple[int, str]] | None = None) -> None:
        self._run = run or _platform_run

    def match(
        self, row: dict[str, Any], since: datetime
    ) -> list[tuple[str, str]] | None:
        import shlex

        start = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        rc, out = self._run(
            f"sacct -n -X -P -S {start} --name {shlex.quote(str(row.get('tag') or ''))} -o JobID,Submit"
        )
        if rc != 0:
            return None
        pairs = [line.split("|", 1) for line in out.splitlines() if "|" in line]
        return [
            (job.strip(), submit.strip())
            for job, submit in pairs
            if job.strip().isdigit()
        ]

    def sacct(self, job_ids: list[str]) -> dict[str, dict[str, Any]] | None:
        from umrs.jobs.hpc_ledger import SACCT_FIELDS, sacct_row

        rc, out = self._run(f"sacct -n -X -P -j {','.join(job_ids)} -o {SACCT_FIELDS}")
        if rc != 0:
            return None
        rows = [sacct_row(line) for line in out.splitlines()]
        return {r["job_id"]: r for r in rows if r}


def _platform_run(command: str) -> tuple[int, str]:
    import asyncio

    from umrs.orchestrator.base import OrchestratorConfig
    from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend

    async def call() -> tuple[int, str]:
        ssh = await RemoteSlurmBackend(OrchestratorConfig())._get_ssh()
        result = await ssh.run_remote(command, timeout=60)
        return int(result.returncode), str(result.stdout)

    return asyncio.run(call())


# ── the timer ────────────────────────────────────────────────────────────────


def start_usage_ledger_sweep(
    venue: str, *, interval: float = INTERVAL_S
) -> threading.Thread | None:
    """Run :class:`UsageLedgerSweep` for *venue* every *interval* seconds on
    a daemon thread (the server's boot; ``UMRS_LEDGER_SWEEP_ENABLED=false``
    turns it off)."""
    import os

    from .usage_ledger import get_usage_ledger

    if not venue or os.environ.get(
        "UMRS_LEDGER_SWEEP_ENABLED", "true"
    ).strip().lower() in ("false", "0", "no"):
        return None
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            try:
                UsageLedgerSweep(
                    get_usage_ledger(), venue, GateResolver(venue), PlatformResolver()
                ).run_once()
            except Exception:
                logger.warning("usage ledger sweep failed", exc_info=True)

    thread = threading.Thread(target=loop, name="usage-ledger-sweep", daemon=True)
    thread.start()
    return thread


__all__ = [
    "GateResolver",
    "PlatformResolver",
    "UsageLedgerSweep",
    "start_usage_ledger_sweep",
    "unit_of",
]
