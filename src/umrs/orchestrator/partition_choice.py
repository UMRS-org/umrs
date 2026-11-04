"""One Slurm partition per submission, chosen from a candidate list.

A site's ``scheduler.partition`` (or a pool roster entry's ``partition``) may
name several candidates as a comma list. Some clusters refuse a list at
submit: the RCCS job_submit policy answers ``sbatch --partition=a,b`` with
"Multiple partitions are not allowed" (2026-10-24), and its per-user,
per-partition quota ("Quota exceeded on partition 'ng-dgx-m0' (limit: 2 x 72
node-hours per user)") rejects a job on one partition while the others idle.

A site opts in with ``scheduler.partition_choice: earliest_start``. UMRS then
keeps the whole candidate list until submit time, asks Slurm with
``sbatch --test-only`` (submits nothing) when the job would start on each
candidate, and submits to the earliest. Starts within ``TIE_TOLERANCE_SEC``
of the earliest count as a tie (concurrent probes of idle partitions answer
with clock seconds a moment apart); a tie goes to the candidate this process
picked least often in the last ``RECENT_PICK_TTL_SEC`` (``--test-only``
simulates only running jobs, so a job submitted seconds ago and not yet
started is invisible to it), then to the configured order. Without the
opt-in (the default, ``first``) the first candidate is used, exactly as
before.

A job that queued anyway (every candidate busy at submit) is not stuck on
its partition (2026-11-04): the janitor asks the backend to move a pending
provision to another of its candidates once that one is *blank*, meaning a
``--test-only`` probe there starts within ``BLANK_SLACK_SEC`` of the
cluster's clock (:func:`blank_target`). The candidates of one entry are
equivalent hardware by configuration (rccs: every ``ng-dgx-s`` /
``ng-dgx-m*`` partition is a GB10 partition), so a move changes only where
the job waits.

Everything here except the in-process ``recent_picks`` record is pure (no
I/O) so the remote backend and the legacy launcher share it; they own the
ssh calls.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Collection, Mapping
from datetime import datetime, timedelta
from typing import Any

PARTITION_CHOICE_FIRST = "first"
PARTITION_CHOICE_EARLIEST = "earliest_start"

# Starts this close to the earliest are a tie (probe answers are clock
# seconds, and concurrent probes are evaluated a moment apart).
TIE_TOLERANCE_SEC = 60
# How long a pick counts against its partition on a tie: covers the gap
# between submit and start (Slurm's main/backfill passes run every ~30-60 s).
RECENT_PICK_TTL_SEC = 120
# A candidate whose probe starts this close to the cluster's clock is blank:
# the job would start there in the next scheduling passes.
BLANK_SLACK_SEC = 120
# Pending reasons a partition move cannot help: the job is held, or waits
# for another job, a begin time or its own reservation.
_UNMOVABLE_REASONS = ("JobHeld", "Dependency", "BeginTime", "Reservation")

# sbatch --test-only success (stderr): "sbatch: Job 426027 to start at
# 2026-10-24T23:01:59 using 20 processors on nodes ng-dgx-m1-10 in partition
# ng-dgx-m1".
_TEST_ONLY_RE = re.compile(
    r"to start at (?P<start>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:.*?in partition (?P<partition>[A-Za-z0-9_.-]+))?"
)
_QUOTA_RE = re.compile(r"Quota exceeded on partition '(?P<partition>[^']+)'")


def split_partitions(value: str | None) -> list[str]:
    """The candidates of a partition value (a name or a comma list), in order."""
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def spreads_partitions(site: Any) -> bool:
    """True when the site opted in to choosing among its candidates."""
    scheduler = getattr(site, "scheduler", None)
    choice = getattr(scheduler, "partition_choice", PARTITION_CHOICE_FIRST)
    return str(choice or "").strip() == PARTITION_CHOICE_EARLIEST


def preferred_partitions(site: Any) -> frozenset[str]:
    """``scheduler.preferred_partitions``: candidates that win a start tie
    over every other candidate, before the recent-pick spreading (rccs:
    ng-dgx-s, so single-node jobs keep the dual-DGX m pairs free and use an
    m partition only when no s node starts as early; 2026-11-03)."""
    raw = getattr(getattr(site, "scheduler", None), "preferred_partitions", "") or ""
    return frozenset(split_partitions(str(raw)))


def default_partition(site: Any) -> str:
    """The partition value a job gets when nothing more specific names one.

    An opted-in site keeps every candidate (normalized ``a,b,c``) so the
    choice happens at submit time; any other site keeps its first candidate.
    """
    raw = getattr(getattr(site, "scheduler", None), "partition", "") or ""
    candidates = split_partitions(raw)
    if spreads_partitions(site):
        return ",".join(candidates)
    return candidates[0] if candidates else ""


def parse_test_only(output: str) -> tuple[datetime, str] | None:
    """``(start, partition)`` from ``sbatch --test-only`` output, else None.

    ``partition`` is "" when the line does not name one.
    """
    match = _TEST_ONLY_RE.search(output or "")
    if match is None:
        return None
    try:
        start = datetime.fromisoformat(match.group("start"))
    except ValueError:
        return None
    return start, match.group("partition") or ""


def quota_exceeded_partition(output: str) -> str | None:
    """The partition an RCCS "Quota exceeded" refusal names, else None."""
    match = _QUOTA_RE.search(output or "")
    return match.group("partition") if match else None


def rank_by_start(
    candidates: list[str],
    starts: Mapping[str, datetime | None],
    load: Mapping[str, int] | None = None,
    preferred: Collection[str] = (),
) -> list[str]:
    """Candidates with a start time, earliest first; refused ones dropped.

    Starts within ``TIE_TOLERANCE_SEC`` of the earliest tie; a tie goes to a
    ``preferred`` partition first, then to the lower ``load`` (recent picks),
    then to the configured order. A preferred partition that starts later
    than the tie window does not win: the job falls back to the earliest.
    """
    known = {cand: start for cand in candidates if (start := starts.get(cand))}
    if not known:
        return []
    earliest = min(known.values())
    window = timedelta(seconds=TIE_TOLERANCE_SEC)
    counts = load or {}

    def _key(cand: str) -> tuple[datetime, bool, int]:
        start = known[cand]
        bucket = earliest if start - earliest <= window else start
        return (bucket, cand not in preferred, counts.get(cand, 0))

    return sorted(known, key=_key)


class RecentPicks:
    """Partitions this process chose in the last ``ttl_sec`` (thread-safe).

    ``sbatch --test-only`` cannot see a job that is still pending, so a burst
    of submits would all tie on the same idle partition; counting the
    process's own recent picks spreads the burst.
    """

    def __init__(
        self,
        ttl_sec: float = RECENT_PICK_TTL_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_sec
        self._clock = clock
        self._picks: list[tuple[float, str]] = []
        self._lock = threading.Lock()

    def note(self, partition: str) -> None:
        """Record that a submit was sent to ``partition`` now."""
        with self._lock:
            self._picks.append((self._clock(), partition))

    def counts(self) -> dict[str, int]:
        """Recent picks per partition (expired ones are forgotten)."""
        cutoff = self._clock() - self._ttl
        with self._lock:
            self._picks = [pick for pick in self._picks if pick[0] > cutoff]
            out: dict[str, int] = {}
            for _, partition in self._picks:
                out[partition] = out.get(partition, 0) + 1
        return out

    def clear(self) -> None:
        """Forget every pick (tests)."""
        with self._lock:
            self._picks.clear()


# One record per process: backends are created per job, the picks must not be.
recent_picks = RecentPicks()


def choose(
    candidates: list[str],
    starts: Mapping[str, datetime | None],
    preferred: Collection[str] = (),
) -> list[str]:
    """Rank by start (ties: fewest recent picks) and record the winner.

    Ranking and recording happen without an await in between, so concurrent
    submits in one event loop see each other's picks.
    """
    ranked = rank_by_start(candidates, starts, recent_picks.counts(), preferred)
    if ranked:
        recent_picks.note(ranked[0])
    return ranked


def redact_exports(args: list[str]) -> str:
    """The command line for a log: ``--export=`` values (may carry keys) hidden."""
    return " ".join(
        "--export=<redacted>" if arg.startswith("--export=") else arg for arg in args
    )


def with_partition(sbatch_args: list[str], partition: str) -> list[str]:
    """``sbatch_args`` with its ``--partition=`` replaced (or added)."""
    flag = f"--partition={partition}"
    out = [flag if arg.startswith("--partition=") else arg for arg in sbatch_args]
    if flag not in out:
        out.insert(1, flag)
    return out


def probe_args(sbatch_args: list[str], partition: str) -> list[str]:
    """The ``sbatch --test-only`` probe for one candidate.

    Same resource directives and script as the real submit, so Slurm checks
    the same limits and quota. ``--parsable`` and ``--export`` are dropped:
    the probe starts nothing, so the job environment is irrelevant, and
    leaving it out keeps the hex-encoded env (it may carry keys) off extra
    command lines.
    """
    kept = [
        arg
        for arg in with_partition(sbatch_args, partition)
        if arg != "--parsable" and not arg.startswith("--export=")
    ]
    return [kept[0], "--test-only", *kept[1:]]


def movable_reason(reason: str | None) -> bool:
    """True when moving a pending job with this ``squeue %r`` reason can help."""
    return not any(mark in (reason or "") for mark in _UNMOVABLE_REASONS)


def parse_clock(text: str) -> datetime | None:
    """The cluster's clock from ``date +%Y-%m-%dT%H:%M:%S`` output, else None."""
    try:
        return datetime.fromisoformat(first_line(text))
    except ValueError:
        return None


def blank_target(
    current: str,
    starts: Mapping[str, datetime | None],
    now: datetime,
    preferred: Collection[str] = (),
) -> str | None:
    """The blank candidate a job pending on ``current`` should move to, else None.

    ``starts`` holds the ``--test-only`` start of every other candidate, in
    configured order. Blank means a start within ``BLANK_SLACK_SEC`` of
    ``now`` (the cluster's clock, the same zone as the probe answers). Among
    blank candidates a ``preferred`` one wins, then the fewest recent picks,
    then the configured order, as at submit.
    """
    slack = timedelta(seconds=BLANK_SLACK_SEC)
    blank = {
        cand: start
        for cand, start in starts.items()
        if cand != current and start is not None and start - now <= slack
    }
    ranked = rank_by_start(list(blank), blank, recent_picks.counts(), preferred)
    return ranked[0] if ranked else None


def first_line(text: str) -> str:
    """The first non-empty line, for a one-line log of why a probe failed."""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""
