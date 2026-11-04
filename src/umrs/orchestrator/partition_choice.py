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


def first_line(text: str) -> str:
    """The first non-empty line, for a one-line log of why a probe failed."""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


# ── per-partition GPU request (ADR-109 D-3) ─────────────────────────────────

#: ``partition_profiles.<p>.gpu_request``: "" keeps the caller's site-wide
#: rule; "none" never asks for GPUs (RCCS GB10: ng-dgx-* reject --gres and
#: --gpus); "gres" asks with ``--gres=<profile gres, else gpu:<count>>``.
GPU_REQUEST_LEGACY = ""
GPU_REQUEST_NONE = "none"
GPU_REQUEST_GRES = "gres"
GPU_REQUEST_STYLES = (GPU_REQUEST_LEGACY, GPU_REQUEST_NONE, GPU_REQUEST_GRES)
#: What a gres value may hold: it goes onto an sbatch command or #SBATCH line.
_GRES_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def gpu_request_style(profile: Any) -> str:
    """The profile's GPU request style; no profile is the legacy style."""
    style = str(getattr(profile, "gpu_request", "") or "").strip()
    if style not in GPU_REQUEST_STYLES:
        raise ValueError(
            f"partition gpu_request {style!r} is not one of none|gres (or empty)"
        )
    return style


def gres_request(style: str, gres: str, count: int, legacy: str | None) -> str | None:
    """The value after ``--gres=`` on one partition, or None for no line.

    ``legacy`` is what the caller's site-wide rule asks; style "" keeps it.
    """
    if style == GPU_REQUEST_NONE:
        return None
    if style != GPU_REQUEST_GRES:
        return legacy
    value = str(gres or "").strip() or (f"gpu:{int(count)}" if count > 0 else "")
    if value and not _GRES_RE.fullmatch(value):
        raise ValueError(f"partition gres {value!r} is not a Slurm gres string")
    return value or None


def gres_gpu_count(gres: str) -> int:
    """GPUs a ``gpu[:type][:count]`` gres string asks for (0: no GPU gres)."""
    parts = str(gres or "").strip().split(":")
    if parts[0] != "gpu":
        return 0
    last = parts[-1] if len(parts) > 1 else ""
    return int(last) if last.isdigit() else 1


def partition_gpus(profile: Any) -> int:
    """GPUs one job on the partition can get: its gres, else ``gpus``."""
    if profile is None:
        return 0
    count = gres_gpu_count(getattr(profile, "gres", ""))
    return count or int(getattr(profile, "gpus", 0) or 0)


def with_gpu_request(sbatch_args: list[str], profile: Any, count: int) -> list[str]:
    """``sbatch_args`` with the ``--gres=`` and ``--exclusive`` the partition's
    profile asks for (``count``: the job's GPUs, for a gres without a string).
    A legacy profile (style "", not exclusive), or none, changes nothing."""
    style = gpu_request_style(profile)
    exclusive = bool(getattr(profile, "exclusive", False))
    if style == GPU_REQUEST_LEGACY and not exclusive:
        return list(sbatch_args)
    legacy = _arg_value(sbatch_args, "--gres=")
    value = gres_request(style, getattr(profile, "gres", ""), count, legacy)
    request = [f"--gres={value}"] if value else []
    if exclusive:
        request.append("--exclusive")
    kept = [
        a for a in sbatch_args if not a.startswith("--gres=") and a != "--exclusive"
    ]
    return _after_partition(kept, request)


def _arg_value(sbatch_args: list[str], prefix: str) -> str | None:
    """The value of the first ``<prefix><value>`` argument, else None."""
    for arg in sbatch_args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _after_partition(sbatch_args: list[str], extra: list[str]) -> list[str]:
    """``extra`` inserted right after ``--partition=`` (else after argv[0])."""
    at = 1
    for index, arg in enumerate(sbatch_args):
        if arg.startswith("--partition="):
            at = index + 1
            break
    return [*sbatch_args[:at], *extra, *sbatch_args[at:]]
