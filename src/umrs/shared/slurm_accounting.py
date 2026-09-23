"""D-36 leg B — Slurm job-cost accounting primitives (ADR-067's
never-approximate rule).

The only GPU-second number that counts is a MEASURED one: the
scheduler's own accounting (``sacct`` Elapsed × AllocTRES%gres). Parse
helpers here are pure and total — garbage answers ``None``/``0``, never
a guess; callers stamp only what was really measured (unmeasured =
absent, never zero-invented).
"""

from __future__ import annotations

import re

# Slurm Elapsed formats: [DD-]HH:MM:SS (days also print as e.g. "2-00:05:00").
_ELAPSED_RE = re.compile(
    r"^(?:(\d+)-)?(?:(\d+):)?(\d{1,2}):(\d{1,2})$"
)

# AllocTRES%gres shapes across Slurm versions: "gres:gpu=2", "gres/gpu=2",
# "gpu=2", "gpu:2". Take the FIRST gpu allocation found.
_GPU_TRES_RE = re.compile(r"(?:^|[,])\s*gres[:/]gpu[=:](\d+)|(?:^|[,])\s*gpu[=:](\d+)")


def parse_slurm_elapsed(text: str) -> float | None:
    """Parse a sacct Elapsed value to seconds; ``None`` when unparseable."""
    raw = str(text or "").strip().upper()
    if not raw or raw in {"INVALID", "UNKNOWN", "NONE"}:
        return None
    match = _ELAPSED_RE.match(raw)
    if match is None:
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def parse_alloc_tres_gpus(text: str) -> int:
    """Count GPUs in an ``AllocTRES%gres`` value; 0 = none (CPU job)."""
    raw = str(text or "").strip()
    if not raw:
        return 0
    match = _GPU_TRES_RE.search(raw)
    if match is None:
        return 0
    first = match.group(1) or match.group(2)
    return int(first) if first else 0


def measured_gpu_seconds(elapsed: float | None, gpus: int) -> float:
    """GPU-seconds for one job (0.0 when nothing was measured/allocated)."""
    if elapsed is None or elapsed <= 0 or gpus <= 0:
        return 0.0
    return float(elapsed) * int(gpus)
