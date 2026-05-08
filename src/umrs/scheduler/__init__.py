"""UMRS Schedule system — cron-based job scheduling.

Allows users to schedule recurring jobs (recipe runs, connector
syncs, maintenance tasks) using cron expressions. Scheduled jobs
are executed as durable UMRS jobs.

No dependency on r4.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScheduleStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScheduleSpec:
    """A scheduled job specification."""

    schedule_id: str = ""
    name: str = ""
    cron_expr: str = ""  # Standard 5-field cron: min hour dom month dow
    job_type: str = ""  # e.g. "connector_sync", "recipe_run", "maintenance"
    job_config: Dict[str, Any] = field(default_factory=dict)
    owner_id: str = ""
    project_id: Optional[str] = None
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: Optional[datetime] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None  # None = unlimited
    timezone_name: str = "UTC"

    def __post_init__(self) -> None:
        if not self.schedule_id and self.name:
            self.schedule_id = hashlib.sha256(
                f"{self.owner_id}:{self.name}".encode()
            ).hexdigest()[:16]


# ── Cron parser (minimal, 5-field) ──────────────────────────────────────────

_CRON_FIELD_RANGES = [
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 6),    # day of week (0=Sun)
]


def validate_cron(expr: str) -> List[str]:
    """Validate a 5-field cron expression. Returns errors."""
    errors: List[str] = []
    parts = expr.strip().split()
    if len(parts) != 5:
        errors.append(f"Expected 5 fields, got {len(parts)}")
        return errors

    field_names = ["minute", "hour", "day_of_month", "month", "day_of_week"]
    for i, (part, (lo, hi)) in enumerate(zip(parts, _CRON_FIELD_RANGES)):
        if part == "*":
            continue
        # Handle */N
        if part.startswith("*/"):
            try:
                step = int(part[2:])
                if step < 1:
                    errors.append(f"{field_names[i]}: step must be >= 1")
            except ValueError:
                errors.append(f"{field_names[i]}: invalid step '{part}'")
            continue
        # Handle ranges and lists
        for token in part.split(","):
            if "-" in token:
                bounds = token.split("-")
                if len(bounds) != 2:
                    errors.append(f"{field_names[i]}: invalid range '{token}'")
                    continue
                try:
                    a, b = int(bounds[0]), int(bounds[1])
                    if a < lo or b > hi or a > b:
                        errors.append(f"{field_names[i]}: range {a}-{b} out of bounds ({lo}-{hi})")
                except ValueError:
                    errors.append(f"{field_names[i]}: non-numeric range '{token}'")
            else:
                try:
                    v = int(token)
                    if v < lo or v > hi:
                        errors.append(f"{field_names[i]}: value {v} out of bounds ({lo}-{hi})")
                except ValueError:
                    errors.append(f"{field_names[i]}: non-numeric value '{token}'")
    return errors


def _cron_field_matches(field_expr: str, value: int) -> bool:
    """Check if a value matches a cron field expression."""
    if field_expr == "*":
        return True
    if field_expr.startswith("*/"):
        step = int(field_expr[2:])
        return value % step == 0
    for token in field_expr.split(","):
        if "-" in token:
            a, b = token.split("-")
            if int(a) <= value <= int(b):
                return True
        elif int(token) == value:
            return True
    return False


def cron_matches(expr: str, dt: datetime) -> bool:
    """Check if a datetime matches a cron expression."""
    parts = expr.strip().split()
    if len(parts) != 5:
        return False
    minute, hour, dom, month, dow = parts
    return (
        _cron_field_matches(minute, dt.minute)
        and _cron_field_matches(hour, dt.hour)
        and _cron_field_matches(dom, dt.day)
        and _cron_field_matches(month, dt.month)
        and _cron_field_matches(dow, dt.weekday())  # Python: Mon=0
    )


# ── Schedule store ──────────────────────────────────────────────────────────


class ScheduleStore:
    """In-memory schedule store. Production uses Postgres."""

    def __init__(self) -> None:
        self._schedules: Dict[str, ScheduleSpec] = {}

    def add(self, spec: ScheduleSpec) -> ScheduleSpec:
        errors = validate_cron(spec.cron_expr)
        if errors:
            raise ValueError(f"Invalid cron: {'; '.join(errors)}")
        if not spec.created_at:
            spec.created_at = datetime.now(timezone.utc)
        self._schedules[spec.schedule_id] = spec
        logger.info("Added schedule: %s (id=%s)", spec.name, spec.schedule_id)
        return spec

    def get(self, schedule_id: str) -> Optional[ScheduleSpec]:
        return self._schedules.get(schedule_id)

    def list_active(self, owner_id: Optional[str] = None) -> List[ScheduleSpec]:
        results: List[ScheduleSpec] = []
        for s in self._schedules.values():
            if s.status != ScheduleStatus.ACTIVE:
                continue
            if owner_id and s.owner_id != owner_id:
                continue
            results.append(s)
        return results

    def pause(self, schedule_id: str) -> bool:
        s = self._schedules.get(schedule_id)
        if s:
            s.status = ScheduleStatus.PAUSED
            return True
        return False

    def resume(self, schedule_id: str) -> bool:
        s = self._schedules.get(schedule_id)
        if s and s.status == ScheduleStatus.PAUSED:
            s.status = ScheduleStatus.ACTIVE
            return True
        return False

    def remove(self, schedule_id: str) -> bool:
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            return True
        return False

    def get_due(self, now: Optional[datetime] = None) -> List[ScheduleSpec]:
        """Return schedules that are due to run at the given time."""
        if now is None:
            now = datetime.now(timezone.utc)
        due: List[ScheduleSpec] = []
        for s in self._schedules.values():
            if s.status != ScheduleStatus.ACTIVE:
                continue
            if s.max_runs is not None and s.run_count >= s.max_runs:
                s.status = ScheduleStatus.COMPLETED
                continue
            if cron_matches(s.cron_expr, now):
                due.append(s)
        return due

    def mark_run(self, schedule_id: str, run_time: Optional[datetime] = None) -> None:
        s = self._schedules.get(schedule_id)
        if s:
            s.last_run = run_time or datetime.now(timezone.utc)
            s.run_count += 1
