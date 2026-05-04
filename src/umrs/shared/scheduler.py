"""
B10 — Scheduled Recurring Jobs

Cron-like scheduler for recurring UMRS jobs.  Supports fixed intervals,
cron expressions, and one-shot delayed execution.  Persists schedules to
survive server restarts.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ScheduleType(Enum):
    FIXED_INTERVAL = "fixed_interval"
    CRON = "cron"
    ONE_SHOT = "one_shot"


class ScheduleStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class Schedule:
    """A recurring or one-shot schedule."""
    schedule_id: str
    name: str
    schedule_type: ScheduleType
    job_spec: dict[str, Any]

    # For fixed interval
    interval_seconds: int = 0

    # For cron (simplified: minute hour day_of_month month day_of_week)
    cron_expression: str = ""

    # For one-shot
    run_at: str | None = None  # ISO-8601

    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_run_at: str | None = None
    next_run_at: str | None = None
    run_count: int = 0
    max_runs: int = 0  # 0 = unlimited
    user_id: str = ""
    project_id: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schedule_type"] = self.schedule_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Schedule:
        d = dict(d)
        d["schedule_type"] = ScheduleType(d["schedule_type"])
        d["status"] = ScheduleStatus(d["status"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ScheduleEvent:
    """Record of a schedule execution."""
    schedule_id: str
    run_number: int
    triggered_at: str
    job_spec: dict[str, Any]
    success: bool = True
    error: str = ""


def parse_interval(expr: str) -> int:
    """Parse 'every 30m', 'every 2h', 'every 1d' → seconds."""
    m = re.match(r"^every\s+(\d+)\s*([smhd])$", expr.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid interval expression: {expr!r}")
    value = int(m.group(1))
    unit = m.group(2).lower()
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


def _cron_field_matches(field_expr: str, value: int) -> bool:
    """Check if a cron field matches a value.

    Supports: * (any), exact number, */step, range (a-b), comma-separated.
    """
    for part in field_expr.split(","):
        part = part.strip()
        if part == "*":
            return True
        if "/" in part:
            base, step_str = part.split("/", 1)
            step = int(step_str)
            if base == "*":
                if value % step == 0:
                    return True
            else:
                base_val = int(base)
                if value >= base_val and (value - base_val) % step == 0:
                    return True
        elif "-" in part:
            lo, hi = part.split("-", 1)
            if int(lo) <= value <= int(hi):
                return True
        else:
            if int(part) == value:
                return True
    return False


def cron_matches(expression: str, dt: datetime) -> bool:
    """Check if datetime matches a cron expression (minute hour dom month dow)."""
    parts = expression.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got {len(parts)}: {expression!r}")

    minute, hour, dom, month, dow = parts
    return (
        _cron_field_matches(minute, dt.minute)
        and _cron_field_matches(hour, dt.hour)
        and _cron_field_matches(dom, dt.day)
        and _cron_field_matches(month, dt.month)
        and _cron_field_matches(dow, dt.weekday())  # 0=Monday
    )


class ScheduleStore:
    """JSON file persistence for schedules."""

    def __init__(self, store_path: str):
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, schedules: dict[str, Schedule]) -> None:
        data = {sid: s.to_dict() for sid, s in schedules.items()}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(self._path)

    def load(self) -> dict[str, Schedule]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text())
        return {sid: Schedule.from_dict(d) for sid, d in data.items()}


class JobScheduler:
    """Manages schedule lifecycle with polling-based execution.

    Usage::

        scheduler = JobScheduler(store_path="schedules.json")

        # Fixed interval
        scheduler.add_interval(
            name="nightly-index",
            interval="every 6h",
            job_spec={"job_type": "index_build", "params": {...}},
        )

        # Cron expression
        scheduler.add_cron(
            name="daily-cleanup",
            cron="0 3 * * *",
            job_spec={"job_type": "cleanup", "params": {...}},
        )

        # One-shot
        scheduler.add_one_shot(
            name="deferred-export",
            run_at="2026-04-10T00:00:00+00:00",
            job_spec={"job_type": "export", "params": {...}},
        )

        # Start polling loop (blocking, or run in thread)
        scheduler.start(callback=my_job_submitter, poll_interval=60)
    """

    def __init__(self, store_path: str | None = None):
        self._schedules: dict[str, Schedule] = {}
        self._store = ScheduleStore(store_path) if store_path else None
        self._event_log: list[ScheduleEvent] = []
        self._lock = threading.Lock()
        self._running = False

        if self._store:
            self._schedules = self._store.load()

    def _save(self) -> None:
        if self._store:
            self._store.save(self._schedules)

    def add_interval(
        self,
        name: str,
        interval: str,
        job_spec: dict[str, Any],
        user_id: str = "",
        project_id: str = "",
        max_runs: int = 0,
        tags: list[str] | None = None,
    ) -> Schedule:
        """Add a fixed-interval schedule."""
        seconds = parse_interval(interval)
        now = datetime.now(timezone.utc)
        sched = Schedule(
            schedule_id=f"sched-{uuid.uuid4().hex[:12]}",
            name=name,
            schedule_type=ScheduleType.FIXED_INTERVAL,
            job_spec=job_spec,
            interval_seconds=seconds,
            next_run_at=(now + timedelta(seconds=seconds)).isoformat(),
            user_id=user_id,
            project_id=project_id,
            max_runs=max_runs,
            tags=tags or [],
        )
        with self._lock:
            self._schedules[sched.schedule_id] = sched
            self._save()
        return sched

    def add_cron(
        self,
        name: str,
        cron: str,
        job_spec: dict[str, Any],
        user_id: str = "",
        project_id: str = "",
        max_runs: int = 0,
        tags: list[str] | None = None,
    ) -> Schedule:
        """Add a cron-based schedule."""
        # Validate expression
        parts = cron.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Cron expression must have 5 fields: {cron!r}")

        sched = Schedule(
            schedule_id=f"sched-{uuid.uuid4().hex[:12]}",
            name=name,
            schedule_type=ScheduleType.CRON,
            job_spec=job_spec,
            cron_expression=cron,
            user_id=user_id,
            project_id=project_id,
            max_runs=max_runs,
            tags=tags or [],
        )
        with self._lock:
            self._schedules[sched.schedule_id] = sched
            self._save()
        return sched

    def add_one_shot(
        self,
        name: str,
        run_at: str,
        job_spec: dict[str, Any],
        user_id: str = "",
        project_id: str = "",
        tags: list[str] | None = None,
    ) -> Schedule:
        """Add a one-shot delayed execution."""
        sched = Schedule(
            schedule_id=f"sched-{uuid.uuid4().hex[:12]}",
            name=name,
            schedule_type=ScheduleType.ONE_SHOT,
            job_spec=job_spec,
            run_at=run_at,
            next_run_at=run_at,
            max_runs=1,
            user_id=user_id,
            project_id=project_id,
            tags=tags or [],
        )
        with self._lock:
            self._schedules[sched.schedule_id] = sched
            self._save()
        return sched

    def pause(self, schedule_id: str) -> None:
        with self._lock:
            s = self._schedules.get(schedule_id)
            if s and s.status == ScheduleStatus.ACTIVE:
                s.status = ScheduleStatus.PAUSED
                self._save()

    def resume(self, schedule_id: str) -> None:
        with self._lock:
            s = self._schedules.get(schedule_id)
            if s and s.status == ScheduleStatus.PAUSED:
                s.status = ScheduleStatus.ACTIVE
                self._save()

    def cancel(self, schedule_id: str) -> None:
        with self._lock:
            s = self._schedules.get(schedule_id)
            if s:
                s.status = ScheduleStatus.CANCELLED
                self._save()

    def remove(self, schedule_id: str) -> bool:
        with self._lock:
            if schedule_id in self._schedules:
                del self._schedules[schedule_id]
                self._save()
                return True
            return False

    def list_schedules(
        self,
        status: ScheduleStatus | None = None,
        user_id: str | None = None,
    ) -> list[Schedule]:
        with self._lock:
            result = list(self._schedules.values())
        if status:
            result = [s for s in result if s.status == status]
        if user_id:
            result = [s for s in result if s.user_id == user_id]
        return result

    def get_schedule(self, schedule_id: str) -> Schedule | None:
        with self._lock:
            return self._schedules.get(schedule_id)

    def get_event_log(self, schedule_id: str | None = None) -> list[ScheduleEvent]:
        if schedule_id:
            return [e for e in self._event_log if e.schedule_id == schedule_id]
        return list(self._event_log)

    def tick(self, callback: Callable[[dict[str, Any]], None]) -> int:
        """Check all schedules and fire due ones. Returns count of fired schedules."""
        now = datetime.now(timezone.utc)
        fired = 0

        with self._lock:
            for sched in list(self._schedules.values()):
                if sched.status != ScheduleStatus.ACTIVE:
                    continue
                if sched.max_runs > 0 and sched.run_count >= sched.max_runs:
                    sched.status = ScheduleStatus.EXPIRED
                    continue

                should_fire = False

                if sched.schedule_type == ScheduleType.FIXED_INTERVAL:
                    if sched.next_run_at:
                        next_dt = datetime.fromisoformat(sched.next_run_at)
                        if next_dt.tzinfo is None:
                            next_dt = next_dt.replace(tzinfo=timezone.utc)
                        if now >= next_dt:
                            should_fire = True
                    else:
                        should_fire = True

                elif sched.schedule_type == ScheduleType.CRON:
                    should_fire = cron_matches(sched.cron_expression, now)
                    # Prevent double-firing within same minute
                    if should_fire and sched.last_run_at:
                        last = datetime.fromisoformat(sched.last_run_at)
                        if last.tzinfo is None:
                            last = last.replace(tzinfo=timezone.utc)
                        if (now - last).total_seconds() < 60:
                            should_fire = False

                elif sched.schedule_type == ScheduleType.ONE_SHOT and sched.run_at:
                    run_dt = datetime.fromisoformat(sched.run_at)
                    if run_dt.tzinfo is None:
                        run_dt = run_dt.replace(tzinfo=timezone.utc)
                    if now >= run_dt:
                        should_fire = True

                if should_fire:
                    event = ScheduleEvent(
                        schedule_id=sched.schedule_id,
                        run_number=sched.run_count + 1,
                        triggered_at=now.isoformat(),
                        job_spec=sched.job_spec,
                    )
                    try:
                        callback(sched.job_spec)
                    except Exception as exc:
                        event.success = False
                        event.error = str(exc)

                    sched.run_count += 1
                    sched.last_run_at = now.isoformat()
                    self._event_log.append(event)
                    fired += 1

                    if sched.schedule_type == ScheduleType.FIXED_INTERVAL:
                        sched.next_run_at = (now + timedelta(seconds=sched.interval_seconds)).isoformat()
                    elif sched.schedule_type == ScheduleType.ONE_SHOT:
                        sched.status = ScheduleStatus.EXPIRED

                    if sched.max_runs > 0 and sched.run_count >= sched.max_runs:
                        sched.status = ScheduleStatus.EXPIRED

            self._save()
        return fired

    def start(
        self,
        callback: Callable[[dict[str, Any]], None],
        poll_interval: int = 60,
    ) -> None:
        """Run polling loop (blocking). Set self._running = False to stop."""
        self._running = True
        while self._running:
            self.tick(callback)
            time.sleep(poll_interval)

    def stop(self) -> None:
        """Signal polling loop to stop."""
        self._running = False
