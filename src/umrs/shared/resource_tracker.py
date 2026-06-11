"""
B5 — Resource Tracking

Track CPU, memory, GPU, storage, and network resource usage per user,
project, and job.  Enables capacity planning and quota enforcement.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceType(str, Enum):
    """Types of tracked resources."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class QuotaAction(str, Enum):
    """Action when quota is exceeded."""

    WARN = "warn"
    THROTTLE = "throttle"
    BLOCK = "block"


@dataclass
class ResourceSnapshot:
    """Point-in-time resource readings."""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_bytes: int = 0
    memory_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_bytes: int = 0
    storage_bytes: int = 0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_bytes": self.memory_bytes,
            "memory_percent": self.memory_percent,
            "gpu_percent": self.gpu_percent,
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "storage_bytes": self.storage_bytes,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
        }


@dataclass
class QuotaPolicy:
    """Configurable per-user resource limits."""

    user_id: str
    max_cpu_percent: float = 100.0
    max_memory_bytes: int = 0  # 0 = unlimited
    max_gpu_count: int = 0
    max_storage_bytes: int = 0
    max_concurrent_jobs: int = 10
    action: QuotaAction = QuotaAction.WARN

    def check(
        self, snapshot: ResourceSnapshot, active_jobs: int = 0
    ) -> list[QuotaViolation]:
        violations = []
        if self.max_cpu_percent > 0 and snapshot.cpu_percent > self.max_cpu_percent:
            violations.append(
                QuotaViolation(
                    resource=ResourceType.CPU,
                    limit=self.max_cpu_percent,
                    actual=snapshot.cpu_percent,
                    action=self.action,
                )
            )
        if self.max_memory_bytes > 0 and snapshot.memory_bytes > self.max_memory_bytes:
            violations.append(
                QuotaViolation(
                    resource=ResourceType.MEMORY,
                    limit=float(self.max_memory_bytes),
                    actual=float(snapshot.memory_bytes),
                    action=self.action,
                )
            )
        if (
            self.max_storage_bytes > 0
            and snapshot.storage_bytes > self.max_storage_bytes
        ):
            violations.append(
                QuotaViolation(
                    resource=ResourceType.STORAGE,
                    limit=float(self.max_storage_bytes),
                    actual=float(snapshot.storage_bytes),
                    action=self.action,
                )
            )
        return violations


@dataclass
class QuotaViolation:
    """A detected quota violation."""

    resource: ResourceType
    limit: float
    actual: float
    action: QuotaAction
    timestamp: float = field(default_factory=time.time)

    @property
    def overage_percent(self) -> float:
        if self.limit <= 0:
            return 0.0
        return ((self.actual - self.limit) / self.limit) * 100.0

    def __str__(self) -> str:
        return (
            f"QuotaViolation({self.resource.value}: "
            f"{self.actual:.1f}/{self.limit:.1f}, "
            f"action={self.action.value})"
        )


@dataclass
class UsageReport:
    """Per-user/project resource usage summary."""

    user_id: str
    project_id: str = ""
    period_start: float = 0.0
    period_end: float = 0.0
    snapshot_count: int = 0
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_memory_bytes: float = 0.0
    max_memory_bytes: int = 0
    total_storage_bytes: int = 0
    total_network_bytes: int = 0
    violations: list[QuotaViolation] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Usage Report for {self.user_id}"
            + (f" ({self.project_id})" if self.project_id else ""),
            f"  Period: {self.period_start:.0f} – {self.period_end:.0f}",
            f"  Snapshots: {self.snapshot_count}",
            f"  CPU:  avg={self.avg_cpu_percent:.1f}% max={self.max_cpu_percent:.1f}%",
            f"  Memory: avg={self.avg_memory_bytes / 1e6:.1f}MB max={self.max_memory_bytes / 1e6:.1f}MB",
            f"  Storage: {self.total_storage_bytes / 1e6:.1f}MB",
            f"  Network: {self.total_network_bytes / 1e6:.1f}MB",
            f"  Violations: {len(self.violations)}",
        ]
        return "\n".join(lines)


class ResourceTracker:
    """Usage collection and aggregation.

    Usage::

        tracker = ResourceTracker()
        tracker.set_quota("user1", QuotaPolicy(user_id="user1", max_cpu_percent=80.0))

        # Record a snapshot
        snap = tracker.take_snapshot("user1")
        violations = tracker.check_quota("user1", snap)

        # Generate report
        report = tracker.generate_report("user1")
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._history: dict[str, list[ResourceSnapshot]] = {}
        self._quotas: dict[str, QuotaPolicy] = {}
        self._active_jobs: dict[str, int] = {}
        self._lock = threading.Lock()
        self._max_history = max_history

    def set_quota(self, user_id: str, policy: QuotaPolicy) -> None:
        with self._lock:
            self._quotas[user_id] = policy

    def get_quota(self, user_id: str) -> QuotaPolicy | None:
        with self._lock:
            return self._quotas.get(user_id)

    def record_snapshot(self, user_id: str, snapshot: ResourceSnapshot) -> None:
        with self._lock:
            if user_id not in self._history:
                self._history[user_id] = []
            history = self._history[user_id]
            history.append(snapshot)
            if len(history) > self._max_history:
                self._history[user_id] = history[-self._max_history :]

    def take_snapshot(self, user_id: str, project_id: str = "") -> ResourceSnapshot:
        """Take a snapshot of current system resources (basic /proc-based)."""
        snapshot = ResourceSnapshot(
            metadata={"user_id": user_id, "project_id": project_id}
        )

        try:
            load = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            snapshot.cpu_percent = (load[0] / cpu_count) * 100.0
        except (OSError, AttributeError):
            pass

        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        meminfo[key] = int(val) * 1024  # kB to bytes
                total = meminfo.get("MemTotal", 1)
                available = meminfo.get("MemAvailable", 0)
                snapshot.memory_bytes = total - available
                snapshot.memory_percent = (
                    ((total - available) / total) * 100.0 if total > 0 else 0.0
                )
        except (OSError, KeyError, ValueError):
            pass

        self.record_snapshot(user_id, snapshot)
        return snapshot

    def check_quota(
        self,
        user_id: str,
        snapshot: ResourceSnapshot | None = None,
    ) -> list[QuotaViolation]:
        with self._lock:
            policy = self._quotas.get(user_id)
        if not policy:
            return []

        if snapshot is None:
            with self._lock:
                history = self._history.get(user_id, [])
            if not history:
                return []
            snapshot = history[-1]

        active = self._active_jobs.get(user_id, 0)
        return policy.check(snapshot, active)

    def set_active_jobs(self, user_id: str, count: int) -> None:
        with self._lock:
            self._active_jobs[user_id] = count

    def generate_report(
        self,
        user_id: str,
        project_id: str = "",
        since: float | None = None,
    ) -> UsageReport:
        with self._lock:
            history = list(self._history.get(user_id, []))

        if since:
            history = [s for s in history if s.timestamp >= since]

        if not history:
            return UsageReport(user_id=user_id, project_id=project_id)

        report = UsageReport(
            user_id=user_id,
            project_id=project_id,
            period_start=history[0].timestamp,
            period_end=history[-1].timestamp,
            snapshot_count=len(history),
            avg_cpu_percent=sum(s.cpu_percent for s in history) / len(history),
            max_cpu_percent=max(s.cpu_percent for s in history),
            avg_memory_bytes=sum(s.memory_bytes for s in history) / len(history),
            max_memory_bytes=max(s.memory_bytes for s in history),
            total_storage_bytes=history[-1].storage_bytes,
            total_network_bytes=sum(
                s.network_bytes_sent + s.network_bytes_recv for s in history
            ),
        )

        # Check violations across all snapshots
        with self._lock:
            policy = self._quotas.get(user_id)
        if policy:
            for snap in history:
                report.violations.extend(policy.check(snap))

        return report

    def get_history(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[ResourceSnapshot]:
        with self._lock:
            return list(self._history.get(user_id, [])[-limit:])

    def clear_history(self, user_id: str) -> None:
        with self._lock:
            self._history.pop(user_id, None)
