"""Tests for B5 — Resource Tracking."""

import time

import pytest

from umrs.shared.resource_tracker import (
    QuotaAction,
    QuotaPolicy,
    QuotaViolation,
    ResourceSnapshot,
    ResourceTracker,
    ResourceType,
    UsageReport,
)


class TestResourceSnapshot:
    def test_creation(self):
        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_bytes=1024 * 1024 * 512,
        )
        assert snap.cpu_percent == 50.0
        assert snap.memory_bytes == 512 * 1024 * 1024

    def test_to_dict(self):
        snap = ResourceSnapshot(cpu_percent=42.0, memory_bytes=999)
        d = snap.to_dict()
        assert d["cpu_percent"] == 42.0
        assert d["memory_bytes"] == 999


class TestQuotaPolicy:
    def test_no_violation(self):
        policy = QuotaPolicy(
            user_id="u1",
            max_cpu_percent=90.0,
            max_memory_bytes=8_000_000_000,
            action=QuotaAction.WARN,
        )
        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_bytes=4_000_000_000,
        )
        violations = policy.check(snap)
        assert len(violations) == 0

    def test_cpu_violation(self):
        policy = QuotaPolicy(
            user_id="u1",
            max_cpu_percent=80.0,
            action=QuotaAction.BLOCK,
        )
        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=95.0,
            memory_bytes=0,
        )
        violations = policy.check(snap)
        cpu_violations = [v for v in violations if v.resource == ResourceType.CPU]
        assert len(cpu_violations) == 1
        assert cpu_violations[0].overage_percent > 0

    def test_memory_violation(self):
        policy = QuotaPolicy(
            user_id="u1",
            max_memory_bytes=1_000_000_000,
            action=QuotaAction.THROTTLE,
        )
        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.0,
            memory_bytes=2_000_000_000,
        )
        violations = policy.check(snap)
        mem_violations = [v for v in violations if v.resource == ResourceType.MEMORY]
        assert len(mem_violations) == 1


class TestResourceTracker:
    def test_record_and_history(self):
        tracker = ResourceTracker()
        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=30.0,
            memory_bytes=1_000_000,
        )
        tracker.record_snapshot("u1", snap)
        history = tracker.get_history("u1")
        assert len(history) == 1
        assert history[0].cpu_percent == 30.0

    def test_set_and_check_quota(self):
        tracker = ResourceTracker()
        policy = QuotaPolicy(
            user_id="u1",
            max_cpu_percent=50.0,
            action=QuotaAction.WARN,
        )
        tracker.set_quota("u1", policy)

        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=80.0,
            memory_bytes=0,
        )
        tracker.record_snapshot("u1", snap)
        violations = tracker.check_quota("u1")
        assert len(violations) > 0

    def test_no_quota_returns_empty(self):
        tracker = ResourceTracker()
        snap = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=99.0,
            memory_bytes=0,
        )
        tracker.record_snapshot("u1", snap)
        violations = tracker.check_quota("u1")
        assert len(violations) == 0

    def test_generate_report(self):
        tracker = ResourceTracker()
        for i in range(5):
            snap = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=20.0 + i * 10,
                memory_bytes=1_000_000 * (i + 1),
            )
            tracker.record_snapshot("u1", snap)

        report = tracker.generate_report("u1")
        assert report is not None
        assert report.snapshot_count == 5
        s = report.summary()
        assert "u1" in s

    def test_generate_report_no_data(self):
        tracker = ResourceTracker()
        report = tracker.generate_report("nobody")
        assert report.snapshot_count == 0


class TestUsageReport:
    def test_summary(self):
        report = UsageReport(
            user_id="u1",
            snapshot_count=10,
            avg_cpu_percent=45.0,
            max_cpu_percent=90.0,
            avg_memory_bytes=2_000_000_000,
            max_memory_bytes=4_000_000_000,
        )
        s = report.summary()
        assert "u1" in s
        assert "45.0" in s or "45" in s
