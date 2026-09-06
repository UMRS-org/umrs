"""Phase timing helpers (WP-12, G-P11-1/G-P1-4)."""

from __future__ import annotations

from datetime import datetime, timezone

from umrs.jobs.timing import attempt_phases, duration_ms

T0 = datetime(2026, 9, 7, 10, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 7, 10, 0, 5, tzinfo=timezone.utc)
T2 = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)


class TestDurationMs:
    def test_known_pair(self):
        assert duration_ms(T0, T1) == 5000

    def test_iso_strings(self):
        assert duration_ms("2026-09-07T10:00:00+00:00", "2026-09-07T10:00:05+00:00") == 5000

    def test_naive_becomes_utc(self):
        assert duration_ms(datetime(2026, 9, 7, 10, 0, 0), datetime(2026, 9, 7, 10, 0, 5)) == 5000

    def test_unknown_side_is_none_not_invented(self):
        assert duration_ms(None, T1) is None
        assert duration_ms(T0, "") is None
        assert duration_ms("garbage", T1) is None

    def test_skew_floors_at_zero(self):
        assert duration_ms(T1, T0) == 0


class TestAttemptPhases:
    def test_running_gives_queue_wait(self):
        phases = attempt_phases(started_at=T0, transitioned_at=T1, phase="running")
        assert phases == {"queue_wait_ms": 5000}

    def test_ended_gives_wall(self):
        phases = attempt_phases(started_at=T0, transitioned_at=T2, phase="ended")
        assert phases == {"wall_ms": 7200000}

    def test_absent_started_at_gives_none_not_zero(self):
        phases = attempt_phases(started_at=None, transitioned_at=T1, phase="running")
        assert phases == {"queue_wait_ms": None}
