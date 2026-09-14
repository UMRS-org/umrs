"""Tests for B10 — Scheduled Recurring Jobs."""

from datetime import datetime, timedelta, timezone

import pytest
from umrs.shared.scheduler import (
    JobScheduler,
    Schedule,
    ScheduleStatus,
    ScheduleStore,
    ScheduleType,
    _cron_field_matches,
    cron_matches,
    parse_interval,
)


class TestParseInterval:
    def test_seconds(self):
        assert parse_interval("every 30s") == 30

    def test_minutes(self):
        assert parse_interval("every 5m") == 300

    def test_hours(self):
        assert parse_interval("every 2h") == 7200

    def test_days(self):
        assert parse_interval("every 1d") == 86400

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_interval("invalid")

    def test_case_insensitive(self):
        assert parse_interval("every 1H") == 3600


class TestCronFieldMatches:
    def test_wildcard(self):
        assert _cron_field_matches("*", 5)
        assert _cron_field_matches("*", 0)

    def test_exact(self):
        assert _cron_field_matches("5", 5)
        assert not _cron_field_matches("5", 6)

    def test_step(self):
        assert _cron_field_matches("*/5", 0)
        assert _cron_field_matches("*/5", 10)
        assert not _cron_field_matches("*/5", 3)

    def test_range(self):
        assert _cron_field_matches("1-5", 3)
        assert not _cron_field_matches("1-5", 6)

    def test_comma_separated(self):
        assert _cron_field_matches("1,5,10", 5)
        assert not _cron_field_matches("1,5,10", 3)


class TestCronMatches:
    def test_every_minute(self):
        dt = datetime(2026, 4, 5, 12, 30, 0)
        assert cron_matches("* * * * *", dt)

    def test_specific_time(self):
        dt = datetime(2026, 4, 5, 3, 0, 0)  # Saturday
        assert cron_matches("0 3 * * *", dt)
        assert not cron_matches("0 4 * * *", dt)

    def test_every_6_hours(self):
        dt = datetime(2026, 4, 5, 6, 0, 0)
        assert cron_matches("0 */6 * * *", dt)

    def test_invalid_expression(self):
        with pytest.raises(ValueError):
            cron_matches("bad cron", datetime.now())


class TestScheduleStore:
    def test_save_and_load(self, tmp_path):
        store_path = str(tmp_path / "schedules.json")
        store = ScheduleStore(store_path)

        sched = Schedule(
            schedule_id="s1",
            name="test",
            schedule_type=ScheduleType.FIXED_INTERVAL,
            job_spec={"type": "cleanup"},
            interval_seconds=3600,
        )
        store.save({"s1": sched})

        loaded = store.load()
        assert "s1" in loaded
        assert loaded["s1"].name == "test"
        assert loaded["s1"].interval_seconds == 3600

    def test_load_nonexistent(self, tmp_path):
        store = ScheduleStore(str(tmp_path / "missing.json"))
        assert store.load() == {}


class TestSchedule:
    def test_roundtrip(self):
        s = Schedule(
            schedule_id="s1",
            name="nightly",
            schedule_type=ScheduleType.CRON,
            job_spec={"type": "index_build"},
            cron_expression="0 3 * * *",
            status=ScheduleStatus.ACTIVE,
        )
        d = s.to_dict()
        s2 = Schedule.from_dict(d)
        assert s2.schedule_id == "s1"
        assert s2.schedule_type == ScheduleType.CRON
        assert s2.cron_expression == "0 3 * * *"


class TestJobScheduler:
    def test_add_interval(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_interval(
            name="periodic",
            interval="every 30m",
            job_spec={"type": "cleanup"},
        )
        assert sched.schedule_type == ScheduleType.FIXED_INTERVAL
        assert sched.interval_seconds == 1800
        assert sched.status == ScheduleStatus.ACTIVE

    def test_add_cron(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_cron(
            name="nightly",
            cron="0 3 * * *",
            job_spec={"type": "index_build"},
        )
        assert sched.schedule_type == ScheduleType.CRON
        assert sched.cron_expression == "0 3 * * *"

    def test_add_cron_invalid(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        with pytest.raises(ValueError):
            scheduler.add_cron(name="bad", cron="not valid", job_spec={})

    def test_add_one_shot(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        sched = scheduler.add_one_shot(
            name="deferred",
            run_at=future,
            job_spec={"type": "export"},
        )
        assert sched.schedule_type == ScheduleType.ONE_SHOT
        assert sched.max_runs == 1

    def test_pause_resume(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_interval(name="p", interval="every 1h", job_spec={})
        scheduler.pause(sched.schedule_id)
        assert scheduler.get_schedule(sched.schedule_id).status == ScheduleStatus.PAUSED
        scheduler.resume(sched.schedule_id)
        assert scheduler.get_schedule(sched.schedule_id).status == ScheduleStatus.ACTIVE

    def test_cancel(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_interval(name="c", interval="every 1h", job_spec={})
        scheduler.cancel(sched.schedule_id)
        assert scheduler.get_schedule(sched.schedule_id).status == ScheduleStatus.CANCELLED

    def test_remove(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_interval(name="r", interval="every 1h", job_spec={})
        assert scheduler.remove(sched.schedule_id)
        assert scheduler.get_schedule(sched.schedule_id) is None
        assert not scheduler.remove("nonexistent")

    def test_list_schedules(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        scheduler.add_interval(name="a", interval="every 1h", job_spec={})
        scheduler.add_interval(name="b", interval="every 2h", job_spec={})
        assert len(scheduler.list_schedules()) == 2

    def test_list_by_status(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        s1 = scheduler.add_interval(name="a", interval="every 1h", job_spec={})
        scheduler.add_interval(name="b", interval="every 2h", job_spec={})
        scheduler.pause(s1.schedule_id)
        active = scheduler.list_schedules(status=ScheduleStatus.ACTIVE)
        assert len(active) == 1

    def test_tick_fires_due_interval(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        # Create schedule with next_run in the past
        sched = scheduler.add_interval(name="now", interval="every 1s", job_spec={"fire": True})
        # Manually set next_run_at to the past
        sched.next_run_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

        fired = []

        def callback(spec):
            fired.append(spec)

        count = scheduler.tick(callback)
        assert count == 1
        assert len(fired) == 1
        assert fired[0]["fire"] is True

    def test_tick_does_not_fire_paused(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_interval(name="paused", interval="every 1s", job_spec={})
        sched.next_run_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        scheduler.pause(sched.schedule_id)

        count = scheduler.tick(lambda spec: None)
        assert count == 0

    def test_tick_one_shot_expires(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        sched = scheduler.add_one_shot(name="once", run_at=past, job_spec={"once": True})

        fired = []
        scheduler.tick(lambda spec: fired.append(spec))
        assert len(fired) == 1

        s = scheduler.get_schedule(sched.schedule_id)
        assert s.status == ScheduleStatus.EXPIRED

        # Second tick should not fire
        fired.clear()
        scheduler.tick(lambda spec: fired.append(spec))
        assert len(fired) == 0

    def test_tick_max_runs_limit(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_interval(
            name="limited",
            interval="every 1s",
            job_spec={},
            max_runs=2,
        )
        sched.next_run_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

        scheduler.tick(lambda spec: None)
        sched.next_run_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        scheduler.tick(lambda spec: None)

        s = scheduler.get_schedule(sched.schedule_id)
        assert s.status == ScheduleStatus.EXPIRED
        assert s.run_count == 2

    def test_event_log(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_interval(name="logged", interval="every 1s", job_spec={"x": 1})
        sched.next_run_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

        scheduler.tick(lambda spec: None)
        log = scheduler.get_event_log(sched.schedule_id)
        assert len(log) == 1
        assert log[0].schedule_id == sched.schedule_id

    def test_callback_error_captured(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        sched = scheduler.add_interval(name="erroring", interval="every 1s", job_spec={})
        sched.next_run_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

        def bad_callback(spec):
            raise RuntimeError("boom")

        scheduler.tick(bad_callback)
        log = scheduler.get_event_log(sched.schedule_id)
        assert len(log) == 1
        assert not log[0].success
        assert "boom" in log[0].error

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "sched.json")
        s1 = JobScheduler(store_path=path)
        s1.add_interval(name="persist", interval="every 1h", job_spec={"p": True})

        s2 = JobScheduler(store_path=path)
        schedules = s2.list_schedules()
        assert len(schedules) == 1
        assert schedules[0].name == "persist"

    def test_stop(self, tmp_path):
        scheduler = JobScheduler(store_path=str(tmp_path / "sched.json"))
        scheduler._running = True
        scheduler.stop()
        assert not scheduler._running
