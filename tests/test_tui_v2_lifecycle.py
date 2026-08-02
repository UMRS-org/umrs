"""Tests for monitor-v2 lifecycle helpers (Wave-2 UX upgrade)."""

from __future__ import annotations

from umrs.tui_v2.widgets.lifecycle import (
    build_status_map,
    detect_transitions,
    filter_by_status,
    stage_tracker,
)


class TestStageTracker:
    def test_queued_shows_first_stage_active(self) -> None:
        line = stage_tracker("queued")
        assert "◐ submitted" in line
        assert "○ provisioning" in line
        assert "→" in line

    def test_provisioning_marks_submitted_done(self) -> None:
        line = stage_tracker("provisioning")
        assert "✓ submitted" in line
        assert "◐ provisioning" in line
        assert "○ serving" in line

    def test_running_all_stages_done(self) -> None:
        line = stage_tracker("running")
        assert line.count("✓") == 3
        assert "◐" not in line and "○" not in line

    def test_failed_status_short_circuits(self) -> None:
        assert stage_tracker("failed") == "[red]✗ provisioning failed[/red]"

    def test_stopped_status_shows_halt(self) -> None:
        assert "stopped" in stage_tracker("stopped")

    def test_unknown_status_empty(self) -> None:
        assert stage_tracker("rebooting") == ""

    def test_case_and_whitespace_insensitive(self) -> None:
        assert "◐ submitted" in stage_tracker("  Queued ")


class TestBuildStatusMap:
    def test_builds_id_to_status(self) -> None:
        items = [{"job_id": "a", "status": "running"}, {"job_id": "b"}]
        assert build_status_map(items) == {"a": "running", "b": ""}

    def test_skips_items_without_id(self) -> None:
        assert build_status_map([{"status": "running"}]) == {}


class TestDetectTransitions:
    def test_first_refresh_is_baseline(self) -> None:
        current = {"j1": "running", "j2": "failed"}
        assert detect_transitions({}, current) == []

    def test_detects_watched_change(self) -> None:
        prev = {"j1": "queued"}
        curr = {"j1": "running"}
        assert detect_transitions(prev, curr) == [("j1", "queued", "running")]

    def test_ignores_unwatched_target_states(self) -> None:
        prev = {"j1": "running"}
        curr = {"j1": "queued"}
        assert detect_transitions(prev, curr) == []

    def test_ignores_unchanged_and_new_ids(self) -> None:
        prev = {"j1": "running"}
        curr = {"j1": "running", "j2": "failed"}
        assert detect_transitions(prev, curr) == []

    def test_custom_watch_set(self) -> None:
        prev = {"j1": "running"}
        curr = {"j1": "succeeded"}
        events = detect_transitions(prev, curr, watch=("succeeded",))
        assert events == [("j1", "running", "succeeded")]


class TestFilterByStatus:
    ITEMS = [
        {"job_id": "a", "status": "running"},
        {"job_id": "b", "status": "failed"},
        {"job_id": "c", "status": "Running"},
    ]

    def test_all_returns_copy_of_everything(self) -> None:
        result = filter_by_status(self.ITEMS, "all")
        assert len(result) == 3

    def test_filter_matches_case_insensitively(self) -> None:
        result = filter_by_status(self.ITEMS, "running")
        assert [r["job_id"] for r in result] == ["a", "c"]

    def test_filter_empty_matches_nothing(self) -> None:
        assert filter_by_status(self.ITEMS, "queued") == []
