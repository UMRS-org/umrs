"""Tests for persistent monitor preferences."""

from __future__ import annotations

from umrs.tui.prefs import MonitorPrefs


def test_prefs_defaults():
    prefs = MonitorPrefs()
    assert prefs.active_tab == "jobs"
    assert prefs.interval == 3.0


def test_prefs_save_load_round_trip(tmp_path):
    p = tmp_path / "monitor.json"
    prefs = MonitorPrefs(
        active_tab="models",
        status_filter="running",
        project_filter="demo",
        interval=5.0,
    )
    prefs.save(str(p))
    loaded = MonitorPrefs.load(str(p))
    assert loaded.active_tab == "models"
    assert loaded.status_filter == "running"
    assert loaded.project_filter == "demo"
    assert loaded.interval == 5.0


def test_prefs_load_missing_returns_defaults(tmp_path):
    p = tmp_path / "nonexistent.json"
    loaded = MonitorPrefs.load(str(p))
    assert loaded.active_tab == "jobs"
    assert loaded.interval == 3.0


def test_prefs_load_invalid_json_returns_defaults(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not valid json")
    loaded = MonitorPrefs.load(str(p))
    assert loaded.active_tab == "jobs"
