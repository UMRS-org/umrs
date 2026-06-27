"""Tests for the JSON/CSV export module."""

from __future__ import annotations

import json

from umrs.tui.export import export_tab


def test_export_jobs_json(tmp_path):
    data = {"jobs": [{"job_id": "j1", "status": "running"}]}
    path = export_tab("jobs", data, fmt="json", out_dir=str(tmp_path))
    loaded = json.loads(open(path).read())
    assert loaded["jobs"][0]["job_id"] == "j1"


def test_export_jobs_csv(tmp_path):
    data = {"jobs": [{"job_id": "j1", "status": "running"}]}
    path = export_tab("jobs", data, fmt="csv", out_dir=str(tmp_path))
    content = open(path).read()
    assert "job_id" in content
    assert "j1" in content


def test_export_endpoints_json(tmp_path):
    data = {"endpoints": [{"job_id": "e1"}]}
    path = export_tab("endpoints", data, fmt="json", out_dir=str(tmp_path))
    loaded = json.loads(open(path).read())
    assert loaded["endpoints"][0]["job_id"] == "e1"


def test_export_list_data(tmp_path):
    data = [{"name": "m1"}, {"name": "m2"}]
    path = export_tab("models", data, fmt="json", out_dir=str(tmp_path))
    loaded = json.loads(open(path).read())
    assert len(loaded) == 2
