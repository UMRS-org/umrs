"""Cycle-2 umrs-api lane (2026-10-21): /health reports the real uptime.

Found live: ``GET /health`` (and ``/umrs/web/dashboard/health``) always
answered ``"uptime_seconds": 0`` minutes after boot. Both read
``app.state.start_time``, which nothing ever set, so a monitor could not
tell a restarted server from a long-running one.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from fastapi.testclient import TestClient


def test_health_uptime_counts_from_app_creation(monkeypatch) -> None:
    from umrs.jobs.service import get_job_service, set_job_service

    monkeypatch.setenv("GATEWAY_ENVIRONMENT", "test")
    prior = get_job_service()
    set_job_service(None)
    try:
        with patch("umrs.server.http_sidecar._build_db_pool", lambda _app: None):
            from umrs.server import http_sidecar

            app = http_sidecar._create_app()
        started = app.state.start_time
        monkeypatch.setattr(time, "time", lambda: started + 42.0)
        r = TestClient(app).get("/health")
    finally:
        set_job_service(prior)
    assert r.status_code == 200
    assert r.json()["uptime_seconds"] == 42.0
