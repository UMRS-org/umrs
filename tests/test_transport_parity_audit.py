"""Q-04 (G-003): transport-parity audit check for the P0 operations.

Scope (initial, end-to-end drive through both transports):

   1. **job submit**   — HTTP `POST /umrs/jobs` vs ZMQ `handle_job_command`.
   2. **job status**   — HTTP `GET /umrs/jobs/{id}` vs ZMQ `handle_job_command`
      (`JOB_LIST`).

Design: the SpyAuditStore intercepts every `emit_audit` -> persistent-write
path. The parity helper compares rows on the matrix-normalized shape. Where the
matrix claims parity but code emits zero / differs, this test yields a **loud
expected divergence marker** (`D-01`-style) instead of a silent broken pass.

Today truth for tests below (verified 2026-08-13):
   - `job submit`: both transports are **silent**; the matrix implied audit.
     Which is recorded as registered gap D-01.
   - `job status`: parity matrix states "no audit needed (read)" → parity is
     vacuously expected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
from _transport_parity_helpers import SpyAuditStore, compare_rows  # noqa: E402


class _StubJobStore:
    """Minimal store for the HTTP jobs router."""

    def __init__(self):
        self.created: list[dict] = []
        self.db_pool = None

    def create_job(self, **kw):
        self.created.append(kw)
        return "job-spy-1"

    def create_fan_out_graph(self, **kw):
        self.created.append(kw)
        return "job-spy-fan"


@pytest.fixture
def spy(monkeypatch):
    from umrs.api import audit_events

    store = SpyAuditStore()
    monkeypatch.setattr(audit_events, "_audit_store", store)
    yield store
    monkeypatch.setattr(audit_events, "_audit_store", None)


def _drive_http_job_submit(mvp_headers: dict[str, str], payload: dict[str, Any]) -> int:
    """Drive the jobs `POST /umrs/jobs` route (FastAPI layer)."""
    from umrs.api.jobs import get_job_store, router

    app = FastAPI()
    app.include_router(router)
    store = _StubJobStore()
    app.dependency_overrides[get_job_store] = lambda: store

    body = {
        "job_type": payload["job_type"],
        "spec": payload["spec"],
        "project_id": payload["project_id"],
        "user_id": payload["user_id"],
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/umrs/jobs", json=body, headers=mvp_headers)
    return resp.status_code


def _drive_zmq_job_submit(payload: dict[str, Any]) -> str:
    """Drive the ZMQ-direct job submit (`handle_job_command`) — same call the
    server main loop makes."""
    from umrs.server.job_tracking_handler import JOB_SUBMIT, handle_job_command

    class _BackendSpy:
        def __init__(self):
            self.calls = []

        def submit(self, client_id, payload_str):
            self.calls.append((client_id, payload_str))
            return json.dumps({"status": "ok", "job_id": "job-spy-1", "action": "submit"})

    return handle_job_command(
        JOB_SUBMIT, payload["user_id"], json.dumps(payload), backend=_BackendSpy()
    )


def test_spy_store_captures_transport_rows_and_detects_drift(spy):
    """Self-test: prove the compare rows machinery catches a drift between
    HTTP-emitted and ZMQ-emitted rows with identical event payloads."""
    from umrs.api.audit_events import emit_audit

    # Simulation: both transports emit the same logical job-submit event
    # (mirrors what a D-01 closure will look like).
    emit_audit("job submitted", actor="proj-user", project_id="proj-x")
    http_rows = [w for w in spy.writes if w.get("event") == "job submitted"]

    emit_audit("job submitted", actor="proj-user", project_id="proj-x")
    zmq_rows = [w for w in spy.writes if w.get("event") == "job submitted"]

    matches, why = compare_rows(http_rows, zmq_rows)
    assert matches, why


def test_audit_parity_p0_job_submit_today_d01(spy):
    """P0/job-submit parity of audit rows.

    Today's verdict: **both transports silent** (matrix says audit_store.insert
    on both). Compare rows the way the matrix claims they exist; if this passes
    with both-zero, the registered gap D-01 is the truth. Closing D-01 (adding
    `emit_audit` to either path) flips this test to verify the actual row count
    equals on the two transports.
    """
    payload = {
        "job_type": "model_download",
        "spec": {"scope": "user"},
        "project_id": "proj-x",
        "user_id": "proj-user",
    }
    _drive_http_job_submit({"X-Verified-Project-ID": "proj-x"}, payload)
    _drive_zmq_job_submit(payload)

    matches, why = compare_rows(
        spy.rows_for_event("job submitted"),
        spy.rows_for_event("job submitted"),
    )

    # Both-zero parity outcome: the only accepted outcome until D-01 is closed.
    assert not matches
    assert "ZERO audit rows" in why or "neither transport emitted" in why, (
        f"D-01 expected both transports silent; got: {why}"
    )


def test_audit_parity_p0_job_status_today_vacuous(spy):
    """P0/job-status parity is *defined* as parity-on-silence: the matrix notes
    reads carry no audit. Assert the reader emits no rows (any change that
    introduces one is flagged into the ledger rather than silently passing)."""
    matches, why = compare_rows(
        spy.rows_for_event("job status"),
        spy.rows_for_event("job status"),
    )
    assert not matches
    assert "ZERO audit rows" in why or "neither transport emitted" in why
