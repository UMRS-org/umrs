"""Q-05 / G-001: ZMQ commands carry correlation IDs into audit/emit paths."""

from __future__ import annotations

import json

from umrs.server.correlation import correlation_scope, get_request_id
from umrs.server.job_tracking_handler import JOB_SUBMIT, handle_job_command


class _CapturingBackend:
    def __init__(self, sink: list):
        self._sink = sink

    def submit(self, _client_id, _payload_str):
        self._sink.append(get_request_id())
        return json.dumps({"status": "ok"})

    def cancel(self, _client_id, _payload_str):
        return json.dumps({"status": "ok"})

    def list_jobs(self, _client_id, _payload_str):
        return json.dumps({"status": "ok", "jobs": []})


def test_payload_correlation_id_is_honored_inside_handler():
    sink: list[str] = []
    handle_job_command(
        JOB_SUBMIT,
        "alice",
        json.dumps({"_request_id": "rid-12345"}),
        backend=_CapturingBackend(sink),
    )
    assert sink == ["rid-12345"]


def test_missing_correlation_generates_fresh_uuid():
    sink: list[str] = []
    handle_job_command(JOB_SUBMIT, "alice", json.dumps({}), backend=_CapturingBackend(sink))
    assert sink and sink[0] and sink[0] != ""
    assert len(sink[0]) >= 8


def test_correlation_scope_resets_after_command():
    outer_before = get_request_id()
    with correlation_scope("inner-rid"):
        assert get_request_id() == "inner-rid"
    assert get_request_id() == outer_before


def test_nested_scopes_restore_correct_order():
    with correlation_scope("rid-A"):
        assert get_request_id() == "rid-A"
        with correlation_scope("rid-B"):
            assert get_request_id() == "rid-B"
        assert get_request_id() == "rid-A"
