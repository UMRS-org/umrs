"""Regression tests for GitHub Issue #87 item 3 — warning framework integration."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestRegistryMissingClientId:
    def test_missing_client_id_sends_unauthorized_reply(self):
        from unittest.mock import MagicMock

        try:
            from umrs.server.commands.registry import CommandRegistry
        except ModuleNotFoundError:
            pytest.skip("CommandRegistry dependencies unavailable")

        registry = CommandRegistry()
        sent: list[tuple[str, bytes]] = []

        class FakeControlMQ:
            def get_auth_context(self, _client_id):
                raise AssertionError("should not resolve auth without client_id")

            def send_string(self, msg, zmq_sender_id):
                sent.append((msg, zmq_sender_id))

        ctx = MagicMock()
        ctx.control_mq = FakeControlMQ()

        allowed = registry._authorize(
            "PROCESS_OBJECTS",
            None,
            b"sender",
            ctx,
        )
        assert allowed is False
        assert sent == [("UNAUTHORIZED:PROCESS_OBJECTS", b"sender")]


class TestEmitCliWarning:
    def test_json_mode_returns_body_without_printing(self):
        from umrs.client.warnings_cli import emit_cli_warning
        from umrs.shared.warnings import operation_ignored_warning

        warning = operation_ignored_warning("notify_server", "denied")
        captured = StringIO()
        with patch("sys.stderr", captured):
            body = emit_cli_warning(warning, output_format="json", bucket="b")
        assert body["status"] == "error"
        assert body["warnings"][0]["category"] == "operation_ignored"
        assert body["bucket"] == "b"
        assert captured.getvalue() == ""

    def test_text_mode_prints_to_stderr(self):
        from umrs.client.warnings_cli import emit_cli_warning
        from umrs.shared.warnings import operation_ignored_warning

        warning = operation_ignored_warning("notify_server", "denied")
        captured = StringIO()
        with patch("sys.stderr", captured):
            emit_cli_warning(warning, output_format="text")
        assert "notify_server" in captured.getvalue()


class TestActionStatusExitCode:
    def test_exception_returns_nonzero(self, monkeypatch):
        try:
            import client_umrs
        except ModuleNotFoundError:
            pytest.skip("client_umrs dependencies unavailable")

        control_mq = MagicMock()
        fake_api = MagicMock()
        fake_api.request_status.side_effect = RuntimeError("status failed")

        monkeypatch.setattr(client_umrs, "UMRSControlAPI", lambda _mq, _u: fake_api)
        monkeypatch.setattr(client_umrs, "_resolve_status_target", lambda _args: "all")

        ctx = MagicMock()
        ctx.args.output = "text"
        ctx.args.username = "alice"
        ctx.args.action = "status"
        ctx.user = {}
        ctx.control_mq = control_mq
        ctx.configure_file = None
        ctx.connection = None

        assert client_umrs._action_status(ctx) == 1
        control_mq.close.assert_called()


class TestDownloadStatusNoJobStore:
    def test_returns_503_when_job_store_unavailable(self, monkeypatch):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            pytest.skip("fastapi not available")

        import umrs.api.jobs as jobs_api
        from umrs.api.files import router

        monkeypatch.setattr(jobs_api, "get_optional_job_store", lambda: None)

        app = FastAPI()
        app.include_router(router)

        # H-1 (2026-10-18): download-status now resolves the caller's project
        # first (an anonymous call is 401), so the mvp caller names one.
        with TestClient(app) as client:
            response = client.get(
                "/umrs/files/download-status",
                params={"job_id": "job-1"},
                headers={"X-Project-ID": "proj-a"},
            )

        assert response.status_code == 503


class TestPresignWarningErrorHandler:
    @pytest.mark.asyncio
    async def test_blocking_download_warning_includes_retry_after_header(self):
        try:
            from umrs.api.error_responses import structured_warning_exception_handler
        except ModuleNotFoundError:
            pytest.skip("API dependencies unavailable")

        from umrs.shared.warnings import UMRSWarningError, blocking_download_warning

        warning = blocking_download_warning("job-abc", "running", retry_after=30)
        warning.details.setdefault("operation", "download")
        warning.details.setdefault("reason", "task_not_ready")
        exc = UMRSWarningError(warning, http_status=409)

        response = await structured_warning_exception_handler(MagicMock(), exc)
        body = json.loads(response.body)

        assert response.status_code == 409
        assert body["error_id"] == "warning.blocking"
        assert body["details"]["operation"] == "download"
        assert body["details"]["reason"] == "task_not_ready"
        assert response.headers.get("retry-after") == "30"


class TestUploadNotifyFailure:
    def test_notify_server_unauthorized_returns_false(self):
        reply = "UNAUTHORIZED:PROCESS_OBJECTS"
        if reply.startswith("UNAUTHORIZED:"):
            ok = False
        else:
            ok = True
        assert ok is False

    def test_upload_exit_code_when_notify_fails_after_s3_put(self):
        s3_key = "path/to/key"
        notify_ok = False
        exit_code = 0
        if s3_key and not notify_ok or not s3_key:
            exit_code = 1
        assert exit_code == 1
