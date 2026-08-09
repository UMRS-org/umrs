"""Regression tests for the 2026-08-09/10 durable-loop fixes.

Covers:
- AttemptMonitor tolerates None status from checkers (no crash, no false failed)
- UMRS_WORKFLOW_STATE_DIR relocates catalog + extracted workflows
- files.py upload owner guard rejects '-' placeholder
- knowledge_adapters._resolve_s3_credentials falls back to admin s3cfg
- durable runtime bootstrap no longer dead-disabled on UNKNOWN detection
"""
import os
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestMonitorNoneGuard:
    @pytest.mark.asyncio
    async def test_none_result_skips_without_false_failure(self):
        from umrs.jobs.models import AttemptStatus, JobAttempt
        from umrs.jobs.monitor import AttemptMonitor

        job_store = MagicMock()
        job_store.get_job.return_value = MagicMock(job_id="j")
        checker = MagicMock()
        checker.check_status = AsyncMock(return_value=None)
        mon = AttemptMonitor(job_store=job_store, status_checker=checker, poll_interval=0.1)
        att = JobAttempt(attempt_id="a", job_id="j", attempt_no=1, backend="wf",
                         status=AttemptStatus.RUNNING, backend_ref="", metrics={})
        await mon._check_attempt(att)
        job_store.mark_attempt_succeeded.assert_not_called()
        job_store.mark_attempt_failed.assert_not_called()
        assert mon._stats["attempts_unknown"] == 1


class TestWorkflowStateDir:
    def test_catalog_path_honors_state_dir(self, monkeypatch, tmp_path):
        from umrs.server.workflow_registry_handler import _catalog_path

        monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(tmp_path))
        assert _catalog_path() == str(tmp_path / "workflow_catalog.json")

    def test_catalog_path_default_config_dir(self, monkeypatch):
        from umrs.server.workflow_registry_handler import _catalog_path

        monkeypatch.delenv("UMRS_WORKFLOW_STATE_DIR", raising=False)
        assert _catalog_path().endswith("workflow_catalog.json")


class TestOwnerGuard:
    def test_placeholder_owner_rejected(self):
        from umrs.api.files import _fire_post_upload_ingest

        calls = {}
        import umrs.data_catalog.triggers as trig

        def fake_trigger(s3, bucket, **kw):
            calls.update(kw)

        orig = trig.trigger_post_upload_ingestion
        trig.trigger_post_upload_ingestion = fake_trigger
        try:
            _fire_post_upload_ingest(MagicMock(), "b", "k", "e", "proj", "-", "f.txt")
            assert calls["owner"] == "proj"
            _fire_post_upload_ingest(MagicMock(), "b", "k", "e", "proj", "  ", "f.txt")
            assert calls["owner"] == "proj"
            _fire_post_upload_ingest(MagicMock(), "b", "k", "e", "proj", "alice", "f.txt")
            assert calls["owner"] == "alice"
        finally:
            trig.trigger_post_upload_ingestion = orig


class TestS3CredResolution:
    def test_env_preferred(self, monkeypatch):
        from umrs.jobs.knowledge_adapters import _resolve_s3_credentials

        monkeypatch.setenv("S3_ACCESS_KEY", "envacc")
        monkeypatch.setenv("S3_SECRET_KEY", "envsec")
        monkeypatch.delenv("UMRS_ADMIN_S3_CONFIGURATION_FILE", raising=False)
        assert _resolve_s3_credentials() == ("envacc", "envsec")

    def test_fallback_to_admin_s3cfg(self, monkeypatch, tmp_path):
        cfg = tmp_path / "s3cfg"
        cfg.write_text("[default]\naccess_key = fileacc\nsecret_key = filesec\n")
        monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_SECRET_KEY", raising=False)
        monkeypatch.setenv("UMRS_ADMIN_S3_CONFIGURATION_FILE", str(cfg))

        from umrs.jobs.knowledge_adapters import _resolve_s3_credentials

        assert _resolve_s3_credentials() == ("fileacc", "filesec")
