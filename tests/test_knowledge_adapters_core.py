"""Direct unit tests for knowledge_adapters pure/environment helpers.

This module (2200+ lines) is the most-touched jobs adapter since August and
was only exercised through mocks. These tests pin the deterministic helpers
that other paths quietly depend on:

- DSN resolution precedence (explicit DSN > component vars; empty-config must
  NOT fabricate a localhost DSN).
- Bucket resolution precedence (job spec > env > default).
- Summary builder and ISO timestamp helpers.
- Sync/async bridge helpers (_fire_coroutine, _run_coroutine_sync) that guard
  the "asyncio.run from a running loop" failure mode.
"""

from __future__ import annotations

import asyncio

import pytest
from umrs.jobs import knowledge_adapters as ka

_COMPONENT_KEYS = (
    "DB_HOST",
    "DB_PORT",
    "UMRS_JOBS_DB",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "DB_SUPERUSER",
    "DB_SUPERUSER_PASSWORD",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


@pytest.fixture(autouse=True)
def _clear_db_env(monkeypatch):
    for var in ("UMRS_DATA_CATALOG_DB", "DATABASE_URL", *_COMPONENT_KEYS):
        monkeypatch.delenv(var, raising=False)
    yield


class TestResolveKnowledgeDSN:
    def test_explicit_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("UMRS_DATA_CATALOG_DB", "postgresql://a:b@h:1/x")
        monkeypatch.setenv("DATABASE_URL", "postgresql://c:d@h:2/y")
        assert ka._resolve_knowledge_dsn() == "postgresql://a:b@h:1/x"

    def test_database_url_second_choice(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://c:d@h:2/y")
        assert ka._resolve_knowledge_dsn() == "postgresql://c:d@h:2/y"

    def test_component_vars_build_dsn(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "db.example")
        monkeypatch.setenv("DB_PORT", "5444")
        monkeypatch.setenv("UMRS_JOBS_DB", "knowdb")
        monkeypatch.setenv("DB_SUPERUSER", "alice")
        monkeypatch.setenv("DB_SUPERUSER_PASSWORD", "s3cret")
        dsn = ka._resolve_knowledge_dsn()
        assert dsn == "postgresql://alice:s3cret@db.example:5444/knowdb"

    def test_empty_config_returns_blank_no_fabrication(self):
        assert ka._resolve_knowledge_dsn() == ""

    def test_explicit_dsn_beats_components(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h:9/z")
        monkeypatch.setenv("DB_HOST", "ignored")
        assert ka._resolve_knowledge_dsn() == "postgresql://x:y@h:9/z"


class TestResolveBucket:
    def test_job_spec_wins(self, monkeypatch):
        monkeypatch.setenv("UMRS_DATA_S3_BUCKET", "envbucket")
        assert ka._resolve_bucket({"bucket": "jobbucket"}) == "jobbucket"

    def test_env_fallback_order(self, monkeypatch):
        monkeypatch.delenv("S3_BUCKET", raising=False)
        monkeypatch.setenv("UMRS_DATA_S3_BUCKET", "umrs-data-x")
        assert ka._resolve_bucket() == "umrs-data-x"
        monkeypatch.delenv("UMRS_DATA_S3_BUCKET", raising=False)
        monkeypatch.setenv("S3_BUCKET", "s3gen")
        assert ka._resolve_bucket() == "s3gen"

    def test_default(self, monkeypatch):
        monkeypatch.delenv("UMRS_DATA_S3_BUCKET", raising=False)
        monkeypatch.delenv("S3_BUCKET", raising=False)
        assert ka._resolve_bucket() == "umrs-data"


class TestBuildSummary:
    def test_all_fields(self):
        parsed = {
            "modules": ["m1", "m2"],
            "classes": ["C"],
            "functions": ["f", "g", "h"],
            "algorithms": ["a"],
            "kernels": ["k"],
        }
        assert (
            ka._build_summary(parsed)
            == "2 module(s) | 1 class(es) | 3 function(s) | 1 algorithm(s) | 1 kernel(s)"
        )

    def test_empty(self):
        assert ka._build_summary({}) == "no structure extracted"


class TestHelpers:
    def test_utcnow_iso_is_timezone_aware(self):
        from datetime import datetime

        parsed = datetime.fromisoformat(ka._utcnow_iso())
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_sha256_hex(self):
        assert ka._sha256_hex(b"abc").startswith("ba7816bf")
        assert len(ka._sha256_hex(b"")) == 64


class TestAsyncBridge:
    async def _coro_value(self):
        return 42

    def test_run_sync_from_plain_sync_context(self):
        assert ka._run_coroutine_sync(self._coro_value(), "plain") == 42

    def test_run_sync_from_running_loop(self):
        async def inner():
            # Inside a running loop — the bridge must offload to a thread.
            return ka._run_coroutine_sync(self._coro_value(), "in-loop")

        assert asyncio.run(inner()) == 42

    def test_run_sync_propagates_errors(self):
        async def boom():
            raise ValueError("kablam")

        with pytest.raises(ValueError, match="kablam"):
            ka._run_coroutine_sync(boom(), "boom")

    def test_fire_coroutine_inside_loop_no_raise(self):
        completed: list[str] = []

        async def side_effect():
            completed.append("done")

        async def inner():
            ka._fire_coroutine(side_effect(), "task-x")
            # Give the fire-and-forget task a moment to run.
            await asyncio.sleep(0.05)

        asyncio.run(inner())
        assert completed == ["done"]

    def test_fire_coroutine_failure_swallowed_with_log(self, caplog):
        async def boom():
            raise RuntimeError("boom-task")

        async def inner():
            ka._fire_coroutine(boom(), "log-task")
            await asyncio.sleep(0.05)

        import logging

        with caplog.at_level(logging.WARNING, logger=ka.logger.name):
            asyncio.run(inner())
        assert any("log-task" in r.getMessage() for r in caplog.records)
