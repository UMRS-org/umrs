"""A knowledge job that wrote nothing says so, or fails and retries.

Defect 2 of defect-wave-2026-10-08: with no embedder staged, KNOWLEDGE_EMBED
"completed successfully" having written no vectors, and nothing on the
user-facing path said so. Two siblings found while fixing it:

- embed with Postgres configured but unreachable was the same silent
  success, so a transient outage lost the document's vectors for good;
- ingest with S3 unavailable returned a ref without counts, so the
  "every source failed → failed" rule (audit B7) never fired and the job
  succeeded having stored nothing.

"No embedder configured" stays a success (embed is an opt-in stage), but the
attempt now carries a detail that `r4 job describe` prints.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.jobs.monitor import AttemptCheckResult, AttemptMonitor


def _attempt(ref: str = ""):
    return SimpleNamespace(backend_ref=ref, attempt_id="att-1", metrics={}, job_id="job-1")


_JOB = SimpleNamespace(job_id="job-1", spec={}, project_id="p")


# ── the embed checker ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_embedder_succeeds_with_a_visible_detail():
    ref = "knowledge_embed|mode=degraded|reason=no_embedder|project=p"
    result = await ka.KnowledgeEmbedStatusChecker().check_status(_attempt(ref), _JOB)
    assert result.status == "succeeded"
    assert result.metrics.get("degraded") is True
    assert "no vectors written" in result.status_detail
    assert "UMRS_KNOWLEDGE_EMBED_MODEL" in result.status_detail


@pytest.mark.asyncio
async def test_unreachable_database_fails_so_the_job_retries():
    ref = "knowledge_embed|mode=degraded|reason=no_pool|project=p"
    result = await ka.KnowledgeEmbedStatusChecker().check_status(_attempt(ref), _JOB)
    assert result.status == "failed"
    assert "unreachable" in result.exit_reason


@pytest.mark.asyncio
async def test_no_database_configured_is_a_visible_no_op():
    ref = "knowledge_embed|mode=degraded|reason=no_dsn|project=p"
    result = await ka.KnowledgeEmbedStatusChecker().check_status(_attempt(ref), _JOB)
    assert result.status == "succeeded"
    assert "no knowledge database configured" in result.status_detail


# ── the embed backend tells the two pool cases apart ────────────────────────


@pytest.mark.parametrize(("dsn", "reason"), [("postgres://db/umrs", "no_pool"), ("", "no_dsn")])
@pytest.mark.asyncio
async def test_embed_backend_names_why_there_is_no_pool(monkeypatch, dsn, reason):
    @asynccontextmanager
    async def no_pool():
        yield None

    monkeypatch.setattr(ka, "_chunk_embedder_for_site", lambda: (object(), ""))
    monkeypatch.setattr(ka, "_build_s3_client", lambda _spec: (None, ""))
    monkeypatch.setattr(ka, "_open_pool", no_pool)
    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: dsn)
    job = SimpleNamespace(job_id="job-1", spec={"sources": [{"artifact_id": "a"}]}, project_id="p")
    ref = await ka.KnowledgeEmbedBackend()._submit_embed(job, _attempt(), "p")
    assert f"reason={reason}" in ref


# ── ingest with S3 down ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_without_s3_is_a_failure(monkeypatch):
    backend = ka.KnowledgeIngestBackend()
    monkeypatch.setattr(backend, "_init_s3", lambda _spec: None)
    sources = [{"kind": "inline_text", "uri": "a.md", "content": "x"}] * 2
    job = SimpleNamespace(job_id="job-1", spec={"sources": sources}, project_id="p")
    ref = await backend.submit_job(job, _attempt())
    result = await ka.KnowledgeIngestStatusChecker().check_status(_attempt(ref), job)
    assert result.status == "failed"
    assert "2 source(s) failed" in result.exit_reason


# ── the monitor keeps the detail of a success ───────────────────────────────


def test_monitor_persists_a_success_detail():
    store = MagicMock()
    store.get_job.return_value = SimpleNamespace(project_id="")
    monitor = AttemptMonitor(job_store=store, status_checker=MagicMock())
    result = AttemptCheckResult.succeeded(metrics={"degraded": True}, detail="degraded: why")
    monitor._apply_success(_attempt(), result)
    store.update_attempt_status_detail.assert_called_once_with("att-1", "degraded: why")


def test_monitor_leaves_a_plain_success_alone():
    store = MagicMock()
    store.get_job.return_value = SimpleNamespace(project_id="")
    monitor = AttemptMonitor(job_store=store, status_checker=MagicMock())
    monitor._apply_success(_attempt(), AttemptCheckResult.succeeded())
    store.update_attempt_status_detail.assert_not_called()
