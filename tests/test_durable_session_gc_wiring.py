"""SESSION_GC wiring in the durable runtime (2026-10-23).

``DurableOrchestrationService.startup`` built "ONE shared backend across
dispatch factory + status checker" from ``from .session_gc import
SessionGcBackend`` — relative to ``umrs.orchestrator``, a module that does
not exist (the backend lives in ``umrs.jobs.session_gc``). The ImportError
was swallowed at debug level, so every boot passed ``session_store=None``:
every session_gc job failed at dispatch "session_gc lacks a shared umrs_db
session store (wiring missing)", and the status checker never saw the
backend's reports. ruff flagged the never-used ``session_gc_backend``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from umrs.jobs import JobType
from umrs.jobs.adapters import create_backend_factory
from umrs.jobs.session_gc import SessionGcBackend
from umrs.orchestrator import durable


def _start_capturing_wiring() -> dict:
    captured: dict = {}

    def _factory(**kwargs):
        captured["factory"] = kwargs
        return MagicMock()

    def _checker(**kwargs):
        captured["checker"] = kwargs
        return MagicMock()

    job_service = MagicMock()
    job_service.startup = AsyncMock()
    with (
        patch.object(durable, "ensure_schema"),
        patch.object(durable, "JobService", return_value=job_service),
        patch.object(durable, "ensure_model_registry_schema"),
        patch.object(durable, "ModelStore"),
        patch.object(durable, "PathValidator"),
        patch.object(durable, "create_backend_factory", side_effect=_factory),
        patch.object(durable, "create_status_checker", side_effect=_checker),
        patch("umrs.jobs.service.set_job_service"),
    ):
        service = durable.DurableOrchestrationService(
            MagicMock(), orchestration_service=MagicMock()
        )
        asyncio.run(service.startup())
    return captured


def test_startup_shares_one_session_gc_backend_backed_by_the_session_store():
    captured = _start_capturing_wiring()
    backend = captured["checker"]["session_gc_backend"]
    assert isinstance(backend, SessionGcBackend)
    assert backend.session_store is not None
    assert captured["factory"]["session_gc_backend"] is backend
    assert captured["factory"]["session_store"] is backend.session_store
    assert captured["checker"]["session_store"] is backend.session_store


def test_the_factory_dispatches_session_gc_to_the_shared_backend():
    shared = SessionGcBackend(session_store=MagicMock())
    factory = create_backend_factory(
        orchestrator=MagicMock(),
        registry=MagicMock(),
        port_manager=MagicMock(),
        session_gc_backend=shared,
    )
    assert factory(JobType.SESSION_GC) is shared


def test_the_factory_still_builds_its_own_backend_from_a_store():
    store = MagicMock()
    factory = create_backend_factory(
        orchestrator=MagicMock(),
        registry=MagicMock(),
        port_manager=MagicMock(),
        session_store=store,
    )
    assert factory(JobType.SESSION_GC).session_store is store
