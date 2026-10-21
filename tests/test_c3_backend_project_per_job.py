"""Cycle-3 mem lane, N7 (2026-10-22): a knowledge job runs under its OWN
project, not the first project the server process saw.

Found live (lane stack, ``.stab-cycle3/mem/repro/n7_sticky_project.py``):
upload a note to project A, then one to project B. B's understand job ran as
``project=A``: B's file node was written into A's graph and B's embed job was
chained under A (twice, fresh server processes). The understand, verify and
export backends are process singletons (jobs/adapters.py) and cached the
first job's project: ``project_id = self._project_id or job.spec[...]``. An
export for B would have bundled A's knowledge.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any

from umrs.jobs import knowledge_adapters as ka


def _job(project: str) -> Any:
    return SimpleNamespace(
        job_id=f"job-{project}", project_id=project, user_id="u", spec={"project_id": project}
    )


def _attempt() -> Any:
    return SimpleNamespace(attempt_id="att", metrics={})


async def _no_pool() -> None:
    return None


def test_understand_runs_each_job_under_its_own_project(monkeypatch):
    seen: list[str] = []

    async def submit(_self: Any, _job: Any, _attempt: Any, project_id: str) -> str:
        seen.append(project_id)
        return "ok"

    monkeypatch.setattr(ka.KnowledgeUnderstandBackend, "_submit_understand", submit)
    backend = ka.KnowledgeUnderstandBackend()
    for project in ("p-a", "p-b"):
        asyncio.run(backend.submit_job(_job(project), _attempt()))
    assert seen == ["p-a", "p-b"]


def test_verify_runs_each_job_under_its_own_project(monkeypatch):
    monkeypatch.setattr(ka, "_created_pool", _no_pool)
    backend = ka.KnowledgeVerifyBackend()
    refs = [asyncio.run(backend.submit_job(_job(p), _attempt())) for p in ("p-a", "p-b")]
    assert "project=p-a|" in refs[0]
    assert "project=p-b|" in refs[1]


def test_export_runs_each_job_under_its_own_project(monkeypatch):
    seen: list[str] = []

    async def export(_self: Any, _pool: Any, project_id: str, *_a: Any, **_k: Any) -> Any:
        seen.append(project_id)
        raise RuntimeError("stop here")

    monkeypatch.setattr(ka, "_created_pool", _no_pool)
    monkeypatch.setattr(ka.KnowledgeExportBackend, "_export_with_integrity", export)
    backend = ka.KnowledgeExportBackend()
    for project in ("p-a", "p-b"):
        with contextlib.suppress(RuntimeError):
            asyncio.run(backend.submit_job(_job(project), _attempt()))
    assert seen == ["p-a", "p-b"]
