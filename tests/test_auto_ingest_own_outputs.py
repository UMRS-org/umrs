"""Auto-ingest acquires the finished job's OWN outputs, never the project's
stored artifacts (ops-watch #612-#620, 2026-10-27).

Every successful ``ingest_results`` job (each workflow_run) re-listed the
whole ``knowledge_artifacts`` table of its project and re-submitted the rows
whose content key the dedup HEAD did not find (rag:// rows carry no bytes,
rows whose object is gone) as KNOWLEDGE_INGEST sources of kind
``document``/``schema``. The source resolver refuses those kinds, so each
run produced a knowledge_ingest that failed every source three times, and a
chained understand/embed that failed "stored object is gone".
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge.sources import EgressDisabledError, SourceResolver

_OWN = {
    "artifact_id": "wfart-own",
    "content_hash": "a" * 64,
    "kind": "s3_uri",
    "source_ref": "s3://r4-data/jobs/j-own/attempt_1/out.json",
}
_STORED_ROWS = [
    {
        "artifact_id": "artifact-rag-legacy-c265",
        "content_hash": "b" * 64,
        "kind": "document",
        "source_ref": "rag://legacy-c265705db359abbcbee7a4d38394fba3",
    },
    {
        "artifact_id": "art-schema-c69c",
        "content_hash": "c" * 64,
        "kind": "schema",
        "source_ref": "projects/default/knowledge/artifacts/" + "c" * 64,
    },
]


class _Pool:
    """``fetchval`` answers the attempt metrics, ``fetch`` the project's
    ``knowledge_artifacts`` rows (the only list query the collector makes
    when the job has a commit attempt)."""

    async def fetchval(self, _query: str, *_args: object) -> dict:
        return {"artifacts": [dict(_OWN)]}

    async def fetch(self, _query: str, *_args: object) -> list[dict]:
        return [dict(row) for row in _STORED_ROWS]


@asynccontextmanager
async def _open_pool():
    yield _Pool()


def _job() -> SimpleNamespace:
    return SimpleNamespace(
        job_id="j-own",
        project_id="default",
        user_id="alice",
        commit_attempt_id="att-1",
        priority=0,
        spec={"ingest_results": True, "workflow": "wf", "step": "*"},
    )


@pytest.fixture
def _pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "postgresql://fake/db")
    monkeypatch.setattr(ka, "_open_pool", _open_pool)


@pytest.mark.usefixtures("_pool")
def test_collect_returns_only_the_jobs_own_outputs() -> None:
    collected = ka._collect_artifact_ids(_job())

    assert [a["artifact_id"] for a in collected] == ["wfart-own"]


class _Service:
    def __init__(self) -> None:
        self.submitted: list[Any] = []

    def submit_job(self, submission: Any) -> str:
        self.submitted.append(submission)
        return f"job-{len(self.submitted)}"


@pytest.mark.usefixtures("_pool")
def test_every_ingest_source_is_resolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _Service()
    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: service)
    # Nothing is in the content store yet: dedup keeps every ref.
    monkeypatch.setattr(ka, "_dedup_artifacts", lambda refs, _p: (list(refs), "bucket"))
    monkeypatch.setattr(ka, "_store_artifact_provenance", lambda *_a, **_k: None)
    monkeypatch.setattr(ka, "_create_produced_by_edges", lambda *_a, **_k: None)

    ka.enqueue_knowledge_ingest(_job(), SimpleNamespace(attempt_id="att-1"), "wf", "*")

    ingest = [s for s in service.submitted if s.job_type.value == "knowledge_ingest"]
    assert len(ingest) == 1
    sources = ingest[0].spec["sources"]
    assert [s["uri"] for s in sources] == [_OWN["source_ref"]]
    resolver = SourceResolver()
    for source in sources:
        try:
            resolver.resolve(source["kind"], source["uri"])
        except EgressDisabledError as exc:  # the 2026-10-27 refusal
            pytest.fail(f"auto-ingest submitted an unresolvable source {source}: {exc}")
