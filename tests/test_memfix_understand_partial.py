"""Cycle-3 memfix #5 (2026-10-22): one text understood from its prefix no
longer marks the whole understand run degraded.

Found by the mem review (``.stab-cycle3/memreview/memreview-v5-0``): an
ingest of 20 files, one of them a text past READ_CAP, set the run-level
``result.degraded``, and ``_run_understand`` then skipped the skill-proposal
leg for all 20 artifacts. Now the cut is named per artifact
(``partial_reads``), the run is not degraded for it, and only the cut
artifact's IR is left out of the proposals.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from tests.test_c3_knowledge_bounded_reads import _S3
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import canonical_identity


def _sources(count: int) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": f"a{i}",
            "content_hash": "h",
            "source_ref": f"k{i}",
            "metadata": {"source_name": f"f{i}.log"},
        }
        for i in range(count)
    ]


def _run(monkeypatch, calls: list[Any]) -> Any:
    monkeypatch.setattr(ka, "_UNDERSTAND_PREFIX_BYTES", 1024)
    monkeypatch.setattr(canonical_identity, "READ_CAP", 4096)
    sources = _sources(5)
    s3 = _S3({("kb", f"k{i}"): (b"x" * 10000 if i == 0 else b"small\n") for i in range(5)})

    async def to_process(_sources: Any, _store: Any, _project: str) -> list[dict]:
        return sources

    async def one(_job, _attempt, src, _p, _g, _store, s3c, bucket, result, **_k):
        ka._understand_bytes(s3c, bucket, src, result)

    async def compile_proposals(_store: Any, project_id: str, **kw: Any) -> int:
        calls.append((project_id, kw.get("exclude")))
        return 0

    monkeypatch.setattr(ka, "_artifacts_to_process", to_process)
    monkeypatch.setattr(ka, "_understand_one_artifact", one)
    monkeypatch.setattr(ka, "_compile_skill_proposals", compile_proposals)
    job = SimpleNamespace(spec={}, job_id="j", project_id="p")
    return asyncio.run(ka._run_understand(job, None, "p", sources, object(), s3, "kb"))


def test_one_cut_text_does_not_degrade_the_run_or_stop_the_proposals(monkeypatch):
    calls: list[Any] = []
    result = _run(monkeypatch, calls)
    assert not result.degraded and result.degraded_reason == ""
    assert [item["artifact_id"] for item in result.partial_reads] == ["a0"]
    assert "understood from its first" in result.partial_reads[0]["reason"]
    assert calls == [("p", {"a0"})]


def test_the_attempt_result_names_the_partial_read(monkeypatch):
    result = _run(monkeypatch, [])
    attempt = SimpleNamespace(metrics={})
    ka._write_result(attempt, result)
    stored = json.loads(attempt.metrics["result"])
    assert stored["degraded"] is False
    assert [item["artifact_id"] for item in stored["partial_reads"]] == ["a0"]


def test_the_proposals_leave_out_only_the_cut_artifacts_ir(monkeypatch):
    import umrs.knowledge.compiled_skill as compiled_skill

    seen: list[list[str]] = []

    def compile_skills(irs: list[Any]) -> list[Any]:
        seen.append([ir.artifact_id for ir in irs])
        return []

    async def fetch(*_args: Any) -> list[dict[str, Any]]:
        return [
            {
                "ir_id": f"ir-{a}",
                "artifact_id": a,
                "content": {},
                "content_hash": "h",
                "ir_type": "document",
                "project_id": "p",
            }
            for a in ("a0", "a1", "a2")
        ]

    monkeypatch.setattr(compiled_skill, "compile_skills_from_irs", compile_skills)
    store = SimpleNamespace(_pool=SimpleNamespace(fetch=fetch))
    assert asyncio.run(ka._compile_skill_proposals(store, "p", exclude={"a0"})) == 0
    assert seen == [["a1", "a2"]]
