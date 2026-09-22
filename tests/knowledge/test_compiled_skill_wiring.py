"""V-09/D-35 pins: the understand job compiles pending_review skill proposals
from the understood IRs and registers them DISABLED (never auto-approved)."""

from __future__ import annotations

import json

import pytest


def _ir_row(ir_id: str, content: dict) -> dict:
    return {
        "ir_id": ir_id,
        "artifact_id": f"art-{ir_id}",
        "content": content,
        "content_hash": f"h-{ir_id}",
        "ir_type": "document",
        "project_id": "proj-a",
    }


_PROCEDURAL = {
    "objective": "Calibrate the microscope stage",
    "methodology": "Load sample, zero the stage, record z-offset",
    "algorithms": ["levenberg-marquardt"],
    "apis": ["set_z"],
}

_PLAIN = {"text": "meeting notes, no method"}


class _Pool:
    def __init__(self, rows):
        self._rows = rows
        self.inserts: list[tuple] = []

    async def fetch(self, query, *args):
        return self._rows

    async def fetchval(self, query, *args):
        self.inserts.append(args)
        return args[0]  # RETURNING skill_id (insert happened)


class _Store:
    def __init__(self, pool):
        self._pool = pool


@pytest.fixture()
def run():
    from umrs.jobs.knowledge_adapters import _compile_skill_proposals

    return _compile_skill_proposals


async def test_procedural_irs_register_disabled_proposals(run):
    pool = _Pool([_ir_row("ir-1", _PROCEDURAL)])
    count = await run(_Store(pool), "proj-a")
    assert count == 1
    (args,) = pool.inserts
    skill_id, name, version, project_id, skill_md, manifest = args
    assert name.startswith("Calibrate") or "Calibrate" in name
    assert project_id == "proj-a"
    manifest = json.loads(manifest)
    assert manifest["_compiled_skill"] is True
    assert manifest["status"] == "pending_review"
    assert manifest["source_ir_ids"] == ["ir-1"]
    assert "zero the stage" in skill_md
    assert "levenberg-marquardt" in skill_md


async def test_plain_text_irs_compile_nothing(run):
    pool = _Pool([_ir_row("ir-1", _PLAIN)])
    count = await run(_Store(pool), "proj-a")
    assert count == 0
    assert pool.inserts == []


async def test_no_irs_short_circuits(run):
    pool = _Pool([])
    assert await run(_Store(pool), "proj-a") == 0


async def test_conflict_noop_counts_nothing(run):
    class _ConflictPool(_Pool):
        async def fetchval(self, query, *args):
            self.inserts.append(args)
            return None  # ON CONFLICT DO NOTHING → no RETURNING

    pool = _ConflictPool([_ir_row("ir-1", _PROCEDURAL)])
    assert await run(_Store(pool), "proj-a") == 0
    assert len(pool.inserts) == 1  # INSERT attempted, idempotent no-op


def test_understand_result_carries_compiled_count():
    from umrs.jobs.knowledge_adapters import KnowledgeUnderstandResult

    result = KnowledgeUnderstandResult()
    assert result.compiled_skill_count == 0
