"""Tests for human-reviewed skill compilation (Phase 3)."""

from __future__ import annotations

import pytest
from umrs.knowledge.compiled_skill import (
    SKILL_STATUS_APPROVED,
    SKILL_STATUS_PENDING,
    SKILL_STATUS_REJECTED,
    CompiledSkill,
    compile_skill_from_ir,
    compile_skills_from_irs,
    effective_status,
    review_decision,
)
from umrs.knowledge.ir import EXTRACTOR_VERSION, build_knowledge_ir
from umrs.knowledge.persistence import (
    read_all_local_compiled_skills,
    read_local_skill_reviews,
    write_local_compiled_skill,
    write_local_skill_reviews,
)


def _ir(artifact_id: str, *, extracted: dict, content: bytes = b"data"):
    payload = {"artifact_id": artifact_id, "artifact_type": "code", "filename": f"{artifact_id}.py"}
    payload.update(extracted)
    return build_knowledge_ir(
        project_id="proj-1",
        artifact_id=artifact_id,
        artifact_type="code",
        content=content,
        extracted=payload,
        extractor_version=EXTRACTOR_VERSION,
    )


class TestCompileSkillFromIR:
    def test_no_procedural_signal_returns_none(self):
        ir = _ir("plain", extracted={"summary": "just a note"})
        assert compile_skill_from_ir(ir) is None

    def test_methodology_produces_skill(self):
        ir = _ir("m1", extracted={"objective": "solve PDE", "methodology": "finite diff"})
        skill = compile_skill_from_ir(ir)
        assert skill is not None
        assert skill.status == SKILL_STATUS_PENDING
        assert skill.objective == "solve PDE"
        assert any("finite diff" in step for step in skill.procedure)
        assert skill.source_ir_ids == [ir.ir_id]
        assert skill.source_artifact_ids == ["m1"]

    def test_algorithms_produce_procedure_steps(self):
        ir = _ir("a1", extracted={"algorithms": ["CG", "GMRES"], "apis": ["solve"]})
        skill = compile_skill_from_ir(ir)
        assert skill is not None
        assert skill.algorithms == ["CG", "GMRES"]
        assert skill.apis == ["solve"]
        assert any("CG" in step for step in skill.procedure)

    def test_skill_id_deterministic(self):
        ir = _ir("d1", extracted={"objective": "x", "methodology": "y"})
        first = compile_skill_from_ir(ir)
        second = compile_skill_from_ir(ir)
        assert first.skill_id == second.skill_id

    def test_title_falls_back_to_filename(self):
        ir = _ir("f1", extracted={"algorithms": ["A"]})
        skill = compile_skill_from_ir(ir)
        assert skill.title == "f1.py"


class TestCompileSkillsFromIRs:
    def test_filters_and_sorts(self):
        irs = [
            _ir("plain", extracted={"summary": "no method"}),
            _ir("m1", extracted={"methodology": "approach one"}),
            _ir("m2", extracted={"methodology": "approach two"}),
        ]
        skills = compile_skills_from_irs(irs)
        assert len(skills) == 2
        assert skills == sorted(skills, key=lambda s: s.skill_id)


class TestReviewDecision:
    def test_approved_decision(self):
        review = review_decision("sid", SKILL_STATUS_APPROVED, reviewer="alice")
        assert review.status == SKILL_STATUS_APPROVED
        assert review.reviewer == "alice"
        assert review.reviewed_at

    def test_rejected_decision(self):
        review = review_decision("sid", SKILL_STATUS_REJECTED, note="wrong")
        assert review.status == SKILL_STATUS_REJECTED
        assert review.note == "wrong"

    def test_pending_is_not_a_decision(self):
        with pytest.raises(ValueError, match="approved.*rejected"):
            review_decision("sid", SKILL_STATUS_PENDING)

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError, match="Invalid skill review status"):
            review_decision("sid", "bogus")


class TestEffectiveStatus:
    def test_defaults_to_skill_status_when_unreviewed(self):
        skill = CompiledSkill(
            skill_id="s1",
            project_id="p",
            title="t",
            objective="",
            procedure=[],
            algorithms=[],
            apis=[],
            source_ir_ids=[],
            source_artifact_ids=[],
        )
        assert effective_status(skill, {}) == SKILL_STATUS_PENDING

    def test_review_overrides_status(self):
        skill = CompiledSkill(
            skill_id="s1",
            project_id="p",
            title="t",
            objective="",
            procedure=[],
            algorithms=[],
            apis=[],
            source_ir_ids=[],
            source_artifact_ids=[],
        )
        review = review_decision("s1", SKILL_STATUS_APPROVED)
        assert effective_status(skill, {"s1": review}) == SKILL_STATUS_APPROVED


class TestPersistence:
    def test_write_once_does_not_overwrite(self, tmp_path):
        ir = _ir("m1", extracted={"methodology": "approach"})
        skill = compile_skill_from_ir(ir)
        path1 = write_local_compiled_skill(tmp_path, skill)
        original = path1.read_text(encoding="utf-8")
        # Recompiling yields identical skill_id; write-once keeps the file intact.
        path2 = write_local_compiled_skill(tmp_path, skill)
        assert path1 == path2
        assert path2.read_text(encoding="utf-8") == original

    def test_round_trip_all_skills(self, tmp_path):
        for aid in ("m1", "m2"):
            skill = compile_skill_from_ir(_ir(aid, extracted={"methodology": aid}))
            write_local_compiled_skill(tmp_path, skill)
        loaded = read_all_local_compiled_skills(tmp_path)
        assert len(loaded) == 2
        assert all(s.status == SKILL_STATUS_PENDING for s in loaded)

    def test_reviews_round_trip(self, tmp_path):
        review = review_decision("s1", SKILL_STATUS_APPROVED, reviewer="bob")
        write_local_skill_reviews(tmp_path, {"s1": review})
        loaded = read_local_skill_reviews(tmp_path)
        assert loaded["s1"].status == SKILL_STATUS_APPROVED
        assert loaded["s1"].reviewer == "bob"

    def test_reviews_missing_returns_empty(self, tmp_path):
        assert read_local_skill_reviews(tmp_path) == {}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
