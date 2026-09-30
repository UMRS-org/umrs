"""Tests for the knowledge IR (canonical persistence record) — Phase 0."""

from __future__ import annotations

import json

from umrs.knowledge.ir import (
    EXTRACTOR_VERSION,
    KnowledgeIR,
    Provenance,
    build_knowledge_ir,
    compute_content_hash,
)
from umrs.knowledge.persistence import (
    knowledge_ir_dir,
    read_local_knowledge_ir,
    write_local_knowledge_ir,
)

CONTENT = b"objective: test the IR layer\nmethodology: unit tests\n"
EXTRACTED = {"artifact_id": "a1", "artifact_type": "text", "summary": "s"}


def _build(**overrides):
    kwargs = {
        "project_id": "proj-1",
        "artifact_id": "a1",
        "artifact_type": "text",
        "content": CONTENT,
        "extracted": EXTRACTED,
        "extractor_version": EXTRACTOR_VERSION,
    }
    kwargs.update(overrides)
    return build_knowledge_ir(**kwargs)


class TestContentHash:
    def test_sha256_hex(self):
        expected = __import__("hashlib").sha256(CONTENT).hexdigest()
        assert compute_content_hash(CONTENT) == expected

    def test_different_content_differs(self):
        assert compute_content_hash(b"a") != compute_content_hash(b"b")


class TestIrIdDeterminism:
    def test_same_inputs_same_id(self):
        ir_a = _build()
        ir_b = _build()
        assert ir_a.ir_id == ir_b.ir_id

    def test_content_change_changes_id(self):
        ir_a = _build()
        ir_b = _build(content=CONTENT + b"more")
        assert ir_a.ir_id != ir_b.ir_id

    def test_extractor_version_change_changes_id(self):
        ir_a = _build(extractor_version="1.0.0")
        ir_b = _build(extractor_version="2.0.0")
        assert ir_a.ir_id != ir_b.ir_id

    def test_artifact_id_change_changes_id(self):
        ir_a = _build(artifact_id="a1")
        ir_b = _build(artifact_id="a2")
        assert ir_a.ir_id != ir_b.ir_id


class TestProvenance:
    def test_regex_extraction_has_no_llm_fields(self):
        ir = _build()
        prov = ir.provenance
        assert prov.content_hash == compute_content_hash(CONTENT)
        assert prov.extractor_version == EXTRACTOR_VERSION
        assert prov.prompt_version is None
        assert prov.llm_model is None
        assert prov.config_hash is None
        assert prov.ocr_engine is None

    def test_llm_fields_are_recorded(self):
        ir = _build(llm_model="glm-4.6", prompt_version="p1", ocr_engine="glm-ocr")
        assert ir.provenance.llm_model == "glm-4.6"
        assert ir.provenance.prompt_version == "p1"
        assert ir.provenance.ocr_engine == "glm-ocr"

    def test_provenance_round_trip(self):
        prov = Provenance(
            content_hash="h",
            extractor_version="1.0.0",
            llm_model="m",
        )
        assert Provenance.from_dict(prov.to_dict()) == prov


class TestKnowledgeIrRoundTrip:
    def test_to_from_dict(self):
        ir = _build()
        restored = KnowledgeIR.from_dict(ir.to_dict())
        assert restored == ir

    def test_content_hash_matches_bytes(self):
        ir = _build()
        assert ir.content_hash == compute_content_hash(CONTENT)

    def test_frozen(self):
        ir = _build()
        try:
            ir.ir_id = "x"  # type: ignore[misc]
        except Exception as exc:
            assert exc.__class__.__name__ == "FrozenInstanceError"
        else:
            raise AssertionError("KnowledgeIR should be immutable")


class TestLocalPersistence:
    def test_write_then_read(self, tmp_path):
        ir_dir = knowledge_ir_dir(tmp_path)
        ir = _build()
        path = write_local_knowledge_ir(ir_dir, ir)
        assert path.is_file()
        loaded = read_local_knowledge_ir(ir_dir, ir.ir_id)
        assert loaded == ir

    def test_write_once_idempotent(self, tmp_path):
        ir_dir = knowledge_ir_dir(tmp_path)
        ir = _build()
        path = write_local_knowledge_ir(ir_dir, ir)
        original = path.read_text(encoding="utf-8")

        # Tamper on disk, then attempt a second write with the same ir_id.
        path.write_text("{}", encoding="utf-8")
        write_local_knowledge_ir(ir_dir, ir)
        # Write-once: existing file left untouched (not restored to original).
        assert path.read_text(encoding="utf-8") == "{}"
        assert original != "{}"

    def test_read_missing_returns_none(self, tmp_path):
        ir_dir = knowledge_ir_dir(tmp_path)
        assert read_local_knowledge_ir(ir_dir, "does-not-exist") is None

    def test_stored_json_is_sorted_and_indented(self, tmp_path):
        ir_dir = knowledge_ir_dir(tmp_path)
        ir = _build()
        path = write_local_knowledge_ir(ir_dir, ir)
        raw = path.read_text(encoding="utf-8")
        assert raw == json.dumps(ir.to_dict(), indent=2, sort_keys=True)
