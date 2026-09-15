"""Knowledge intermediate representation (IR) — the canonical persistence record.

Phase 0 of the knowledge persistence layer.

``KnowledgeIR`` is the immutable, versioned, provenance-anchored record that
sits between the Understanding stage (``ExtractedKnowledge``) and the
Compilation stage (graph / vector / SQL indexes). It is the canonical
intermediate representation: every derived index is regenerable from it.

Design invariants:
  - **Canonical, not the graph.** Relationships (heuristic ``KnowledgeLinker``)
    are the least-reliable output. Truth is anchored to the immutable source
    bytes + extraction record, so derived indexes can be regenerated when the
    extractor or linker improves.
  - **Content-addressed, deterministic id.** ``ir_id`` is derived from
    ``(artifact_id, content_hash, extractor_version)`` so re-running extraction
    on unchanged content with the same extractor version yields the same id.
    This makes write-once persistence idempotent.
  - **Honest reproducibility.** Provenance enables traceability and targeted
    invalidation, not bit-identical regeneration (LLM/OCR extraction is not a
    pure function).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# Semantic version of the extraction pipeline. Bump when extraction output for
# the same input changes (new fields, new patterns, LLM/OCR backends). Anchors
# KnowledgeIR provenance and the deterministic ir_id.
EXTRACTOR_VERSION = "1.0.0"


def compute_content_hash(content: bytes) -> str:
    """Return the sha256 hex digest of raw source bytes."""
    return hashlib.sha256(content).hexdigest()


def _compute_ir_id(artifact_id: str, content_hash: str, extractor_version: str) -> str:
    """Deterministic id from (artifact_id, content_hash, extractor_version)."""
    seed = f"{artifact_id}:{content_hash}:{extractor_version}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Provenance:
    """Lineage of how a ``KnowledgeIR`` was produced.

    Enables traceability and targeted invalidation. Fields that describe an
    LLM/OCR pipeline are ``None`` until those pipelines are introduced.
    """

    content_hash: str
    extractor_version: str
    prompt_version: str | None = None
    llm_model: str | None = None
    config_hash: str | None = None
    ocr_engine: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Provenance:
        return cls(
            content_hash=payload["content_hash"],
            extractor_version=payload["extractor_version"],
            prompt_version=payload.get("prompt_version"),
            llm_model=payload.get("llm_model"),
            config_hash=payload.get("config_hash"),
            ocr_engine=payload.get("ocr_engine"),
        )


@dataclass(frozen=True)
class KnowledgeIR:
    """Immutable canonical record for one extracted artifact."""

    ir_id: str
    artifact_id: str
    project_id: str
    artifact_type: str
    content_hash: str
    extractor_version: str
    extracted: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(
        default_factory=lambda: Provenance(content_hash="", extractor_version="")
    )
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ir_id": self.ir_id,
            "artifact_id": self.artifact_id,
            "project_id": self.project_id,
            "artifact_type": self.artifact_type,
            "content_hash": self.content_hash,
            "extractor_version": self.extractor_version,
            "extracted": self.extracted,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KnowledgeIR:
        return cls(
            ir_id=payload["ir_id"],
            artifact_id=payload["artifact_id"],
            project_id=payload["project_id"],
            artifact_type=payload["artifact_type"],
            content_hash=payload["content_hash"],
            extractor_version=payload["extractor_version"],
            extracted=payload.get("extracted", {}),
            provenance=Provenance.from_dict(payload["provenance"]),
            created_at=payload.get("created_at", ""),
        )


def build_knowledge_ir(
    *,
    project_id: str,
    artifact_id: str,
    artifact_type: str,
    content: bytes,
    extracted: dict[str, Any],
    extractor_version: str,
    prompt_version: str | None = None,
    llm_model: str | None = None,
    config_hash: str | None = None,
    ocr_engine: str | None = None,
) -> KnowledgeIR:
    """Build a ``KnowledgeIR`` from raw source bytes and extraction output.

    ``content`` is the raw source bytes (the canonical input that must be
    retained for regeneration). ``extracted`` is the lossless
    ``ExtractedKnowledge.to_dict()`` output.
    """
    content_hash = compute_content_hash(content)
    ir_id = _compute_ir_id(artifact_id, content_hash, extractor_version)
    provenance = Provenance(
        content_hash=content_hash,
        extractor_version=extractor_version,
        prompt_version=prompt_version,
        llm_model=llm_model,
        config_hash=config_hash,
        ocr_engine=ocr_engine,
    )
    return KnowledgeIR(
        ir_id=ir_id,
        artifact_id=artifact_id,
        project_id=project_id,
        artifact_type=artifact_type,
        content_hash=content_hash,
        extractor_version=extractor_version,
        extracted=extracted,
        provenance=provenance,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
