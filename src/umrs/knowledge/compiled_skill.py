"""Human-reviewed skill compilation (Phase 3: continuous learning).

The knowledge pipeline turns artifacts into structured ``KnowledgeIR`` records
and a derived knowledge graph. Phase 3 adds a *continuous learning* step: it
compiles procedural knowledge — reusable "skills" — out of what the project has
understood, so recurring methods become explicit, cite-able procedures.

Design invariants
------------------
* **Human-in-the-loop.** Compilation only ever produces skills in the
  ``pending_review`` state. Promotion to ``approved`` (or ``rejected``) is an
  explicit human decision recorded separately — the platform never
  auto-approves learned skills.
* **Deterministic + idempotent.** ``skill_id`` is a content hash of the source
  IR lineage, so recompiling the same understanding yields the same skill id
  (write-once persistence therefore never duplicates).
* **Lineage-anchored.** Every compiled skill records the ``KnowledgeIR`` ids it
  was distilled from, keeping it traceable back to canonical, provenance-anchored
  records.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .ir import KnowledgeIR

SKILL_STATUS_PENDING = "pending_review"
SKILL_STATUS_APPROVED = "approved"
SKILL_STATUS_REJECTED = "rejected"

VALID_SKILL_STATUSES = frozenset(
    {SKILL_STATUS_PENDING, SKILL_STATUS_APPROVED, SKILL_STATUS_REJECTED}
)


def _compute_skill_id(project_id: str, title: str, source_ir_ids: Sequence[str]) -> str:
    """Deterministic id from project, title, and sorted source lineage."""
    basis = f"{project_id}:{title}:{'|'.join(sorted(source_ir_ids))}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CompiledSkill:
    """A reusable procedural knowledge unit awaiting human review."""

    skill_id: str
    project_id: str
    title: str
    objective: str
    procedure: list[str]
    algorithms: list[str]
    apis: list[str]
    source_ir_ids: list[str]
    source_artifact_ids: list[str]
    status: str = SKILL_STATUS_PENDING
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "project_id": self.project_id,
            "title": self.title,
            "objective": self.objective,
            "procedure": list(self.procedure),
            "algorithms": list(self.algorithms),
            "apis": list(self.apis),
            "source_ir_ids": list(self.source_ir_ids),
            "source_artifact_ids": list(self.source_artifact_ids),
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CompiledSkill:
        return cls(
            skill_id=payload["skill_id"],
            project_id=payload["project_id"],
            title=payload["title"],
            objective=payload.get("objective", ""),
            procedure=list(payload.get("procedure", [])),
            algorithms=list(payload.get("algorithms", [])),
            apis=list(payload.get("apis", [])),
            source_ir_ids=list(payload.get("source_ir_ids", [])),
            source_artifact_ids=list(payload.get("source_artifact_ids", [])),
            status=payload.get("status", SKILL_STATUS_PENDING),
            created_at=payload.get("created_at", ""),
        )


@dataclass
class SkillReview:
    """A human review decision for a compiled skill (mutable, latest-wins)."""

    skill_id: str
    status: str
    reviewer: str = ""
    note: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "status": self.status,
            "reviewer": self.reviewer,
            "note": self.note,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillReview:
        return cls(
            skill_id=payload["skill_id"],
            status=payload["status"],
            reviewer=payload.get("reviewer", ""),
            note=payload.get("note", ""),
            reviewed_at=payload.get("reviewed_at", ""),
        )


def _procedure_from_extracted(extracted: dict[str, Any]) -> list[str]:
    """Assemble ordered procedure steps from an extracted-knowledge dict."""
    steps: list[str] = []
    methodology = (extracted.get("methodology") or "").strip()
    if methodology:
        steps.append(f"Method: {methodology}")
    execution_flow = (extracted.get("execution_flow") or "").strip()
    if execution_flow:
        steps.append(f"Execution: {execution_flow}")
    for algorithm in extracted.get("algorithms", []):
        steps.append(f"Apply algorithm: {algorithm}")
    experiments = extracted.get("experiments", [])
    if experiments:
        steps.append("Run experiments: " + "; ".join(str(e) for e in experiments))
    return steps


def _has_procedural_signal(extracted: dict[str, Any]) -> bool:
    """True when an artifact carries enough method signal to form a skill."""
    return bool(
        (extracted.get("methodology") or "").strip()
        or (extracted.get("execution_flow") or "").strip()
        or extracted.get("algorithms")
    )


def _skill_title(extracted: dict[str, Any], fallback: str) -> str:
    for key in ("objective", "summary"):
        value = (extracted.get(key) or "").strip()
        if value:
            return value[:120]
    return extracted.get("filename") or fallback


def compile_skill_from_ir(ir: KnowledgeIR) -> CompiledSkill | None:
    """Compile a single ``pending_review`` skill from one IR, or ``None``.

    Returns ``None`` when the artifact lacks procedural signal (e.g. a plain
    text note with no method, algorithms, or execution flow).
    """
    extracted = ir.extracted or {}
    if not _has_procedural_signal(extracted):
        return None
    title = _skill_title(extracted, fallback=ir.artifact_id)
    return CompiledSkill(
        skill_id=_compute_skill_id(ir.project_id, title, [ir.ir_id]),
        project_id=ir.project_id,
        title=title,
        objective=(extracted.get("objective") or "").strip(),
        procedure=_procedure_from_extracted(extracted),
        algorithms=list(extracted.get("algorithms", [])),
        apis=list(extracted.get("apis", [])),
        source_ir_ids=[ir.ir_id],
        source_artifact_ids=[ir.artifact_id],
        status=SKILL_STATUS_PENDING,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def compile_skills_from_irs(irs: Sequence[KnowledgeIR]) -> list[CompiledSkill]:
    """Compile pending-review skills from a set of canonical IRs.

    Skills are returned sorted by ``skill_id`` for deterministic output.
    """
    skills = [s for s in (compile_skill_from_ir(ir) for ir in irs) if s is not None]
    skills.sort(key=lambda s: s.skill_id)
    return skills


def review_decision(
    skill_id: str,
    status: str,
    *,
    reviewer: str = "",
    note: str = "",
) -> SkillReview:
    """Build a validated human review decision for a compiled skill."""
    if status not in VALID_SKILL_STATUSES:
        raise ValueError(f"Invalid skill review status: {status!r}")
    if status == SKILL_STATUS_PENDING:
        raise ValueError("A review decision must be 'approved' or 'rejected'")
    return SkillReview(
        skill_id=skill_id,
        status=status,
        reviewer=reviewer,
        note=note,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )


def effective_status(skill: CompiledSkill, reviews: dict[str, SkillReview]) -> str:
    """Return a skill's status after applying any human review decision."""
    review = reviews.get(skill.skill_id)
    return review.status if review is not None else skill.status


# Keep an explicit re-export surface for the field helper (documents intent).
__all__ = [
    "SKILL_STATUS_APPROVED",
    "SKILL_STATUS_PENDING",
    "SKILL_STATUS_REJECTED",
    "VALID_SKILL_STATUSES",
    "CompiledSkill",
    "SkillReview",
    "compile_skill_from_ir",
    "compile_skills_from_irs",
    "effective_status",
    "review_decision",
]
