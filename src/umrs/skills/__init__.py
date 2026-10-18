"""UMRS Skill Registry.

A skill is a bundle of:
- SKILL.md    — human-readable description and instructions
- mcp.json    — optional MCP server manifest
- config.json — metadata (name, version, capabilities, dependencies)

Skills are registered in UMRS so they can be discovered and loaded
by any client (r4, the VS Code extension, n8n, etc.). UMRS stores skill metadata
and validates skill bundles at registration time.

No dependency on r4.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from r4_contracts.skills import SkillManifest, SkillScope

logger = logging.getLogger(__name__)


def project_key(scope: SkillScope, project_id: str | None) -> str:
    """The ``skills.project_key`` value (migration 0123): the project of a
    project-scope skill, else ``''``."""
    return (project_id or "") if scope == SkillScope.PROJECT else ""


def skill_key(
    scope: SkillScope, owner_id: str, project_id: str | None, name: str
) -> str:
    """Identity string a skill id hashes; it follows the registry key.

    User and system skills keep the pre-0123 ``scope:owner:name`` form, so
    their stored ids stay valid; a project skill adds its project.
    """
    key = project_key(scope, project_id)
    if key:
        return f"{scope.value}:{owner_id}:{key}:{name}"
    return f"{scope.value}:{owner_id}:{name}"


@dataclass
class RegisteredSkill:
    """A skill stored in the UMRS registry."""

    skill_id: str = ""
    manifest: SkillManifest = field(default_factory=lambda: SkillManifest(name=""))
    scope: SkillScope = SkillScope.USER
    owner_id: str = ""
    project_id: str | None = None
    skill_md: str = ""  # Content of SKILL.md
    registered_at: datetime | None = None
    enabled: bool = True
    review_state: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # The id follows the registry key: a project skill is one per project
        # (migration 0123; it was one per owner, so registering it in a second
        # project replaced the first, H-10.1 follow-up 2026-10-18).
        if not self.skill_id and self.manifest.name:
            identity = skill_key(
                self.scope, self.owner_id, self.project_id, self.manifest.name
            )
            self.skill_id = hashlib.sha256(identity.encode()).hexdigest()[:16]


REVIEW_STATUS_PENDING = "pending_review"
REVIEW_DECISIONS = frozenset({"approved", "rejected"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_pending_proposal(skill: RegisteredSkill) -> bool:
    """A loop-proposed skill awaiting human review (D-27 edit plane)."""
    return (
        skill.enabled is False
        and (skill.review_state or {}).get("status") == REVIEW_STATUS_PENDING
    )


class SkillValidationError(Exception):
    """Raised when a skill bundle fails validation."""


class SkillRegistry:
    """Registry for skill bundles.

    In production, backed by Postgres (``SkillStore``).  This in-memory
    implementation provides the API contract and is used for tests and
    ephemeral deployments.
    """

    def __init__(self) -> None:
        self._skills: dict[str, RegisteredSkill] = {}

    def validate_manifest(self, manifest: SkillManifest) -> list[str]:
        """Validate a skill manifest. Returns list of errors."""
        errors: list[str] = []
        if not manifest.name:
            errors.append("name is required")
        if not manifest.version:
            errors.append("version is required")
        if (
            manifest.name
            and not manifest.name.replace("-", "").replace("_", "").isalnum()
        ):
            errors.append("name must be alphanumeric (hyphens/underscores allowed)")
        return errors

    def register(
        self,
        manifest: SkillManifest,
        skill_md: str,
        scope: SkillScope = SkillScope.USER,
        owner_id: str = "",
        project_id: str | None = None,
        enabled: bool = True,
        review_state: dict[str, Any] | None = None,
    ) -> RegisteredSkill:
        """Register a skill bundle."""
        errors = self.validate_manifest(manifest)
        if errors:
            raise SkillValidationError("; ".join(errors))

        skill = RegisteredSkill(
            manifest=manifest,
            scope=scope,
            owner_id=owner_id,
            project_id=project_id,
            skill_md=skill_md,
            registered_at=datetime.utcnow(),
            enabled=enabled,
            review_state=review_state,
        )
        self._skills[skill.skill_id] = skill
        logger.info("Registered skill: %s (id=%s)", manifest.name, skill.skill_id)
        return skill

    def get(self, skill_id: str) -> RegisteredSkill | None:
        return self._skills.get(skill_id)

    def list_skills(
        self,
        scope: SkillScope | None = None,
        owner_id: str | None = None,
        tag: str | None = None,
    ) -> list[RegisteredSkill]:
        """List skills, optionally filtered."""
        results: list[RegisteredSkill] = []
        for skill in self._skills.values():
            if not skill.enabled:
                continue
            if scope and skill.scope != scope:
                continue
            if owner_id and skill.owner_id != owner_id:
                continue
            if tag and tag not in skill.manifest.tags:
                continue
            results.append(skill)
        return results

    def unregister(self, skill_id: str) -> bool:
        """Remove a skill from the registry."""
        if skill_id in self._skills:
            del self._skills[skill_id]
            return True
        return False

    def search(self, query: str, limit: int = 20) -> list[RegisteredSkill]:
        """Search skills by name/description/tags."""
        q = query.lower()
        scored: list[tuple[float, RegisteredSkill]] = []
        for skill in self._skills.values():
            if not skill.enabled:
                continue
            score = 0.0
            if q in skill.manifest.name.lower():
                score += 2.0
            if q in skill.manifest.description.lower():
                score += 1.0
            for tag in skill.manifest.tags:
                if q in tag.lower():
                    score += 0.5
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def list_pending(self) -> list[RegisteredSkill]:
        """Loop-proposed skills awaiting human review (D-27 edit plane)."""
        return [s for s in self._skills.values() if is_pending_proposal(s)]

    def decide_review(
        self,
        skill_id: str,
        decision: str,
        *,
        reviewer: str,
        note: str = "",
        test_receipt: dict[str, Any] | None = None,
    ) -> RegisteredSkill | None:
        """Merge a human review decision into a pending proposal's state.

        Approve is the merge: the only transition that enables a compiled
        proposal. Returns the updated skill, or ``None`` when the skill is
        absent or no longer pending (the caller maps that to a conflict).
        """
        skill = self._skills.get(skill_id)
        if skill is None or not is_pending_proposal(skill):
            return None
        state = dict(skill.review_state or {})
        state["status"] = decision
        state["decision"] = decision
        state["reviewer"] = reviewer
        state["decided_at"] = _now_iso()
        if note:
            state["note"] = note
        if test_receipt is not None:
            state["test_receipt"] = test_receipt
        skill.review_state = state
        if decision == "approved":
            skill.enabled = True
        return skill


__all__ = [
    "REVIEW_DECISIONS",
    "REVIEW_STATUS_PENDING",
    "RegisteredSkill",
    "SkillRegistry",
    "SkillScope",
    "SkillValidationError",
    "is_pending_proposal",
    "project_key",
    "skill_key",
]
