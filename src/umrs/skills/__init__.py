"""UMRS Skill Registry.

A skill is a bundle of:
- SKILL.md    — human-readable description and instructions
- mcp.json    — optional MCP server manifest
- config.json — metadata (name, version, capabilities, dependencies)

Skills are registered in UMRS so they can be discovered and loaded
by any client (r4, OpenCode, n8n, etc.). UMRS stores skill metadata
and validates skill bundles at registration time.

No dependency on r4.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from r4_contracts.skills import SkillManifest, SkillScope

logger = logging.getLogger(__name__)


@dataclass
class RegisteredSkill:
    """A skill stored in the UMRS registry."""

    skill_id: str = ""
    manifest: SkillManifest = field(default_factory=lambda: SkillManifest(name=""))
    scope: SkillScope = SkillScope.USER
    owner_id: str = ""
    project_id: Optional[str] = None
    skill_md: str = ""  # Content of SKILL.md
    registered_at: Optional[datetime] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.skill_id and self.manifest.name:
            self.skill_id = hashlib.sha256(
                f"{self.scope.value}:{self.owner_id}:{self.manifest.name}".encode()
            ).hexdigest()[:16]


class SkillValidationError(Exception):
    """Raised when a skill bundle fails validation."""


class SkillRegistry:
    """Registry for skill bundles.

    In production, backed by Postgres. This in-memory implementation
    provides the API contract.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, RegisteredSkill] = {}

    def validate_manifest(self, manifest: SkillManifest) -> List[str]:
        """Validate a skill manifest. Returns list of errors."""
        errors: List[str] = []
        if not manifest.name:
            errors.append("name is required")
        if not manifest.version:
            errors.append("version is required")
        if manifest.name and not manifest.name.replace("-", "").replace("_", "").isalnum():
            errors.append("name must be alphanumeric (hyphens/underscores allowed)")
        return errors

    def register(
        self,
        manifest: SkillManifest,
        skill_md: str,
        scope: SkillScope = SkillScope.USER,
        owner_id: str = "",
        project_id: Optional[str] = None,
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
        )
        self._skills[skill.skill_id] = skill
        logger.info("Registered skill: %s (id=%s)", manifest.name, skill.skill_id)
        return skill

    def get(self, skill_id: str) -> Optional[RegisteredSkill]:
        return self._skills.get(skill_id)

    def list_skills(
        self,
        scope: Optional[SkillScope] = None,
        owner_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[RegisteredSkill]:
        """List skills, optionally filtered."""
        results: List[RegisteredSkill] = []
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

    def search(self, query: str, limit: int = 20) -> List[RegisteredSkill]:
        """Search skills by name/description/tags."""
        q = query.lower()
        scored: List[tuple[float, RegisteredSkill]] = []
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
