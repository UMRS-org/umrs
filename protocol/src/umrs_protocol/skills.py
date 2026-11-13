"""Skill manifest schema (client-visible).

Mirrors `umrs.skills.SkillManifest` and `SkillScope`. Server-side
registry, validation policy, and persistence remain in UMRS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["SkillScope", "SkillManifest"]


class SkillScope(Enum):
    SYSTEM = "system"
    PROJECT = "project"
    USER = "user"


@dataclass
class SkillManifest:
    """Parsed `config.json` for a skill bundle."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    capabilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    mcp_server: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillManifest:
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            capabilities=data.get("capabilities", []),
            dependencies=data.get("dependencies", []),
            mcp_server=data.get("mcp_server"),
            tags=data.get("tags", []),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }
        if self.mcp_server:
            d["mcp_server"] = self.mcp_server
        return d
