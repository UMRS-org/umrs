"""Base class for knowledge verifiers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..store import KnowledgeStore

# Severity levels, most-to-least impactful.
BLOCKING = "blocking"
WARNING = "warning"
INFO = "info"


class Severity(str, Enum):
    """Severity levels for verifier findings."""

    BLOCKING = "blocking"
    WARNING = "warning"
    INFO = "info"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Finding:
    """Single finding reported by a verifier."""

    verifier: str
    artifact_id: str | None  # None if finding is project-scoped
    severity: str  # blocking | warning | info
    rule: str  # machine-readable rule ID
    message: str  # human-readable description


class VerifierBase:
    """Abstract base for all knowledge verifiers."""

    name: str = "base"
    description: str = ""
    severity: str = WARNING

    def __repr__(self) -> str:
        return f"<Verifier {self.name}: {self.description}>"

    async def run(
        self,
        knowledge_store: KnowledgeStore,
        project_id: str,
    ) -> list[Finding]:
        """Run verifier and return list of findings.

        Subclasses MUST implement.
        """
        raise NotImplementedError

    def can_run(self, project_id: str) -> bool:
        """Return True if this verifier can run on the given project.

        Default: always runnable. Override to skip (e.g. project not ready).
        """
        return True
