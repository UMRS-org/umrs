"""IncompleteContextVerifier — checks project-level coverage thresholds.

Verifies:
  - Minimum artifact count per project.
  - Minimum IR count per project.
  - Minimum node count in the knowledge graph.

Reported as WARNING when thresholds not met.
"""

from __future__ import annotations

import logging

from .base import Finding, Severity, VerifierBase

logger = logging.getLogger(__name__)

# Default thresholds.
_DEFAULT_MIN_ARTIFACTS = 3
_DEFAULT_MIN_IRS = 1
_DEFAULT_MIN_NODES = 0


class IncompleteContextVerifier(VerifierBase):
    """Check project coverage against configurable thresholds."""

    name = "incomplete_context"
    description = "Project-level coverage below thresholds"
    severity = Severity.WARNING

    def __init__(
        self,
        min_artifacts: int = _DEFAULT_MIN_ARTIFACTS,
        min_irs: int = _DEFAULT_MIN_IRS,
        min_nodes: int = _DEFAULT_MIN_NODES,
    ) -> None:
        """Configure thresholds.

        Args:
            min_artifacts: Minimum artifacts required.
            min_irs: Minimum IRs required.
            min_nodes: Minimum graph nodes required.
        """
        self.min_artifacts = min_artifacts
        self.min_irs = min_irs
        self.min_nodes = min_nodes

    async def run(
        self,
        knowledge_store,
        project_id: str,
    ) -> list[Finding]:
        """Count project objects and report if below thresholds."""
        findings: list[Finding] = []

        # Count artifacts
        artifacts, _ = await knowledge_store.list_artifacts(project_id)
        artifact_count = len(artifacts) if artifacts else 0

        if artifact_count < self.min_artifacts:
            findings.append(
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity=self.severity,
                    rule="low_artifacts",
                    message=(
                        f"Project {project_id} has {artifact_count}"
                        f" artifacts (minimum {self.min_artifacts})"
                    ),
                )
            )

        # Count IRs
        irs = await knowledge_store.list_irs(project_id)
        ir_count = len(irs)

        if ir_count < self.min_irs:
            findings.append(
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity=self.severity,
                    rule="low_irs",
                    message=(
                        f"Project {project_id} has {ir_count} IRs"
                        f" (minimum {self.min_irs})"
                    ),
                )
            )

        # Count nodes
        nodes = await knowledge_store.list_nodes(project_id)
        node_count = len(nodes)

        if self.min_nodes > 0 and node_count < self.min_nodes:
            findings.append(
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity=self.severity,
                    rule="low_nodes",
                    message=(
                        f"Project {project_id} has {node_count}"
                        f" nodes (minimum {self.min_nodes})"
                    ),
                )
            )

        return findings
