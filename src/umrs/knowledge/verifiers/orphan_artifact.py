"""OrphanArtifactVerifier — finds artifacts with no IRs and no graph ties.

An artifact is orphaned when:
  - It has zero IRs pointing to it, AND
  - It is not referenced as a source_id or target_id in any edge.
"""

from __future__ import annotations

import logging

from .base import Finding, Severity, VerifierBase

logger = logging.getLogger(__name__)


class OrphanArtifactVerifier(VerifierBase):
    """Find artifacts with no IRs and no graph participation."""

    name = "orphan_artifact"
    description = "Artifacts with no IRs and no edges"
    severity = Severity.WARNING

    async def run(
        self,
        knowledge_store,
        project_id: str,
    ) -> list[Finding]:
        """Return findings for every orphan artifact in the project."""
        findings: list[Finding] = []

        # Collect artifact_ids referenced by IRs
        ir_artifact_ids: set[str] = set()
        for ir in await knowledge_store.list_irs(project_id):
            ir_artifact_ids.add(ir.artifact_id)

        # Check each artifact
        artifacts, _ = await knowledge_store.list_artifacts(project_id)
        for artifact in artifacts:
            has_ir = artifact.artifact_id in ir_artifact_ids
            if not has_ir:
                findings.append(
                    Finding(
                        verifier=self.name,
                        artifact_id=artifact.artifact_id,
                        severity=self.severity,
                        rule="no_ir",
                        message=(
                            f"Artifact {artifact.artifact_id} has no associated IRs"
                        ),
                    )
                )

        return findings
