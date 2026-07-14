"""MissingMetadataVerifier — finds artifacts without required metadata.

Checks:
  - Artifacts missing mime_type (kind is not 'embedding').
  - Artifacts missing description in metadata dict.
"""

from __future__ import annotations

import logging

from .base import Finding, Severity, VerifierBase

logger = logging.getLogger(__name__)

# Kinds where mime_type is naturally empty/optional.
MIME_EXEMPT_KINDS = {"embedding", "graph", "index"}


class MissingMetadataVerifier(VerifierBase):
    """Find artifacts lacking mime_type or description metadata."""

    name = "missing_metadata"
    description = "Artifacts without mime_type or description"
    severity = Severity.INFO

    async def run(
        self,
        knowledge_store,
        project_id: str,
    ) -> list[Finding]:
        """Check every artifact for missing metadata fields."""
        findings: list[Finding] = []

        artifacts, _ = await knowledge_store.list_artifacts(project_id)
        for artifact in artifacts:
            # Check mime_type
            if artifact.kind not in MIME_EXEMPT_KINDS and not artifact.mime_type:
                findings.append(
                    Finding(
                        verifier=self.name,
                        artifact_id=artifact.artifact_id,
                        severity=self.severity,
                        rule="no_mime_type",
                        message=(
                            f"Artifact {artifact.artifact_id} (kind={artifact.kind})"
                            f" has no mime_type"
                        ),
                    )
                )

            # Check description in metadata
            metadata = artifact.metadata if artifact.metadata else {}
            if not metadata.get("description"):
                findings.append(
                    Finding(
                        verifier=self.name,
                        artifact_id=artifact.artifact_id,
                        severity=self.severity,
                        rule="no_description",
                        message=(
                            f"Artifact {artifact.artifact_id} has no"
                            f" description in metadata"
                        ),
                    )
                )

        return findings
