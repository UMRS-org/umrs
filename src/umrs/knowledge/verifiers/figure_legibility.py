"""FigureLegibilityVerifier — VLM-assisted figure quality audit.

Uses an optional VLM to assess figure captions and legibility.
When VLM unavailable: skip with degradation note (confidence cap 0.5).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Finding, Severity, VerifierBase

if TYPE_CHECKING:
    from ..store import KnowledgeStore

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Analyze this figure and assess whether the caption is legible, "
    "descriptive, and matches the visual content.\n\n"
    "Return JSON:\n"
    "  - legible: boolean\n"
    "  - quality: low|medium|high\n"
    "  - issues: array of strings\n"
    "  - confidence: float 0-1\n"
)


class FigureLegibilityVerifier(VerifierBase):
    """Optional VLM-based figure quality check."""

    name = "figure_legibility"
    description = "VLM-assisted figure caption quality check"
    severity = Severity.INFO

    def __init__(self, vision_client=None):
        """Initialize with optional VLM client for testing.

        Args:
            vision_client: Optional VisionClient instance (from r4).
        """
        self._vision_client = vision_client
        self._vlm_available = vision_client is not None

    def can_run(self, project_id: str) -> bool:
        """Check if VLM endpoint is available.

        Returns True only when a VLM client was explicitly provided.
        Auto-probing r4.vision import is unreliable in tests.
        """
        if self._vision_client is not None:
            self._vlm_available = True
            return True
        return False

    async def run(self, knowledge_store: "KnowledgeStore", project_id: str) -> list[Finding]:
        """Run figure legibility check on image artifacts."""
        if not self.can_run(project_id):
            return [self._degraded_finding()]

        images = await self._gather_images(knowledge_store, project_id)
        if not images:
            return []

        findings: list[Finding] = []
        for image in images:
            extracted = knowledge_store.get_extracted_knowledge(image.artifact_id)
            result = await self._assess_image(image, extracted)
            if result and not result.get("legible", True):
                for issue in result.get("issues", []):
                    findings.append(
                        Finding(
                            verifier=self.name,
                            artifact_id=image.artifact_id,
                            severity=self.severity,
                            rule="figure_issue",
                            message=f"Figure quality issue: {issue}",
                        )
                    )
        return findings

    def _degraded_finding(self) -> Finding:
        """Return degraded skip finding."""
        return Finding(
            verifier=self.name,
            artifact_id=None,
            severity=Severity.INFO,
            rule="vlm_unavailable",
            message="Figure legibility check skipped: VLM unavailable [DEGRADED]",
        )

    async def _gather_images(self, knowledge_store: "KnowledgeStore", project_id: str) -> list:
        """Gather image artifacts from knowledge store."""
        artifacts, _ = await knowledge_store.list_artifacts(project_id)
        return [a for a in artifacts if a.kind in ("image", "figure", "diagram")]

    async def _assess_image(
        self,
        artifact,
        extracted: dict | None,
    ) -> dict | None:
        """Assess a single image via VLM."""
        if self._vision_client is not None:
            return await self._vlm_assess(artifact, extracted)
        # Fallback: no vision client available for VLM calls.
        return None

    async def _vlm_assess(self, artifact, extracted: dict | None) -> dict | None:
        """Use VisionClient to assess an image artifact."""
        try:
            # VisionClient.submit expects an image URI or path.
            # We rely on stored artifact_uri from knowledge store.
            uri = artifact.source_ref if hasattr(artifact, "source_ref") else ""
            if not uri:
                return None

            result = self._vision_client.submit(uri)
            # Return parsed result.
            if isinstance(result, dict):
                return {
                    "legible": result.get("legible", True),
                    "quality": result.get("quality", "medium"),
                    "issues": result.get("issues", []),
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM assessment failed for %s: %s", artifact.artifact_id, exc)
        return None
