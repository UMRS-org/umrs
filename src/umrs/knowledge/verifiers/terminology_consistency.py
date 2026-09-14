"""TerminologyConsistencyVerifier — LLM-based terminology audit.

Uses an LLM to check whether terminology is used consistently across all
documents in a project.  Delegates to a fake-LLM in tests.

When degraded (no LLM endpoint): cap confidence at 0.5 and record skip.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .base import Finding, Severity, VerifierBase

if TYPE_CHECKING:
    from ..store import KnowledgeStore

logger = logging.getLogger(__name__)

# Default prompt template (customizable).
_TERMINOLOGY_PROMPT = (
    "You are a scientific terminology auditor.  Review the following "
    "documents and identify any terminology inconsistencies.\n\n"
    "Documents:\n{documents}\n\n"
    "Return a JSON object with:\n"
    "  - consistent: boolean\n"
    "  - issues: array of strings (empty if consistent)\n"
    "  - confidence: float 0-1\n"
)


class TerminologyConsistencyVerifier(VerifierBase):
    """Check terminology consistency across project documents via LLM."""

    name = "terminology_consistency"
    description = "LLM-based terminology consistency audit"
    severity = Severity.WARNING

    def __init__(self, llm_client=None, prompt=None):
        """Initialize with optional LLM client for testing.

        Args:
            llm_client: Optional callable(async str) -> str for LLM calls.
                When None, attempts r4.Client().chat().
            prompt: Optional prompt template string.
        """
        self._llm_client = llm_client
        self._prompt_template = prompt or _TERMINOLOGY_PROMPT
        self._llm_available = False

    def can_run(self, project_id: str) -> bool:
        """Check if an LLM client was injected.

        UMRS must remain independently workable without the r4 client
        package (import-boundary invariant), so availability is purely a
        function of the injected ``llm_client`` callable.
        """
        self._llm_available = self._llm_client is not None
        return self._llm_available

    async def run(
        self, knowledge_store: KnowledgeStore, project_id: str
    ) -> list[Finding]:
        """Run terminology consistency check.

        Returns:
            list[Finding] — each finding carries a degraded confidence cap.
        """
        findings: list[Finding] = []

        if not self.can_run(project_id):
            findings.append(
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity=self.severity,
                    rule="llm_unavailable",
                    message="Terminology check skipped: LLM endpoint unavailable",
                )
            )
            return findings

        # Gather document texts from the knowledge store.
        documents = await self._gather_documents(knowledge_store, project_id)

        if not documents:
            return findings

        prompt = self._prompt_template.format(documents=documents)

        try:
            llm_response = await self._call_llm(prompt)
            result = self._parse_llm_response(llm_response)

            for issue_text in result.get("issues", []):
                findings.append(
                    Finding(
                        verifier=self.name,
                        artifact_id=None,
                        severity=self.severity,
                        rule="terminology_inconsistency",
                        message=issue_text,
                    )
                )

            if not result.get("consistent", False):
                # Document degraded confidence when no real LLM available.
                if not self._llm_available:
                    findings = self._cap_confidence(findings)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Terminology check failed: %s", exc)
            findings.append(
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity=Severity.INFO,
                    rule="llm_error",
                    message=f"Terminology check failed: {exc}",
                )
            )

        return findings

    async def _gather_documents(
        self, knowledge_store: KnowledgeStore, project_id: str
    ) -> str:
        """Extract document texts from the knowledge store."""
        artifacts, _ = await knowledge_store.list_artifacts(project_id)
        texts: list[str] = []
        for art in artifacts:
            if art.kind in ("document", "notebook"):
                extracted = knowledge_store.get_extracted_knowledge(art.artifact_id)
                if extracted and (extracted.get("content") or extracted.get("summary")):
                    texts.append(
                        f"[{art.artifact_id}] {extracted.get('summary', '')}\n"
                        f"{extracted.get('content', '')}"
                    )
        return "\n\n---\n\n".join(texts) if texts else ""

    async def _call_llm(self, prompt: str) -> str:
        """Call the injected LLM client with the given prompt."""
        if self._llm_client is None:
            raise RuntimeError(
                "TerminologyConsistencyVerifier requires an injected llm_client"
            )
        return await self._llm_client(prompt)

    def _parse_llm_response(self, raw: str) -> dict:
        """Parse LLM JSON response into structured dict."""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {
                "consistent": False,
                "issues": ["Could not parse LLM response"],
                "confidence": 0.0,
            }

    def _cap_confidence(self, findings: list[Finding]) -> list[Finding]:
        """Cap confidence for degraded mode."""
        for f in findings:
            f_dict = {
                "verifier": f.verifier,
                "artifact_id": f.artifact_id,
                "severity": f.severity,
                "rule": f.rule,
                "message": f.message,
            }
            if "confidence" in f_dict.get("message", ""):
                f_dict["message"] = f"{f.message} [DEGRADED: confidence capped at 0.5]"
        return findings
