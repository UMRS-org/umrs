"""UnsupportedClaimVerifier — RAG retrieve + LLM judge.

For each claim extracted from a document, retrieves supporting docs
via RAG and uses an LLM to judge whether the claim is supported.

When degraded (RAG/LLM unavailable): cap confidence at 0.5 and record skip.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Finding, Severity, VerifierBase

if TYPE_CHECKING:
    from ..store import KnowledgeStore

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = (
    "You are a claim verifier.  Given a claim and supporting evidence, "
    "determine if the claim is supported.\n\n"
    "Claim: {claim}\n\n"
    "Supporting Evidence:\n{evidence}\n\n"
    "Return JSON:\n"
    "  - supported: boolean\n"
    "  - reason: string explanation\n"
    "  - confidence: float 0-1\n"
)

# How many top-k RAG results to gather per claim.
_RAG_TOP_K = 5


class UnsupportedClaimVerifier(VerifierBase):
    """RAG retrieve + LLM judge for unsupported claims."""

    name = "unsupported_claim"
    description = "RAG+LLM judge for unsupported claims"
    severity = Severity.WARNING

    def __init__(self, llm_client=None, rag_query=None):
        """Initialize with optional clients for testing.

        Args:
            llm_client: Optional callable(async str) -> str for LLM calls.
            rag_query: Optional callable(async str) -> list[str] for RAG.
        """
        self._llm_client = llm_client
        self._rag_query = rag_query
        self._llm_available = False
        self._rag_available = False

    def can_run(self, project_id: str) -> bool:
        """Check if RAG and LLM clients were injected.

        UMRS must remain independently workable without the r4 client
        package (import-boundary invariant), so availability is purely a
        function of the injected ``llm_client`` / ``rag_query`` callables.
        """
        available = self._llm_client is not None and self._rag_query is not None
        self._llm_available = available
        self._rag_available = available
        return available

    async def run(
        self, knowledge_store: KnowledgeStore, project_id: str
    ) -> list[Finding]:
        """Run unsupported claim check."""
        findings: list[Finding] = []

        if not self.can_run(project_id):
            findings.append(
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity=self.severity,
                    rule="llm_unavailable",
                    message="Unsupported claim check skipped: LLM/RAG unavailable",
                )
            )
            return findings

        artifacts, _ = await knowledge_store.list_artifacts(project_id)
        if not artifacts:
            return findings

        for artifact in artifacts:
            extracted = knowledge_store.get_extracted_knowledge(artifact.artifact_id)
            if not extracted:
                continue

            claims = self._extract_claims(extracted)
            for claim in claims:
                evidence = await self._retrieve_evidence(claim, project_id)
                supported = await self._judge_claim(claim, evidence)

                if not supported:
                    findings.append(
                        Finding(
                            verifier=self.name,
                            artifact_id=artifact.artifact_id,
                            severity=self.severity,
                            rule="unsupported_claim",
                            message=f"Claim may be unsupported: {claim}",
                        )
                    )

        return findings

    def _extract_claims(self, extracted: dict) -> list[str]:
        """Extract claims from extracted knowledge.

        Looks for a 'claims' key, or falls back to sentence-splitting.
        Only extracts sentences with 5+ words as potential claims.
        """
        claims = extracted.get("claims", [])
        if not claims:
            content = extracted.get("content", "")
            # Naive sentence split as fallback — only long sentences.
            claims = [
                s.strip()
                for s in content.split(".")
                if len(s.split()) >= 5  # at least 5 words
            ]
        return claims[:10]  # Limit to 10 claims per artifact.

    async def _retrieve_evidence(self, claim: str, project_id: str) -> list[str]:
        """Retrieve supporting evidence via the injected RAG callable."""
        if self._rag_query is None:
            return []
        return self._rag_query(claim)

    async def _judge_claim(self, claim: str, evidence: list[str]) -> bool:
        """Ask LLM to judge if claim is supported by evidence."""
        if self._llm_client is None:
            # No injected LLM — conservative: treat as unsupported.
            return False
        evidence_text = "\n\n".join(evidence) if evidence else "(no evidence found)"
        prompt = _JUDGE_PROMPT.format(claim=claim, evidence=evidence_text)

        try:
            raw = await self._llm_client(prompt)

            import json

            result = json.loads(raw)
            return bool(result.get("supported", False))
        except Exception:  # noqa: BLE001
            # On LLM error, assume unsupported (conservative).
            return False
