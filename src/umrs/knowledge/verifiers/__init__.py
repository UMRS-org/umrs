"""KnowledgeVerifier registry package.

Cheap static analysers over the knowledge store:
- unresolved_reference: IR references → existing artifacts/nodes/edges
- orphan_artifact: artifacts with no IRs and no graph participation
- duplicate_knowledge: content-duplicate artifacts by hash tier
- missing_metadata: artifacts without mime_type or description
- incomplete_context: project coverage thresholds

Model-backed verifiers (degrade gracefully when LLM/VLM unavailable):
- terminology_consistency: LLM-based terminology consistency audit
- unsupported_claim: RAG retrieve + LLM judge for unsupported claims
- near_duplicate: embedding cosine similarity for near-duplicates
- figure_legibility: optional VLM-assisted figure quality check

MCP-based verifiers (runtime tool-call path):
- mcp_adapter: wraps custom MCP tool as verifier (KNOWLEDGE_VERIFY)

No umrs.api or umrs.server imports.
"""

from __future__ import annotations

from typing import Optional

from .base import Finding, Severity, VerifierBase

# -- Cheap static verifiers --
from .duplicate_knowledge import (
    DuplicateKnowledgeVerifier,
)

# -- Model-backed verifiers --
from .figure_legibility import FigureLegibilityVerifier
from .incomplete_context import (
    IncompleteContextVerifier,
)

# -- MCP adapter verifiers --
from .mcp_adapter import (
    MCPVerifier,
    MCPVerifierConfig,
    register_mcp_verifier,
)
from .missing_metadata import (
    MissingMetadataVerifier,
)
from .near_duplicate import NearDuplicateVerifier
from .orphan_artifact import (
    OrphanArtifactVerifier,
)
from .terminology_consistency import TerminologyConsistencyVerifier
from .unresolved_reference import (
    UnresolvedReferenceVerifier,
)
from .unsupported_claim import UnsupportedClaimVerifier

# -- Registry --

VERIFIER_REGISTRY: dict[str, VerifierBase] = {}


def register_verifier(verifier: VerifierBase) -> None:
    """Register a verifier instance in the global registry."""
    if verifier.name in VERIFIER_REGISTRY:
        raise KeyError(f"Verifier already registered: {verifier.name}")
    VERIFIER_REGISTRY[verifier.name] = verifier


def get_verifier(name: str) -> VerifierBase | None:
    """Return a registered verifier by name, or None."""
    return VERIFIER_REGISTRY.get(name)


def get_all_verifiers() -> list[VerifierBase]:
    """Return a list of all registered verifiers."""
    return list(VERIFIER_REGISTRY.values())


def _register_all() -> None:
    """Auto-register all verifiers defined in this package."""
    for _mod in (
        DuplicateKnowledgeVerifier,
        IncompleteContextVerifier,
        MissingMetadataVerifier,
        OrphanArtifactVerifier,
        UnresolvedReferenceVerifier,
        FigureLegibilityVerifier,
        NearDuplicateVerifier,
        TerminologyConsistencyVerifier,
        UnsupportedClaimVerifier,
        # MCPVerifier is registered manually via register_mcp_verifier()
        # because it needs a config (tool_name, config_path) per instance.
    ):
        try:
            register_verifier(_mod())
        except KeyError:
            pass  # already registered (idempotent)


_register_all()


__all__ = [
    "VERIFIER_REGISTRY",
    "DuplicateKnowledgeVerifier",
    "FigureLegibilityVerifier",
    "Finding",
    "IncompleteContextVerifier",
    "MCPVerifier",
    "MCPVerifierConfig",
    "MissingMetadataVerifier",
    "NearDuplicateVerifier",
    "OrphanArtifactVerifier",
    "Severity",
    "TerminologyConsistencyVerifier",
    "UnresolvedReferenceVerifier",
    "UnsupportedClaimVerifier",
    "VerifierBase",
    "get_all_verifiers",
    "get_verifier",
    "register_mcp_verifier",
    "register_verifier",
]
