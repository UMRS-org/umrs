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

from .duplicate_knowledge import (  # noqa: F401
    DuplicateKnowledgeVerifier,
)
from .incomplete_context import (  # noqa: F401
    IncompleteContextVerifier,
)
from .missing_metadata import (  # noqa: F401
    MissingMetadataVerifier,
)
from .orphan_artifact import (  # noqa: F401
    OrphanArtifactVerifier,
)
from .unresolved_reference import (  # noqa: F401
    UnresolvedReferenceVerifier,
)

# -- Model-backed verifiers --

from .figure_legibility import FigureLegibilityVerifier  # noqa: F401
from .near_duplicate import NearDuplicateVerifier  # noqa: F401
from .terminology_consistency import TerminologyConsistencyVerifier  # noqa: F401
from .unsupported_claim import UnsupportedClaimVerifier  # noqa: F401

# -- MCP adapter verifiers --

from .mcp_adapter import (  # noqa: F401
    MCPVerifier,
    MCPVerifierConfig,
    register_mcp_verifier,
)

# -- Registry --

VERIFIER_REGISTRY: dict[str, "VerifierBase"] = {}


def register_verifier(verifier: VerifierBase) -> None:
    """Register a verifier instance in the global registry."""
    if verifier.name in VERIFIER_REGISTRY:
        raise KeyError(f"Verifier already registered: {verifier.name}")
    VERIFIER_REGISTRY[verifier.name] = verifier


def get_verifier(name: str) -> Optional[VerifierBase]:
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
    "Finding",
    "Severity",
    "VERIFIER_REGISTRY",
    "VerifierBase",
    "get_all_verifiers",
    "get_verifier",
    "register_verifier",
    "DuplicateKnowledgeVerifier",
    "IncompleteContextVerifier",
    "MissingMetadataVerifier",
    "OrphanArtifactVerifier",
    "UnresolvedReferenceVerifier",
    "FigureLegibilityVerifier",
    "NearDuplicateVerifier",
    "TerminologyConsistencyVerifier",
    "UnsupportedClaimVerifier",
    "MCPVerifier",
    "MCPVerifierConfig",
    "register_mcp_verifier",
]
