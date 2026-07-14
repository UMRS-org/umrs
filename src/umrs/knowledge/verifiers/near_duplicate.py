"""NearDuplicateVerifier — embedding cosine similarity.

Compares artifacts via cosine similarity of their embeddings from the
knowledge store.  Configurable threshold (default 0.95).

No LLM needed — uses vector store for similarity comparison.
When degraded (no embeddings): cap confidence at 0.5 and record skip.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .base import Finding, Severity, VerifierBase

if TYPE_CHECKING:
    from ..store import KnowledgeStore

logger = logging.getLogger(__name__)

# Default cosine similarity threshold for near-duplicates.
_DEFAULT_THRESHOLD = 0.95


class NearDuplicateVerifier(VerifierBase):
    """Find near-duplicate artifacts via embedding similarity."""

    name = "near_duplicate"
    description = "Near-duplicate detection via embedding similarity"
    severity = Severity.WARNING

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD):
        """Initialize near-duplicate verifier.

        Args:
            threshold: Cosine similarity threshold (0.0–1.0).
                Default 0.95.  Higher = stricter.
        """
        self.threshold = max(0.0, min(1.0, threshold))

    def can_run(self, project_id: str) -> bool:
        """Check if embeddings are available."""
        # We'll know after attempting to list embeddings.
        return True  # always attempt; degrade if no embeddings found.

    async def run(self, knowledge_store: "KnowledgeStore", project_id: str) -> list[Finding]:
        """Run near-duplicate detection."""
        findings: list[Finding] = []

        artifacts, _ = await knowledge_store.list_artifacts(project_id)
        if not artifacts:
            return findings

        # Gather embeddings.
        embeddings_map = await self._collect_embeddings(knowledge_store, project_id, artifacts)

        if not embeddings_map:
            # No embeddings — degrade gracefully.
            findings.append(
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity=Severity.INFO,
                    rule="no_embeddings",
                    message="Near-duplicate check skipped: no embeddings available [DEGRADED]",
                )
            )
            return findings

        # Compare all pairs via cosine similarity.
        artifact_ids = list(embeddings_map.keys())
        seen_pairs: set[tuple[str, str]] = set()

        for i, id_a in enumerate(artifact_ids):
            for j, id_b in enumerate(artifact_ids):
                if j <= i:
                    continue
                pair = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                sim = self._cosine_similarity(
                    embeddings_map[id_a], embeddings_map[id_b]
                )

                if sim >= self.threshold:
                    findings.append(
                        Finding(
                            verifier=self.name,
                            artifact_id=pair[1],
                            severity=self.severity,
                            rule="near_duplicate",
                            message=(
                                f"Artifact {pair[1]} is near-duplicate of "
                                f"{pair[0]} (cosine similarity {sim:.4f}, "
                                f"threshold {self.threshold})"
                            ),
                        )
                    )

        return findings

    async def _collect_embeddings(
        self,
        knowledge_store: "KnowledgeStore",
        project_id: str,
        artifacts,
    ) -> dict[str, list[float]]:
        """Collect vector embeddings for all artifacts."""
        embeddings_map: dict[str, list[float]] = {}
        for art in artifacts:
            extracted = knowledge_store.get_extracted_knowledge(art.artifact_id)
            if extracted and "embedding" in extracted:
                vec = extracted["embedding"]
                if isinstance(vec, list) and len(vec) > 0:
                    try:
                        vec = [float(v) for v in vec]
                        embeddings_map[art.artifact_id] = vec
                    except (TypeError, ValueError):
                        pass
        return embeddings_map

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
