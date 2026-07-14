"""DuplicateKnowledgeVerifier — finds duplicate content by hash tier.

Detects artifacts sharing the same content_hash (exact duplicates) or
same hash prefix at configurable tiers (length-8, length-16, etc.)
indicating likely near-duplicates.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from .base import Finding, Severity, VerifierBase

logger = logging.getLogger(__name__)

# Tier prefix lengths for grouping.
_HASH_TIERS = (8, 16)


class DuplicateKnowledgeVerifier(VerifierBase):
    """Find duplicate artifacts by content hash."""

    name = "duplicate_knowledge"
    description = "Artifacts with duplicate content hashes"
    severity = Severity.WARNING

    async def run(
        self,
        knowledge_store,
        project_id: str,
    ) -> list[Finding]:
        """Find artifacts sharing content hashes (exact and prefix tiers)."""
        findings: list[Finding] = []

        artifacts, _ = await knowledge_store.list_artifacts(project_id)

        if not artifacts:
            return findings

        # Exact duplicates
        exact = self._find_exact_duplicates(artifacts)
        findings.extend(exact)

        # Exact dup IDs for prefix-tier dedup
        exact_dup_ids = {f.artifact_id for f in exact if f.rule == "exact_duplicate"}

        # Prefix-tier duplicates
        for tier_len in _HASH_TIERS:
            prefix_finds = self._find_prefix_duplicates(
                artifacts, tier_len, exact_dup_ids
            )
            findings.extend(prefix_finds)

        return findings

    def _find_exact_duplicates(self, artifacts: list) -> list[Finding]:
        """Find artifacts sharing identical content_hash."""
        findings: list[Finding] = []
        by_hash: dict[str, list] = defaultdict(list)
        for a in artifacts:
            if a.content_hash:
                by_hash[a.content_hash].append(a)

        for h, group in by_hash.items():
            if len(group) > 1:
                leader = group[0].artifact_id
                seen = set()
                for dupe in group[1:]:
                    if dupe.artifact_id not in seen:
                        seen.add(dupe.artifact_id)
                        findings.append(
                            Finding(
                                verifier=self.name,
                                artifact_id=dupe.artifact_id,
                                severity=self.severity,
                                rule="exact_duplicate",
                                message=(
                                    f"Artifact {dupe.artifact_id} is a"
                                    f" duplicate of {leader} (hash {h})"
                                ),
                            )
                        )
        return findings

    def _find_prefix_duplicates(
        self,
        artifacts: list,
        tier_len: int,
        skip_exact: set[str],
    ) -> list[Finding]:
        """Find artifacts sharing hash prefix, excluding exact dups."""
        findings: list[Finding] = []
        by_prefix: dict[str, list] = defaultdict(list)
        for a in artifacts:
            prefix = a.content_hash[:tier_len] if a.content_hash else ""
            if prefix:
                by_prefix[prefix].append(a)

        for prefix, group in by_prefix.items():
            if len(group) > 1:
                leader = group[0].artifact_id
                for dupe in group[1:]:
                    if dupe.artifact_id in skip_exact:
                        continue
                    findings.append(
                        Finding(
                            verifier=self.name,
                            artifact_id=dupe.artifact_id,
                            severity=self.severity,
                            rule=f"prefix_duplicate_{tier_len}",
                            message=(
                                f"Artifact {dupe.artifact_id} may be"
                                f" similar to {leader}"
                                f" (hash prefix {prefix})"
                            ),
                        )
                    )
        return findings
