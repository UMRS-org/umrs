"""UnresolvedReferenceVerifier — IR references must point to existing targets.

Checks:
  - IR content references (files, functions, classes) that map to artifact_ids
    or node_ids that no longer exist in the store.
  - Edges whose source_id or target_id node does not exist.
"""

from __future__ import annotations

import logging

from .base import Finding, Severity, VerifierBase

logger = logging.getLogger(__name__)


class UnresolvedReferenceVerifier(VerifierBase):
    """Find IR references and graph edges that point to missing targets."""

    name = "unresolved_reference"
    description = "IR and edge references to missing artifacts or nodes"
    severity = Severity.BLOCKING

    async def run(
        self,
        knowledge_store,
        project_id: str,
    ) -> list[Finding]:
        """Check all IRs and edges for unresolved references."""
        findings: list[Finding] = []

        # Collect all known artifact_ids and node_ids
        known_artifacts = set()
        known_nodes = set()

        artifacts, _ = await knowledge_store.list_artifacts(project_id)
        for artifact in artifacts:
            known_artifacts.add(artifact.artifact_id)

        for node in await knowledge_store.list_nodes(project_id):
            known_nodes.add(node.node_id)

        # Check IR content references
        irs = await knowledge_store.list_irs(project_id)
        for ir in irs:
            ir_findings = self._check_ir(ir, known_artifacts, known_nodes)
            findings.extend(ir_findings)

        # Check edges for missing endpoints
        for edge in await knowledge_store.list_edges(project_id):
            ef = self._check_edge(edge, known_nodes)
            if ef:
                findings.append(ef)

        return findings

    def _check_ir(
        self,
        ir,
        known_artifacts: set,
        known_nodes: set,
    ) -> list[Finding]:
        """Check an IR's content dict for dangling references."""
        findings: list[Finding] = []
        content = ir.content if isinstance(ir.content, dict) else {}

        for ref_field in (
            "artifact_ref",
            "node_ref",
            "file_ref",
            "class_ref",
            "func_ref",
        ):
            ref_val = content.get(ref_field)
            if not ref_val:
                continue
            if isinstance(ref_val, str) and ref_val in known_artifacts:
                continue
            if isinstance(ref_val, str) and ref_val in known_nodes:
                continue
            findings.append(
                Finding(
                    verifier=self.name,
                    artifact_id=ir.ir_id,
                    severity=self.severity,
                    rule=f"unresolved_{ref_field}",
                    message=f"IR {ir.ir_id} references '{ref_val}' which does not exist",
                )
            )

        return findings

    def _check_edge(
        self,
        edge,
        known_nodes: set,
    ) -> Finding | None:
        """Check that both endpoints of an edge exist."""
        if edge.source_id not in known_nodes:
            return Finding(
                verifier=self.name,
                artifact_id=edge.edge_id,
                severity=self.severity,
                rule="edge_source_missing",
                message=f"Edge {edge.edge_id} source {edge.source_id} not found",
            )
        if edge.target_id not in known_nodes:
            return Finding(
                verifier=self.name,
                artifact_id=edge.edge_id,
                severity=self.severity,
                rule="edge_target_missing",
                message=f"Edge {edge.edge_id} target {edge.target_id} not found",
            )
        return None
