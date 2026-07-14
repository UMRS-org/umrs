"""Tests for KnowledgeVerifier package.

Planted-defect fixtures produce exactly expected findings.
Each verifier is tested in isolation.
Registry registration/deregistration is tested.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from umrs.knowledge.store import (
    KnowledgeArtifact,
    KnowledgeEdge,
    KnowledgeIRRow,
    KnowledgeNode,
    KnowledgeStore,
)
from umrs.knowledge.verifiers import (
    VERIFIER_REGISTRY,
    Severity,
    VerifierBase,
    get_all_verifiers,
    get_verifier,
    register_verifier,
)
from umrs.knowledge.verifiers.duplicate_knowledge import (
    DuplicateKnowledgeVerifier,
)
from umrs.knowledge.verifiers.incomplete_context import (
    IncompleteContextVerifier,
)
from umrs.knowledge.verifiers.missing_metadata import (
    MissingMetadataVerifier,
)
from umrs.knowledge.verifiers.orphan_artifact import (
    OrphanArtifactVerifier,
)
from umrs.knowledge.verifiers.unresolved_reference import (
    UnresolvedReferenceVerifier,
)

PROJECT_ID = "test-project"
_TS = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)


# -- Helpers --


def _artifact(
    artifact_id: str = "art-001",
    content_hash: str = "abc123def456",
    kind: str = "document",
    project_id: str = PROJECT_ID,
    mime_type: str = "text/plain",
    metadata: dict | None = None,
) -> KnowledgeArtifact:
    return KnowledgeArtifact(
        artifact_id=artifact_id,
        content_hash=content_hash,
        kind=kind,
        project_id=project_id,
        mime_type=mime_type,
        metadata=metadata or {},
    )


def _ir(
    ir_id: str = "ir-001",
    artifact_id: str = "art-001",
    content: dict[str, Any] | None = None,
) -> KnowledgeIRRow:
    return KnowledgeIRRow(
        ir_id=ir_id,
        artifact_id=artifact_id,
        ir_type="document",
        content=content or {},
        content_hash="hash1",
    )


def _node(
    node_id: str = "node-001",
    project_id: str = PROJECT_ID,
) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=node_id,
        project_id=project_id,
        label="test",
        node_type="concept",
        content={},
        content_hash="ch1",
    )


def _edge(
    edge_id: str = "edge-001",
    project_id: str = PROJECT_ID,
    source_id: str = "node-001",
    target_id: str = "node-002",
) -> KnowledgeEdge:
    return KnowledgeEdge(
        edge_id=edge_id,
        project_id=project_id,
        source_id=source_id,
        target_id=target_id,
        relation_type="related_to",
        content={},
    )


def _make_store(
    artifacts: list | None = None,
    irs: list | None = None,
    nodes: list | None = None,
    edges: list | None = None,
) -> KnowledgeStore:
    """Build a KnowledgeStore with mocked list_* methods returning the given data."""
    artifacts = artifacts or []
    irs = irs or []
    nodes = nodes or []
    edges = edges or []

    store = MagicMock(spec=KnowledgeStore)
    store.list_artifacts = AsyncMock(return_value=(artifacts, len(artifacts)))
    store.list_irs = AsyncMock(return_value=irs)
    store.list_nodes = AsyncMock(return_value=nodes)
    store.list_edges = AsyncMock(return_value=edges)

    # Patch the method on the class so verifiers can call it as bound methods
    return store


# -- Registry tests --


class TestRegistry:
    """Test registration, lookup, and deregistration."""

    def test_register_and_lookup(self):
        VERIFIER_REGISTRY.pop("unresolved_reference", None)
        ver = UnresolvedReferenceVerifier()
        register_verifier(ver)
        assert get_verifier("unresolved_reference") is ver
        del VERIFIER_REGISTRY["unresolved_reference"]

    def test_register_duplicate_raises(self):
        VERIFIER_REGISTRY.pop("unresolved_reference", None)
        ver = UnresolvedReferenceVerifier()
        register_verifier(ver)
        with pytest.raises(KeyError, match="already registered"):
            register_verifier(ver)
        del VERIFIER_REGISTRY["unresolved_reference"]

    def test_get_all_returns_registered(self):
        VERIFIER_REGISTRY.pop("orphan_artifact", None)
        ver = OrphanArtifactVerifier()
        register_verifier(ver)
        all_v = get_all_verifiers()
        assert ver in all_v
        del VERIFIER_REGISTRY["orphan_artifact"]

    def test_get_nonexistent_returns_none(self):
        assert get_verifier("nonexistent") is None


# -- UnresolvedReferenceVerifier tests --


class TestUnresolvedReference:
    """Plant: IR references a non-existent artifact and node."""

    def test_finds_ir_reference_to_missing_artifact(self):
        artifacts = [
            _artifact("art-001", "hash1"),
        ]
        irs = [
            _ir("ir-001", "art-001", {"artifact_ref": "art-missing"}),
        ]
        nodes = [_node("node-001")]
        edges = []

        store = _make_store(artifacts=artifacts, irs=irs, nodes=nodes, edges=edges)
        ver = UnresolvedReferenceVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        assert len(findings) == 1
        assert findings[0].rule == "unresolved_artifact_ref"
        assert findings[0].severity == Severity.BLOCKING

    def test_finds_edge_with_missing_endpoint(self):
        artifacts = []
        irs = []
        nodes = [_node("node-001")]
        edges = [
            _edge("edge-001", source_id="node-001", target_id="node-gone"),
        ]

        store = _make_store(artifacts=artifacts, irs=irs, nodes=nodes, edges=edges)
        ver = UnresolvedReferenceVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        assert len(findings) == 1
        assert findings[0].rule == "edge_target_missing"

    def test_no_findings_when_all_refs_valid(self):
        artifacts = [_artifact("art-001", "hash1")]
        irs = [_ir("ir-001", "art-001", {"artifact_ref": "art-001"})]
        nodes = [_node("node-001")]
        edges = [
            _edge("edge-001", source_id="node-001", target_id="node-001"),
        ]

        store = _make_store(artifacts=artifacts, irs=irs, nodes=nodes, edges=edges)
        ver = UnresolvedReferenceVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        assert len(findings) == 0


# -- OrphanArtifactVerifier tests --


class TestOrphanArtifact:
    """Plant: one artifact with no IR pointing to it."""

    def test_finds_orphan_artifact(self):
        artifacts = [
            _artifact("art-001", "hash1"),
            _artifact("art-orphan", "hash2"),  # no IR references this
        ]
        irs = [
            _ir("ir-001", "art-001"),
        ]
        nodes = []
        edges = []

        store = _make_store(artifacts=artifacts, irs=irs, nodes=nodes, edges=edges)
        ver = OrphanArtifactVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        assert len(findings) == 1
        assert findings[0].artifact_id == "art-orphan"
        assert findings[0].rule == "no_ir"
        assert findings[0].severity == Severity.WARNING

    def test_no_findings_when_all_artifacts_have_irs(self):
        artifacts = [
            _artifact("art-001", "hash1"),
            _artifact("art-002", "hash2"),
        ]
        irs = [
            _ir("ir-001", "art-001"),
            _ir("ir-002", "art-002"),
        ]

        store = _make_store(artifacts=artifacts, irs=irs)
        ver = OrphanArtifactVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        assert len(findings) == 0


# -- DuplicateKnowledgeVerifier tests --


class TestDuplicateKnowledge:
    """Plant: two artifacts with identical content hash."""

    def test_finds_exact_duplicate(self):
        artifacts = [
            _artifact("art-001", "samehash", mime_type="text/plain"),
            _artifact("art-002", "samehash", mime_type="text/plain"),  # dup
            _artifact("art-003", "otherhash", mime_type="text/plain"),
        ]

        store = _make_store(artifacts=artifacts)
        ver = DuplicateKnowledgeVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        dup_ids = {f.artifact_id for f in findings}
        assert "art-002" in dup_ids
        assert all(f.rule == "exact_duplicate" for f in findings)
        assert len(findings) == 1

    def test_prefix_tier_detection(self):
        """Two hashes sharing the first 8 characters."""
        artifacts = [
            _artifact("art-001", "abcdef01xxxx", mime_type="text/plain"),
            _artifact("art-002", "abcdef01yyyy", mime_type="text/plain"),
        ]

        store = _make_store(artifacts=artifacts)
        ver = DuplicateKnowledgeVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        # Should find prefix duplicate at tier 8
        prefix_findings = [f for f in findings if "prefix_duplicate_8" in f.rule]
        assert len(prefix_findings) == 1

    def test_no_duplicates(self):
        artifacts = [
            _artifact("art-001", "hash1", mime_type="text/plain"),
            _artifact("art-002", "hash2", mime_type="text/plain"),
            _artifact("art-003", "hash3", mime_type="text/plain"),
        ]

        store = _make_store(artifacts=artifacts)
        ver = DuplicateKnowledgeVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        assert len(findings) == 0


# -- MissingMetadataVerifier tests --


class TestMissingMetadata:
    """Plant: artifacts without mime_type and without description."""

    def test_finds_missing_mime_type(self):
        artifacts = [
            _artifact("art-001", "hash1", kind="document", mime_type=""),
            _artifact("art-002", "hash2", kind="embedding", mime_type=""),  # exempt
        ]

        store = _make_store(artifacts=artifacts)
        ver = MissingMetadataVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        mime_findings = [f for f in findings if f.rule == "no_mime_type"]
        assert len(mime_findings) == 1
        assert mime_findings[0].artifact_id == "art-001"

    def test_finds_missing_description(self):
        artifacts = [
            _artifact("art-001", "hash1", mime_type="text/plain", metadata={}),
            _artifact("art-002", "hash2", mime_type="text/plain", metadata={}),
        ]

        store = _make_store(artifacts=artifacts)
        ver = MissingMetadataVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        desc_findings = [f for f in findings if f.rule == "no_description"]
        assert len(desc_findings) == 2

    def test_no_findings_when_metadata_complete(self):
        artifacts = [
            _artifact(
                "art-001",
                "hash1",
                mime_type="text/plain",
                metadata={"description": "A fine document"},
            ),
        ]

        store = _make_store(artifacts=artifacts)
        ver = MissingMetadataVerifier()
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        assert len(findings) == 0


# -- IncompleteContextVerifier tests --


class TestIncompleteContext:
    """Plant: project with too few artifacts and IRs."""

    def test_finds_low_artifacts(self):
        artifacts = [
            _artifact("art-001", "hash1", mime_type="text/plain"),
        ]  # only 1, below default min 3

        store = _make_store(artifacts=artifacts)
        ver = IncompleteContextVerifier(min_artifacts=3, min_irs=0, min_nodes=0)
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        low_art = [f for f in findings if f.rule == "low_artifacts"]
        assert len(low_art) == 1
        assert "1 artifacts" in low_art[0].message

    def test_finds_low_irs(self):
        artifacts = [
            _artifact("art-001", "hash1", mime_type="text/plain"),
            _artifact("art-002", "hash2", mime_type="text/plain"),
        ]
        irs = []  # zero IRs

        store = _make_store(artifacts=artifacts, irs=irs)
        ver = IncompleteContextVerifier(min_artifacts=1, min_irs=2, min_nodes=0)
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        low_ir = [f for f in findings if f.rule == "low_irs"]
        assert len(low_ir) == 1

    def test_no_findings_when_above_thresholds(self):
        artifacts = [
            _artifact(f"art-{i}", f"hash{i}", mime_type="text/plain")
            for i in range(5)
        ]
        irs = [
            _ir(f"ir-{i}", f"art-{i}") for i in range(3)
        ]
        nodes = [
            _node(f"node-{i}") for i in range(4)
        ]

        store = _make_store(artifacts=artifacts, irs=irs, nodes=nodes)
        ver = IncompleteContextVerifier(min_artifacts=3, min_irs=2, min_nodes=2)
        findings = asyncio.get_event_loop().run_until_complete(
            ver.run(store, PROJECT_ID)
        )

        assert len(findings) == 0


# -- VerifierBase tests --


class TestVerifierBase:
    """Test base class properties and default behavior."""

    def test_verifier_repres(self):
        ver = UnresolvedReferenceVerifier()
        assert "unresolved_reference" in repr(ver)

    def test_can_run_default_true(self):
        ver = OrphanArtifactVerifier()
        assert ver.can_run("any-project") is True

    def test_can_run_can_be_overridden(self):
        class ReadyOnlyVerifier(VerifierBase):
            name = "ready_only"
            description = "only runs when ready"

            def can_run(self, project_id: str) -> bool:
                return project_id.startswith("prod-")

            async def run(self, *_args, **_kwargs):
                return []

        ver = ReadyOnlyVerifier()
        assert not ver.can_run("dev-project")
        assert ver.can_run("prod-project")


# -- can_run tests per verifier --


class TestCanRun:
    """Verify each verifier's can_run default."""

    def test_unresolved_can_run(self):
        ver = UnresolvedReferenceVerifier()
        assert ver.can_run("x") is True

    def test_orphan_can_run(self):
        ver = OrphanArtifactVerifier()
        assert ver.can_run("x") is True

    def test_duplicate_can_run(self):
        ver = DuplicateKnowledgeVerifier()
        assert ver.can_run("x") is True

    def test_missing_metadata_can_run(self):
        ver = MissingMetadataVerifier()
        assert ver.can_run("x") is True

    def test_incomplete_can_run(self):
        ver = IncompleteContextVerifier()
        assert ver.can_run("x") is True
