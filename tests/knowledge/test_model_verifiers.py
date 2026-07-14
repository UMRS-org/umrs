"""Tests for model-backed verifiers.

Covers:
- TerminologyConsistencyVerifier (LLM-based)
- UnsupportedClaimVerifier (RAG retrieve + LLM judge)
- NearDuplicateVerifier (embedding cosine similarity)
- FigureLegibilityVerifier (VLM optional)

Each verifier is tested:
- With mock LLM/VLM (happy path)
- In degraded mode (LLM/VLM unavailable → cap confidence, record skip)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ID = "test-project"
_TS = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)


# -- Fixtures --


def _artifact(
    artifact_id: str = "art-001",
    content_hash: str = "abc123def456",
    kind: str = "document",
    project_id: str = PROJECT_ID,
    mime_type: str = "text/plain",
    source_ref: str = "",
    metadata: dict | None = None,
) -> MagicMock:
    a = MagicMock()
    a.artifact_id = artifact_id
    a.content_hash = content_hash
    a.kind = kind
    a.project_id = project_id
    a.mime_type = mime_type
    a.source_ref = source_ref
    a.metadata = metadata or {}
    return a


def _mock_knowledge_store(
    artifacts: list,
    extracted: dict[str, dict] | None = None,
) -> MagicMock:
    """Build a mock KnowledgeStore with given artifacts and extracted knowledge."""
    store = AsyncMock()
    store.list_artifacts = AsyncMock(return_value=(artifacts, 0))

    extracted = extracted or {}

    def get_extracted(artifact_id):
        return extracted.get(artifact_id)

    store.get_extracted_knowledge = get_extracted
    return store


def _extracted_doc(
    artifact_id: str,
    content: str = "",
    summary: str = "",
    claims: list[str] | None = None,
    embedding: list[float] | None = None,
    caption: str = "",
) -> dict:
    """Build extracted knowledge dict for an artifact."""
    result: dict[str, Any] = {
        "content": content,
        "summary": summary,
    }
    if claims is not None:
        result["claims"] = claims
    if embedding is not None:
        result["embedding"] = embedding
    if caption:
        result["caption"] = caption
    return result


# -- LLM/ VLM mocks --


class FakeLLM:
    """Deterministic fake LLM for testing."""

    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None):
        """
        Args:
            responses: Mapping of prompt substrings -> response strings (JSON).
            default: Default response when no substring matches.
        """
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, value in self.responses.items():
            if key in prompt:
                return value
        return self.default or '{"consistent": true, "issues": [], "confidence": 1.0}'


class FakeVLM:
    """Deterministic fake VLM for testing."""

    def __init__(self, responses: dict[str, dict] | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def submit(self, image_uri: str) -> dict:
        self.calls.append(image_uri)
        return self.responses.get(image_uri, {"legible": True, "quality": "high", "issues": []})


class FakeRAG:
    """Deterministic fake RAG for testing."""

    def __init__(self, responses: dict[str, list[str]] | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def query(self, query: str) -> dict:
        self.calls.append(query)
        sources = self.responses.get(query, [])
        return {"answer": "", "sources": sources}


# -- TerminologyConsistencyVerifier tests --


class TestTerminologyConsistencyVerifier:
    """Tests for TerminologyConsistencyVerifier."""

    def test_llm_available_can_run(self):
        fake_llm = FakeLLM()
        from umrs.knowledge.verifiers.terminology_consistency import (
            TerminologyConsistencyVerifier,
        )

        v = TerminologyConsistencyVerifier(llm_client=fake_llm)
        assert v.can_run("any-project") is True

    @pytest.mark.asyncio
    async def test_consistent_documents(self):
        """When all docs use same terms → no findings."""
        from umrs.knowledge.verifiers.terminology_consistency import (
            TerminologyConsistencyVerifier,
        )

        fake_llm = FakeLLM(default='{"consistent": true, "issues": [], "confidence": 0.95}')
        v = TerminologyConsistencyVerifier(llm_client=fake_llm)

        art = _artifact("art-001", kind="document")
        store = _mock_knowledge_store(
            [art],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    content="The neural network was trained for 100 epochs.",
                    summary="Training report",
                )
            },
        )

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0
        assert len(fake_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_inconsistent_documents(self):
        """When docs use conflicting terms → findings."""
        from umrs.knowledge.verifiers.terminology_consistency import (
            TerminologyConsistencyVerifier,
        )

        fake_llm = FakeLLM(
            default=json.dumps(
                {
                    "consistent": False,
                    "issues": [
                        "Term inconsistency: model vs neural net",
                        "Term inconsistency: epoch vs iteration",
                    ],
                    "confidence": 0.9,
                }
            )
        )
        v = TerminologyConsistencyVerifier(llm_client=fake_llm)

        art1 = _artifact("art-001", kind="document")
        art2 = _artifact("art-002", kind="document")
        store = _mock_knowledge_store(
            [art1, art2],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    content="The neural network was trained for 100 epochs.",
                    summary="Training report A",
                ),
                "art-002": _extracted_doc(
                    "art-002",
                    content="The neural net converged after 100 iterations.",
                    summary="Training report B",
                ),
            },
        )

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 2
        assert all(f.rule == "terminology_inconsistency" for f in findings)

    @pytest.mark.asyncio
    async def test_no_documents(self):
        """No artifacts → no findings."""
        from umrs.knowledge.verifiers.terminology_consistency import (
            TerminologyConsistencyVerifier,
        )

        fake_llm = FakeLLM()
        v = TerminologyConsistencyVerifier(llm_client=fake_llm)
        store = _mock_knowledge_store([])

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0
        assert len(fake_llm.calls) == 0

    @pytest.mark.asyncio
    async def test_no_extraced_content(self):
        """Artifacts with no extracted knowledge → no LLM call."""
        from umrs.knowledge.verifiers.terminology_consistency import (
            TerminologyConsistencyVerifier,
        )

        fake_llm = FakeLLM()
        v = TerminologyConsistencyVerifier(llm_client=fake_llm)
        art = _artifact("art-001", kind="document")
        store = _mock_knowledge_store([art], {})  # no extracted knowledge

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0
        assert len(fake_llm.calls) == 0

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_info(self):
        """LLM error → info finding, not crash."""
        from umrs.knowledge.verifiers.terminology_consistency import (
            TerminologyConsistencyVerifier,
        )

        async def broken_llm(prompt: str) -> str:
            raise RuntimeError("LLM timeout")

        v = TerminologyConsistencyVerifier(llm_client=broken_llm)

        art = _artifact("art-001", kind="document")
        store = _mock_knowledge_store(
            [art],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    content="Some content.",
                    summary="Summary",
                )
            },
        )

        findings = await v.run(store, PROJECT_ID)
        info_findings = [f for f in findings if f.rule == "llm_error"]
        assert len(info_findings) >= 1

    @pytest.mark.asyncio
    async def test_llm_unavailable_degraded(self):
        """No LLM client provided and r4 unavailable → degraded skip."""
        from unittest.mock import patch

        from umrs.knowledge.verifiers.terminology_consistency import (
            TerminologyConsistencyVerifier,
        )

        # No llm_client → verifier probes r4.Client, which we block.
        v = TerminologyConsistencyVerifier(llm_client=None)
        with patch("r4.Client") as mock_client_cls:
            mock_client_cls.side_effect = ImportError("no r4")
            assert v.can_run(PROJECT_ID) is False

            art = _artifact("art-001", kind="document")
            store = _mock_knowledge_store([art])

            findings = await v.run(store, PROJECT_ID)
            skip = [f for f in findings if f.rule == "llm_unavailable"]
            assert len(skip) == 1
            assert "DEGRADED" in skip[0].message or "unavailable" in skip[0].message.lower()

    @pytest.mark.asyncio
    async def test_prompt_snapshot(self):
        """Capture current prompt template for regression tracking."""
        from umrs.knowledge.verifiers.terminology_consistency import (
            TerminologyConsistencyVerifier,
        )

        v = TerminologyConsistencyVerifier()
        assert "scientific terminology auditor" in v._prompt_template
        assert "consistent" in v._prompt_template
        assert "issues" in v._prompt_template
        assert "confidence" in v._prompt_template
        assert "{documents}" in v._prompt_template


# -- UnsupportedClaimVerifier tests --


class TestUnsupportedClaimVerifier:
    """Tests for UnsupportedClaimVerifier."""

    def test_llm_and_rag_available_can_run(self):
        fake_llm = FakeLLM()
        fake_rag = FakeRAG()
        from umrs.knowledge.verifiers.unsupported_claim import (
            UnsupportedClaimVerifier,
        )

        v = UnsupportedClaimVerifier(llm_client=fake_llm, rag_query=fake_rag.query)
        assert v.can_run("any-project") is True

    @pytest.mark.asyncio
    async def test_supported_claim(self):
        """Claim supported by RAG evidence → no finding."""
        from umrs.knowledge.verifiers.unsupported_claim import (
            UnsupportedClaimVerifier,
        )

        fake_llm = FakeLLM(
            default='{"supported": true, "reason": "Evidence supports this claim", "confidence": 0.9}'
        )
        fake_rag = FakeRAG(
            responses={
                "model was trained on ImageNet": [
                    {"content": "The model was trained on the ImageNet dataset."},
                    {"content": "ImageNet training data covers 1000 classes."},
                ]
            }
        )
        v = UnsupportedClaimVerifier(llm_client=fake_llm, rag_query=fake_rag.query)

        art = _artifact("art-001", kind="document")
        store = _mock_knowledge_store(
            [art],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    content="The model was trained on ImageNet.",
                    claims=["The model was trained on ImageNet."],
                )
            },
        )

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0
        assert len(fake_rag.calls) == 1
        assert len(fake_llm.calls) == 1

    @pytest.mark.asyncio
    async def test_unsupported_claim(self):
        """Claim NOT supported by RAG evidence → finding."""
        from umrs.knowledge.verifiers.unsupported_claim import (
            UnsupportedClaimVerifier,
        )

        fake_llm = FakeLLM(
            default='{"supported": false, "reason": "No evidence found in RAG results", "confidence": 0.8}'
        )
        fake_rag = FakeRAG(
            responses={
                "model achieves 99% accuracy": [
                    {"content": "The model was tested on a small dataset."},
                    {"content": "Results were 95% on test set A."},
                ]
            }
        )
        v = UnsupportedClaimVerifier(llm_client=fake_llm, rag_query=fake_rag.query)

        art = _artifact("art-001", kind="document")
        store = _mock_knowledge_store(
            [art],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    content="The model achieves 99% accuracy on all benchmarks.",
                    claims=["The model achieves 99% accuracy on all benchmarks."],
                )
            },
        )

        findings = await v.run(store, PROJECT_ID)
        unsupported = [f for f in findings if f.rule == "unsupported_claim"]
        assert len(unsupported) == 1

    @pytest.mark.asyncio
    async def test_no_artifacts(self):
        """No artifacts → no findings."""
        from umrs.knowledge.verifiers.unsupported_claim import (
            UnsupportedClaimVerifier,
        )

        fake_llm = FakeLLM()
        fake_rag = FakeRAG()
        v = UnsupportedClaimVerifier(llm_client=fake_llm, rag_query=fake_rag.query)
        store = _mock_knowledge_store([])

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0

    @pytest.mark.asyncio
    async def test_no_claims_in_artifact(self):
        """Artifact with no claims → no LLM calls."""
        from umrs.knowledge.verifiers.unsupported_claim import (
            UnsupportedClaimVerifier,
        )

        fake_llm = FakeLLM()
        fake_rag = FakeRAG()
        v = UnsupportedClaimVerifier(llm_client=fake_llm, rag_query=fake_rag.query)
        art = _artifact("art-001", kind="document")
        store = _mock_knowledge_store(
            [art],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    content="Blob.",
                )
            },
        )

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0
        assert len(fake_llm.calls) == 0

    @pytest.mark.asyncio
    async def test_llm_error_conservative(self):
        """LLM error → assume unsupported (conservative)."""
        from umrs.knowledge.verifiers.unsupported_claim import (
            UnsupportedClaimVerifier,
        )

        async def broken_llm(prompt: str) -> str:
            raise RuntimeError("LLM down")

        fake_rag = FakeRAG()
        v = UnsupportedClaimVerifier(llm_client=broken_llm, rag_query=fake_rag.query)

        art = _artifact("art-001", kind="document")
        store = _mock_knowledge_store(
            [art],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    content="This is a claim.",
                    claims=["This is a claim."],
                )
            },
        )

        findings = await v.run(store, PROJECT_ID)
        # LLM error → conservative: assume unsupported.
        unsupported = [f for f in findings if f.rule == "unsupported_claim"]
        assert len(unsupported) == 1

    @pytest.mark.asyncio
    async def test_llm_and_rag_unavailable_degraded(self):
        """Both LLM and RAG unavailable → degraded skip finding."""
        from unittest.mock import patch

        from umrs.knowledge.verifiers.unsupported_claim import (
            UnsupportedClaimVerifier,
        )

        v = UnsupportedClaimVerifier(llm_client=None, rag_query=None)
        with patch("r4.Client") as mock_client_cls:
            mock_client_cls.side_effect = ImportError("no r4")
            assert v.can_run(PROJECT_ID) is False

            art = _artifact("art-001", kind="document")
            store = _mock_knowledge_store([art])

            findings = await v.run(store, PROJECT_ID)
            skip = [f for f in findings if f.rule == "llm_unavailable"]
            assert len(skip) == 1
            assert "unavailable" in skip[0].message.lower()

    @pytest.mark.asyncio
    async def test_prompt_snapshot(self):
        """Capture current judge prompt template."""
        from umrs.knowledge.verifiers.unsupported_claim import (
            _JUDGE_PROMPT,
        )

        assert "claim verifier" in _JUDGE_PROMPT
        assert "supported" in _JUDGE_PROMPT
        assert "reason" in _JUDGE_PROMPT
        assert "confidence" in _JUDGE_PROMPT
        assert "{claim}" in _JUDGE_PROMPT
        assert "{evidence}" in _JUDGE_PROMPT


# -- NearDuplicateVerifier tests --


class TestNearDuplicateVerifier:
    """Tests for NearDuplicateVerifier."""

    def test_cosine_similarity_identity(self):
        """Same vector → cosine similarity = 1.0."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        vec = [1.0, 2.0, 3.0, 4.0]
        sim = NearDuplicateVerifier._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors → cosine similarity = 0.0."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = NearDuplicateVerifier._cosine_similarity(a, b)
        assert abs(sim) < 1e-9

    def test_cosine_similarity_opposite(self):
        """Opposite vectors → cosine similarity = -1.0."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        a = [1.0, 2.0]
        b = [-1.0, -2.0]
        sim = NearDuplicateVerifier._cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-9

    def test_cosine_similarity_mismatched_lengths(self):
        """Different length vectors → cosine similarity = 0.0."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0]
        sim = NearDuplicateVerifier._cosine_similarity(a, b)
        assert abs(sim) < 1e-9

    @pytest.mark.asyncio
    async def test_near_duplicates_detected(self):
        """Two similar embeddings → near-duplicate finding."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        v = NearDuplicateVerifier(threshold=0.95)

        # Two nearly identical vectors (0.99 similarity).
        vec1 = [1.0, 0.1, 0.05, 0.02]
        vec2 = [1.01, 0.11, 0.05, 0.02]

        art1 = _artifact("art-001", kind="document")
        art2 = _artifact("art-002", kind="document")
        store = _mock_knowledge_store(
            [art1, art2],
            {
                "art-001": _extracted_doc("art-001", embedding=vec1),
                "art-002": _extracted_doc("art-002", embedding=vec2),
            },
        )

        findings = await v.run(store, PROJECT_ID)
        near_dups = [f for f in findings if f.rule == "near_duplicate"]
        assert len(near_dups) == 1

    @pytest.mark.asyncio
    async def test_different_embeddings_not_duplicate(self):
        """Dissimilar embeddings → no finding."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        v = NearDuplicateVerifier(threshold=0.95)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        art1 = _artifact("art-001", kind="document")
        art2 = _artifact("art-002", kind="document")
        store = _mock_knowledge_store(
            [art1, art2],
            {
                "art-001": _extracted_doc("art-001", embedding=vec1),
                "art-002": _extracted_doc("art-002", embedding=vec2),
            },
        )

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0

    @pytest.mark.asyncio
    async def test_no_embeddings_degraded(self):
        """No embeddings available → degraded info finding."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        v = NearDuplicateVerifier()

        art = _artifact("art-001", kind="document")
        store = _mock_knowledge_store(
            [art],
            {"art-001": _extracted_doc("art-001")},  # no embedding
        )

        findings = await v.run(store, PROJECT_ID)
        info = [f for f in findings if f.rule == "no_embeddings"]
        assert len(info) == 1
        assert "DEGRADED" in info[0].message

    @pytest.mark.asyncio
    async def test_no_artifacts(self):
        """No artifacts → no findings."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        v = NearDuplicateVerifier()
        store = _mock_knowledge_store([])

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0

    @pytest.mark.asyncio
    async def test_property_same_embedding_is_duplicate(self):
        """Property test: same content hash → identical embedding → duplicate."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        v = NearDuplicateVerifier(threshold=0.95)

        # Same embedding (identical content).
        vec = [0.5, 0.3, 0.2, 0.1, 0.4]

        art1 = _artifact("art-001", kind="document")
        art2 = _artifact("art-002", kind="document")
        store = _mock_knowledge_store(
            [art1, art2],
            {
                "art-001": _extracted_doc("art-001", embedding=vec),
                "art-002": _extracted_doc("art-002", embedding=vec),
            },
        )

        findings = await v.run(store, PROJECT_ID)
        near_dups = [f for f in findings if f.rule == "near_duplicate"]
        assert len(near_dups) == 1
        # Verify it's the second artifact flagged as duplicate of first.
        assert near_dups[0].artifact_id == "art-002"

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        """Lower threshold → more near-duplicates detected."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        v = NearDuplicateVerifier(threshold=0.90)

        # Moderately similar vectors (sim ~0.92).
        vec1 = [1.0, 0.3, 0.1, 0.05]
        vec2 = [0.95, 0.35, 0.08, 0.06]

        art1 = _artifact("art-001", kind="document")
        art2 = _artifact("art-002", kind="document")
        store = _mock_knowledge_store(
            [art1, art2],
            {
                "art-001": _extracted_doc("art-001", embedding=vec1),
                "art-002": _extracted_doc("art-002", embedding=vec2),
            },
        )

        findings = await v.run(store, PROJECT_ID)
        near_dups = [f for f in findings if f.rule == "near_duplicate"]
        assert len(near_dups) >= 1

    @pytest.mark.asyncio
    async def test_embedding_typed_mixed(self):
        """Mix of valid float vectors and invalid/non-list extracted → skip invalid."""
        from umrs.knowledge.verifiers.near_duplicate import (
            NearDuplicateVerifier,
        )

        v = NearDuplicateVerifier()

        art1 = _artifact("art-001", kind="document")
        art2 = _artifact("art-002", kind="document")
        art3 = _artifact("art-003", kind="document")
        store = _mock_knowledge_store(
            [art1, art2, art3],
            {
                "art-001": _extracted_doc("art-001", embedding=[1.0, 0.1, 0.05]),
                "art-002": _extracted_doc("art-002", embedding=[1.01, 0.11, 0.05]),
                # art-003 has no embedding or invalid embedding
                "art-003": _extracted_doc("art-003"),
            },
        )

        findings = await v.run(store, PROJECT_ID)
        near_dups = [f for f in findings if f.rule == "near_duplicate"]
        assert len(near_dups) == 1  # only art-001/art-002 pair


# -- FigureLegibilityVerifier tests --


class TestFigureLegibilityVerifier:
    """Tests for FigureLegibilityVerifier."""

    def test_vision_client_available_can_run(self):
        fake_vision = FakeVLM()
        from umrs.knowledge.verifiers.figure_legibility import (
            FigureLegibilityVerifier,
        )

        v = FigureLegibilityVerifier(vision_client=fake_vision)
        assert v.can_run("any-project") is True

    @pytest.mark.asyncio
    async def test_vlm_unavailable_degraded(self):
        """No VLM → skip with degradation note."""
        from umrs.knowledge.verifiers.figure_legibility import (
            FigureLegibilityVerifier,
        )

        # Patch r4.vision import to simulate unavailability.
        v = FigureLegibilityVerifier(vision_client=None)
        with patch(
            "r4.vision.VisionClient",
            side_effect=ImportError("no vision"),
        ):
            assert v.can_run(PROJECT_ID) is False

            art = _artifact("art-001", kind="document")
            store = _mock_knowledge_store([art])

            findings = await v.run(store, PROJECT_ID)
            skip = [f for f in findings if f.rule == "vlm_unavailable"]
            assert len(skip) == 1
            assert "DEGRADED" in skip[0].message

    @pytest.mark.asyncio
    async def test_vlm_reports_healthy_figure(self):
        """VLM says figure is legible → no findings."""
        from umrs.knowledge.verifiers.figure_legibility import (
            FigureLegibilityVerifier,
        )

        fake_vision = FakeVLM(
            responses={
                "s3://figures/fig1.png": {
                    "legible": True,
                    "quality": "high",
                    "issues": [],
                }
            }
        )
        v = FigureLegibilityVerifier(vision_client=fake_vision)

        art = _artifact("art-001", kind="image", source_ref="s3://figures/fig1.png")
        store = _mock_knowledge_store(
            [art],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    caption="Figure 1: Training accuracy over 100 epochs.",
                )
            },
        )

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0

    @pytest.mark.asyncio
    async def test_vlm_reports_figure_issues(self):
        """VLM finds figure quality issues → findings."""
        from umrs.knowledge.verifiers.figure_legibility import (
            FigureLegibilityVerifier,
        )

        fake_vision = FakeVLM(
            responses={
                "s3://figures/fig2.png": {
                    "legible": False,
                    "quality": "low",
                    "issues": [
                        "Caption text too small to read",
                        "Axis labels are missing",
                    ],
                }
            }
        )
        v = FigureLegibilityVerifier(vision_client=fake_vision)

        art = _artifact("art-001", kind="image", source_ref="s3://figures/fig2.png")
        store = _mock_knowledge_store(
            [art],
            {
                "art-001": _extracted_doc(
                    "art-001",
                    caption="Figure 2: Some plot.",
                )
            },
        )

        findings = await v.run(store, PROJECT_ID)
        fig_issues = [f for f in findings if f.rule == "figure_issue"]
        assert len(fig_issues) == 2

    @pytest.mark.asyncio
    async def test_no_image_artifacts(self):
        """No image artifacts → no findings."""
        from umrs.knowledge.verifiers.figure_legibility import (
            FigureLegibilityVerifier,
        )

        fake_vision = FakeVLM()
        v = FigureLegibilityVerifier(vision_client=fake_vision)

        art = _artifact("art-001", kind="document")  # not an image
        store = _mock_knowledge_store([art])

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0

    @pytest.mark.asyncio
    async def test_no_artifacts(self):
        """No artifacts → no findings."""
        from umrs.knowledge.verifiers.figure_legibility import (
            FigureLegibilityVerifier,
        )

        fake_vision = FakeVLM()
        v = FigureLegibilityVerifier(vision_client=fake_vision)
        store = _mock_knowledge_store([])

        findings = await v.run(store, PROJECT_ID)
        assert len(findings) == 0


# -- Registry tests --


class TestModelVerifierRegistry:
    """Verify all model-backed verifiers are registered."""

    def test_all_model_verifiers_registered(self):
        from umrs.knowledge.verifiers import VERIFIER_REGISTRY

        assert "terminology_consistency" in VERIFIER_REGISTRY
        assert "unsupported_claim" in VERIFIER_REGISTRY
        assert "near_duplicate" in VERIFIER_REGISTRY
        assert "figure_legibility" in VERIFIER_REGISTRY

    def test_all_verifiers_total_count(self):
        """Check that all 9 verifiers are registered.

        This test re-registers verifiers deleted by TestRegistry tests
        to ensure a complete count.
        """
        from umrs.knowledge.verifiers import (
            VERIFIER_REGISTRY,
            OrphanArtifactVerifier,
            UnresolvedReferenceVerifier,
            get_all_verifiers,
        )

        # Restore verifiers that TestRegistry tests may have deleted.
        if "unresolved_reference" not in VERIFIER_REGISTRY:
            VERIFIER_REGISTRY["unresolved_reference"] = UnresolvedReferenceVerifier()
        if "orphan_artifact" not in VERIFIER_REGISTRY:
            VERIFIER_REGISTRY["orphan_artifact"] = OrphanArtifactVerifier()

        all_v = get_all_verifiers()
        names = {v.name for v in all_v}
        expected = {
            "unresolved_reference",
            "orphan_artifact",
            "duplicate_knowledge",
            "missing_metadata",
            "incomplete_context",
            "terminology_consistency",
            "unsupported_claim",
            "near_duplicate",
            "figure_legibility",
        }
        assert names == expected, f"Missing: {expected - names}, Extra: {names - expected}"

    def test_get_verifier_by_name(self):
        from umrs.knowledge.verifiers import get_verifier

        v = get_verifier("terminology_consistency")
        assert v is not None
        assert v.name == "terminology_consistency"

        v = get_verifier("near_duplicate")
        assert v is not None
        assert v.name == "near_duplicate"

        v = get_verifier("nonexistent")
        assert v is None
