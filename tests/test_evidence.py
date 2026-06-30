"""Tests for the evidence subsystem (Phase 1, Step 2.5)."""

from __future__ import annotations

import pytest

from umrs.evidence import (
    DuplicateEvidenceError,
    Evidence,
    EvidenceNotFoundError,
    EvidenceSourceType,
    EvidenceStatus,
    InMemoryEvidenceStore,
)


# ── Enum Tests ────────────────────────────────────────────────────────────


def test_evidence_source_type_enum():
    """Test EvidenceSourceType enum values."""
    assert EvidenceSourceType.UPLOADED_FILE == "uploaded_file"
    assert EvidenceSourceType.NOTEBOOK_CELL == "notebook_cell"
    assert EvidenceSourceType.SOURCE_CODE == "source_code"
    assert EvidenceSourceType.EXPERIMENT_RESULT == "experiment_result"
    assert EvidenceSourceType.EXTERNAL_PAPER == "external_paper"
    assert EvidenceSourceType.WEB_SEARCH == "web_search"
    assert EvidenceSourceType.DATASET == "dataset"
    assert EvidenceSourceType.OBSERVATION == "observation"
    assert EvidenceSourceType.CONCLUSION == "conclusion"
    assert EvidenceSourceType.REFERENCE == "reference"
    assert EvidenceSourceType.CHAT_RESPONSE == "chat_response"
    assert EvidenceSourceType.MANUAL == "manual"


def test_evidence_status_enum():
    """Test EvidenceStatus enum values."""
    assert EvidenceStatus.ACTIVE == "active"
    assert EvidenceStatus.RETRACTED == "retracted"
    assert EvidenceStatus.SUPERSEDED == "superseded"


# ── Evidence Tests ───────────────────────────────────────────────────────


def test_evidence_defaults():
    """Test Evidence defaults."""
    evd = Evidence()

    assert evd.evidence_id != ""  # Auto-generated
    assert evd.project_id == ""
    assert evd.claim_id == ""
    assert evd.source_type == EvidenceSourceType.MANUAL
    assert evd.source_id == ""
    assert evd.content == ""
    assert evd.content_uri == ""
    assert evd.relevance_score == 0.0
    assert evd.citation == ""
    assert evd.status == EvidenceStatus.ACTIVE
    assert evd.metadata == {}
    assert evd.created_at != ""
    assert evd.updated_at != ""
    assert evd.created_by == ""


def test_evidence_to_dict():
    """Test Evidence to_dict."""
    evd = Evidence(
        evidence_id="evd-001",
        project_id="proj-001",
        claim_id="claim-001",
        source_type=EvidenceSourceType.UPLOADED_FILE,
        source_id="file-001",
        content="This paper shows...",
        relevance_score=0.8,
        citation="Smith et al. (2023)",
        status=EvidenceStatus.ACTIVE,
    )

    data = evd.to_dict()

    assert data["evidence_id"] == "evd-001"
    assert data["project_id"] == "proj-001"
    assert data["claim_id"] == "claim-001"
    assert data["source_type"] == "uploaded_file"
    assert data["source_id"] == "file-001"
    assert data["content"] == "This paper shows..."
    assert data["relevance_score"] == 0.8
    assert data["citation"] == "Smith et al. (2023)"
    assert data["status"] == "active"


def test_evidence_from_dict():
    """Test Evidence from_dict."""
    data = {
        "evidence_id": "evd-001",
        "project_id": "proj-001",
        "claim_id": "claim-001",
        "source_type": "uploaded_file",
        "source_id": "file-001",
        "content": "This paper shows...",
        "content_uri": "",
        "relevance_score": 0.8,
        "citation": "Smith et al. (2023)",
        "status": "active",
        "metadata": {},
        "created_at": "2026-06-30T10:00:00Z",
        "updated_at": "2026-06-30T10:00:00Z",
        "created_by": "user-001",
    }

    evd = Evidence.from_dict(data)

    assert evd.evidence_id == "evd-001"
    assert evd.project_id == "proj-001"
    assert evd.claim_id == "claim-001"
    assert evd.source_type == EvidenceSourceType.UPLOADED_FILE
    assert evd.source_id == "file-001"
    assert evd.content == "This paper shows..."
    assert evd.relevance_score == 0.8
    assert evd.citation == "Smith et al. (2023)"
    assert evd.status == EvidenceStatus.ACTIVE
    assert evd.created_by == "user-001"


def test_evidence_round_trip():
    """Test that to_dict -> from_dict -> to_dict produces the same result."""
    evd = Evidence(
        project_id="proj-001",
        claim_id="claim-001",
        source_type=EvidenceSourceType.EXTERNAL_PAPER,
        content="Evidence content",
        relevance_score=0.9,
        citation="Smith et al. (2023)",
    )

    data1 = evd.to_dict()
    evd2 = Evidence.from_dict(data1)
    data2 = evd2.to_dict()

    assert data1 == data2


# ── InMemoryEvidenceStore Tests ──────────────────────────────────────────


def test_store_add_and_get():
    """Test adding and getting evidence."""
    store = InMemoryEvidenceStore()
    evd = Evidence(
        project_id="proj-001",
        claim_id="claim-001",
        source_type=EvidenceSourceType.UPLOADED_FILE,
        content="Evidence content",
    )
    evd_id = store.add_evidence(evd)

    assert evd_id == evd.evidence_id
    retrieved = store.get_evidence(evd_id)
    assert retrieved is not None
    assert retrieved.content == "Evidence content"


def test_store_add_duplicate_raises():
    """Test that adding duplicate evidence raises an error."""
    store = InMemoryEvidenceStore()
    evd = Evidence(evidence_id="evd-001", project_id="proj-001")
    store.add_evidence(evd)

    with pytest.raises(DuplicateEvidenceError):
        store.add_evidence(evd)


def test_store_get_nonexistent():
    """Test getting non-existent evidence returns None."""
    store = InMemoryEvidenceStore()
    assert store.get_evidence("nonexistent") is None


def test_store_list_evidence():
    """Test listing evidence with filters."""
    store = InMemoryEvidenceStore()
    store.add_evidence(Evidence(evidence_id="evd-001", project_id="proj-001", source_type=EvidenceSourceType.UPLOADED_FILE, claim_id="claim-001", relevance_score=0.8))
    store.add_evidence(Evidence(evidence_id="evd-002", project_id="proj-001", source_type=EvidenceSourceType.EXTERNAL_PAPER, claim_id="claim-001", relevance_score=0.9))
    store.add_evidence(Evidence(evidence_id="evd-003", project_id="proj-002", source_type=EvidenceSourceType.UPLOADED_FILE, claim_id="claim-002", relevance_score=0.7))

    # All active evidence
    all_evd = store.list_evidence()
    assert len(all_evd) == 3

    # Filter by project
    proj1_evd = store.list_evidence(project_id="proj-001")
    assert len(proj1_evd) == 2

    # Filter by source type
    uploaded_evd = store.list_evidence(source_type=EvidenceSourceType.UPLOADED_FILE)
    assert len(uploaded_evd) == 2

    # Filter by claim
    claim1_evd = store.list_evidence(claim_id="claim-001")
    assert len(claim1_evd) == 2

    # Combined filters
    proj1_uploaded = store.list_evidence(
        project_id="proj-001",
        source_type=EvidenceSourceType.UPLOADED_FILE,
    )
    assert len(proj1_uploaded) == 1


def test_store_list_evidence_sorted_by_relevance():
    """Test that list_evidence returns items sorted by relevance_score descending."""
    store = InMemoryEvidenceStore()
    store.add_evidence(Evidence(evidence_id="evd-001", project_id="p1", relevance_score=0.5))
    store.add_evidence(Evidence(evidence_id="evd-002", project_id="p1", relevance_score=0.9))
    store.add_evidence(Evidence(evidence_id="evd-003", project_id="p1", relevance_score=0.7))

    results = store.list_evidence(project_id="p1")

    # Most relevant first
    assert results[0].evidence_id == "evd-002"  # 0.9
    assert results[1].evidence_id == "evd-003"  # 0.7
    assert results[2].evidence_id == "evd-001"  # 0.5


def test_store_list_evidence_limit():
    """Test that limit is respected."""
    store = InMemoryEvidenceStore()
    for i in range(10):
        store.add_evidence(Evidence(evidence_id=f"evd-{i:03d}", project_id="proj-001", relevance_score=0.5))

    limited = store.list_evidence(limit=5)
    assert len(limited) == 5


def test_store_search_evidence():
    """Test searching evidence."""
    store = InMemoryEvidenceStore()
    store.add_evidence(Evidence(evidence_id="evd-001", project_id="p1", content="This is about machine learning", citation="Smith 2023"))
    store.add_evidence(Evidence(evidence_id="evd-002", project_id="p1", content="This is about quantum computing", citation="Jones 2024"))
    store.add_evidence(Evidence(evidence_id="evd-003", project_id="p2", content="Machine learning is powerful", citation="Brown 2023"))

    # Search by content
    results = store.search_evidence("machine learning")
    assert len(results) == 2

    # Search by citation
    results = store.search_evidence("Jones")
    assert len(results) == 1
    assert results[0].evidence_id == "evd-002"

    # Search with project filter
    results = store.search_evidence("machine learning", project_id="p1")
    assert len(results) == 1
    assert results[0].evidence_id == "evd-001"


def test_store_get_evidence_for_claim():
    """Test getting evidence for a specific claim."""
    store = InMemoryEvidenceStore()
    store.add_evidence(Evidence(evidence_id="evd-001", project_id="p1", claim_id="claim-001", relevance_score=0.8))
    store.add_evidence(Evidence(evidence_id="evd-002", project_id="p1", claim_id="claim-001", relevance_score=0.9))
    store.add_evidence(Evidence(evidence_id="evd-003", project_id="p1", claim_id="claim-002", relevance_score=0.7))

    claim1_evd = store.get_evidence_for_claim("claim-001")

    assert len(claim1_evd) == 2
    # Sorted by relevance descending
    assert claim1_evd[0].evidence_id == "evd-002"  # 0.9
    assert claim1_evd[1].evidence_id == "evd-001"  # 0.8


def test_store_get_evidence_for_claim_no_claim():
    """Test getting evidence for a claim with no evidence."""
    store = InMemoryEvidenceStore()
    claim_evd = store.get_evidence_for_claim("nonexistent-claim")
    assert len(claim_evd) == 0


def test_store_update_status():
    """Test updating evidence status."""
    store = InMemoryEvidenceStore()
    evd_id = store.add_evidence(Evidence(project_id="p1", content="test"))

    result = store.update_status(evd_id, EvidenceStatus.RETRACTED)
    assert result is True
    assert store.get_evidence(evd_id).status == EvidenceStatus.RETRACTED


def test_store_update_status_nonexistent():
    """Test updating status of non-existent evidence returns False."""
    store = InMemoryEvidenceStore()
    assert store.update_status("nonexistent", EvidenceStatus.RETRACTED) is False


def test_store_count():
    """Test counting evidence."""
    store = InMemoryEvidenceStore()
    store.add_evidence(Evidence(evidence_id="evd-001", project_id="p1", source_type=EvidenceSourceType.UPLOADED_FILE))
    store.add_evidence(Evidence(evidence_id="evd-002", project_id="p1", source_type=EvidenceSourceType.EXTERNAL_PAPER))
    store.add_evidence(Evidence(evidence_id="evd-003", project_id="p1", source_type=EvidenceSourceType.UPLOADED_FILE))
    store.add_evidence(Evidence(evidence_id="evd-004", project_id="p2", source_type=EvidenceSourceType.UPLOADED_FILE))

    assert store.count() == 4
    assert store.count(project_id="p1") == 3
    assert store.count(source_type=EvidenceSourceType.UPLOADED_FILE) == 3
    assert store.count(project_id="p1", source_type=EvidenceSourceType.UPLOADED_FILE) == 2


def test_store_list_excludes_retracted():
    """Test that list_evidence excludes retracted evidence by default."""
    store = InMemoryEvidenceStore()
    evd_id = store.add_evidence(Evidence(project_id="p1", content="test"))
    store.add_evidence(Evidence(evidence_id="evd-002", project_id="p1", content="test2"))

    # Retract one
    store.update_status(evd_id, EvidenceStatus.RETRACTED)

    # Should only return 1 active evidence
    active_evd = store.list_evidence(project_id="p1")
    assert len(active_evd) == 1


def test_store_get_evidence_for_claim_excludes_retracted():
    """Test that get_evidence_for_claim excludes retracted evidence."""
    store = InMemoryEvidenceStore()
    evd1_id = store.add_evidence(Evidence(evidence_id="evd-001", project_id="p1", claim_id="claim-001", relevance_score=0.8))
    evd2_id = store.add_evidence(Evidence(evidence_id="evd-002", project_id="p1", claim_id="claim-001", relevance_score=0.9))

    # Retract one
    store.update_status(evd1_id, EvidenceStatus.RETRACTED)

    claim_evd = store.get_evidence_for_claim("claim-001")
    assert len(claim_evd) == 1
    assert claim_evd[0].evidence_id == "evd-002"
