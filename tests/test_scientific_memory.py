"""Tests for the scientific_memory subsystem (Phase 1, Step 2.1)."""

from __future__ import annotations

import pytest
from umrs.scientific_memory import (
    DuplicateMemoryEntryError,
    InMemoryScientificMemoryStore,
    MemoryEntry,
    MemoryEntryStatus,
    MemoryEntryType,
)

# ── Enum Tests ────────────────────────────────────────────────────────────


def test_memory_entry_type_enum():
    """Test MemoryEntryType enum values."""
    assert MemoryEntryType.GENERATED_CODE == "generated_code"
    assert MemoryEntryType.EXPERIMENT_LOG == "experiment_log"
    assert MemoryEntryType.PLOT_FIGURE == "plot_figure"
    assert MemoryEntryType.ANALYSIS_RESULT == "analysis_result"
    assert MemoryEntryType.NEW_HYPOTHESIS == "new_hypothesis"
    assert MemoryEntryType.REVIEW_COMMENT == "review_comment"
    assert MemoryEntryType.ACCEPTED_FINDING == "accepted_finding"
    assert MemoryEntryType.FAILED_EXPERIMENT == "failed_experiment"
    assert MemoryEntryType.OBSERVATION == "observation"
    assert MemoryEntryType.CONCLUSION == "conclusion"
    assert MemoryEntryType.QUESTION == "question"
    assert MemoryEntryType.ASSUMPTION == "assumption"
    assert MemoryEntryType.REFERENCE == "reference"
    assert MemoryEntryType.NOTE == "note"


def test_memory_entry_status_enum():
    """Test MemoryEntryStatus enum values."""
    assert MemoryEntryStatus.ACTIVE == "active"
    assert MemoryEntryStatus.ARCHIVED == "archived"
    assert MemoryEntryStatus.DEPRECATED == "deprecated"
    assert MemoryEntryStatus.RETRACTED == "retracted"


# ── MemoryEntry Tests ─────────────────────────────────────────────────────


def test_memory_entry_defaults():
    """Test MemoryEntry defaults."""
    entry = MemoryEntry()

    assert entry.entry_id != ""  # Auto-generated
    assert entry.project_id == ""
    assert entry.experiment_id == ""
    assert entry.entry_type == MemoryEntryType.NOTE
    assert entry.source == "system"
    assert entry.content == ""
    assert entry.content_uri == ""
    assert entry.summary == ""
    assert entry.metadata == {}
    assert entry.tags == []
    assert entry.status == MemoryEntryStatus.ACTIVE
    assert entry.confidence == 0.0
    assert entry.references == []
    assert entry.created_at != ""  # Auto-generated
    assert entry.updated_at != ""  # Auto-generated
    assert entry.created_by == ""


def test_memory_entry_to_dict():
    """Test MemoryEntry to_dict."""
    entry = MemoryEntry(
        entry_id="mem-001",
        project_id="proj-001",
        entry_type=MemoryEntryType.GENERATED_CODE,
        source="agent",
        content="print('hello')",
        summary="Generated Python code",
        tags=["python", "generated"],
        confidence=0.8,
    )

    data = entry.to_dict()

    assert data["entry_id"] == "mem-001"
    assert data["project_id"] == "proj-001"
    assert data["entry_type"] == "generated_code"
    assert data["source"] == "agent"
    assert data["content"] == "print('hello')"
    assert data["summary"] == "Generated Python code"
    assert data["tags"] == ["python", "generated"]
    assert data["confidence"] == 0.8
    assert data["status"] == "active"


def test_memory_entry_from_dict():
    """Test MemoryEntry from_dict."""
    data = {
        "entry_id": "mem-001",
        "project_id": "proj-001",
        "experiment_id": "exp-001",
        "entry_type": "experiment_log",
        "source": "system",
        "content": "Starting experiment...\nDone.",
        "summary": "Experiment log",
        "metadata": {"key": "value"},
        "tags": ["experiment", "log"],
        "status": "active",
        "confidence": 1.0,
        "references": ["ref-001"],
        "created_at": "2026-06-30T10:00:00Z",
        "updated_at": "2026-06-30T10:00:00Z",
        "created_by": "user-001",
    }

    entry = MemoryEntry.from_dict(data)

    assert entry.entry_id == "mem-001"
    assert entry.project_id == "proj-001"
    assert entry.experiment_id == "exp-001"
    assert entry.entry_type == MemoryEntryType.EXPERIMENT_LOG
    assert entry.source == "system"
    assert entry.content == "Starting experiment...\nDone."
    assert entry.summary == "Experiment log"
    assert entry.metadata == {"key": "value"}
    assert entry.tags == ["experiment", "log"]
    assert entry.status == MemoryEntryStatus.ACTIVE
    assert entry.confidence == 1.0
    assert entry.references == ["ref-001"]
    assert entry.created_by == "user-001"


def test_memory_entry_round_trip():
    """Test that to_dict -> from_dict -> to_dict produces the same result."""
    entry = MemoryEntry(
        project_id="proj-001",
        entry_type=MemoryEntryType.NEW_HYPOTHESIS,
        content="This is a hypothesis",
        summary="A hypothesis",
        tags=["hypothesis"],
        confidence=0.6,
    )

    data1 = entry.to_dict()
    entry2 = MemoryEntry.from_dict(data1)
    data2 = entry2.to_dict()

    assert data1 == data2


# ── InMemoryScientificMemoryStore Tests ────────────────────────────────────


def test_store_add_and_get():
    """Test adding and getting entries."""
    store = InMemoryScientificMemoryStore()
    entry = MemoryEntry(
        project_id="proj-001",
        content="Test content",
    )
    entry_id = store.add(entry)

    assert entry_id == entry.entry_id
    retrieved = store.get(entry_id)
    assert retrieved is not None
    assert retrieved.content == "Test content"


def test_store_add_duplicate_raises():
    """Test that adding a duplicate entry raises an error."""
    store = InMemoryScientificMemoryStore()
    entry = MemoryEntry(entry_id="mem-001", project_id="proj-001")
    store.add(entry)

    with pytest.raises(DuplicateMemoryEntryError):
        store.add(entry)


def test_store_get_nonexistent():
    """Test getting a non-existent entry returns None."""
    store = InMemoryScientificMemoryStore()
    assert store.get("nonexistent") is None


def test_store_list_entries():
    """Test listing entries with filters."""
    store = InMemoryScientificMemoryStore()
    store.add(
        MemoryEntry(
            entry_id="code-001",
            project_id="proj-001",
            entry_type=MemoryEntryType.GENERATED_CODE,
            content="code1",
        )
    )
    store.add(
        MemoryEntry(
            entry_id="code-002",
            project_id="proj-001",
            entry_type=MemoryEntryType.GENERATED_CODE,
            content="code2",
        )
    )
    store.add(
        MemoryEntry(
            entry_id="log-001",
            project_id="proj-001",
            entry_type=MemoryEntryType.EXPERIMENT_LOG,
            content="log1",
        )
    )
    store.add(
        MemoryEntry(
            entry_id="code-003",
            project_id="proj-002",
            entry_type=MemoryEntryType.GENERATED_CODE,
            content="code3",
        )
    )

    # All entries
    all_entries = store.list_entries()
    assert len(all_entries) == 4

    # Filter by project
    proj1_entries = store.list_entries(project_id="proj-001")
    assert len(proj1_entries) == 3

    # Filter by type
    code_entries = store.list_entries(entry_type=MemoryEntryType.GENERATED_CODE)
    assert len(code_entries) == 3

    # Filter by project and type
    proj1_code = store.list_entries(
        project_id="proj-001", entry_type=MemoryEntryType.GENERATED_CODE
    )
    assert len(proj1_code) == 2

    # Filter by experiment_id
    store.add(
        MemoryEntry(
            entry_id="exp-001", project_id="proj-001", experiment_id="exp-123", content="exp"
        )
    )
    exp_entries = store.list_entries(experiment_id="exp-123")
    assert len(exp_entries) == 1
    assert exp_entries[0].entry_id == "exp-001"


def test_store_list_entries_limit():
    """Test that limit is respected."""
    store = InMemoryScientificMemoryStore()
    for i in range(10):
        store.add(
            MemoryEntry(entry_id=f"mem-{i:03d}", project_id="proj-001", content=f"content-{i}")
        )

    limited = store.list_entries(limit=5)
    assert len(limited) == 5


def test_store_search():
    """Test searching entries."""
    store = InMemoryScientificMemoryStore()
    store.add(
        MemoryEntry(
            entry_id="n1",
            project_id="p1",
            content="This is about machine learning",
            tags=["ml", "ai"],
        )
    )
    store.add(
        MemoryEntry(
            entry_id="n2",
            project_id="p1",
            content="This is about quantum computing",
            tags=["quantum"],
        )
    )
    store.add(
        MemoryEntry(
            entry_id="n3", project_id="p2", content="Machine learning is powerful", tags=["ml"]
        )
    )

    # Search by content
    results = store.search("machine learning")
    assert len(results) == 2

    # Search by tag
    results = store.search("quantum")
    assert len(results) == 1
    assert results[0].entry_id == "n2"

    # Search with project filter
    results = store.search("machine learning", project_id="p1")
    assert len(results) == 1
    assert results[0].entry_id == "n1"


def test_store_update_status():
    """Test updating entry status."""
    store = InMemoryScientificMemoryStore()
    entry = MemoryEntry(entry_id="mem-001", project_id="proj-001")
    store.add(entry)

    result = store.update_status("mem-001", MemoryEntryStatus.ARCHIVED)
    assert result is True
    assert store.get("mem-001").status == MemoryEntryStatus.ARCHIVED


def test_store_update_status_nonexistent():
    """Test updating status of non-existent entry returns False."""
    store = InMemoryScientificMemoryStore()
    assert store.update_status("nonexistent", MemoryEntryStatus.ARCHIVED) is False


def test_store_count():
    """Test counting entries."""
    store = InMemoryScientificMemoryStore()
    store.add(
        MemoryEntry(entry_id="n1", project_id="p1", entry_type=MemoryEntryType.GENERATED_CODE)
    )
    store.add(
        MemoryEntry(entry_id="n2", project_id="p1", entry_type=MemoryEntryType.EXPERIMENT_LOG)
    )
    store.add(
        MemoryEntry(entry_id="n3", project_id="p1", entry_type=MemoryEntryType.GENERATED_CODE)
    )
    store.add(
        MemoryEntry(entry_id="n4", project_id="p2", entry_type=MemoryEntryType.GENERATED_CODE)
    )

    assert store.count() == 4
    assert store.count(project_id="p1") == 3
    assert store.count(entry_type=MemoryEntryType.GENERATED_CODE) == 3
    assert store.count(project_id="p1", entry_type=MemoryEntryType.GENERATED_CODE) == 2


def test_store_get_timeline():
    """Test getting a timeline of entries."""
    store = InMemoryScientificMemoryStore()
    # Add entries with different created_at timestamps
    entry1 = MemoryEntry(entry_id="n1", project_id="p1", content="First")
    entry1.created_at = "2026-06-30T10:00:00Z"
    entry2 = MemoryEntry(entry_id="n2", project_id="p1", content="Second")
    entry2.created_at = "2026-06-30T11:00:00Z"
    entry3 = MemoryEntry(entry_id="n3", project_id="p1", content="Third")
    entry3.created_at = "2026-06-30T09:00:00Z"  # Earliest
    entry4 = MemoryEntry(entry_id="n4", project_id="p2", content="Other project")
    entry4.created_at = "2026-06-30T12:00:00Z"

    store.add(entry1)
    store.add(entry2)
    store.add(entry3)
    store.add(entry4)

    timeline = store.get_timeline("p1")

    # Should be sorted by created_at ascending (oldest first)
    assert len(timeline) == 3
    assert timeline[0].entry_id == "n3"  # 09:00
    assert timeline[1].entry_id == "n1"  # 10:00
    assert timeline[2].entry_id == "n2"  # 11:00


def test_store_list_entries_sorted_by_created_at_descending():
    """Test that list_entries returns entries sorted by created_at descending."""
    store = InMemoryScientificMemoryStore()
    entry1 = MemoryEntry(entry_id="n1", project_id="p1")
    entry1.created_at = "2026-06-30T10:00:00Z"
    entry2 = MemoryEntry(entry_id="n2", project_id="p1")
    entry2.created_at = "2026-06-30T12:00:00Z"
    entry3 = MemoryEntry(entry_id="n3", project_id="p1")
    entry3.created_at = "2026-06-30T11:00:00Z"

    store.add(entry1)
    store.add(entry2)
    store.add(entry3)

    results = store.list_entries(project_id="p1")

    # Newest first
    assert results[0].entry_id == "n2"  # 12:00
    assert results[1].entry_id == "n3"  # 11:00
    assert results[2].entry_id == "n1"  # 10:00
