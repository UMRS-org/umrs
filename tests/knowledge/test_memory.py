"""Tests for the scientific memory module (Phase 9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.knowledge.memory import (
    MemoryEntry,
    MemoryEntryStatus,
    MemoryEntryType,
    ScientificMemoryManager,
    ScientificMemoryStore,
    create_scientific_memory,
)

# ── Enum Tests ─────────────────────────────────────────────────────────────


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
    assert MemoryEntryType.NOTE == "note"


def test_memory_entry_status_enum():
    """Test MemoryEntryStatus enum values."""
    assert MemoryEntryStatus.ACTIVE == "active"
    assert MemoryEntryStatus.ARCHIVED == "archived"
    assert MemoryEntryStatus.DEPRECATED == "deprecated"
    assert MemoryEntryStatus.RETRACTED == "retracted"


# ── MemoryEntry Tests ──────────────────────────────────────────────────────


def test_memory_entry_defaults():
    """Test MemoryEntry defaults."""
    entry = MemoryEntry(
        entry_id="test-001",
        entry_type=MemoryEntryType.NOTE,
    )

    assert entry.entry_id == "test-001"
    assert entry.entry_type == MemoryEntryType.NOTE
    assert entry.project_id == ""
    assert entry.source == "system"
    assert entry.content == ""
    assert entry.status == MemoryEntryStatus.ACTIVE
    assert entry.confidence == 0.0
    assert entry.references == []
    assert entry.tags == []
    assert entry.created_at != ""  # Should be set by __post_init__
    assert entry.updated_at != ""


def test_memory_entry_to_dict():
    """Test MemoryEntry to_dict."""
    entry = MemoryEntry(
        entry_id="test-001",
        entry_type=MemoryEntryType.GENERATED_CODE,
        project_id="proj-001",
        content="print('hello')",
        summary="Generated Python code",
        tags=["python", "generated"],
        confidence=0.8,
    )

    data = entry.to_dict()
    assert data["entry_id"] == "test-001"
    assert data["entry_type"] == "generated_code"
    assert data["project_id"] == "proj-001"
    assert data["content"] == "print('hello')"
    assert data["summary"] == "Generated Python code"
    assert data["tags"] == ["python", "generated"]
    assert data["confidence"] == 0.8
    assert data["status"] == "active"


def test_memory_entry_from_dict():
    """Test MemoryEntry from_dict."""
    data = {
        "entry_id": "test-001",
        "entry_type": "note",
        "project_id": "proj-001",
        "source": "user",
        "content": "This is a note",
        "summary": "A note",
        "metadata": {"key": "value"},
        "tags": ["note"],
        "status": "active",
        "created_at": "2026-06-30T10:00:00Z",
        "updated_at": "2026-06-30T10:00:00Z",
        "created_by": "user-001",
        "confidence": 0.5,
        "references": ["ref-001"],
    }

    entry = MemoryEntry.from_dict(data)

    assert entry.entry_id == "test-001"
    assert entry.entry_type == MemoryEntryType.NOTE
    assert entry.project_id == "proj-001"
    assert entry.source == "user"
    assert entry.content == "This is a note"
    assert entry.summary == "A note"
    assert entry.metadata == {"key": "value"}
    assert entry.tags == ["note"]
    assert entry.status == MemoryEntryStatus.ACTIVE
    assert entry.created_by == "user-001"
    assert entry.confidence == 0.5
    assert entry.references == ["ref-001"]


def test_memory_entry_round_trip():
    """Test that to_dict -> from_dict -> to_dict produces the same result."""
    entry = MemoryEntry(
        entry_id="test-001",
        entry_type=MemoryEntryType.NEW_HYPOTHESIS,
        project_id="proj-001",
        content="This is a hypothesis",
        summary="A hypothesis",
        tags=["hypothesis"],
        confidence=0.6,
    )

    data1 = entry.to_dict()
    entry2 = MemoryEntry.from_dict(data1)
    data2 = entry2.to_dict()

    assert data1 == data2


# ── ScientificMemoryStore Tests ────────────────────────────────────────────


def test_store_add_and_get():
    """Test adding and getting entries."""
    store = ScientificMemoryStore()
    entry = MemoryEntry(
        entry_id="test-001",
        entry_type=MemoryEntryType.NOTE,
        content="Test content",
    )
    store.add(entry)

    retrieved = store.get("test-001")
    assert retrieved is not None
    assert retrieved.content == "Test content"


def test_store_add_duplicate_raises():
    """Test that adding a duplicate entry raises an error."""
    store = ScientificMemoryStore()
    entry = MemoryEntry(
        entry_id="test-001",
        entry_type=MemoryEntryType.NOTE,
    )
    store.add(entry)

    with pytest.raises(ValueError, match="already exists"):
        store.add(entry)


def test_store_list_entries():
    """Test listing entries with filters."""
    store = ScientificMemoryStore()
    store.add(
        MemoryEntry(
            entry_id="note-001",
            entry_type=MemoryEntryType.NOTE,
            project_id="proj-001",
        )
    )
    store.add(
        MemoryEntry(
            entry_id="code-001",
            entry_type=MemoryEntryType.GENERATED_CODE,
            project_id="proj-001",
        )
    )
    store.add(
        MemoryEntry(
            entry_id="note-002",
            entry_type=MemoryEntryType.NOTE,
            project_id="proj-002",
        )
    )

    # All entries
    all_entries = store.list_entries()
    assert len(all_entries) == 3

    # Filter by type
    notes = store.list_entries(entry_type=MemoryEntryType.NOTE)
    assert len(notes) == 2

    # Filter by project
    proj1_entries = store.list_entries(project_id="proj-001")
    assert len(proj1_entries) == 2


def test_store_search():
    """Test searching entries."""
    store = ScientificMemoryStore()
    store.add(
        MemoryEntry(
            entry_id="note-001",
            entry_type=MemoryEntryType.NOTE,
            content="This is about machine learning",
            tags=["ml", "ai"],
        )
    )
    store.add(
        MemoryEntry(
            entry_id="note-002",
            entry_type=MemoryEntryType.NOTE,
            content="This is about quantum computing",
            tags=["quantum"],
        )
    )

    # Search by content
    results = store.search("machine learning")
    assert len(results) == 1
    assert results[0].entry_id == "note-001"

    # Search by tag
    results = store.search("quantum")
    assert len(results) == 1
    assert results[0].entry_id == "note-002"


def test_store_update_status():
    """Test updating entry status."""
    store = ScientificMemoryStore()
    entry = MemoryEntry(
        entry_id="test-001",
        entry_type=MemoryEntryType.NOTE,
    )
    store.add(entry)

    result = store.update_status("test-001", MemoryEntryStatus.ARCHIVED)
    assert result is True
    assert store.get("test-001").status == MemoryEntryStatus.ARCHIVED


def test_store_count():
    """Test counting entries."""
    store = ScientificMemoryStore()
    store.add(MemoryEntry(entry_id="n1", entry_type=MemoryEntryType.NOTE, project_id="p1"))
    store.add(MemoryEntry(entry_id="n2", entry_type=MemoryEntryType.NOTE, project_id="p1"))
    store.add(
        MemoryEntry(entry_id="c1", entry_type=MemoryEntryType.GENERATED_CODE, project_id="p2")
    )

    assert store.count() == 3
    assert store.count(entry_type=MemoryEntryType.NOTE) == 2
    assert store.count(project_id="p1") == 2


def test_store_persistence(tmp_path: Path):
    """Test that entries persist to disk."""
    storage_file = tmp_path / "memory.json"
    store1 = ScientificMemoryStore(storage_path=storage_file)
    store1.add(
        MemoryEntry(
            entry_id="test-001",
            entry_type=MemoryEntryType.NOTE,
            content="Test content",
        )
    )

    # Create a new store and load from disk
    store2 = ScientificMemoryStore(storage_path=storage_file)
    store2.load()

    assert store2.get("test-001") is not None
    assert store2.get("test-001").content == "Test content"


# ── ScientificMemoryManager Tests ─────────────────────────────────────────


def test_manager_init():
    """Test ScientificMemoryManager initialization."""
    manager = ScientificMemoryManager()
    assert manager.store is not None


def test_manager_register_generated_code():
    """Test registering generated code."""
    manager = ScientificMemoryManager()
    entry_id = manager.register_generated_code(
        project_id="proj-001",
        code="def hello(): return 'world'",
        summary="Generated Python function",
        language="python",
        created_by="agent-001",
    )

    entry = manager.store.get(entry_id)
    assert entry is not None
    assert entry.entry_type == MemoryEntryType.GENERATED_CODE
    assert entry.content == "def hello(): return 'world'"
    assert entry.metadata["language"] == "python"
    assert entry.created_by == "agent-001"


def test_manager_register_experiment_log():
    """Test registering an experiment log."""
    manager = ScientificMemoryManager()
    entry_id = manager.register_experiment_log(
        project_id="proj-001",
        experiment_id="exp-001",
        log_content="Starting experiment...\nDone.",
        summary="Experiment log",
    )

    entry = manager.store.get(entry_id)
    assert entry is not None
    assert entry.entry_type == MemoryEntryType.EXPERIMENT_LOG
    assert entry.experiment_id == "exp-001"


def test_manager_register_plot_figure():
    """Test registering a plot figure."""
    manager = ScientificMemoryManager()
    entry_id = manager.register_plot_figure(
        project_id="proj-001",
        figure_uri="s3://bucket/figure.png",
        summary="Loss curve",
    )

    entry = manager.store.get(entry_id)
    assert entry is not None
    assert entry.entry_type == MemoryEntryType.PLOT_FIGURE
    assert entry.content_uri == "s3://bucket/figure.png"


def test_manager_register_analysis_result():
    """Test registering an analysis result."""
    manager = ScientificMemoryManager()
    entry_id = manager.register_analysis_result(
        project_id="proj-001",
        analysis="The results show a 20% improvement.",
        summary="Analysis of results",
    )

    entry = manager.store.get(entry_id)
    assert entry is not None
    assert entry.entry_type == MemoryEntryType.ANALYSIS_RESULT
    assert "20% improvement" in entry.content


def test_manager_register_new_hypothesis():
    """Test registering a new hypothesis."""
    manager = ScientificMemoryManager()
    entry_id = manager.register_new_hypothesis(
        project_id="proj-001",
        hypothesis="Increasing learning rate will improve convergence.",
        confidence=0.6,
    )

    entry = manager.store.get(entry_id)
    assert entry is not None
    assert entry.entry_type == MemoryEntryType.NEW_HYPOTHESIS
    assert entry.confidence == 0.6


def test_manager_register_review_comment():
    """Test registering a review comment."""
    manager = ScientificMemoryManager()
    entry_id = manager.register_review_comment(
        project_id="proj-001",
        comment="This approach looks promising.",
        target_entry_id="entry-001",
    )

    entry = manager.store.get(entry_id)
    assert entry is not None
    assert entry.entry_type == MemoryEntryType.REVIEW_COMMENT
    assert entry.references == ["entry-001"]


def test_manager_register_accepted_finding():
    """Test registering an accepted finding."""
    manager = ScientificMemoryManager()
    entry_id = manager.register_accepted_finding(
        project_id="proj-001",
        finding="The transformer architecture outperforms RNNs.",
        confidence=0.9,
    )

    entry = manager.store.get(entry_id)
    assert entry is not None
    assert entry.entry_type == MemoryEntryType.ACCEPTED_FINDING
    assert entry.confidence == 0.9


def test_manager_register_failed_experiment():
    """Test registering a failed experiment."""
    manager = ScientificMemoryManager()
    entry_id = manager.register_failed_experiment(
        project_id="proj-001",
        experiment_id="exp-001",
        failure_reason="Out of memory",
        log_content="OOM error at step 100",
    )

    entry = manager.store.get(entry_id)
    assert entry is not None
    assert entry.entry_type == MemoryEntryType.FAILED_EXPERIMENT
    assert "Out of memory" in entry.content


def test_manager_search():
    """Test searching through the manager."""
    manager = ScientificMemoryManager()
    manager.register_new_hypothesis(
        project_id="proj-001",
        hypothesis="Machine learning can predict molecular properties.",
    )
    manager.register_accepted_finding(
        project_id="proj-001",
        finding="The transformer architecture works well.",
    )

    results = manager.search("machine learning")
    assert len(results) >= 1
    assert any("machine learning" in r.content.lower() for r in results)


def test_manager_list_by_type():
    """Test listing entries by type."""
    manager = ScientificMemoryManager()
    manager.register_new_hypothesis(project_id="proj-001", hypothesis="H1")
    manager.register_new_hypothesis(project_id="proj-001", hypothesis="H2")
    manager.register_accepted_finding(project_id="proj-001", finding="F1")

    hypotheses = manager.list_by_type(MemoryEntryType.NEW_HYPOTHESIS)
    assert len(hypotheses) == 2

    findings = manager.list_by_type(MemoryEntryType.ACCEPTED_FINDING)
    assert len(findings) == 1


# ── Convenience Function Tests ─────────────────────────────────────────────


def test_create_scientific_memory():
    """Test the create_scientific_memory convenience function."""
    manager = create_scientific_memory()
    assert isinstance(manager, ScientificMemoryManager)
    assert manager.store is not None


def test_create_scientific_memory_with_path(tmp_path: Path):
    """Test creating scientific memory with a storage path."""
    storage_file = tmp_path / "memory.json"
    manager = create_scientific_memory(storage_path=storage_file)
    assert isinstance(manager, ScientificMemoryManager)

    # Add an entry
    manager.register_new_hypothesis(
        project_id="proj-001",
        hypothesis="Test hypothesis",
    )

    # Create a new manager with the same storage path
    manager2 = create_scientific_memory(storage_path=storage_file)

    # Should be able to find the entry
    results = manager2.search("Test hypothesis")
    assert len(results) >= 1
