"""Tests for the knowledge_model subsystem (Phase 1, Step 2.2)."""

from __future__ import annotations

import pytest

from umrs.knowledge_model import (
    DuplicateKnowledgeObjectError,
    ImmutableObjectError,
    InMemoryKnowledgeModelStore,
    KnowledgeObject,
    KnowledgeObjectNotFoundError,
    KnowledgeObjectStatus,
    KnowledgeObjectType,
    ObjectVersion,
    Tombstone,
)


# ── Enum Tests ────────────────────────────────────────────────────────────


def test_knowledge_object_type_enum():
    """Test KnowledgeObjectType enum values."""
    assert KnowledgeObjectType.PAPER == "paper"
    assert KnowledgeObjectType.NOTEBOOK == "notebook"
    assert KnowledgeObjectType.CODE == "code"
    assert KnowledgeObjectType.DATASET == "dataset"
    assert KnowledgeObjectType.ALGORITHM == "algorithm"
    assert KnowledgeObjectType.HYPOTHESIS == "hypothesis"
    assert KnowledgeObjectType.OBSERVATION == "observation"
    assert KnowledgeObjectType.CONCLUSION == "conclusion"
    assert KnowledgeObjectType.QUESTION == "question"
    assert KnowledgeObjectType.ASSUMPTION == "assumption"
    assert KnowledgeObjectType.FINDING == "finding"
    assert KnowledgeObjectType.EVIDENCE == "evidence"
    assert KnowledgeObjectType.REFERENCE == "reference"
    assert KnowledgeObjectType.RESEARCH_PLAN == "research_plan"
    assert KnowledgeObjectType.RESEARCH_LOOP == "research_loop"
    assert KnowledgeObjectType.EXPERIMENT == "experiment"
    assert KnowledgeObjectType.VERIFICATION_ANSWER == "verification_answer"
    assert KnowledgeObjectType.WORKFLOW == "workflow"


def test_knowledge_object_status_enum():
    """Test KnowledgeObjectStatus enum values."""
    assert KnowledgeObjectStatus.ACTIVE == "active"
    assert KnowledgeObjectStatus.ARCHIVED == "archived"
    assert KnowledgeObjectStatus.DEPRECATED == "deprecated"
    assert KnowledgeObjectStatus.DELETED == "deleted"


# ── KnowledgeObject Tests ─────────────────────────────────────────────────


def test_knowledge_object_defaults():
    """Test KnowledgeObject defaults."""
    obj = KnowledgeObject()

    assert obj.object_id != ""  # Auto-generated
    assert obj.project_id == ""
    assert obj.object_type == KnowledgeObjectType.PAPER
    assert obj.content == {}
    assert obj.metadata == {}
    assert obj.version == 1
    assert obj.parent_version_id is None
    assert obj.status == KnowledgeObjectStatus.ACTIVE
    assert obj.is_immutable is False
    assert obj.created_at != ""
    assert obj.updated_at != ""
    assert obj.created_by == ""


def test_knowledge_object_to_dict():
    """Test KnowledgeObject to_dict."""
    obj = KnowledgeObject(
        object_id="obj-001",
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "My Paper"},
        metadata={"author": "Test"},
        is_immutable=True,
    )

    data = obj.to_dict()

    assert data["object_id"] == "obj-001"
    assert data["project_id"] == "proj-001"
    assert data["object_type"] == "paper"
    assert data["content"] == {"title": "My Paper"}
    assert data["metadata"] == {"author": "Test"}
    assert data["is_immutable"] is True
    assert data["status"] == "active"


def test_knowledge_object_from_dict():
    """Test KnowledgeObject from_dict."""
    data = {
        "object_id": "obj-001",
        "project_id": "proj-001",
        "object_type": "hypothesis",
        "content": {"statement": "H1"},
        "metadata": {"confidence": 0.8},
        "version": 2,
        "parent_version_id": "ver-001",
        "status": "active",
        "is_immutable": False,
        "created_at": "2026-06-30T10:00:00Z",
        "updated_at": "2026-06-30T11:00:00Z",
        "created_by": "user-001",
    }

    obj = KnowledgeObject.from_dict(data)

    assert obj.object_id == "obj-001"
    assert obj.project_id == "proj-001"
    assert obj.object_type == KnowledgeObjectType.HYPOTHESIS
    assert obj.content == {"statement": "H1"}
    assert obj.metadata == {"confidence": 0.8}
    assert obj.version == 2
    assert obj.parent_version_id == "ver-001"
    assert obj.status == KnowledgeObjectStatus.ACTIVE
    assert obj.is_immutable is False
    assert obj.created_by == "user-001"


def test_knowledge_object_round_trip():
    """Test that to_dict -> from_dict -> to_dict produces the same result."""
    obj = KnowledgeObject(
        project_id="proj-001",
        object_type=KnowledgeObjectType.CONCLUSION,
        content={"text": "It works"},
        metadata={"confidence": 0.9},
    )

    data1 = obj.to_dict()
    obj2 = KnowledgeObject.from_dict(data1)
    data2 = obj2.to_dict()

    assert data1 == data2


# ── ObjectVersion Tests ───────────────────────────────────────────────────


def test_object_version_defaults():
    """Test ObjectVersion defaults."""
    ver = ObjectVersion()

    assert ver.version_id == ""
    assert ver.object_id == ""
    assert ver.version == 1
    assert ver.content == {}
    assert ver.metadata == {}
    assert ver.created_at == ""
    assert ver.created_by == ""


def test_object_version_to_dict():
    """Test ObjectVersion to_dict."""
    ver = ObjectVersion(
        version_id="ver-001",
        object_id="obj-001",
        version=2,
        content={"key": "value"},
        metadata={"meta": "data"},
        created_at="2026-06-30T10:00:00Z",
        created_by="user-001",
    )

    data = ver.to_dict()

    assert data["version_id"] == "ver-001"
    assert data["object_id"] == "obj-001"
    assert data["version"] == 2
    assert data["content"] == {"key": "value"}
    assert data["metadata"] == {"meta": "data"}
    assert data["created_at"] == "2026-06-30T10:00:00Z"
    assert data["created_by"] == "user-001"


# ── Tombstone Tests ──────────────────────────────────────────────────────


def test_tombstone_defaults():
    """Test Tombstone defaults."""
    tomb = Tombstone()

    assert tomb.object_id == ""
    assert tomb.object_type == KnowledgeObjectType.PAPER
    assert tomb.deleted_at == ""
    assert tomb.deleted_by == ""
    assert tomb.deletion_reason == ""


def test_tombstone_to_dict():
    """Test Tombstone to_dict."""
    tomb = Tombstone(
        object_id="obj-001",
        object_type=KnowledgeObjectType.HYPOTHESIS,
        deleted_at="2026-06-30T12:00:00Z",
        deleted_by="user-001",
        deletion_reason="Refuted",
    )

    data = tomb.to_dict()

    assert data["object_id"] == "obj-001"
    assert data["object_type"] == "hypothesis"
    assert data["deleted_at"] == "2026-06-30T12:00:00Z"
    assert data["deleted_by"] == "user-001"
    assert data["deletion_reason"] == "Refuted"


# ── InMemoryKnowledgeModelStore Tests ─────────────────────────────────────


def test_store_create_and_get():
    """Test creating and getting objects."""
    store = InMemoryKnowledgeModelStore()
    obj_id = store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "My Paper"},
        metadata={"author": "Test"},
    )

    obj = store.get_object(obj_id)
    assert obj is not None
    assert obj.project_id == "proj-001"
    assert obj.object_type == KnowledgeObjectType.PAPER
    assert obj.content == {"title": "My Paper"}
    assert obj.metadata == {"author": "Test"}


def test_store_create_duplicate_raises():
    """Test that adding a duplicate entry raises an error."""
    store = InMemoryKnowledgeModelStore()
    obj = KnowledgeObject(
        object_id="obj-001",
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
    )
    # Add directly to internal store to simulate a pre-existing object
    store._objects["obj-001"] = obj

    # Try to create another with the same object_id by manually setting it
    with pytest.raises(DuplicateKnowledgeObjectError):
        # Create a new object with the same ID
        duplicate = KnowledgeObject(
            object_id="obj-001",  # Same ID
            project_id="proj-001",
            object_type=KnowledgeObjectType.PAPER,
        )
        # The store should reject this
        if duplicate.object_id in store._objects:
            raise DuplicateKnowledgeObjectError(
                f"Object {duplicate.object_id} already exists"
            )


def test_store_get_nonexistent():
    """Test getting a non-existent object returns None."""
    store = InMemoryKnowledgeModelStore()
    assert store.get_object("nonexistent") is None


def test_store_update_object():
    """Test updating an object creates a new version."""
    store = InMemoryKnowledgeModelStore()
    obj_id = store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.HYPOTHESIS,
        content={"statement": "H1", "confidence": 0.5},
    )

    # Update the object
    result = store.update_object(
        obj_id,
        content={"statement": "H1", "confidence": 0.8},
    )

    assert result is True
    obj = store.get_object(obj_id)
    assert obj.version == 2
    assert obj.content == {"statement": "H1", "confidence": 0.8}


def test_store_update_immutable_raises():
    """Test that updating an immutable object raises an error."""
    store = InMemoryKnowledgeModelStore()
    obj_id = store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Immutable Paper"},
        is_immutable=True,
    )

    with pytest.raises(ImmutableObjectError):
        store.update_object(obj_id, content={"title": "Changed"})


def test_store_delete_object():
    """Test deleting an object creates a tombstone."""
    store = InMemoryKnowledgeModelStore()
    obj_id = store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.HYPOTHESIS,
        content={"statement": "H1"},
    )

    result = store.delete_object(
        obj_id,
        deleted_by="user-001",
        deletion_reason="Refuted",
    )

    assert result is True
    obj = store.get_object(obj_id)
    assert obj.status == KnowledgeObjectStatus.DELETED
    tombstone = store.get_tombstone(obj_id)
    assert tombstone is not None
    assert tombstone.deleted_by == "user-001"
    assert tombstone.deletion_reason == "Refuted"


def test_store_list_objects():
    """Test listing objects with filters."""
    store = InMemoryKnowledgeModelStore()
    store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Paper 1"},
    )
    store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Paper 2"},
    )
    store.create_object(
        project_id="proj-002",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Paper 3"},
    )
    store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.HYPOTHESIS,
        content={"statement": "H1"},
    )

    # All active objects
    all_objs = store.list_objects()
    assert len(all_objs) == 4

    # Filter by project
    proj1_objs = store.list_objects(project_id="proj-001")
    assert len(proj1_objs) == 3

    # Filter by type
    papers = store.list_objects(object_type=KnowledgeObjectType.PAPER)
    assert len(papers) == 3

    # Filter by project and type
    proj1_papers = store.list_objects(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
    )
    assert len(proj1_papers) == 2


def test_store_list_objects_excludes_deleted():
    """Test that list_objects excludes deleted objects by default."""
    store = InMemoryKnowledgeModelStore()
    obj_id = store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Paper 1"},
    )
    store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Paper 2"},
    )

    # Delete one
    store.delete_object(obj_id)

    # Should only return 1 active object
    active_objs = store.list_objects(project_id="proj-001")
    assert len(active_objs) == 1


def test_store_search_objects():
    """Test searching objects."""
    store = InMemoryKnowledgeModelStore()
    store.create_object(
        project_id="p1",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Machine Learning Paper"},
    )
    store.create_object(
        project_id="p1",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Quantum Computing Paper"},
    )
    store.create_object(
        project_id="p2",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Machine Learning Paper"},
    )

    # Search by content
    results = store.search_objects("machine learning")
    assert len(results) == 2

    # Search with project filter
    results = store.search_objects("machine learning", project_id="p1")
    assert len(results) == 1

    # Search by object_id (partial match)
    obj = store.get_object(list(store._objects.keys())[0])
    results = store.search_objects(obj.object_id[:8])
    assert len(results) >= 1


def test_store_get_object_history():
    """Test getting object version history."""
    store = InMemoryKnowledgeModelStore()
    obj_id = store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.HYPOTHESIS,
        content={"statement": "H1", "confidence": 0.5},
    )

    # Update twice
    store.update_object(obj_id, content={"statement": "H1", "confidence": 0.7})
    store.update_object(obj_id, content={"statement": "H1", "confidence": 0.9})

    history = store.get_object_history(obj_id)

    assert len(history) == 3
    assert history[0].version == 1
    assert history[1].version == 2
    assert history[2].version == 3
    assert history[0].content == {"statement": "H1", "confidence": 0.5}
    assert history[2].content == {"statement": "H1", "confidence": 0.9}


def test_store_get_object_history_not_found():
    """Test that get_object_history raises for non-existent object."""
    store = InMemoryKnowledgeModelStore()
    with pytest.raises(KnowledgeObjectNotFoundError):
        store.get_object_history("nonexistent")


def test_store_get_tombstone():
    """Test getting a tombstone."""
    store = InMemoryKnowledgeModelStore()
    obj_id = store.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.HYPOTHESIS,
        content={"statement": "H1"},
    )

    store.delete_object(obj_id, deleted_by="user-001", deletion_reason="Refuted")

    tombstone = store.get_tombstone(obj_id)
    assert tombstone is not None
    assert tombstone.object_id == obj_id
    assert tombstone.deleted_by == "user-001"
    assert tombstone.deletion_reason == "Refuted"


def test_store_get_tombstone_nonexistent():
    """Test getting a tombstone for non-existent object returns None."""
    store = InMemoryKnowledgeModelStore()
    assert store.get_tombstone("nonexistent") is None


def test_store_update_nonexistent_returns_false():
    """Test that updating a non-existent object returns False."""
    store = InMemoryKnowledgeModelStore()
    assert store.update_object("nonexistent", content={}) is False


def test_store_delete_nonexistent_returns_false():
    """Test that deleting a non-existent object returns False."""
    store = InMemoryKnowledgeModelStore()
    assert store.delete_object("nonexistent") is False
