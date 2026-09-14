"""Tests for the embeddings_api subsystem (Phase 1, Step 2.7)."""

from __future__ import annotations

from umrs.embeddings_api import (
    Embedding,
    EmbeddingMatch,
    EmbeddingStatus,
    InMemoryEmbeddingStore,
)

# ── Enum Tests ────────────────────────────────────────────────────────────


def test_embedding_status_enum():
    """Test EmbeddingStatus enum values."""
    assert EmbeddingStatus.ACTIVE == "active"
    assert EmbeddingStatus.ARCHIVED == "archived"
    assert EmbeddingStatus.DEPRECATED == "deprecated"


# ── Embedding Tests ──────────────────────────────────────────────────────


def test_embedding_defaults():
    """Test Embedding defaults."""
    emb = Embedding()

    assert emb.embedding_id != ""  # Auto-generated
    assert emb.project_id == ""
    assert emb.object_id == ""
    assert emb.model == ""
    assert emb.vector == []
    assert emb.text == ""
    assert emb.metadata == {}
    assert emb.status == EmbeddingStatus.ACTIVE
    assert emb.created_at != ""
    assert emb.updated_at != ""


def test_embedding_to_dict():
    """Test Embedding to_dict."""
    emb = Embedding(
        embedding_id="emb-001",
        project_id="proj-001",
        object_id="obj-001",
        model="sentence-transformers/all-MiniLM-L6-v2",
        vector=[0.1, 0.2, 0.3],
        text="This is a test",
    )

    data = emb.to_dict()

    assert data["embedding_id"] == "emb-001"
    assert data["project_id"] == "proj-001"
    assert data["object_id"] == "obj-001"
    assert data["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert data["vector"] == [0.1, 0.2, 0.3]
    assert data["text"] == "This is a test"
    assert data["status"] == "active"


def test_embedding_from_dict():
    """Test Embedding from_dict."""
    data = {
        "embedding_id": "emb-001",
        "project_id": "proj-001",
        "object_id": "obj-001",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector": [0.1, 0.2, 0.3],
        "text": "This is a test",
        "metadata": {"key": "value"},
        "status": "active",
        "created_at": "2026-06-30T10:00:00Z",
        "updated_at": "2026-06-30T10:00:00Z",
    }

    emb = Embedding.from_dict(data)

    assert emb.embedding_id == "emb-001"
    assert emb.project_id == "proj-001"
    assert emb.object_id == "obj-001"
    assert emb.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert emb.vector == [0.1, 0.2, 0.3]
    assert emb.text == "This is a test"
    assert emb.metadata == {"key": "value"}
    assert emb.status == EmbeddingStatus.ACTIVE


def test_embedding_round_trip():
    """Test that to_dict -> from_dict -> to_dict produces the same result."""
    emb = Embedding(
        project_id="proj-001",
        object_id="obj-001",
        model="test-model",
        vector=[0.1, 0.2, 0.3],
        text="test text",
    )

    data1 = emb.to_dict()
    emb2 = Embedding.from_dict(data1)
    data2 = emb2.to_dict()

    assert data1 == data2


# ── EmbeddingMatch Tests ─────────────────────────────────────────────────


def test_embedding_match_defaults():
    """Test EmbeddingMatch defaults."""
    match = EmbeddingMatch()

    assert match.embedding_id == ""
    assert match.object_id == ""
    assert match.score == 0.0
    assert match.text == ""
    assert match.metadata == {}


def test_embedding_match_to_dict():
    """Test EmbeddingMatch to_dict."""
    match = EmbeddingMatch(
        embedding_id="emb-001",
        object_id="obj-001",
        score=0.95,
        text="test text",
    )

    data = match.to_dict()

    assert data["embedding_id"] == "emb-001"
    assert data["object_id"] == "obj-001"
    assert data["score"] == 0.95
    assert data["text"] == "test text"


# ── InMemoryEmbeddingStore Tests ─────────────────────────────────────────


def test_store_create_and_get():
    """Test creating and getting embeddings."""
    store = InMemoryEmbeddingStore()
    emb_id = store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="This is a test",
    )

    emb = store.get_embedding(emb_id)
    assert emb is not None
    assert emb.project_id == "proj-001"
    assert emb.object_id == "obj-001"
    assert emb.text == "This is a test"
    assert len(emb.vector) > 0


def test_store_get_nonexistent():
    """Test getting a non-existent embedding returns None."""
    store = InMemoryEmbeddingStore()
    assert store.get_embedding("nonexistent") is None


def test_store_search_similar():
    """Test searching for similar embeddings."""
    store = InMemoryEmbeddingStore()
    emb1_id = store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="machine learning",
    )
    emb2_id = store.create_embedding(
        project_id="proj-001",
        object_id="obj-002",
        text="quantum computing",
    )

    # Get the first embedding's vector
    emb1 = store.get_embedding(emb1_id)
    query_embedding = emb1.vector

    # Search for similar embeddings
    results = store.search_similar(
        query_embedding=query_embedding,
        top_k=10,
        project_id="proj-001",
    )

    assert len(results) > 0
    # The first result should be the most similar (emb1 itself)
    assert results[0].embedding_id == emb1_id
    assert results[0].score > 0.0


def test_store_search_similar_project_filter():
    """Test that search_similar respects project_id filter."""
    store = InMemoryEmbeddingStore()
    store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="machine learning",
    )
    store.create_embedding(
        project_id="proj-002",
        object_id="obj-002",
        text="machine learning",
    )

    # Get a query embedding
    emb = store.get_embedding(list(store._embeddings.keys())[0])
    query_embedding = emb.vector

    # Search only in proj-001
    results = store.search_similar(
        query_embedding=query_embedding,
        top_k=10,
        project_id="proj-001",
    )

    # Should only return results from proj-001
    assert len(results) > 0
    for result in results:
        assert result.embedding_id in store._embeddings
        assert store._embeddings[result.embedding_id].project_id == "proj-001"


def test_store_search_similar_top_k():
    """Test that top_k limits the number of results."""
    store = InMemoryEmbeddingStore()
    for i in range(10):
        store.create_embedding(
            project_id="proj-001",
            object_id=f"obj-{i:03d}",
            text=f"text-{i}",
        )

    # Get a query embedding
    emb = store.get_embedding(list(store._embeddings.keys())[0])
    query_embedding = emb.vector

    # Search with top_k=5
    results = store.search_similar(
        query_embedding=query_embedding,
        top_k=5,
        project_id="proj-001",
    )

    assert len(results) == 5


def test_store_update_embedding():
    """Test updating an embedding's text."""
    store = InMemoryEmbeddingStore()
    emb_id = store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="original text",
    )

    # Update the text
    result = store.update_embedding(emb_id, text="updated text")

    assert result is True
    emb = store.get_embedding(emb_id)
    assert emb.text == "updated text"
    assert len(emb.vector) > 0


def test_store_update_embedding_nonexistent():
    """Test that updating a non-existent embedding returns False."""
    store = InMemoryEmbeddingStore()
    assert store.update_embedding("nonexistent", text="test") is False


def test_store_delete_embedding():
    """Test deleting an embedding."""
    store = InMemoryEmbeddingStore()
    emb_id = store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="test",
    )

    result = store.delete_embedding(emb_id)
    assert result is True
    assert store.get_embedding(emb_id) is None


def test_store_delete_embedding_nonexistent():
    """Test that deleting a non-existent embedding returns False."""
    store = InMemoryEmbeddingStore()
    assert store.delete_embedding("nonexistent") is False


def test_store_list_embeddings():
    """Test listing embeddings with filters."""
    store = InMemoryEmbeddingStore()
    store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="test1",
    )
    store.create_embedding(
        project_id="proj-001",
        object_id="obj-002",
        text="test2",
    )
    store.create_embedding(
        project_id="proj-002",
        object_id="obj-003",
        text="test3",
    )

    # All active embeddings
    all_embs = store.list_embeddings()
    assert len(all_embs) == 3

    # Filter by project
    proj1_embs = store.list_embeddings(project_id="proj-001")
    assert len(proj1_embs) == 2

    # Filter by project
    proj2_embs = store.list_embeddings(project_id="proj-002")
    assert len(proj2_embs) == 1


def test_store_list_embeddings_limit():
    """Test that limit is respected."""
    store = InMemoryEmbeddingStore()
    for i in range(10):
        store.create_embedding(
            project_id=f"proj-{i:03d}",
            object_id=f"obj-{i:03d}",
            text=f"text-{i}",
        )

    limited = store.list_embeddings(limit=5)
    assert len(limited) == 5


def test_store_count():
    """Test counting embeddings."""
    store = InMemoryEmbeddingStore()
    store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="test1",
    )
    store.create_embedding(
        project_id="proj-001",
        object_id="obj-002",
        text="test2",
    )
    store.create_embedding(
        project_id="proj-002",
        object_id="obj-003",
        text="test3",
    )

    assert store.count() == 3
    assert store.count(project_id="proj-001") == 2
    assert store.count(project_id="proj-002") == 1


def test_store_with_custom_embedder():
    """Test that the store can use a custom embedder."""

    class MockEmbedder:
        @property
        def model_name(self) -> str:
            return "mock-embedder"

        def embed(self, text: str) -> list[float]:
            # Return a simple deterministic vector
            return [float(len(text)), float(ord(text[0]))]

    store = InMemoryEmbeddingStore(embedder=MockEmbedder())
    emb_id = store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="hello",
    )

    emb = store.get_embedding(emb_id)
    assert emb is not None
    assert emb.model == "mock-embedder"
    assert emb.vector == [5.0, 104.0]  # len("hello")=5, ord('h')=104


def test_store_search_similar_excludes_archived():
    """Test that search_similar excludes archived embeddings."""
    store = InMemoryEmbeddingStore()
    emb1_id = store.create_embedding(
        project_id="proj-001",
        object_id="obj-001",
        text="test1",
    )
    emb2_id = store.create_embedding(
        project_id="proj-001",
        object_id="obj-002",
        text="test2",
    )

    # Archive emb2
    emb2 = store.get_embedding(emb2_id)
    emb2.status = EmbeddingStatus.ARCHIVED

    # Get a query embedding
    emb1 = store.get_embedding(emb1_id)
    query_embedding = emb1.vector

    # Search for similar embeddings
    results = store.search_similar(
        query_embedding=query_embedding,
        top_k=10,
        project_id="proj-001",
    )

    # Should only return active embeddings (emb1)
    assert len(results) == 1
    assert results[0].embedding_id == emb1_id
