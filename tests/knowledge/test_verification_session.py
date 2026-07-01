"""Tests for the verification session (M3 Phase 3)."""

from __future__ import annotations

from umrs.knowledge.verification_session import (
    VerificationSession,
    VerificationSessionStore,
)


def test_verification_session_defaults():
    """Test VerificationSession defaults."""
    session = VerificationSession()

    assert session.session_id != ""  # Auto-generated
    assert session.project_id == ""
    assert session.question == ""
    assert session.evidence_ids == []
    assert session.graph_traversal == []
    assert session.reasoning_context is None
    assert session.timestamp != ""  # Auto-generated
    assert session.answer == ""
    assert session.confidence == 0.0


def test_verification_session_to_persistent_dict():
    """Test that to_persistent_dict excludes the ephemeral answer."""
    session = VerificationSession(
        project_id="proj-001",
        question="Explain my project",
        evidence_ids=["paper-001", "code-001"],
        graph_traversal=["paper-001", "code-001"],
        answer="This is the LLM-generated answer",
        confidence=0.85,
    )

    data = session.to_persistent_dict()

    # Persistent data should be present
    assert data["session_id"] == session.session_id
    assert data["project_id"] == "proj-001"
    assert data["question"] == "Explain my project"
    assert data["evidence_ids"] == ["paper-001", "code-001"]
    assert data["graph_traversal"] == ["paper-001", "code-001"]
    assert data["confidence"] == 0.85

    # Ephemeral answer should NOT be in persistent dict
    assert "answer" not in data


def test_verification_session_to_dict_includes_answer():
    """Test that to_dict includes the ephemeral answer."""
    session = VerificationSession(
        project_id="proj-001",
        question="Explain my project",
        answer="This is the LLM-generated answer",
    )

    data = session.to_dict()

    assert data["answer"] == "This is the LLM-generated answer"


def test_verification_session_store_save_and_get():
    """Test saving and getting a verification session."""
    store = VerificationSessionStore()
    session = VerificationSession(
        project_id="proj-001",
        question="Explain my project",
        evidence_ids=["paper-001"],
        graph_traversal=["paper-001"],
        answer="LLM answer",
        confidence=0.9,
    )

    session_id = store.save(session)
    retrieved = store.get(session_id)

    assert retrieved is not None
    assert retrieved["question"] == "Explain my project"
    assert retrieved["evidence_ids"] == ["paper-001"]
    # Answer should NOT be persisted
    assert "answer" not in retrieved


def test_verification_session_store_get_nonexistent():
    """Test getting a non-existent session returns None."""
    store = VerificationSessionStore()
    assert store.get("nonexistent") is None


def test_verification_session_store_list_sessions():
    """Test listing sessions with filters."""
    store = VerificationSessionStore()
    store.save(VerificationSession(
        session_id="s1", project_id="p1", question="Q1",
    ))
    store.save(VerificationSession(
        session_id="s2", project_id="p1", question="Q2",
    ))
    store.save(VerificationSession(
        session_id="s3", project_id="p2", question="Q3",
    ))

    all_sessions = store.list_sessions()
    assert len(all_sessions) == 3

    proj1_sessions = store.list_sessions(project_id="p1")
    assert len(proj1_sessions) == 2


def test_verification_session_store_delete():
    """Test deleting a session."""
    store = VerificationSessionStore()
    session = VerificationSession(
        session_id="s1", project_id="p1", question="Q1",
    )
    store.save(session)

    result = store.delete("s1")
    assert result is True
    assert store.get("s1") is None


def test_verification_session_store_delete_nonexistent():
    """Test deleting a non-existent session returns False."""
    store = VerificationSessionStore()
    assert store.delete("nonexistent") is False
