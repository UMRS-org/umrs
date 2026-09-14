"""Tests for the project state machine (M3 Phase 1)."""

from __future__ import annotations

import pytest
from umrs.project_state import (
    InvalidStateTransitionError,
    ProjectState,
    ProjectStateMachine,
)


def test_project_state_enum():
    """Test ProjectState enum values."""
    assert ProjectState.CREATED == "created"
    assert ProjectState.UPLOADING == "uploading"
    assert ProjectState.INGESTING == "ingesting"
    assert ProjectState.KNOWLEDGE_BUILDING == "knowledge_building"
    assert ProjectState.VERIFICATION == "verification"
    assert ProjectState.READY == "ready"
    assert ProjectState.PLANNING == "planning"
    assert ProjectState.RUNNING == "running"
    assert ProjectState.PAUSED == "paused"
    assert ProjectState.COMPLETED == "completed"
    assert ProjectState.PUBLISHED == "published"


def test_state_machine_init():
    """Test state machine initialization."""
    sm = ProjectStateMachine()
    assert sm.state == ProjectState.CREATED


def test_state_machine_init_custom():
    """Test state machine with custom initial state."""
    sm = ProjectStateMachine(initial_state=ProjectState.READY)
    assert sm.state == ProjectState.READY


def test_valid_transition():
    """Test a valid state transition."""
    sm = ProjectStateMachine()
    sm.transition(ProjectState.UPLOADING)
    assert sm.state == ProjectState.UPLOADING


def test_invalid_transition():
    """Test that an invalid transition raises an error."""
    sm = ProjectStateMachine()
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(ProjectState.RUNNING)  # Can't go Created → Running


def test_full_lifecycle():
    """Test the full project lifecycle."""
    sm = ProjectStateMachine()

    # Created → Uploading → Ingesting → Knowledge Building →
    # Verification → Ready → Planning → Running → Completed → Published
    sm.transition(ProjectState.UPLOADING)
    sm.transition(ProjectState.INGESTING)
    sm.transition(ProjectState.KNOWLEDGE_BUILDING)
    sm.transition(ProjectState.VERIFICATION)
    sm.transition(ProjectState.READY)
    sm.transition(ProjectState.PLANNING)
    sm.transition(ProjectState.RUNNING)
    sm.transition(ProjectState.COMPLETED)
    sm.transition(ProjectState.PUBLISHED)

    assert sm.state == ProjectState.PUBLISHED


def test_pause_resume():
    """Test pausing and resuming."""
    sm = ProjectStateMachine()

    # Get to RUNNING state
    sm.transition(ProjectState.UPLOADING)
    sm.transition(ProjectState.INGESTING)
    sm.transition(ProjectState.KNOWLEDGE_BUILDING)
    sm.transition(ProjectState.VERIFICATION)
    sm.transition(ProjectState.READY)
    sm.transition(ProjectState.PLANNING)
    sm.transition(ProjectState.RUNNING)

    # Pause
    sm.transition(ProjectState.PAUSED)
    assert sm.state == ProjectState.PAUSED

    # Resume
    sm.transition(ProjectState.RUNNING)
    assert sm.state == ProjectState.RUNNING


def test_history():
    """Test that transition history is recorded."""
    sm = ProjectStateMachine()
    sm.transition(ProjectState.UPLOADING)
    sm.transition(ProjectState.INGESTING)

    assert len(sm.history) == 2
    assert sm.history[0] == (ProjectState.CREATED, ProjectState.UPLOADING)
    assert sm.history[1] == (ProjectState.UPLOADING, ProjectState.INGESTING)


def test_can_transition():
    """Test can_transition method."""
    sm = ProjectStateMachine()
    assert sm.can_transition(ProjectState.UPLOADING) is True
    assert sm.can_transition(ProjectState.RUNNING) is False


def test_reset():
    """Test reset method."""
    sm = ProjectStateMachine()
    sm.transition(ProjectState.UPLOADING)
    sm.reset()
    assert sm.state == ProjectState.CREATED
    assert len(sm.history) == 0


def test_published_is_terminal():
    """Test that PUBLISHED is a terminal state."""
    sm = ProjectStateMachine()
    # Get to PUBLISHED
    for state in [
        ProjectState.UPLOADING,
        ProjectState.INGESTING,
        ProjectState.KNOWLEDGE_BUILDING,
        ProjectState.VERIFICATION,
        ProjectState.READY,
        ProjectState.PLANNING,
        ProjectState.RUNNING,
        ProjectState.COMPLETED,
        ProjectState.PUBLISHED,
    ]:
        sm.transition(state)

    # Can't transition from PUBLISHED
    assert sm.can_transition(ProjectState.RUNNING) is False
    assert sm.can_transition(ProjectState.CREATED) is False
