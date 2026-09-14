"""Project state machine — explicit lifecycle for ResearchProject.

M3 Phase 1: Persistent Scientific Model.

The project state machine makes the project lifecycle explicit and
enables event-driven automation.

States:
  Created → Uploading → Ingesting → Knowledge Building →
  Verification → Ready → Planning → Running → Paused →
  Completed → Published

Each state transition is validated. Invalid transitions raise an error.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "ProjectState",
    "InvalidStateTransitionError",
    "ProjectStateMachine",
]


class ProjectState(str, Enum):
    """States of a ResearchProject."""

    CREATED = "created"
    UPLOADING = "uploading"
    INGESTING = "ingesting"
    KNOWLEDGE_BUILDING = "knowledge_building"
    VERIFICATION = "verification"
    READY = "ready"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    PUBLISHED = "published"


class InvalidStateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        from_state: ProjectState,
        to_state: ProjectState,
    ) -> None:
        super().__init__(f"Invalid transition: {from_state.value} → {to_state.value}")
        self.from_state = from_state
        self.to_state = to_state


class ProjectStateMachine:
    """State machine for ResearchProject.

    Enforces valid state transitions. Each transition can trigger
    callbacks for event-driven automation.

    Valid transitions:
      Created         → Uploading
      Uploading       → Ingesting
      Ingesting       → KnowledgeBuilding
      KnowledgeBuilding → Verification
      Verification    → Ready
      Ready           → Planning
      Planning        → Running
      Running         → Paused | Completed
      Paused          → Running | Completed
      Completed       → Published
    """

    # Valid state transitions
    _TRANSITIONS: dict[ProjectState, set[ProjectState]] = {
        ProjectState.CREATED: {
            ProjectState.UPLOADING,
        },
        ProjectState.UPLOADING: {
            ProjectState.INGESTING,
            ProjectState.CREATED,
        },
        ProjectState.INGESTING: {
            ProjectState.KNOWLEDGE_BUILDING,
            ProjectState.UPLOADING,
        },
        ProjectState.KNOWLEDGE_BUILDING: {
            ProjectState.VERIFICATION,
            ProjectState.INGESTING,
        },
        ProjectState.VERIFICATION: {
            ProjectState.READY,
            ProjectState.KNOWLEDGE_BUILDING,
        },
        ProjectState.READY: {
            ProjectState.PLANNING,
            ProjectState.VERIFICATION,
        },
        ProjectState.PLANNING: {
            ProjectState.RUNNING,
            ProjectState.READY,
        },
        ProjectState.RUNNING: {
            ProjectState.PAUSED,
            ProjectState.COMPLETED,
            ProjectState.PLANNING,
        },
        ProjectState.PAUSED: {
            ProjectState.RUNNING,
            ProjectState.COMPLETED,
        },
        ProjectState.COMPLETED: {
            ProjectState.PUBLISHED,
            ProjectState.RUNNING,
        },
        ProjectState.PUBLISHED: set(),
    }

    def __init__(self, initial_state: ProjectState = ProjectState.CREATED) -> None:
        self._state = initial_state
        self._history: list[tuple[ProjectState, ProjectState]] = []

    @property
    def state(self) -> ProjectState:
        """Current state."""
        return self._state

    @property
    def history(self) -> list[tuple[ProjectState, ProjectState]]:
        """Transition history as (from, to) pairs."""
        return self._history

    def can_transition(self, to_state: ProjectState) -> bool:
        """Check if a transition is valid."""
        return to_state in self._TRANSITIONS.get(self._state, set())

    def transition(self, to_state: ProjectState) -> ProjectState:
        """Transition to a new state.

        Args:
            to_state: The target state

        Returns:
            The new state

        Raises:
            InvalidStateTransitionError: If the transition is invalid
        """
        if not self.can_transition(to_state):
            raise InvalidStateTransitionError(self._state, to_state)
        self._history.append((self._state, to_state))
        self._state = to_state
        return self._state

    def reset(self, state: ProjectState = ProjectState.CREATED) -> None:
        """Reset the state machine."""
        self._state = state
        self._history = []
