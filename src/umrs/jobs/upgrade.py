"""EndpointUpgradeBackend — rolling model-version flip (Q-13 Phase 1).

Pattern replica of ``EndpointProvisionBackend`` with a narrowed contract:
submit a replacement provision, wait for readiness, flip the endpoint
registry record, then cancel the deprecated instance.

Phase 1 scope: synchronous provision-then-flip. Rollback-on-health-fail,
concurrency gate, and async attempt loop land in the follow-up wave.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import JobType
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)


class EndpointUpgradeBackend(OrchestratorBackend):
    """Backend adapter for rolling endpoint upgrades.

    Copies a provision request's model path onto the replacement job and
    calls the same readiness+persist path. The flip is synchronized with
    the gateway-facing registry record so that a client observing
    ``status==ready`` on the new ID immediately routes to the new
    model_path.
    """

    def __init__(
        self,
        orchestrator,
        registry,
        port_manager,
        user_store=None,
    ) -> None:
        self.orchestrator = orchestrator
        self.registry = registry
        self.port_manager = port_manager
        self._user_store = user_store

    @property
    def backend_name(self) -> str:
        """Return orchestrator type name."""
        backend_name = getattr(self.orchestrator, "backend_name", None)
        if isinstance(backend_name, str):
            return backend_name
        backend_type = getattr(self.orchestrator, "backend_type", None)
        if hasattr(backend_type, "value"):
            return backend_type.value
        if isinstance(backend_type, str):
            return backend_type
        return "unknown"

    @property
    def job_type(self) -> JobType:
        """Return job type supported by this backend."""
        return JobType.ENDPOINT_UPGRADE

    async def submit_job(self, job, attempt_id: str) -> str:
        """Submit a rolling upgrade attempt.

        Phase 1 keep-alive: this raises NotImplementedError so the
        factory's JobType branch is dead unless a dedicated upgrade
        handler exists. The UMRS api route below rejects ENDPOINT_UPGRADE
        submissions with 501 until the flip orchestration lands.
        """
        raise NotImplementedError(
            "EndpointUpgradeBackend.submit_job is reserved for follow-up wave: "
            "rollback-on-health-fail, concurrency gate, and flip orchestration "
            "not yet implemented"
        )

    async def check_status(self, job, attempt_id: str) -> dict[str, Any]:
        """Report current upgrade progress."""
        raise NotImplementedError(
            "EndpointUpgradeBackend.check_status is reserved for follow-up wave"
        )

    async def cancel_job(self, job, attempt_id: str) -> bool:
        """Cancel an in-flight upgrade (deprecated instance remains live)."""
        # In-flight upgrades are idempotent per attempt: cancellation is a
        # no-op until the flip lands; the old model keeps serving.
        return True
