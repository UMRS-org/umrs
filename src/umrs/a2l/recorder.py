"""A2lRecorder — narrow event port between the execution spine and a2l.

The execution spine (jobs/attempt lifecycle handlers) emits recorder events
*in addition to* its existing behavior; it never reads a2l tables. The spine
should hold a module-level ``NullA2lRecorder`` by default and have the real
recorder injected at startup when a store pool is available — a2l must stay
degradable, never a hard dependency of job progress (fail-closed on writes
is acceptable; fail-blocking job execution is not).

Event keying contract (ADR-034): every event carries ``job_id`` and
``attempt_id``; artifact events also carry the deterministic content hash.
The recorder persists conserved *additions* only — it does not copy execution
state.

Loop affinity (2026-10-20): the sidecar builds the asyncpg pool on the
uvicorn loop, but the durable job runtime drives workflow checkers on its
own loop thread. asyncpg connections are loop-affine — awaited from the
wrong loop they logged "Future attached to a different loop", "another
operation is in progress", "connection was closed in the middle of
operation" (ops-watch 2026-10-17/18). ``StoreA2lRecorder`` therefore bounces
store calls to the pool's owner loop when the caller runs elsewhere.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class A2lRecorder(Protocol):
    """Port the execution spine calls to conserve execution history."""

    async def record_attempt_artifact(
        self,
        *,
        job_id: str,
        attempt_id: str,
        artifact_uri: str,
        artifact_type: str,
        user_id: str,
        content_hash: str,
        project_id: str | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        """Conserve one attempt-produced artifact identity. Returns a2l id or None."""
        ...

    async def record_decision(
        self,
        *,
        decided_by: str,
        decision: str,
        rationale: str = "",
        job_id: str = "",
        attempt_id: str = "",
        artifact_id: str = "",
        payload: dict | None = None,
    ) -> str | None:
        """Conserve one human/operator decision. Returns decision id or None."""
        ...


class NullA2lRecorder:
    """No-op recorder — default when a2l store is unavailable."""

    async def record_attempt_artifact(self, **kwargs: object) -> None:
        return None

    async def record_decision(self, **kwargs: object) -> None:
        return None


# ── Process-global recorder ────────────────────────────────────────────────
# Callees (job checkers, route handlers) always take the global; composition
# (http_sidecar lifespan) swaps in a StoreA2lRecorder when a DSN exists.
# Default is Null — a2l must never become a hard dependency of job progress.

_DEFAULT_RECORDER: A2lRecorder = NullA2lRecorder()


def get_recorder() -> A2lRecorder:
    """Return the process-global recorder (Null unless configured)."""
    return _DEFAULT_RECORDER


def configure_recorder(recorder: A2lRecorder) -> None:
    """Install the process-global recorder (sidecar startup path)."""
    global _DEFAULT_RECORDER
    _DEFAULT_RECORDER = recorder


class StoreA2lRecorder:
    """Recorder backed by the durable ConservationStore (asyncpg)."""

    def __init__(
        self, store: object, owner_loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        # typed loosely to avoid a hard import cycle with store module state
        self._store = store
        # The loop the store's asyncpg pool belongs to; store calls from any
        # other loop bounce through run_coroutine_threadsafe (module docstring).
        self._owner_loop = owner_loop

    async def _on_owner_loop(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Await *coro* on the pool's loop, wherever the caller runs."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        owner = self._owner_loop
        if owner is None or running is owner or not owner.is_running():
            return await coro
        future = asyncio.run_coroutine_threadsafe(coro, owner)
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=30)

    async def record_attempt_artifact(self, **kwargs: object) -> str | None:
        from datetime import datetime, timezone

        from .models import ArtifactDerive, ArtifactRecord
        from .store import compute_artifact_id

        kw = dict(kwargs)
        try:
            parents = list(kw.pop("parents", []) or [])
            operation = str(kw.pop("operation", "job_artifact"))
            job_id = str(kw["job_id"])
            attempt_id = str(kw["attempt_id"])
            if parents:
                derive = ArtifactDerive(
                    parents=parents,
                    operation=operation,
                    type=str(kw["artifact_type"]),
                    uri=str(kw["artifact_uri"]),
                    user_id=str(kw["user_id"]),
                    input_data_hash=str(kw["content_hash"]),
                    metadata=dict(kw.pop("metadata", None) or {}),
                    project_id=kw.get("project_id"),  # type: ignore[arg-type]
                    job_id=job_id,
                    attempt_id=attempt_id,
                )
                record = await self._on_owner_loop(self._store.derive_artifact(derive))  # type: ignore[attr-defined]
            else:
                # Parent-less conservation (ingest-style): direct row.
                record = await self._on_owner_loop(
                    self._store.create_artifact(  # type: ignore[attr-defined]
                        ArtifactRecord(
                            id=compute_artifact_id(
                                str(kw["content_hash"]), operation, str(kw["artifact_type"])
                            ),
                            type=str(kw["artifact_type"]),
                            uri=str(kw["artifact_uri"]),
                            user_id=str(kw["user_id"]),
                            metadata=dict(kw.pop("metadata", None) or {}),
                            project_id=kw.get("project_id"),  # type: ignore[arg-type]
                            job_id=job_id,
                            attempt_id=attempt_id,
                            created_at=datetime.now(timezone.utc).isoformat(),
                        )
                    )
                )
            return record.id
        except Exception:
            logger.warning(
                "a2l recorder: artifact conservation failed (job %s)",
                kw.get("job_id"),
                exc_info=True,
            )
            return None

    async def record_decision(self, **kwargs: object) -> str | None:
        from .models import DecisionCreate

        kw = dict(kwargs)
        try:
            record = await self._on_owner_loop(
                self._store.create_decision(  # type: ignore[attr-defined]
                    DecisionCreate(
                        decided_by=str(kw["decided_by"]),
                        decision=str(kw["decision"]),
                        rationale=str(kw.get("rationale") or ""),
                        job_id=str(kw.get("job_id") or ""),
                        attempt_id=str(kw.get("attempt_id") or ""),
                        artifact_id=str(kw.get("artifact_id") or ""),
                        payload=dict(kw.get("payload") or {}),
                    )
                )
            )
            return record.id
        except Exception:
            logger.warning("a2l recorder: decision conservation failed", exc_info=True)
            return None
