"""Durable ``OBJECT_VERIFY`` job (ADR-115 D-2/D-5, slice MB-S2c).

One job re-reads one object inside UMRS, hashes it, and records the verdict
on the object's provenance row (``umrs.data_catalog.provenance``):

* ``verified``: the bytes match the version's ETag, the hash UMRS recorded
  when it stored them, and the hash the spec declares (if any);
* ``mismatch``: one of those differs (a tampered or corrupted object); the
  reason is named;
* a legacy object (no row) is attested on demand: it gets a row with the
  hash UMRS just computed.

Spec: ``{"bucket": ..., "key": ..., "sha256": <optional declared hash>}``.
The object is read from the store holding the bucket for the job's own
project (a bound project's store, ADR-077), the same resolver as
``DATA_INGESTION``. The submit paths admit only a bucket the caller may use
(``job_tracking_handler.spec_resource_refusal``).

A verdict is the job's result, so a ``mismatch`` is a *completed* job whose
metrics say ``sha256_state=mismatch``; the job fails only when there is no
verdict (no database, object gone, object changed while read).

The submit starts the re-read and returns (the scheduler submits one job
after another, so it must never wait on a read); the status checker
collects the verdict. A malformed spec or a missing database fails at
submit; the HTTP seam refuses a malformed spec with 400 (``api.jobs``).

Registered in ``jobs.adapters`` (a lazy lane); ``jobs.store`` is untouched.
Later slices call :func:`umrs.data_catalog.provenance.verify_object_pooled`
(selection on a legacy ref, MB-S4) or enqueue this job (intake complete,
MB-S5).
"""

from __future__ import annotations

import asyncio
import logging
import re
import weakref
from typing import Any

from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)

BACKEND_NAME = "object_verify"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_spec(spec: dict[str, Any]) -> None:
    """Raise ValueError unless *spec* names a bucket and key (and, when it
    declares one, a lowercase hex SHA-256). Shared with the submit seam."""
    for field in ("bucket", "key"):
        value = spec.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"object_verify spec.{field} is required")
    declared = spec.get("sha256", "")
    if declared and not (isinstance(declared, str) and _SHA256.match(declared)):
        raise ValueError("object_verify spec.sha256 must be 64 lowercase hex characters")


def _ref(attempt_id: str) -> str:
    """The backend ref names the attempt, never the key (a key may hold
    ``|``); the verdict is collected by the status checker."""
    return f"{BACKEND_NAME}|attempt={attempt_id}"


def parse_ref(ref: str) -> dict[str, str]:
    """``{"attempt": ...}`` from a backend ref."""
    parts = ref.split("|")[1:]
    return dict(part.split("=", 1) for part in parts if "=" in part)


def _verify(job: Job) -> Any:
    from umrs.data_catalog.provenance import provenance_pool, verify_object_pooled

    from .data_ingestion import _ingest_s3_client

    spec = job.spec or {}
    bucket, key = spec["bucket"].strip(), spec["key"].strip()
    client = _ingest_s3_client((job.project_id or "").strip(), bucket)
    return verify_object_pooled(
        provenance_pool(), client, bucket, key, declared_sha256=spec.get("sha256") or ""
    )


def _preflight(job: Job) -> None:
    """What fails at once, before any byte moves: a malformed spec, no
    provenance database."""
    from umrs.data_catalog.provenance import ObjectUnreadable, provenance_pool

    validate_spec(job.spec or {})
    if provenance_pool() is None:
        raise ObjectUnreadable("no provenance database: OBJECT_VERIFY needs one")


#: Re-reads in flight in this process, by attempt id. The scheduler awaits
#: every submit in turn, so a submit that awaited an 8 GiB re-read held every
#: other job of the front for that long (review of MB-S2c, 2026-11-27): the
#: submit starts the re-read and returns, the status checker collects it. A
#: restart loses the entry, and the checker fails the attempt for a retry.
_IN_FLIGHT: dict[str, asyncio.Task[Any]] = {}
#: At most two re-reads per event loop wait for a worker thread (the
#: provenance semaphore bounds the reads; this bounds the waiting threads).
_LOOP_SLOTS: weakref.WeakKeyDictionary[Any, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _slots() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    slots = _LOOP_SLOTS.get(loop)
    if slots is None:
        slots = _LOOP_SLOTS[loop] = asyncio.Semaphore(2)
    return slots


async def _verify_in_thread(job: Job) -> Any:
    async with _slots():
        outcome = await asyncio.to_thread(_verify, job)
    logger.info(
        "OBJECT_VERIFY %s: job=%s state=%s reason=%s",
        outcome.bucket,
        job.job_id,
        outcome.sha256_state,
        outcome.reason,
    )
    return outcome


class ObjectVerifyBackend(OrchestratorBackend):
    """Starts the re-read in a worker thread (up to 8 GiB per object) and
    returns at once: neither the scheduler nor the event loop waits on it."""

    @property
    def backend_name(self) -> str:
        return BACKEND_NAME

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        _preflight(job)
        attempt_id = str(getattr(attempt, "attempt_id", "") or job.job_id)
        _IN_FLIGHT[attempt_id] = asyncio.create_task(_verify_in_thread(job))
        return _ref(attempt_id)


def _verdict_of(task: asyncio.Task[Any]) -> AttemptCheckResult:
    """A verified or mismatched object both complete the job (the metrics
    carry the state); no verdict fails the attempt."""
    if task.cancelled():
        return AttemptCheckResult.failed("object_verify re-read was cancelled")
    error = task.exception()
    if error is not None:
        return AttemptCheckResult.failed(f"object_verify: no verdict: {error}")
    outcome = task.result()
    return AttemptCheckResult.succeeded(
        metrics={
            "sha256_state": outcome.sha256_state,
            "sha256": outcome.observed_sha256,
            "reason": outcome.reason,
            "source_kind": outcome.source_kind,
        }
    )


class ObjectVerifyStatusChecker(BackendStatusChecker):
    """Collects the verdict of a re-read this process started."""

    @property
    def job_type(self) -> JobType:
        return JobType.OBJECT_VERIFY

    async def check_status(self, attempt: JobAttempt, job: Job) -> AttemptCheckResult:
        ref = attempt.backend_ref or ""
        if not ref.startswith(f"{BACKEND_NAME}|"):
            return AttemptCheckResult.failed("Unexpected backend reference")
        attempt_id = parse_ref(ref).get("attempt", "")
        task = _IN_FLIGHT.get(attempt_id)
        if task is None:
            return AttemptCheckResult.failed(
                "object_verify re-read is not running in this process "
                "(lost to a restart): retried"
            )
        if not task.done():
            return AttemptCheckResult.running("re-reading the object")
        _IN_FLIGHT.pop(attempt_id, None)
        return _verdict_of(task)
