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

Registered in ``jobs.adapters`` (a lazy lane); ``jobs.store`` is untouched.
Later slices call :func:`umrs.data_catalog.provenance.verify_object_pooled`
(selection on a legacy ref, MB-S4) or enqueue this job (intake complete,
MB-S5).
"""

from __future__ import annotations

import asyncio
import logging
import re
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


def _ref(outcome: Any) -> str:
    """The backend ref: the verdict, never the key (a key may hold ``|``)."""
    return (
        f"{BACKEND_NAME}|state={outcome.sha256_state}|sha256={outcome.observed_sha256}"
        f"|reason={outcome.reason or ''}|source={outcome.source_kind}"
    )


def parse_ref(ref: str) -> dict[str, str]:
    """``{"state", "sha256", "reason", "source"}`` from a backend ref."""
    parts = ref.split("|")[1:]
    return dict(part.split("=", 1) for part in parts if "=" in part)


def _verify(job: Job) -> Any:
    from umrs.data_catalog.provenance import provenance_pool, verify_object_pooled

    from .data_ingestion import _ingest_s3_client

    spec = job.spec or {}
    validate_spec(spec)
    bucket, key = spec["bucket"].strip(), spec["key"].strip()
    client = _ingest_s3_client((job.project_id or "").strip(), bucket)
    return verify_object_pooled(
        provenance_pool(), client, bucket, key, declared_sha256=spec.get("sha256") or ""
    )


class ObjectVerifyBackend(OrchestratorBackend):
    """Synchronous re-read in a worker thread (up to 8 GiB per object; the
    event loop and the API workers stay free)."""

    @property
    def backend_name(self) -> str:
        return BACKEND_NAME

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        outcome = await asyncio.to_thread(_verify, job)
        logger.info(
            "OBJECT_VERIFY %s: job=%s state=%s reason=%s",
            outcome.bucket,
            job.job_id,
            outcome.sha256_state,
            outcome.reason,
        )
        return _ref(outcome)


class ObjectVerifyStatusChecker(BackendStatusChecker):
    """The verdict is in the ref: a verified or mismatched object both
    complete the job, and the metrics carry the state."""

    @property
    def job_type(self) -> JobType:
        return JobType.OBJECT_VERIFY

    async def check_status(self, attempt: JobAttempt, job: Job) -> AttemptCheckResult:
        ref = attempt.backend_ref or ""
        if not ref.startswith(f"{BACKEND_NAME}|"):
            return AttemptCheckResult.failed("Unexpected backend reference")
        verdict = parse_ref(ref)
        return AttemptCheckResult.succeeded(
            metrics={
                "sha256_state": verdict.get("state", ""),
                "sha256": verdict.get("sha256", ""),
                "reason": verdict.get("reason") or None,
                "source_kind": verdict.get("source", ""),
            }
        )
