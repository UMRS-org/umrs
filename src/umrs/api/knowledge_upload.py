"""Bytes-to-central-ingest upload (WP-15, dual-corpus dwell).

Context: rag facade uploads used to proxy bytes to the DEPLOYED mini-rivault
RAG endpoint (endpoint corpus). The ADR-027/WP-15 target is the central
node-keyed corpus, and the gap's remaining leg was "upload facade →
knowledge_ingest ownership move". The owner decision (2026-09-11) is
**dual-corpus dwell**: old endpoint-corpus rows stay readable as long-
standing minted data; NEW uploads route into the central durable ingest
plane (this module).

Route shape: accept bytes → stage at
``knowledge-uploads/<project>/<uuid>/<basename>`` in the configured S3
bucket → submit a KNOWLEDGE_INGEST durable job with an ``s3_uri`` source
(native kind in the ingest source registry). Returns 202 + job_id.
Failure semantics: s3 staging or runtime unavailability = honest 503;
nothing is silently dual-written (a "both routes succeeded" mash was the
pre-dwell ambiguity this leg closes).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Request,
    UploadFile,
)
from fastapi import status as http_status

from ..shared.storage.s3_config import get_s3_config
from ._knowledge_mutation_guard import (
    enforce_graph_mutation_authorization,
    request_actor,
)
from .files import get_project_id
from .governance import guard_ingest
from .knowledge_v2 import _resolve_ingest_target
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)

_BUCKET_PREFIX = "knowledge-uploads"
_SAFE_BASENAME_MAX = 200


def _safe_basename(name: str) -> str:
    """Strip path components + cap length (uploaded name is caller-shaped)."""
    base = (name or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = base.replace("\x00", "").strip() or "upload.bin"
    return base[:_SAFE_BASENAME_MAX]


_KNOWLEDGE_UPLOAD_MAX_BYTES = 512 * 1024 * 1024


def _refuse_oversize_upload(size: int | None) -> None:
    """Typed 413 when a known upload size passes the knowledge upload cap."""
    if isinstance(size, int) and size > _KNOWLEDGE_UPLOAD_MAX_BYTES:
        raise_typed(
            PlatformError.knowledge2(
                "upload_too_large",
                f"Upload exceeds {_KNOWLEDGE_UPLOAD_MAX_BYTES // (1024 * 1024)} MiB limit",
            )
        )


class _UploadTooLarge(Exception):
    """The streamed part passed the cap (a part whose size was not known)."""


class _StagingReader:
    """File-like view boto3 stages the part through: sha256 and size as it
    streams, refused once the cap is passed. Only one chunk is held at a time."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self.sha256 = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self.size += len(chunk)
        if self.size > _KNOWLEDGE_UPLOAD_MAX_BYTES:
            raise _UploadTooLarge
        self.sha256.update(chunk)
        return chunk


async def _stage_upload_bytes(
    file: UploadFile, *, bucket: str, staged_key: str, client: Any
) -> tuple[str, int]:
    """Size-check + stream the part to the object store (venue faults are
    honest 503s, oversize 413s).

    Returns ``(content_sha256, byte_size)`` of the staged payload so the
    per-user disk ledger (ADR-051 D-3) can charge the uploader once per
    unique object. The part is streamed, never read whole: ``await
    file.read()`` held the upload in memory (cycle-3 N4: a 400 MiB
    /umrs/rag/upload, +412 MiB peak RSS).
    """
    # The part's size first: refusing only after read() held the whole
    # oversize body in memory (2026-10-21: a 700 MiB upload, +683 MiB RSS).
    _refuse_oversize_upload(getattr(file, "size", None))
    reader = _StagingReader(file.file)
    try:
        await asyncio.to_thread(
            client.upload_fileobj,
            reader,
            bucket,
            staged_key,
            ExtraArgs={"ContentType": file.content_type or "application/octet-stream"},
        )
    except _UploadTooLarge:
        _cleanup_staged_bytes(client=client, bucket=bucket, staged_key=staged_key)
        _refuse_oversize_upload(reader.size)
    except Exception as exc:  # venue fault: fail honest, never silent-fallback
        raise_typed(
            PlatformError.knowledge2("unavailable", f"S3 staging unavailable: {exc}"),
            cause=exc,
        )
    return reader.sha256.hexdigest(), reader.size


def _cleanup_staged_bytes(*, client: Any, bucket: str, staged_key: str) -> None:
    """Best-effort delete of bytes staged for a refused upload.

    Never masks the typed quota refusal — a cleanup failure is logged and
    swallowed (the follow-up sweep still exists for catastrophic residue).
    """
    try:
        client.delete_object(Bucket=bucket, Key=staged_key)
    except Exception:  # noqa: BLE001 — never mask the typed quota refusal; sweep owns residue
        logger.warning("staged-bytes cleanup failed for refused upload: %s", staged_key)


async def _submit_upload_ingest(
    *,
    bucket: str,
    staged_key: str,
    basename: str,
    content_type: str,
    proposal_id: str | None,
    project_id: str,
    target_graph_id: str,
    user_id: str = "",
    origin_actor: str = "user",
) -> dict[str, Any]:
    """Submit the KNOWLEDGE_INGEST job against the staged key (503 on no runtime).

    ``origin_actor`` rides the spec to the understand chain (D-35, K1.1(a)):
    an agent upload, admitted only with an approved proposal, lands its
    graph rows ``proposed``, never ``established``. The actor is the
    ``x-r4-actor`` header, so this holds for clients that set it (K-12).
    """
    from umrs.jobs.models import JobType
    from umrs.server.durable_job_runtime import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise_typed(
            PlatformError.knowledge2(
                "unavailable",
                (
                    "UMRS durable job runtime not available "
                    "(staged object retained for operator cleanup)"
                ),
            )
        )
    spec: dict[str, Any] = {
        "sources": [
            {
                "kind": "s3_uri",
                "uri": f"s3://{bucket}/{staged_key}",
                "filename": basename,
                "content_type": content_type,
            }
        ],
        "allow_domains": [],
        "project_id": project_id,
        "graph_id": target_graph_id,
        "source": "knowledge_upload",
        "origin_actor": origin_actor,
    }
    if proposal_id:
        spec["proposal_id"] = proposal_id
    try:
        # The uploader owns the job (and its chained understand/embed jobs):
        # an ownerless job is invisible to `r4 job list` and 403 on describe.
        job_id = await runtime.submit_job_sync(
            job_type=JobType.KNOWLEDGE_INGEST,
            project_id=project_id,
            spec=spec,
            user_id=user_id,
        )
    except Exception as exc:
        raise_typed(
            PlatformError.knowledge2("failure", f"Ingest submit failed: {exc}"),
            cause=exc,
        )
    logger.info(
        "knowledge upload staged → ingest queued: %s (key=%s graph=%s)",
        job_id,
        staged_key,
        target_graph_id,
    )
    return {
        "status": "queued",
        "job_id": job_id,
        "job_type": JobType.KNOWLEDGE_INGEST.value,
        "s3_key": staged_key,
        "graph_id": target_graph_id,
        "dwell_contract": "central-write/endpoint-read-only",
    }


@router.post("/upload", status_code=http_status.HTTP_202_ACCEPTED)
async def knowledge_upload(
    request: Request,
    file: UploadFile = File(...),
    graph_id: str | None = None,
    project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Stage bytes → central durable ingest (dual-corpus dwell contract).
    # auth: verified

    202 with the durable job id; the job writes knowledge_nodes/edges into
    the resolved graph (default: the project's default graph).
    """
    # 1. Identity + filename sanity.
    if not (file.filename or "").strip():
        raise_typed(PlatformError.knowledge2("bad_request", "Filename required"))

    user_id = request.headers.get("x-user-id", "")

    # 2. Ingest governance + mutation authority (same gates as /ingest).
    guard_ingest(
        project_id,
        "knowledge.upload",
        user_id=user_id,
        details={"filename": file.filename},
    )
    target_graph_id = _resolve_ingest_target(request, project_id, graph_id)
    enforce_graph_mutation_authorization(
        request,
        target_graph_id=target_graph_id,
        proposal_id=request.headers.get("x-proposal-id"),
    )

    # 3. Stage bytes → charge per-user disk ledger (ADR-051 D-3) → submit.
    config = get_s3_config()
    basename = _safe_basename(file.filename)
    staged_key = f"{_BUCKET_PREFIX}/{project_id}/{uuid4()}/{basename}"
    content_hash, byte_size = await _stage_upload_bytes(
        file,
        bucket=config.bucket,
        staged_key=staged_key,
        client=config.get_client(),
    )
    # Disk budget: per-user, charged once per unique content hash, refusal
    # is typed quota.disk_bytes_exceeded. Staged bytes are cleaned up on
    # refusal — otherwise an attacker could inflate shared S3 storage by
    # deliberately exceeding their own quota (follow-up closed 2026-09-15).
    if user_id:
        from umrs.api.disk_ledger import DiskLedger

        pool = getattr(request.app.state, "umrs_db_pool", None)
        if pool is not None:
            conn = pool.getconn()
            try:
                DiskLedger(conn).charge(
                    user_id, content_hash=content_hash, byte_size=byte_size
                )
            except Exception:
                _cleanup_staged_bytes(
                    client=config.get_client(), bucket=config.bucket, staged_key=staged_key
                )
                raise
            finally:
                pool.putconn(conn)
    return await _submit_upload_ingest(
        bucket=config.bucket,
        staged_key=staged_key,
        basename=basename,
        content_type=file.content_type or "application/octet-stream",
        proposal_id=request.headers.get("x-proposal-id"),
        project_id=project_id,
        target_graph_id=target_graph_id,
        user_id=user_id,
        origin_actor=request_actor(request),
    )
