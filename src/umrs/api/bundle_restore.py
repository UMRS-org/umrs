"""K4: synchronous .kb bundle restore + import routes.

`POST /umrs/knowledge/restore` — restores an exported knowledge bundle
from a server-side directory after manifest integrity verification
(when `UMRS_SECRET_KEY` configures HMAC); admin-only operator facility.
`POST /umrs/knowledge/import` — the user-facing twin: the bundle arrives
as a ``.kb.tar.gz`` upload and lands in the caller's own project (D-40).
All restores are idempotent (re-runnable for DR repair); rows are
individually fault-isolated.

Synchronous by design, mirroring the export route's in-process model;
a durable-job variant can layer on later without changing the engine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import IO, Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from umrs.knowledge.graph_registry import default_graph_id

from ._knowledge_mutation_guard import (
    enforce_graph_mutation_authorization,
    mark_proposal_executed,
    request_actor,
    request_session_id,
)
from .files import get_project_id
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge-restore"])


class RestoreRequest(BaseModel):
    """Bundle restore request."""

    bundle_dir: str = Field(
        ...,
        min_length=1,
        description="Server-side path to the bundle directory (contains manifest.json)",
    )
    project_id: str | None = Field(
        default=None,
        description="Retarget all restored rows to this project (defaults to the bundle's own project)",
    )
    verify: bool = Field(
        default=True,
        description="Verify manifest hash + HMAC before restoring (needs UMRS_SECRET_KEY)",
    )
    allow_unsigned: bool = Field(
        default=False,
        description="Restore an unsigned/unverifiable bundle anyway (explicit opt-out)",
    )
    write_upgrade: bool = Field(
        default=False,
        description="Rewrite a 1.x bundle's manifest at v1.2 after restore (D-41; drops the stale signature)",
    )
    proposal_id: str | None = Field(
        default=None,
        description="Approved graph-extension proposal (required for agent actors, 0069)",
    )


def _require_operator(request: Request) -> None:
    """Server-path restore is an operator facility: ``bundle_dir`` names a
    directory on the UMRS host (read with server rights; ``write_upgrade``
    rewrites its manifest) and ``project_id`` retargets into any project —
    so only admins (or the legacy anonymous dev scope) may call it. A
    user-facing import takes the bundle's bytes, not a server path."""
    from umrs.api._scope_identity import request_caller_scope

    scope = request_caller_scope(request)
    if not (scope.is_admin or scope.anonymous):
        raise_typed(
            PlatformError.bundle(
                "forbidden",
                "bundle restore from a server-side directory is admin-only "
                "(bundle_dir resolves on the UMRS host, not your workstation)",
            )
        )


def _resolve_target_project(body: RestoreRequest) -> str:
    """Project whose default graph the restore mutates (explicit retarget wins)."""
    if body.project_id:
        return body.project_id
    manifest_path = os.path.join(body.bundle_dir, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise_typed(
            PlatformError.bundle(
                "bad_request",
                f"cannot resolve restore target: unreadable manifest ({exc})",
            ),
            cause=exc,
        )
    project = manifest.get("project_id")
    if not project:
        raise_typed(
            PlatformError.bundle(
                "bad_request",
                "bundle manifest has no project_id — pass project_id explicitly",
            ),
        )
    return str(project)


class _CatalogRestoreAdapter:
    """Owns the CatalogSpec shape for the import engine (ADR-053 fence:
    ``umrs.knowledge`` never imports ``umrs.api``)."""

    def __init__(self, store) -> None:  # type: ignore[no-untyped-def]
        from .scientific_catalogs import ScientificCatalogStore

        self._store: ScientificCatalogStore = store

    def register_from_bundle(self, spec: dict, *, owner_id: str) -> dict:
        from .scientific_catalogs import CatalogSpec

        return self._store.create(spec=CatalogSpec(**spec), owner_id=owner_id)


def _reference_stores(request: Request):
    """Skill/MCP stores for the V-05 reference-consumption leg (context manager).

    One pooled psycopg2 connection for the life of the ``with`` block,
    always returned (the 2026-09-09 exhaustion class). Absent pool yields
    an empty dict — the engine counts the refs as deferred, never crashes.
    """
    from contextlib import contextmanager

    @contextmanager
    def _provide():
        pool = getattr(request.app.state, "umrs_db_pool", None)
        if pool is None or not hasattr(pool, "getconn"):
            yield {}
            return
        from umrs.mcp.store import MCPServerStore
        from umrs.skills.store import SkillStore

        from .scientific_catalogs import ScientificCatalogStore

        conn = pool.getconn()
        try:
            yield {
                "skill_store": SkillStore(conn),
                "mcp_store": MCPServerStore(conn),
                "catalog_store": _CatalogRestoreAdapter(ScientificCatalogStore(conn)),
            }
        finally:
            pool.putconn(conn)

    return _provide()


async def _import_bundle_dir(
    request: Request,
    pool: Any,
    bundle_dir: str,
    *,
    target_project_id: str | None,
    verify: bool,
    allow_unsigned: bool,
    write_upgrade: bool,
):
    """Run the import engine over one bundle directory (restore + import)."""
    from umrs.knowledge.bundle_import import BundleFormatError, KnowledgeImportEngine
    from umrs.knowledge.store import KnowledgeStore

    with _reference_stores(request) as ref_stores:
        engine = KnowledgeImportEngine(
            store=KnowledgeStore(pool),
            secret_key=os.getenv("UMRS_SECRET_KEY", ""),
            **ref_stores,
        )
        try:
            return await engine.import_project(
                bundle_dir,
                target_project_id=target_project_id,
                verify=verify,
                allow_unsigned=allow_unsigned,
                write_upgrade=write_upgrade,
            )
        except BundleFormatError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/restore", response_model=dict[str, Any])
async def restore_bundle(request: Request, body: RestoreRequest) -> dict[str, Any]:
    """K4: restore a .kb bundle with manifest verification.

    Restores write persistent graph content, so agent actors
    (``x-r4-actor: agent``) need an approved graph-extension proposal
    targeting the target project's default graph (0069 guard); human
    restores keep the pre-existing behavior.
    """
    from umrs.api._asyncpg_pool import knowledge_asyncpg_pool

    _require_operator(request)
    actor = request_actor(request)
    if actor == "agent":
        target = default_graph_id(_resolve_target_project(body))
        enforce_graph_mutation_authorization(
            request, target_graph_id=target, proposal_id=body.proposal_id
        )

    pool = await knowledge_asyncpg_pool(request)
    result = await _import_bundle_dir(
        request,
        pool,
        body.bundle_dir,
        target_project_id=body.project_id,
        verify=body.verify,
        allow_unsigned=body.allow_unsigned,
        write_upgrade=body.write_upgrade,
    )

    if actor == "agent":
        mark_proposal_executed(
            request,
            body.proposal_id,
            actor=actor,
            session_id=request_session_id(request),
            operation="knowledge_restore",
        )

    return {"status": "ok", "restore": result.to_dict()}


# ── D-40 transport: import a bundle by its bytes (user-facing) ───────────────

_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _stage_archive(source: IO[bytes], dest: Path) -> None:
    """Copy the upload's spooled body to a private temp file (size-capped)."""
    written = 0
    with open(dest, "wb") as out:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            if written > _MAX_ARCHIVE_BYTES:
                raise_typed(
                    PlatformError.bundle(
                        "bad_request",
                        f"bundle archive exceeds {_MAX_ARCHIVE_BYTES // 2**30} GiB",
                    )
                )
            out.write(chunk)


def _unpack(archive: Path, dest: Path) -> Path:
    from umrs.knowledge.bundle_archive import BundleArchiveError, safe_extract

    try:
        with open(archive, "rb") as handle:
            return safe_extract(handle, dest)
    except BundleArchiveError as exc:
        raise_typed(PlatformError.bundle("bad_request", str(exc)), cause=exc)


@router.post("/import", response_model=dict[str, Any])
async def import_bundle_archive(
    request: Request,
    file: UploadFile = File(...),
    allow_unsigned: bool = False,
    project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Import a ``.kb.tar.gz`` bundle into the caller's own project.
    # auth: verified

    The user-facing half of "sharing = export/import": the archive's bytes
    arrive in the request (never a server path), unpack into a private
    temporary directory (hostile-input safe), and every row retargets to the
    caller's verified project. The manifest is verified; an unsigned or
    foreign-key bundle needs the explicit ``allow_unsigned=true`` opt-out
    (recorded as unverified in the result). Agent actors need an approved
    graph-extension proposal (``x-proposal-id``), as for every write.
    """
    import asyncio
    import tempfile

    from umrs.api._asyncpg_pool import knowledge_asyncpg_pool

    from .governance import guard_ingest

    actor = request_actor(request)
    proposal_id = request.headers.get("x-proposal-id")
    if actor == "agent":
        enforce_graph_mutation_authorization(
            request,
            target_graph_id=default_graph_id(project_id),
            proposal_id=proposal_id,
        )
    guard_ingest(
        project_id,
        "knowledge.import",
        user_id=request.headers.get("x-user-id", ""),
        details={"filename": file.filename, "allow_unsigned": allow_unsigned},
    )
    pool = await knowledge_asyncpg_pool(request)
    with tempfile.TemporaryDirectory(prefix="kb-import-") as tmp:
        archive = Path(tmp) / "bundle.kb.tar.gz"
        await asyncio.to_thread(_stage_archive, file.file, archive)
        unpacked = Path(tmp) / "unpacked"
        unpacked.mkdir()
        root = await asyncio.to_thread(_unpack, archive, unpacked)
        result = await _import_bundle_dir(
            request,
            pool,
            str(root),
            target_project_id=project_id,
            verify=True,
            allow_unsigned=allow_unsigned,
            write_upgrade=False,
        )
    if actor == "agent":
        mark_proposal_executed(
            request,
            proposal_id,
            actor=actor,
            session_id=request_session_id(request),
            operation="knowledge_import",
        )
    return {"status": "ok", "project_id": project_id, "import": result.to_dict()}
