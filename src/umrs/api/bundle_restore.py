"""K4: synchronous .kb bundle restore route.

`POST /umrs/knowledge/restore` — restores an exported knowledge bundle
from a server-side directory after manifest integrity verification
(when `UMRS_SECRET_KEY` configures HMAC). All restores are idempotent
(re-runnable for DR repair); rows are individually fault-isolated.

Synchronous by design, mirroring the export route's in-process model;
a durable-job variant can layer on later without changing the engine.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from umrs.knowledge.graph_registry import default_graph_id

from ._knowledge_mutation_guard import (
    enforce_graph_mutation_authorization,
    mark_proposal_executed,
    request_actor,
    request_session_id,
)
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
    proposal_id: str | None = Field(
        default=None,
        description="Approved graph-extension proposal (required for agent actors, 0069)",
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


@router.post("/restore", response_model=dict[str, Any])
async def restore_bundle(request: Request, body: RestoreRequest) -> dict[str, Any]:
    """K4: restore a .kb bundle with manifest verification.

    Restores write persistent graph content, so agent actors
    (``x-r4-actor: agent``) need an approved graph-extension proposal
    targeting the target project's default graph (0069 guard); human
    restores keep the pre-existing behavior.
    """
    from umrs.api._asyncpg_pool import knowledge_asyncpg_pool

    actor = request_actor(request)
    if actor == "agent":
        target = default_graph_id(_resolve_target_project(body))
        enforce_graph_mutation_authorization(
            request, target_graph_id=target, proposal_id=body.proposal_id
        )

    pool = await knowledge_asyncpg_pool(request)

    from umrs.knowledge.bundle_import import BundleFormatError, KnowledgeImportEngine
    from umrs.knowledge.store import KnowledgeStore

    with _reference_stores(request) as ref_stores:
        engine = KnowledgeImportEngine(
            store=KnowledgeStore(pool),
            secret_key=os.getenv("UMRS_SECRET_KEY", ""),
            **ref_stores,
        )
        try:
            result = await engine.import_project(
                body.bundle_dir,
                target_project_id=body.project_id,
                verify=body.verify,
                allow_unsigned=body.allow_unsigned,
            )
        except BundleFormatError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if actor == "agent":
        mark_proposal_executed(
            request,
            body.proposal_id,
            actor=actor,
            session_id=request_session_id(request),
            operation="knowledge_restore",
        )

    return {"status": "ok", "restore": result.to_dict()}
