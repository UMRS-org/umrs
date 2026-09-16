"""Quota status read model (ADR-051 D-2/D-3).

One honest summary route: the verified caller sees their own two resource
poles — CPU-process units (live agent jobs) and disk bytes (catalog-attributed
unique content). Read-only, no accounting decisions here; caps come straight
from the same env knobs the enforcement edges read, so the number shown is
exactly the number enforced.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ._scope_identity import get_caller_scope
from .agent_quota import agent_max_processes, count_active_agent_units
from .disk_ledger import DiskLedger, disk_bytes_max
from .typed_errors import PlatformError, raise_typed


class PoleRow(BaseModel):
    current: int = Field(..., description="Live usage right now")
    limit: int = Field(..., description="Resolved site cap (0 = uncapped)")


class QuotaUnitsResponse(BaseModel):
    user_id: str
    cpu_process: PoleRow
    disk_bytes: PoleRow


router = APIRouter(prefix="/umrs/quota", tags=["quota"])


def _pool(request: Request) -> Any:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.quota("unavailable", "database pool unavailable"))
    return pool


@router.get("/units", response_model=QuotaUnitsResponse)
def quota_units(request: Request) -> QuotaUnitsResponse:
    """Per-user gauge for both ADR-051 caps (caller-attributed, no fabrication).

    # auth: verified
    # Caller-attributed; anonymous refused (see body).

    The route requires a verified identity: in strict mode the shared scope
    layer answers anonymous callers with 401 before this body runs; in mvp
    (dev) mode an anonymous scope arrives with no user — refuse it with the
    same canonical code the other verified lanes emit. An anonymous gauge
    would be fabricated data.
    """
    caller = get_caller_scope(request)
    if not caller.user_id:
        raise_typed(
            PlatformError.scopeidentity(
                "unauthorized", "verified identity required for quota summary"
            )
        )
    pool = _pool(request)
    cpu_current = count_active_agent_units(pool, user_id=caller.user_id)
    conn = pool.getconn()
    try:
        disk_usage = DiskLedger(conn).usage(caller.user_id)
    finally:
        pool.putconn(conn)
    return QuotaUnitsResponse(
        user_id=caller.user_id,
        cpu_process=PoleRow(current=cpu_current, limit=agent_max_processes()),
        disk_bytes=PoleRow(current=disk_usage, limit=disk_bytes_max()),
    )


__all__ = ["router"]
