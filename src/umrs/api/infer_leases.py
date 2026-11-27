"""Workstation inference leases (ADR-062 amendment 2, DEC-037, S-D37-3).

    POST   /umrs/infer/leases              mint a short-lived kind=workstation lease
    DELETE /umrs/infer/leases/{lease_id}   revoke one of the caller's own

A platform-bound R4 calls a public provider through the deployment's
``umrs-infer`` instead of dialling it with a key on the workstation. It mints
a lease here (signed gateway identity), then sends provider calls to the edge
route ``/gw/infer/<session_id>/...`` with the lease token as its API key; the
edge relays them to ``umrs-infer`` ``/w/<session_id>/...``, which unseals the
provider key server-side (``umrs.infer_proxy``).

Rules (each re-checked by ``umrs-infer`` on every request, so a lease minted
through the generic ``/umrs/secrets`` API gains nothing):

- the owner is the verified caller, never a body field, admins included;
- the project is one the caller may write (``writable_project_for``);
- every source is the caller's own sealed row, named by its name;
- a public provider needs the project to read ``non_sensitive`` unless its
  host is on the site's institutional allowlist (403, refused by name);
- the lease is short (``ttl_seconds`` ≤ 3600, lifetime capped at issue, so a
  renewal cannot stretch it) and carries no egress IP: a workstation's
  address is not stable, the token and the session id are the binding.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from umrs.infer_proxy.leases import (
    WORKSTATION_KIND,
    WORKSTATION_MAX_LIFETIME_SECONDS,
    WORKSTATION_SECRET_NAME,
)
from umrs.knowledge.access_scope import CallerScope, writable_project_for
from umrs.shared.project_data_class import (
    PUBLIC_PROVIDER_KINDS,
    data_class_for_project,
    is_institutional_url,
)

from ._scope_identity import get_caller_scope
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/umrs/infer/leases",
    tags=["infer-leases"],
    dependencies=[Depends(get_caller_scope)],
)

DEFAULT_TTL_SECONDS = 900
MAX_TTL_SECONDS = WORKSTATION_MAX_LIFETIME_SECONDS
MAX_SOURCES = 8
# The edge route the client sends provider calls to (gateway-relative).
RELAY_PREFIX = "/gw/infer"


class WorkstationLeaseRequest(BaseModel):
    project: str = ""
    sources: list[str] = Field(..., min_length=1, max_length=MAX_SOURCES)
    default_source: str = ""
    ttl_seconds: int = Field(DEFAULT_TTL_SECONDS, ge=60, le=MAX_TTL_SECONDS)


class WorkstationLeaseResponse(BaseModel):
    lease_id: str
    token: str  # plaintext — returned once, never stored
    session_id: str
    relay_path: str
    project: str
    source_ids: list[str]
    default_source_id: str
    expires_at: str | None = None


def _broker() -> Any:
    from .secret_broker import get_broker

    return get_broker()


def _pool(request: Request) -> Any:
    return getattr(request.app.state, "umrs_db_pool", None)


def _sealed_store(request: Request) -> Any:
    from umrs.security.sealed_sources import SealedSourceStore, SealedStoreUnavailable

    pool = _pool(request)
    if pool is None:
        raise_typed(
            PlatformError.usersources(
                "unavailable", "workstation leases require the shared umrs_db pool"
            )
        )
    try:
        return SealedSourceStore(pool)
    except SealedStoreUnavailable as exc:
        raise_typed(PlatformError.usersources("unavailable", str(exc)), cause=exc)


def _caller_user(caller: CallerScope) -> str:
    if not caller.user_id:
        raise_typed(
            PlatformError.secretbroker(
                "unauthorized", "a workstation lease needs a verified user"
            )
        )
    return caller.user_id


def _bound_project(requested: str, caller: CallerScope) -> str:
    project = writable_project_for(requested, caller)
    if not project:
        raise_typed(
            PlatformError.secretbroker(
                "forbidden",
                f"project_refused: no write access to project {requested!r}",
                reason="project_refused",
            )
        )
    return project


def _owned_sources(store: Any, owner: str, names: list[str]) -> list[Any]:
    """The caller's live sealed rows, in request order (404 on any miss)."""
    metas = []
    for name in dict.fromkeys(names):
        meta = store.get_meta(owner, name)
        if meta is None:
            raise_typed(
                PlatformError.secretbroker(
                    "not_found", f"source {name!r} not found for the caller"
                )
            )
        metas.append(meta)
    return metas


def _needs_non_sensitive(meta: Any) -> bool:
    if meta.provider_kind not in PUBLIC_PROVIDER_KINDS:
        return False
    return not is_institutional_url(meta.allow_host)


def _refuse_class(pool: Any, project: str, metas: list[Any]) -> None:
    """ADR-062: a public provider only for a project that reads non_sensitive."""
    public = [meta.name for meta in metas if _needs_non_sensitive(meta)]
    if not public or data_class_for_project(pool, project) == "non_sensitive":
        return
    raise_typed(
        PlatformError.secretbroker(
            "forbidden",
            f"source_class_refused: project {project!r} is not non_sensitive, so "
            f"the public sources {public} are refused (ADR-062)",
            reason="source_class_refused",
        )
    )


def _default_source_id(metas: list[Any], wanted: str) -> str:
    if not wanted:
        return str(metas[0].id)
    for meta in metas:
        if wanted in (meta.name, str(meta.id)):
            return str(meta.id)
    raise_typed(
        PlatformError.secretbroker(
            "bad_request", f"default_source {wanted!r} is not one of the sources"
        )
    )


@router.post(
    "",
    response_model=WorkstationLeaseResponse,
    status_code=status.HTTP_201_CREATED,
)
def issue_workstation_lease(
    req: WorkstationLeaseRequest,
    request: Request,
    caller: CallerScope = Depends(get_caller_scope),
) -> WorkstationLeaseResponse:
    """# auth: verified"""
    owner = _caller_user(caller)
    project = _bound_project(req.project, caller)
    metas = _owned_sources(_sealed_store(request), owner, req.sources)
    _refuse_class(_pool(request), project, metas)
    session_id = f"ws-{secrets.token_hex(12)}"
    lease, token = _broker().issue_lease(
        f"workstation-{session_id}",
        WORKSTATION_SECRET_NAME,
        "inference",
        lease_seconds=req.ttl_seconds,
        tier=WORKSTATION_KIND,
        owner_id=owner,
        session_id=session_id,
        source_ids=tuple(str(meta.id) for meta in metas),
        default_source_id=_default_source_id(metas, req.default_source),
        project=project,
        max_lifetime_seconds=req.ttl_seconds,
    )
    logger.info(
        "workstation lease %s issued owner=%s project=%s sources=%d ttl=%ds",
        lease.lease_id,
        owner,
        project,
        len(metas),
        req.ttl_seconds,
    )
    return WorkstationLeaseResponse(
        lease_id=lease.lease_id,
        token=token,
        session_id=session_id,
        relay_path=f"{RELAY_PREFIX}/{session_id}",
        project=project,
        source_ids=list(lease.source_ids),
        default_source_id=lease.default_source_id,
        expires_at=lease.expires_at.isoformat() if lease.expires_at else None,
    )


@router.delete("/{lease_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_workstation_lease(
    lease_id: str, caller: CallerScope = Depends(get_caller_scope)
) -> None:
    """# auth: verified"""
    owner = _caller_user(caller)
    broker = _broker()
    lease = broker.get_lease(lease_id)
    if lease is None or lease.tier != WORKSTATION_KIND or lease.owner_id != owner:
        raise_typed(PlatformError.secretbroker("not_found", "Lease not found"))
    broker.revoke_lease(lease_id, actor=owner)
