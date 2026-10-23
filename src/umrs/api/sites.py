"""The caller's view of this site (multi-HPC plan S2, 2026-10-24).

``GET /umrs/sites/me`` answers one question for a verified caller: what can
*I* do at this site? It returns

* the site's non-secret capabilities (the same facts ``/health`` already
  carries in ``site_capabilities``, plus whether the staged step lane and its
  GPU gate are open);
* the site's declared HPC access mode (``scheduler.hpc_access``, added by the
  access-modes work, S1b). Until a site declares one the answer is
  ``"undeclared"`` and clients must not route work here from another site;
* the caller's **own** user row at this venue: present, disabled or absent,
  and whether a Slurm account is set. Never another user's row, never the
  account name, never a key path or consent reference.

``/health`` stays unauthenticated and unchanged (plan §3.8): everything
caller-specific lives only behind this verified route.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ._scope_identity import get_caller_scope
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/sites", tags=["sites"])

UNDECLARED = "undeclared"


class SiteAccess(BaseModel):
    mode: str = Field(
        UNDECLARED, description="The site's default mode, or 'undeclared'"
    )
    declared: bool = False
    modes_allowed: list[str] = Field(default_factory=list)
    operator: str | None = None


class SiteAccount(BaseModel):
    status: str = Field(..., description="present | disabled | absent | unknown")
    slurm_account: bool = Field(
        False, description="A Slurm account is set for this site"
    )


class SiteCapabilities(BaseModel):
    scheduler_kind: str | None = None
    staged_lane: bool = False
    gpu: bool = False


class SiteMeResponse(BaseModel):
    site: str | None = None
    user_id: str
    access: SiteAccess
    account: SiteAccount
    capabilities: SiteCapabilities


def _load_site() -> Any:
    """This server's site config, or None (a site without one runs unscoped)."""
    try:
        from umrs.orchestrator.site_config import load_site_config

        return load_site_config()
    except Exception:  # noqa: BLE001 — no site.yaml is a valid deployment
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def access_of(site: Any) -> SiteAccess:
    """The declared ``scheduler.hpc_access`` block, or ``undeclared``.

    Declared means a non-empty ``modes_allowed`` and a ``default_mode`` in it
    (the S1b contract, AM §4.1); anything less is reported as undeclared.
    """
    hpc = _as_mapping(getattr(getattr(site, "scheduler", None), "hpc_access", None))
    allowed = [str(m) for m in hpc.get("modes_allowed") or [] if str(m)]
    default = str(hpc.get("default_mode") or "")
    operator = str(hpc.get("operator") or "") or None
    if not allowed or default not in allowed:
        return SiteAccess(modes_allowed=allowed, operator=operator)
    return SiteAccess(
        mode=default, declared=True, modes_allowed=allowed, operator=operator
    )


def capabilities_of(site: Any) -> SiteCapabilities:
    if site is None:
        return SiteCapabilities()
    step = getattr(site, "step_executor", None)
    return SiteCapabilities(
        scheduler_kind=str(site.scheduler.kind),
        staged_lane=bool(getattr(step, "remote_enabled", False)),
        gpu=bool(
            getattr(step, "remote_enabled", False) and getattr(step, "gpu", False)
        ),
    )


def _user_store() -> Any:
    from umrs.admin.user_store import UserStore

    return UserStore()


def account_of(user_id: str, site_name: str | None) -> SiteAccount:
    """The caller's own row at this venue (PG-first, JSON fallback)."""
    from umrs.admin.user_store import UserStoreError

    try:
        record = _user_store().resolve_user(user_id)
    except UserStoreError:
        return SiteAccount(status="absent")
    except Exception:  # an unreadable store is unknown, never absent
        logger.warning("user row unreadable for sites/me", exc_info=True)
        return SiteAccount(status="unknown")
    status = "disabled" if record.disabled else "present"
    return SiteAccount(
        status=status, slurm_account=bool(record.resolve_slurm_account(site_name))
    )


@router.get("/me", response_model=SiteMeResponse)
def sites_me(request: Request) -> SiteMeResponse:
    """The verified caller's access, account row and capabilities here.

    # auth: verified
    # Caller-attributed; anonymous refused (see body).
    """
    caller = get_caller_scope(request)
    if not caller.user_id:
        raise_typed(
            PlatformError.scopeidentity(
                "unauthorized", "verified identity required for the site summary"
            )
        )
    site = _load_site()
    name = str(getattr(site, "name", "") or "") or None
    return SiteMeResponse(
        site=name,
        user_id=caller.user_id,
        access=access_of(site),
        account=account_of(caller.user_id, name),
        capabilities=capabilities_of(site),
    )
