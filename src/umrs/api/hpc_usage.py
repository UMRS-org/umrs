"""The caller's own HPC usage at this site (ADR-112 wave D; plan §6 C/D.0 D4).

``GET /umrs/sites/me/hpc/usage?project=&since=&limit=``: the caller's
current ledger rows at this venue and their totals. Served in both access
modes (no ``mode_undeclared``): work on the platform login is ledgered too.
The caller is always the verified gateway identity; nobody reads another
person's rows here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ._verified_headers import CallerIdentity, get_verified_caller
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/sites/me/hpc", tags=["sites"])


def _venue() -> str:
    from umrs.shared.hpc_user_work_access import current_site

    return str(getattr(current_site(), "name", "") or "")


def _ledger() -> Any:
    from umrs.hpc_accounts.usage_ledger import LedgerUnavailable, get_usage_ledger

    try:
        return get_usage_ledger()
    except LedgerUnavailable as exc:
        raise_typed(PlatformError.hpcaccess("ledger_unavailable", str(exc)), cause=exc)


def _since(text: str | None) -> Any:
    from umrs.hpc_accounts.usage_reports import parse_since

    try:
        return parse_since(text)
    except ValueError:
        raise_typed(PlatformError.projects("bad_request", f"since: not an ISO date or time: {text!r}"))


@router.get("/usage")
def my_hpc_usage(
    project: str | None = None,
    since: str | None = None,
    limit: int = 200,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """The caller's own usage rows at this site and their totals.

    # auth: verified
    """
    from umrs.hpc_accounts.usage_reports import my_usage

    return my_usage(
        _ledger(), _venue(), caller.user_id, project=project, since=_since(since), limit=limit
    )


__all__ = ["router"]
