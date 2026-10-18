"""UMRS API routes for M3.5 verification findings: retired (410 Gone).

Retired 2026-10-18 (hardening plan H-10.1): ``POST /umrs/verification/chat``
and ``GET /umrs/verification/sessions[/<id>]``. The chat reasoned over the
module-global M3.5 graph (project "default", shared with the retired
knowledge explorer) whatever project the caller had, so a project B caller
read project "default"'s nodes and graph paths. Its controller had no
session store, so the sessions routes always answered ``[]`` / ``null``.
No client calls them and the edge has no ``/gw/verification`` location.

Retired 2026-10-18 (H-10.1 follow-up): ``POST /umrs/verification/chat/resolve``,
``GET /umrs/verification/findings`` and ``PATCH /umrs/verification/findings/<key>``
(TASK-046). Their store was injected by ``set_store``, which nothing called,
so all three answered 503. Wiring it would not have made them work:

- the TUI's resolve (``packages/r4-tui-v3/src/stores/agent-runtime.ts``)
  names report ``"manual"``, and ``verification_findings.report_id`` is a
  foreign key to ``verification_reports``, which has no such row (the insert
  fails, a 500); its key hashes a question the TUI answered locally, so it
  names no finding a verification run produced;
- the two findings routes were admin-only and projectless copies of the
  per-project lane, ``GET`` / ``PATCH /umrs/knowledge/findings``, which reads
  the caller's readable project and writes only a writable one.

The routes still resolve the caller first (strict mode: 401 without the
signed tuple), then answer 410 with that pointer. Decision and evidence:
``docs/development/dormant-modules.md``, "Module-global per-project state".
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ._verified_headers import CallerIdentity, get_caller, get_verified_caller

router = APIRouter(prefix="/umrs/verification", tags=["verification"])

_FINDINGS_GONE = (
    "The /umrs/verification findings routes are retired: findings belong to a "
    "project's verification reports. List them with GET "
    "/umrs/knowledge/findings?project_id=<project> and move one with PATCH "
    "/umrs/knowledge/findings/<finding_key> "
    '{"from_status", "new_status", "evidence"}.'
)


def _findings_gone() -> None:
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_FINDINGS_GONE)


@router.post("/chat/resolve", status_code=status.HTTP_410_GONE)
async def resolve_finding(
    _caller: CallerIdentity | None = Depends(get_caller),
) -> None:
    """Retired: a chat-resolved finding had no report and no project."""
    _findings_gone()


@router.get("/findings", status_code=status.HTTP_410_GONE)
async def list_findings(
    _caller: CallerIdentity | None = Depends(get_caller),
) -> None:
    """Retired: use the per-project ``GET /umrs/knowledge/findings``."""
    _findings_gone()


@router.patch("/findings/{finding_key}", status_code=status.HTTP_410_GONE)
async def transition_finding(
    finding_key: str,
    _caller: CallerIdentity = Depends(get_verified_caller),
) -> None:
    """Retired: use ``PATCH /umrs/knowledge/findings/{finding_key}``."""
    del finding_key
    _findings_gone()
