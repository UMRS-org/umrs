"""UMRS Companion HTTP surface: the core router (wave T3b m5).

Routes under ``/umrs/companion`` (deployed as ``/gw/companion/*`` behind NGINX
+ auth sidecar). Core serves the pending-approval inbox over durable rows
(ADR-036 (a), WP-05; ``api/approvals.py``, mounted here): owner-scoped list +
terminal decisions that parked loop drivers consume; another user's approval
reads as 404.

U3 S4a (D6, 2026-11-15 count: one ``companion_worker`` job ever): the worker
lifecycle (``/workers``) and turn-attempt (``/chat``, ``/turns``) surfaces are
retired; the 0070/0071 tables stay in core migrations and sit idle.

U3 S4b: the trust allowlist (``/trust*``) and the session archive tombstone
(``/sessions/{session_id}/archive``) are the r4 extension's
(``umrs_r4.api.trust_routes``), mounted through ``add_router``.

Strict-mode identity gate: ``identity.strict_user_header``, the sidecar's
signed identity tuple only, per ADR-027's claim-carrier line.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from .identity import strict_user_header

logger = logging.getLogger(__name__)

# Core since U3 S1 item 8; the same object, so dependency overrides still apply.
_strict_headers = strict_user_header

router = APIRouter(prefix="/umrs/companion", tags=["companion"])

# ── tool-approval inbox (core since U3 S1 item 8) ───────────────────────────
from .approvals import register as _register_approvals

_register_approvals(router)
