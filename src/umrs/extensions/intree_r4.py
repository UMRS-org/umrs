"""Transitional: the r4 runtime parts still in core, registered as extension r4.

U3 S3 proves the seams while the code is still in-tree (u3a map §6): these
parts no longer mount or start from ``http_sidecar`` directly; they register
here through the same ExtensionContext hooks the umrs-r4 package uses. S4 and
S5 move each part into ``umrs_r4`` and delete its line here; S5 empties this
module (and drops the loader's transitional ``r4`` default).
"""

from __future__ import annotations

from typing import Any

from umrs.extensions.context import ExtensionContext


def _routers(ext: ExtensionContext) -> None:
    from umrs.api.admin_sessions import router as admin_sessions_router
    from umrs.api.companion_chat_approvals import router as chat_approvals_router
    from umrs.api.companion_sessions import router as companion_sessions_router
    from umrs.api.companion_venue import router as companion_venue_router
    from umrs.api.companion_venue_parity import router as venue_parity_router
    from umrs.api.plan_run_venue import router as plan_run_venue_router

    # S5 cluster: remote sessions and the frontend-venue runtime + worker rail
    # (wave 2026-09-21, ADR-046 A2), admin sessions (RS-0 C-1), plan-run venue.
    ext.add_router(companion_sessions_router, family="companion")
    ext.add_router(chat_approvals_router, family="companion")
    ext.add_router(companion_venue_router, family="companion")
    # ADR-099 D-8 (RS-1 lane A): remote ask_user questions + agent settings.
    ext.add_router(venue_parity_router, family="companion")
    ext.add_router(admin_sessions_router, family="admin")
    ext.add_router(plan_run_venue_router, family="traces")  # ADR-092 slice 8


def init_loop_plan_host(db_pool: Any) -> None:
    """ADR-092 slice 8b: the server-hosted loop's plan-run host (the venue
    verbs and the proposal lane). Unwired, a plan loop's iteration stops
    by name (``plan_host_unavailable``), never runs anything else."""
    from umrs.api.plan_run_venue import build_loop_plan_host
    from umrs.knowledge import plan_iteration

    plan_iteration.configure(build_loop_plan_host(db_pool))


def register(ext: ExtensionContext) -> None:
    _routers(ext)
    ext.add_startup(init_loop_plan_host)
