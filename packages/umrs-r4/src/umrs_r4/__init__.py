"""umrs-r4: the R4 runtime hosted by UMRS, as a UMRS extension (ADR-117 §6).

Core ``umrs`` never imports this package; the core loader finds it through the
``umrs.extensions`` entry point and calls :func:`register`.

Contents:
- U3 S2: the MCP knowledge-verifier adapter (``umrs_r4.verifiers``), no hook.
- U3 S4b: the companion trust ledger (``umrs_r4.trust``): its routes, its
  startup, the session-archive participant (ADR-035 item 8) and the approval
  listener (D7).
- U3 S5: the remote-session, venue and plan-run cluster (ADR-046 A2,
  ADR-092, ADR-099): the companion session / venue / admin-session / plan-run
  routes, the ``frontend_session`` job lane, the ``plan`` loop iteration kind,
  the plan-loop host startup, ``umrs_r4.venues``, the session cap.
"""

from __future__ import annotations

from typing import Any

# The ExtensionContext API version this extension targets (checked at load).
EXTENSION_API_VERSION = 1


def init_trust_store(db_pool: Any) -> None:
    """The pool-backed trust store (schema ensured by core beforehand)."""
    from umrs_r4.trust.access import set_trust_store
    from umrs_r4.trust.store import TrustAllowlistStore

    # Borrow per call, never a boot lease (http_sidecar._init_mcp_store).
    set_trust_store(TrustAllowlistStore(db_pool))


def init_loop_plan_host(db_pool: Any) -> None:
    """ADR-092 slice 8b: the server-hosted loop's plan-run host (the venue
    verbs and the proposal lane). Unwired, a plan loop's iteration stops
    by name (``plan_host_unavailable``), never runs anything else."""
    from umrs_r4.api.plan_run_venue import build_loop_plan_host
    from umrs_r4.knowledge import plan_iteration

    plan_iteration.configure(build_loop_plan_host(db_pool))


def _routers(ext: Any) -> None:
    from umrs_r4.api.admin_sessions import router as admin_sessions_router
    from umrs_r4.api.companion_chat_approvals import router as chat_approvals_router
    from umrs_r4.api.companion_sessions import router as companion_sessions_router
    from umrs_r4.api.companion_venue import router as companion_venue_router
    from umrs_r4.api.companion_venue_parity import router as venue_parity_router
    from umrs_r4.api.plan_run_traces import router as plan_run_traces_router
    from umrs_r4.api.plan_run_venue import router as plan_run_venue_router
    from umrs_r4.api.trust_routes import router as trust_router

    # Remote sessions and the frontend-venue runtime + worker rail (wave
    # 2026-09-21, ADR-046 A2); ADR-099 D-8 questions + agent settings.
    ext.add_router(companion_sessions_router, family="companion")
    ext.add_router(chat_approvals_router, family="companion")
    ext.add_router(companion_venue_router, family="companion")
    ext.add_router(venue_parity_router, family="companion")
    ext.add_router(admin_sessions_router, family="admin")  # RS-0 C-1
    ext.add_router(plan_run_traces_router, family="traces")  # ADR-092 slice 7
    ext.add_router(plan_run_venue_router, family="traces")  # ADR-092 slice 8
    ext.add_router(trust_router, family="companion")  # wave T3c, ADR-035


def _job_types(ext: Any) -> None:
    from umrs.extensions.context import JobTypeRegistration

    from umrs_r4.jobs.frontend_session_backend import build_frontend_session_lane

    # ADR-046 A2: remote companion sandboxes; the lane is None where the venue
    # is not configured (UMRS_VENUE_ENABLED), and the claim then skips the rows.
    ext.add_job_type(
        JobTypeRegistration(value="frontend_session", build_lane=build_frontend_session_lane)
    )


def register(ext: Any) -> None:
    """Register the r4 extension's parts with the core ``ExtensionContext``."""
    from umrs_r4.api.loop_plan_iteration import plan_kind
    from umrs_r4.trust.approval_listener import mint_session_trust_on_always
    from umrs_r4.trust.archive_participant import revoke_trust_on_archive
    from umrs_r4.venues.host_sessions import build_host_sessions

    _routers(ext)
    _job_types(ext)
    ext.add_loop_iteration_kind(plan_kind())  # ADR-092 slice 8b
    ext.add_startup(init_trust_store)
    ext.add_startup(init_loop_plan_host)
    ext.add_session_archive_participant(revoke_trust_on_archive)
    ext.add_approval_listener(mint_session_trust_on_always)
    ext.add_host_sessions(build_host_sessions)  # SH-H1a hosts janitor
