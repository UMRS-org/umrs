"""
HTTP sidecar for the UMRS server.

Serves the endpoint registry REST API alongside the ZMQ server,
allowing the Gateway to discover active endpoints via HTTP.

Usage:
    from umrs.server.http_sidecar import start_http_sidecar

    thread = start_http_sidecar(port=8082)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import pathlib
import threading
import time
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.staticfiles import StaticFiles

from ..api.web_dashboard import router as web_dashboard_router

if TYPE_CHECKING:
    from umrs.admin.account_cascade import AccountStandingSweep

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8082


def _is_tunnel_mode() -> bool:
    """Return True when UMRS_TUNNEL_MODE is enabled."""
    return os.environ.get("UMRS_TUNNEL_MODE", "false").lower() in ("true", "1", "yes")


def _resolve_bind_host(host: str | None) -> str:
    """Resolve the HTTP sidecar bind address.

    In tunnel mode (UMRS_TUNNEL_MODE=true) the sidecar MUST bind to
    127.0.0.1 only — matching the ZMQ invariant in Platform Contract §8.1/§8.4.
    An explicit ``UMRS_HTTP_HOST`` env var can override the default but is
    still clamped to loopback in tunnel mode.
    """
    env_host = os.environ.get("UMRS_HTTP_HOST")
    if _is_tunnel_mode():
        if env_host and env_host.strip() not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "UMRS_TUNNEL_MODE=true — ignoring UMRS_HTTP_HOST=%s, binding 127.0.0.1",
                env_host,
            )
        return "127.0.0.1"
    if env_host:
        return env_host.strip()
    if host is not None:
        return host
    return "127.0.0.1"


def _build_db_pool(app):
    """Create the shared Postgres pool and attach it to app.state."""
    try:
        from umrs.server.job_tracking_handler import (
            build_durable_job_db_pool_from_env,
        )

        db_pool = build_durable_job_db_pool_from_env()
        app.state.umrs_db_pool = db_pool
        return db_pool
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning("Shared DB pool unavailable in HTTP sidecar")
        return None


def _start_durable_runtime() -> None:
    """Start the process-owned durable jobs runtime (best-effort).

    ``start_durable_job_runtime_from_env()`` is a no-op when no orchestrator
    backend (Slurm/K8s/Docker) is detected. Without this call the durable-job
    dispatch scheduler never starts on this deployment and durable jobs
    (workflow_run, knowledge_*, data_ingestion) queue forever.
    """
    try:
        from umrs.server.durable_job_runtime import start_durable_job_runtime_from_env

        start_durable_job_runtime_from_env()
    except Exception:
        logger.warning("Durable job runtime start skipped", exc_info=True)


def _init_catalog_meta_store(db_pool):
    """Initialize the catalog metadata store."""
    from umrs.api.catalog_store import init_catalog_meta_store

    if init_catalog_meta_store(db_pool) is None:
        logger.warning(
            "Catalog store unavailable in HTTP sidecar — /umrs/catalog/* will return 503"
        )


def _init_job_store(db_pool):
    """Initialize the UMRS job store."""
    try:
        from umrs.api.jobs import set_job_store
        from umrs.jobs.store import JobStore

        set_job_store(JobStore(db_pool))
        logger.info("Job store initialized for HTTP sidecar")
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning(
            "Job store unavailable in HTTP sidecar — /umrs/jobs/* will return 503"
        )


def _init_model_store(app, db_pool):
    """Initialize the model registry store."""
    try:
        from umrs.api.models import set_model_store
        from umrs.model_registry import (
            ensure_schema as ensure_model_registry_schema,
        )
        from umrs.model_registry.store import PathValidator, PooledModelStore

        _ensure_schema_on_borrowed_conn(db_pool, ensure_model_registry_schema)
        # WP-06 r2 P10-C3 (console badges): wire the grant-reader seam so the
        # ACL's read_reason surface isn't scope-only — without it every listing
        # stays a scope query (badge chips never carry 'grant' on live rows).
        from umrs.grants import GrantStore
        from umrs.model_registry.store import ACLEnforcer

        def _read_grants_for_acl(asset_kind: str, asset_id: str):
            """Borrow-and-return per call (fail-open grants absent = no grants)."""
            if db_pool is None:
                return []
            conn = db_pool.getconn()
            try:
                return GrantStore(conn).list_for_asset(asset_kind, asset_id)
            finally:
                db_pool.putconn(conn)

        acl = ACLEnforcer(grant_reader=_read_grants_for_acl)
        # Borrow per call, never a boot lease (see _init_mcp_store): the leased
        # connection sat idle in a read transaction until the R-14 reaper
        # killed it, and the next /umrs/models or endpoint-provision lookup
        # 500'd (rccs/bdrt 2026-10-07, found by Ops Watch).
        set_model_store(
            PooledModelStore(db_pool, path_validator=PathValidator(), acl_enforcer=acl)
        )
        logger.info("Model store initialized for HTTP sidecar")
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning(
            "Model store unavailable in HTTP sidecar — /umrs/models/* will return 503"
        )


def _init_mcp_store(app, db_pool):
    """Initialize the MCP server store."""
    try:
        from umrs.api.mcp import set_mcp_store
        from umrs.mcp import ensure_schema as ensure_mcp_schema
        from umrs.mcp.store import MCPServerStore

        schema_conn = db_pool.getconn()
        try:
            ensure_mcp_schema(schema_conn)
            schema_conn.commit()
        finally:
            db_pool.putconn(schema_conn)
        # Borrow per call, never a boot lease: the leased connection sat idle
        # in a read transaction until the R-14 reaper killed it and every
        # /umrs/mcp/* call 500'd (live bdrt 2026-09-27).
        mcp_store = MCPServerStore(pool=db_pool)
        set_mcp_store(mcp_store)
        # ADR-053: the jobs-lane D-22 registry rides the same store instance
        # (installed on the jobs adapters module, never an api import).
        from umrs.jobs.adapters import set_mcp_registry

        set_mcp_registry(mcp_store)
        logger.info("MCP server store initialized for HTTP sidecar")
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning(
            "MCP store unavailable in HTTP sidecar — /umrs/mcp/* will return 503"
        )


def _ensure_schema_on_borrowed_conn(db_pool, ensure_schema) -> None:
    """Run a store's boot DDL on a borrowed connection and hand it back."""
    schema_conn = db_pool.getconn()
    try:
        ensure_schema(schema_conn)
        schema_conn.commit()
    finally:
        db_pool.putconn(schema_conn)


def _init_audit_store(app, db_pool):
    """Initialize the audit store."""
    try:
        from umrs.api.audit_events import set_audit_store
        from umrs.api.audit_store import AuditStore, ensure_audit_schema

        _ensure_schema_on_borrowed_conn(db_pool, ensure_audit_schema)
        # Borrow per call, never a boot lease (see _init_mcp_store).
        set_audit_store(AuditStore(pool=db_pool))
        logger.info("Audit store initialized for HTTP sidecar")
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning("Audit store unavailable — using in-memory buffer only")


def _init_governance_store(app, db_pool):
    """Initialize the governance store."""
    try:
        from umrs.api.governance import (
            GovernanceStore,
            ensure_governance_schema,
            set_governance_store,
        )

        gov_conn = db_pool.getconn()
        ensure_governance_schema(gov_conn)
        gov_conn.commit()
        app.state.umrs_governance_store_conn = gov_conn
        gov_store = GovernanceStore(gov_conn, pool=db_pool)
        app.state.umrs_governance_store = gov_store
        set_governance_store(gov_store)
        logger.info("Governance store initialized for HTTP sidecar")
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning("Governance store unavailable — quota enforcement disabled")


def _init_research_sessions_store(app, db_pool):
    """Initialize the research-session store (ADR-025)."""
    try:
        from umrs.api.research_sessions import set_session_store
        from umrs.research_sessions import ensure_schema
        from umrs.research_sessions.store import SessionStore

        _ensure_schema_on_borrowed_conn(db_pool, ensure_schema)
        # Borrow per call, never a boot lease (see _init_mcp_store): on the
        # leased connection no write ever committed on Postgres, and R-14's
        # reaper rolled them back (2026-09-27).
        set_session_store(SessionStore(db_pool))
        logger.info("Research session store initialized for HTTP sidecar")
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning(
            "Research session store unavailable in HTTP sidecar — "
            "/umrs/research/sessions/* will return 503"
        )


def _init_deep_stores(app, db_pool):
    """Pool-backed durable stores that live behind one umbrella call.

    Kept in a helper to hold ``_create_app`` under the McCabe gate (9).
    Order matters for boot logs only; each init is independently fallible.
    """
    _init_job_store(db_pool)
    _init_catalog_meta_store(db_pool)
    _init_model_store(app, db_pool)
    _init_mcp_store(app, db_pool)
    _init_audit_store(app, db_pool)
    _init_governance_store(app, db_pool)
    _init_research_sessions_store(app, db_pool)
    _init_companion_trust_store(app, db_pool)
    _init_usage_pool(db_pool)
    _init_loop_plan_host(db_pool)
    _init_skill_store(db_pool)


def _init_skill_store(db_pool) -> None:
    """Skill registry on the ``skills`` table (H-10.1, 2026-10-18).

    The REST layer served an in-memory registry: registrations died with the
    process and the D-27 proposals the knowledge jobs write into ``skills``
    never reached ``/umrs/skills/pending``. Borrows per call (see
    ``_init_mcp_store``); the table comes from migrations 0052/0112.
    """
    try:
        from umrs.api.skills import set_skill_store

        set_skill_store(db_pool)
        logger.info("Skill registry backed by Postgres in HTTP sidecar")
    except Exception:  # noqa: BLE001 — boot must survive per-component failures
        logger.warning("Skill store unavailable — in-memory skill registry")


def _init_loop_plan_host(db_pool) -> None:
    """ADR-092 slice 8b: the server-hosted loop's plan-run host (the venue
    verbs and the proposal lane). Unwired, a plan loop's iteration stops
    by name (``plan_host_unavailable``), never runs anything else."""
    try:
        from umrs.api.plan_run_venue import build_loop_plan_host
        from umrs.knowledge import plan_iteration

        plan_iteration.configure(build_loop_plan_host(db_pool))
    except Exception:  # boot must survive per-component failures
        logger.warning("plan-loop host unavailable in HTTP sidecar", exc_info=True)


def _init_companion_trust_store(app, db_pool):
    """Initialize the companion trust allowlist store (wave T3c, ADR-035)."""
    try:
        from umrs.api.companion import set_trust_store
        from umrs.companion_trust import ensure_schema
        from umrs.companion_trust.store import TrustAllowlistStore

        _ensure_schema_on_borrowed_conn(db_pool, ensure_schema)
        # Borrow per call, never a boot lease (see _init_mcp_store); the trust
        # store and the approval inbox used to share one leased connection.
        set_trust_store(TrustAllowlistStore(db_pool))
        logger.info("Companion trust store initialized for HTTP sidecar")
        _init_approval_inbox(db_pool)
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning(
            "Companion trust store unavailable in HTTP sidecar — "
            "/umrs/companion/trust* will return 503; approval inbox stays process-local"
        )


def _init_approval_inbox(db_pool) -> None:
    """Durable approval inbox (ADR-036 (a), WP-05) on the shared pool.

    Failure leaves the process-local inbox in place — visible as
    ``durability: process`` on the approvals route, never silent.
    """
    try:
        from umrs.companion_trust.approval_inbox import ApprovalInboxStore
        from umrs.knowledge import tool_approvals

        tool_approvals.configure(ApprovalInboxStore(db_pool))
        logger.info("Companion approval inbox initialized (postgres)")
    except Exception:
        logger.warning(
            "Approval inbox unavailable — staying process-local", exc_info=True
        )


def _init_usage_pool(db_pool):
    """Wire the LLM usage recording pool."""
    try:
        from umrs.api.llm import set_usage_db_pool

        set_usage_db_pool(db_pool)
        logger.info("LLM usage recording pool initialized")
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.warning("LLM usage recording unavailable — usage will not be tracked")


def _site_capabilities() -> dict[str, str | int]:
    """Collect non-secret site capabilities for client discovery."""
    caps: dict[str, str | int] = {}
    try:
        from umrs.orchestrator.site_config import load_site_config

        site = load_site_config()
        caps["scheduler_kind"] = site.scheduler.kind
        caps["slurm_delegation"] = site.scheduler.slurm_delegation
        caps["site_name"] = site.name
        if site.scheduler.slurm_delegation == "per_user_ssh":
            caps["service_pubkey_path"] = site.scheduler.service_pubkey_path
            caps["slurm_proxy_path"] = site.scheduler.slurm_proxy_path
    except Exception:  # noqa: BLE001, S110 — boot/teardown must survive per-component failures
        pass
    zmq_port = os.environ.get("UMRS_CONTROL_PORT")
    if zmq_port:
        with contextlib.suppress(ValueError):
            caps["zmq_control_port"] = int(zmq_port)
    return caps


def _check_database(app):
    """Return the database component status for /health."""
    db_pool = getattr(app.state, "umrs_db_pool", None)
    if db_pool is None:
        return {"status": "not_configured"}
    try:
        conn = db_pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        finally:
            db_pool.putconn(conn)
        return {"status": "ok"}
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "unavailable"}


def _check_registry_health():
    """Return endpoint registry status for /health."""
    try:
        from umrs.orchestrator.endpoint_registry import get_registry

        reg = get_registry()
        all_eps = reg.list_all()
        by_status: dict[str, int] = {}
        for ep in all_eps:
            by_status[ep.status.value] = by_status.get(ep.status.value, 0) + 1
        return {"status": "ok", "total": len(all_eps), **by_status}
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "unavailable"}


def _check_models(app):
    """Return the model store component status."""
    try:
        from umrs.api.models import get_optional_model_store

        store = get_optional_model_store()
        if store is not None:
            return {"status": "ok"}
        return {"status": "not_configured"}
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "unavailable"}


def _check_durable_jobs_health():
    """Return durable jobs component status for /health."""
    try:
        from umrs.jobs.service import get_job_service

        svc = get_job_service()
        if svc is not None and svc._started:
            # Task 2-delta: include liveness heartbeat
            from umrs.server.durable_job_runtime import get_runtime

            rt = get_runtime()
            if rt is not None:
                status = rt.thread_status()
                return {"status": "ok", "liveness": status}
            return {"status": "ok"}
        return {"status": "not_started"}
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "unavailable"}


async def _health(request: Request):
    """Liveness + component summary probe."""
    import time

    app = request.app
    start_time = getattr(app.state, "start_time", None) or 0
    uptime = time.time() - float(start_time) if start_time else 0
    components = {
        "database": _check_database(app),
        "registry": _check_registry_health(),
        "models": _check_models(app),
        "durable_jobs": _check_durable_jobs_health(),
    }
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "site_capabilities": _site_capabilities(),
        "components": components,
    }


async def _metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def _check_postgres(app):
    """Return Postgres status and whether it is a critical failure."""
    db_pool = getattr(app.state, "umrs_db_pool", None)
    if db_pool is None:
        return {"status": "not_configured"}, True
    try:
        conn = db_pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        finally:
            db_pool.putconn(conn)
        return {"status": "ok"}, False
    except Exception as exc:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "unavailable", "error": str(exc)[:120]}, True


def _check_job_store(app):
    """Return job store status and whether it is a critical failure."""
    from umrs.api.jobs import get_optional_job_store

    job_store = get_optional_job_store()
    if job_store is None:
        return {"status": "unavailable"}, True
    try:
        pending = (
            job_store.count_by_status("queued")
            if hasattr(job_store, "count_by_status")
            else None
        )
        return {
            "status": "ok",
            "pending_jobs": str(pending) if pending is not None else "unknown",
        }, False
    except Exception as exc:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "error", "error": str(exc)[:120]}, True


def _check_durable_jobs_readiness():
    """Return durable jobs status and whether it is a critical failure."""
    try:
        from umrs.jobs.service import get_job_service

        svc = get_job_service()
        if svc is not None and svc._started:
            # Task 2-delta: include liveness heartbeat
            from umrs.server.durable_job_runtime import get_runtime

            rt = get_runtime()
            if rt is not None:
                status = rt.thread_status()
                if status["error"] is not None:
                    return {"status": "error", "liveness": status}, True
                return {"status": "ok", "liveness": status}, False
            return {"status": "ok"}, False
        return {"status": "not_started"}, True
    except Exception as exc:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "error", "error": str(exc)[:120]}, True


def _check_registry_readiness():
    """Return endpoint registry status for /readiness."""
    try:
        from umrs.orchestrator.endpoint_registry import get_registry

        reg = get_registry()
        all_eps = reg.list_all()
        ready_count = sum(1 for ep in all_eps if ep.status.value == "ready")
        return {
            "status": "ok",
            "total_endpoints": str(len(all_eps)),
            "ready_endpoints": str(ready_count),
        }
    except Exception as exc:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "error", "error": str(exc)[:120]}


def _check_backend():
    """Return backend/site-config status for /readiness."""
    try:
        from umrs.orchestrator.site_config import load_site_config

        site = load_site_config()
        return {"status": "ok", "type": site.scheduler.kind}
    except Exception as exc:  # noqa: BLE001 — boot/teardown must survive per-component failures
        return {"status": "unknown", "error": str(exc)[:120]}


async def _readiness(request: Request):
    """Readiness probe — checks subsystem availability."""
    app = request.app
    components: dict[str, dict[str, str]] = {}
    critical_failures: list[str] = []

    postgres, failed = _check_postgres(app)
    components["postgres"] = postgres
    if failed:
        critical_failures.append("postgres")

    job_store, failed = _check_job_store(app)
    components["job_store"] = job_store
    if failed:
        critical_failures.append("job_store")

    components["model_store"] = _check_models(app)

    durable_jobs, failed = _check_durable_jobs_readiness()
    components["durable_jobs"] = durable_jobs
    if failed:
        critical_failures.append("durable_jobs")

    components["registry"] = _check_registry_readiness()
    components["backend"] = _check_backend()

    status = "degraded" if critical_failures else "ready"
    result = {
        "ready": status == "ready",
        "status": status,
        "components": components,
    }
    if critical_failures:
        result["unavailable_required"] = critical_failures
        from fastapi.responses import JSONResponse

        return JSONResponse(content=result, status_code=503)
    return result


async def _wire_a2l_recorder(application) -> None:
    """ADR-034 Phase 2: wire the process-global a2l conservation recorder.

    Pool creation is eager here so the same pool is reused by the lazy
    knowledge pool dependency (same app.state key).
    """
    from umrs.api._asyncpg_pool import _init_json_codecs, _resolve_knowledge_dsn

    try:
        from umrs.a2l.recorder import StoreA2lRecorder, configure_recorder
        from umrs.a2l.store import init_store as init_a2l_store

        dsn, is_local = _resolve_knowledge_dsn()
        if dsn:
            import asyncpg

            from umrs.shared.pg_settings import asyncpg_pool_kwargs

            kwargs: dict = asyncpg_pool_kwargs(
                "a2l-recorder", dsn=dsn, init=_init_json_codecs
            )
            if is_local:
                kwargs["ssl"] = "disable"  # jobs pool parity
            pool = await asyncpg.create_pool(**kwargs)
            application.state.umrs_knowledge_asyncpg_pool = pool
            # The durable runtime's checkers run on their own loop thread;
            # asyncpg is loop-affine, so the recorder bounces store calls
            # home (2026-10-20, ops-watch cross-loop errors).
            configure_recorder(
                StoreA2lRecorder(
                    init_a2l_store(pool), owner_loop=asyncio.get_running_loop()
                )
            )
            logger.info("a2l conservation recorder wired (ADR-034)")
        else:
            logger.info("a2l conservation recorder: no DSN — Null recorder")
    except Exception:
        logger.warning(
            "a2l recorder init failed — conservation disabled", exc_info=True
        )


def _upgrade_loop_store() -> None:
    """Durable loops: upgrade to psycopg2 persistence.
    Independent of the a2l pool (psycopg2 for thread safety).
    """
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn

    try:
        dsn_sync, _ = _resolve_knowledge_dsn()
        if dsn_sync:
            from umrs.api.loop import configure_loop_store_sync

            configure_loop_store_sync(dsn_sync)
            import psycopg2 as _psycopg2

            from umrs.shared.pg_settings import pg_connect_kwargs

            # D-06 accounts: sync JSON users -> Postgres; the PG lane goes
            # read-first (the JSON files stay the bootstrap/recovery lane).
            try:
                from umrs.admin.user_store import configure_user_store_pg, site_venue

                # 0113: rows scoped to this site (umrs-front's servers share umrs_db).
                configure_user_store_pg(
                    _SingleConnPool(
                        lambda: _psycopg2.connect(
                            dsn_sync, **pg_connect_kwargs("user-store")
                        )
                    ),
                    venue=site_venue(),
                )
            except Exception:  # noqa: BLE001 — accounts lane fails open to JSON by contract
                logger.warning(
                    "user store PG lane not wired (JSON stays authoritative)"
                )

            # D-36 leg B: the notification emitter rides the same DSN seam
            # (budget-stop notifications surface through the durable store).
            from umrs.platform_notifications import set_emitter

            set_emitter(
                _SingleConnPool(
                    lambda: _psycopg2.connect(
                        dsn_sync, **pg_connect_kwargs("notifications")
                    )
                )
            )
    except Exception:
        logger.warning("loop store upgrade failed", exc_info=True)


def _start_account_sweep() -> AccountStandingSweep | None:
    """RS-0 S-2: the account standing sweep. Disabled accounts and, under
    ``UMRS_PROJECT_MEMBERSHIP=enforce``, ex-members lose their leases, open
    jobs and CURVE sessions within ``UMRS_ACCOUNT_SWEEP_SEC`` (30 s, at most
    60); their remote sessions end on the next status check through the
    revoked lease. Needs the site database; ``UMRS_ACCOUNT_SWEEP_ENABLED=false``
    turns it off."""
    flag = os.environ.get("UMRS_ACCOUNT_SWEEP_ENABLED", "true").strip().lower()
    if flag in ("false", "0", "no"):
        logger.warning("account standing sweep disabled (UMRS_ACCOUNT_SWEEP_ENABLED)")
        return None
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn

    dsn, _ = _resolve_knowledge_dsn()
    if not dsn:
        logger.info("account standing sweep: no DSN, not started")
        return None
    try:
        import psycopg2 as _psycopg2

        from umrs.admin.account_cascade import AccountStandingSweep
        from umrs.jobs.store import JobStore
        from umrs.shared.pg_settings import pg_connect_kwargs

        pool = _SingleConnPool(
            lambda: _psycopg2.connect(dsn, **pg_connect_kwargs("account-sweep"))
        )
        sweep = AccountStandingSweep(
            pool=pool, job_store_factory=lambda: JobStore(pool)
        )
        sweep.start()
        return sweep
    except Exception:
        logger.warning("account standing sweep not started", exc_info=True)
        return None


class _SingleConnPool:
    """Minimal psycopg2 pool: fresh conn per lease (cheap, always-returned)."""

    def __init__(self, factory):
        self._factory = factory

    def getconn(self):
        return self._factory()

    def putconn(self, conn):
        try:
            conn.close()
        except Exception:  # noqa: BLE001, S110 — boot/teardown must survive per-component failures
            pass


def _rehydrate_loop_drivers() -> None:
    """Re-spawn drivers for loops RUNNING/PAUSED at the last shutdown.

    Runs AFTER the durable runtime start (ADR-036 D1): job-hosted loops are
    resumed by reconciler → scheduler → loop_hosting.host, thread-hosted
    ones here; without a runtime, job-hosted loops fall back to threads.
    """
    try:
        from umrs.api.loop import reregister_loop_drivers

        reregister_loop_drivers()
    except Exception:
        logger.warning("loop driver re-registration failed", exc_info=True)


def _boot_reconcile_enabled() -> bool:
    return os.environ.get("UMRS_BOOT_RECONCILE_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


#: Seconds the pre-bind boot pass probes each stuck endpoint. Short on
#: purpose: a row whose job is alive but not ready is deferred, and
#: _resume_interrupted_provisions finishes it on the server loop with the full
#: budget. The old 300 s default ran serially before uvicorn bound the port,
#: so bdrt's API was down 15 min after the 2026-10-22 restart while its four
#: endpoints answered 502 (UMRS_RECONCILE_TIMEOUT_SEC still overrides).
_BOOT_RECONCILE_PROBE_SEC = 10


def _boot_reconcile_timeout_sec() -> int:
    raw = os.environ.get("UMRS_RECONCILE_TIMEOUT_SEC", "")
    return int(raw) if raw.strip() else _BOOT_RECONCILE_PROBE_SEC


def _run_boot_reconcile() -> None:
    """Boot-time reconciler: re-resolve endpoints stuck in PROVISIONING/
    STARTING after a UMRS restart (Task 6 Gap B), before uvicorn binds.

    Each row gets a short probe (:func:`_boot_reconcile_timeout_sec`): ready
    rows are promoted, dead ones failed, live-but-loading ones deferred to
    :func:`_resume_interrupted_provisions`. Non-fatal: the sidecar starts
    either way. UMRS_BOOT_RECONCILE_ENABLED gates it ("true" by default).
    """
    if not _boot_reconcile_enabled():
        logger.info("Boot-time reconciler disabled by UMRS_BOOT_RECONCILE_ENABLED")
        return
    try:
        from ..orchestrator.service import get_orchestration_service

        service = get_orchestration_service()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                service.reconcile_stale_endpoints(
                    timeout_sec=_boot_reconcile_timeout_sec()
                )
            )
            logger.info(
                "Boot-time reconciler: checked=%d ready=%d failed=%d errors=%d",
                result["checked"],
                result["promoted_to_ready"],
                result["marked_failed"],
                len(result["errors"]),
            )
        finally:
            loop.close()
    except Exception:
        logger.exception(
            "Boot-time reconciler failed (non-fatal — sidecar still starts)"
        )


async def _resume_interrupted_provisions() -> None:
    """Finish provisions a restart cut short (the boot reconciler defers rows
    whose backend job is still queued or loading; this runs them to READY or
    to FAILED-and-cancelled on the server's own loop)."""
    if not _boot_reconcile_enabled():
        return
    try:
        from umrs.orchestrator.service import get_orchestration_service

        await get_orchestration_service().resume_unfinished_provisions()
    except Exception:
        logger.warning("resuming interrupted provisions failed", exc_info=True)


def _tag_metrics_source() -> None:
    """Tag all LLM metrics with the operational source (site name)."""
    try:
        from umrs.metrics import METRICS
        from umrs.orchestrator.site_config import load_site_config

        METRICS.set_source(load_site_config().name)
        logger.info("LLM metrics source tagged: %s", METRICS._source)
    except Exception:  # noqa: BLE001 — boot/teardown must survive per-component failures
        logger.debug("No site config available — LLM metrics will have empty source")


def _create_app():
    """Create a minimal FastAPI app mounting the Gateway-facing UMRS routers."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from umrs.api.knowledge_v2 import router as knowledge_v2_router
    from umrs.knowledge.events_sse import router as events_sse_router
    from umrs.server.structured_logging import configure_structured_logging

    configure_structured_logging()

    from umrs.api.a2l import router as a2l_router
    from umrs.api.a2l_ledger import router as a2l_ledger_router
    from umrs.api.admin_session_policy import router as admin_session_policy_router
    from umrs.api.admin_sessions import router as admin_sessions_router
    from umrs.api.admin_usage import router as admin_usage_router
    from umrs.api.artifact_routes import router as artifact_router
    from umrs.api.assets import router as assets_router
    from umrs.api.buckets import router as buckets_router
    from umrs.api.bundle_restore import router as bundle_restore_router
    from umrs.api.catalog import router as catalog_router
    from umrs.api.chat import router as chat_router
    from umrs.api.companion_chat_approvals import (
        router as companion_chat_approvals_router,
    )
    from umrs.api.companion_sessions import router as companion_sessions_router
    from umrs.api.companion_venue import router as companion_venue_router
    from umrs.api.egress_audit import router as egress_audit_router
    from umrs.api.endpoints import router as endpoints_router
    from umrs.api.evidence import router as evidence_router
    from umrs.api.experiment_registry import router as experiment_registry_router
    from umrs.api.files import router as files_router
    from umrs.api.grants import router as grants_router
    from umrs.api.intake_uploads import router as intake_uploads_router
    from umrs.api.jobs import router as jobs_router
    from umrs.api.knowledge import router as knowledge_router
    from umrs.api.knowledge_artifact_delete import (
        router as knowledge_artifact_delete_router,
    )
    from umrs.api.knowledge_attrition import router as knowledge_attrition_router
    from umrs.api.knowledge_graphs import router as knowledge_graphs_router
    from umrs.api.knowledge_lineage import router as knowledge_lineage_router
    from umrs.api.knowledge_proposals import router as knowledge_proposals_router
    from umrs.api.knowledge_query import router as knowledge_query_router
    from umrs.api.knowledge_upload import router as knowledge_upload_router
    from umrs.api.llm import router as llm_router
    from umrs.api.loop import router as loops_router
    from umrs.api.mcp import router as mcp_router
    from umrs.api.models import router as models_router
    from umrs.api.ops import router as ops_router
    from umrs.api.pipelines import router as pipelines_router
    from umrs.api.plan_run_venue import router as plan_run_venue_router
    from umrs.api.provenance import router as provenance_router
    from umrs.api.rag import router as rag_router
    from umrs.api.reflective_memory import router as reflective_memory_router
    from umrs.api.research_sessions import router as research_sessions_router
    from umrs.api.scientific_memory import router as scientific_memory_router
    from umrs.api.secret_broker import router as secret_broker_router
    from umrs.api.skills import router as skills_router
    from umrs.api.source_grants import router as source_grants_router
    from umrs.api.traces import router as traces_router
    from umrs.api.usage import router as usage_router
    from umrs.api.user_sources import router as user_sources_router
    from umrs.api.vision import router as vision_router
    from umrs.api.workflows import router as workflows_router
    from umrs.api.zmq_sessions import router as zmq_sessions_router

    environment = os.environ.get("GATEWAY_ENVIRONMENT", "development").lower()
    enable_docs = environment in ("development", "dev", "test")

    @asynccontextmanager
    async def _lifespan(application: FastAPI):
        _tag_metrics_source()
        from umrs.orchestrator.janitor import start_janitor, stop_janitor
        from umrs.scheduler.service import start_scheduler, stop_scheduler

        await _wire_a2l_recorder(application)
        _upgrade_loop_store()

        start_janitor()
        logger.info("Janitor started via lifespan hook")
        from umrs.server.endpoint_relay import start_endpoint_relay, stop_endpoint_relay

        await _resume_interrupted_provisions()  # before the relay plans on them
        if start_endpoint_relay():
            logger.info("Endpoint walltime relay started via lifespan hook")
        if start_scheduler():
            logger.info("Scheduler service started via lifespan hook")
        _start_durable_runtime()
        _rehydrate_loop_drivers()
        account_sweep = _start_account_sweep()
        try:
            yield
        finally:
            if account_sweep is not None:
                account_sweep.stop()
            stop_scheduler()
            stop_endpoint_relay()
            stop_janitor()
            logger.info("Scheduler service + janitor stopped via lifespan hook")

    app = FastAPI(
        title="UMRS Endpoint Discovery API",
        description="HTTP sidecar for Gateway endpoint discovery, LLM/RAG/MCP proxy",
        version="0.3.0",
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        lifespan=_lifespan,
    )
    # /health and the dashboard health read this; nothing set it, so both
    # reported uptime_seconds 0 for the life of the process (2026-10-21).
    app.state.start_time = time.time()

    # WP-16 r2 problem+json (opt-in): RFC 7807 projection applies only to
    # responses carrying x-platform-error-code; all other shapes unchanged.
    from umrs.api.problem_json import enabled_by_env, install_problem_json_handler

    if enabled_by_env():
        install_problem_json_handler(app)
        logger.info("problem+json projection enabled (UMRS_PROBLEM_JSON)")

    # WP-12 round 2: optional OTel (no-op unless OTEL_EXPORTER_OTLP_ENDPOINT).
    from umrs.server.otel import setup_otel

    setup_otel(app, service_label="umrs-http-sidecar")

    # Correlation ID middleware: extracts X-Request-ID from Gateway (NGINX)
    # or X-Correlation-ID from R4 SDK; makes it available via get_request_id().
    from umrs.server.correlation import CorrelationIDMiddleware

    app.add_middleware(CorrelationIDMiddleware)

    # JSON bodies above UMRS_MAX_JSON_BODY_BYTES (default 64 MiB) are a 413
    # before they are read (2026-10-21: a 300 MB body took ~1 GB of RSS).
    from umrs.server.body_limit import JsonBodyLimitMiddleware

    app.add_middleware(JsonBodyLimitMiddleware)

    # Structured error responses: all HTTPException responses include
    # error_id and request_id for distributed tracing and automation.
    from fastapi import HTTPException as _HTTPException

    from umrs.api.error_responses import (
        structured_http_exception_handler,
        structured_warning_exception_handler,
        unhandled_exception_handler,
    )
    from umrs.shared.warnings import UMRSWarningError

    app.add_exception_handler(_HTTPException, structured_http_exception_handler)
    app.add_exception_handler(UMRSWarningError, structured_warning_exception_handler)
    # Anything no route caught: typed 503 for a database outage, 400 for a
    # NUL Postgres cannot store, typed 500 otherwise (was plain text).
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(catalog_router)
    app.include_router(a2l_router)
    app.include_router(a2l_ledger_router)
    app.include_router(artifact_router)
    app.include_router(assets_router)
    app.include_router(bundle_restore_router)
    app.include_router(chat_router)
    app.include_router(companion_sessions_router)
    app.include_router(companion_chat_approvals_router)
    # Wave 2026-09-21 (ADR-046 A2): frontend-venue runtime + worker rail.
    app.include_router(companion_venue_router)
    # RS-0 C-1: admin session list/show/stop over the same venue service.
    app.include_router(admin_sessions_router)
    app.include_router(admin_session_policy_router)  # ADR-099 session policy rules
    # ADR-099 (RS-1 lane Q): remote-session token usage, admin only.
    app.include_router(admin_usage_router)
    app.include_router(egress_audit_router)
    app.include_router(grants_router)
    app.include_router(endpoints_router)
    app.include_router(files_router)
    app.include_router(buckets_router)  # ADR-079 personal buckets
    app.include_router(intake_uploads_router)
    app.include_router(jobs_router)
    app.include_router(knowledge_router)
    app.include_router(knowledge_graphs_router)
    app.include_router(knowledge_lineage_router)
    app.include_router(knowledge_proposals_router)
    app.include_router(knowledge_query_router)
    app.include_router(knowledge_upload_router)
    app.include_router(knowledge_attrition_router)
    app.include_router(knowledge_artifact_delete_router)
    app.include_router(knowledge_v2_router)
    app.include_router(events_sse_router)
    app.include_router(llm_router)
    app.include_router(loops_router)
    app.include_router(mcp_router)
    # NOTE: no separate policy_router — policy endpoints live on the models
    # router itself; mounting it twice was a duplicate-registration of every
    # /umrs/models route (route_inventory captured 19 dupes, belt red 2026-09-16).
    app.include_router(models_router)
    app.include_router(ops_router)
    app.include_router(pipelines_router)
    app.include_router(rag_router)
    app.include_router(skills_router)
    app.include_router(vision_router)
    app.include_router(provenance_router)
    app.include_router(research_sessions_router)
    app.include_router(traces_router)
    app.include_router(plan_run_venue_router)  # ADR-092 slice 8
    app.include_router(usage_router)
    app.include_router(workflows_router)
    app.include_router(zmq_sessions_router)
    app.include_router(web_dashboard_router)
    # Wave T3b m5: venue companion HTTP surface (registry + turn-attempt
    # surface; deployment path is behind /gw/companion/* via NGINX sidecar).
    from umrs.api.companion import router as companion_router

    app.include_router(companion_router)
    # Previously-dormant satellite routers (now durable when DB env present).
    app.include_router(scientific_memory_router)
    app.include_router(reflective_memory_router)
    app.include_router(evidence_router)
    app.include_router(experiment_registry_router)
    # R0/R1 secrets surfaces (enveloped identity enforced router-wide, G-6).
    app.include_router(secret_broker_router)
    app.include_router(user_sources_router)
    app.include_router(source_grants_router)
    # ADR-051 (2026-09-14): self-service durable projects registry.
    from umrs.api.projects import router as projects_router

    app.include_router(projects_router)

    # D-39 (2026-09-22): Scientific Catalog registry (gap row D-39, Phase 1).
    from umrs.api.scientific_catalogs import router as scientific_catalogs_router

    app.include_router(scientific_catalogs_router)

    # ADR-051 (2026-09-16): caller-attributed quota gauge (D-2/D-3 read model).
    from umrs.api.quota_status import router as quota_status_router

    app.include_router(quota_status_router)

    # ADR-086 (2026-10-05): circles — project-scoped context/audience cells
    # (supersedes the ADR-050 §3 groups authority).
    from umrs.api.circles import router as circles_router

    app.include_router(circles_router)

    # Satellite stores auto-resolve via the shared factories (Postgres when
    # a pool is resolvable, else in-memory). Routers return 503 on failure.
    from umrs.api.evidence import set_store as set_evidence_store
    from umrs.api.experiment_registry import set_registry as set_exp_registry
    from umrs.api.scientific_memory import set_store as set_scientific_memory_store
    from umrs.evidence import build_default_store as build_evidence_store
    from umrs.experiment_registry import build_default_store as build_registry_store
    from umrs.scientific_memory import build_default_store as build_sm_store

    for starter, builder, name in (
        (set_scientific_memory_store, build_sm_store, "scientific memory"),
        (set_evidence_store, build_evidence_store, "evidence"),
        (set_exp_registry, build_registry_store, "experiment registry"),
    ):
        try:
            starter(builder())
        except Exception:
            logger.warning("%s satellite store init failed", name, exc_info=True)

    db_pool = _build_db_pool(app)
    if db_pool is not None:
        _init_deep_stores(app, db_pool)

    app.get("/health")(_health)
    app.get("/metrics")(_metrics)
    app.get("/readiness")(_readiness)

    dashboard_dir = pathlib.Path(__file__).resolve().parent / "web_gui"
    if dashboard_dir.is_dir():
        app.mount(
            "/umrs/web",
            StaticFiles(directory=str(dashboard_dir), html=True),
            name="umrs-web",
        )

    return app


def start_http_sidecar(
    port: int | None = None,
    host: str | None = None,
) -> threading.Thread | None:
    """
    Start the HTTP sidecar in a daemon thread.

    Returns the thread, or None if disabled via UMRS_HTTP_SIDECAR_ENABLED=false.

    The bind address respects ``UMRS_TUNNEL_MODE``: when tunnel mode is
    enabled the sidecar is forced to ``127.0.0.1`` regardless of the
    *host* argument (Platform Contract §8.4).
    """
    enabled = os.environ.get("UMRS_HTTP_SIDECAR_ENABLED", "true").lower()
    if enabled not in ("true", "1", "yes"):
        logger.info("HTTP sidecar disabled (UMRS_HTTP_SIDECAR_ENABLED=%s)", enabled)
        return None

    resolved_host = _resolve_bind_host(host)
    port = port or int(os.environ.get("UMRS_HTTP_PORT", str(_DEFAULT_PORT)))

    def _run():
        import uvicorn

        try:
            app = _create_app()
        except Exception as exc:  # noqa: BLE001 — the daemon runs on without its sidecar
            logger.warning("HTTP sidecar disabled — app creation failed: %s", exc)
            return
        logger.info("Starting HTTP sidecar on %s:%d", resolved_host, port)

        _run_boot_reconcile()
        uvicorn.run(app, host=resolved_host, port=port, log_level="warning")

    thread = threading.Thread(
        target=_run,
        name="umrs-http-sidecar",
        daemon=True,
    )
    thread.start()
    logger.info("HTTP sidecar thread started on port %d", port)
    return thread
