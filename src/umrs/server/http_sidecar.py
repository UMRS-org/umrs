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
import logging
import os
import pathlib
import threading

from fastapi import Request
from fastapi.staticfiles import StaticFiles

from ..api.web_dashboard import router as web_dashboard_router

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
    except Exception:
        logger.warning("Shared DB pool unavailable in HTTP sidecar")
        return None


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
    except Exception:
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
        from umrs.model_registry.store import ModelStore, PathValidator

        model_conn = db_pool.getconn()
        ensure_model_registry_schema(model_conn)
        model_conn.commit()
        app.state.umrs_model_store_conn = model_conn
        set_model_store(
            ModelStore(model_conn, path_validator=PathValidator(), pool=db_pool)
        )
        logger.info("Model store initialized for HTTP sidecar")
    except Exception:
        logger.warning(
            "Model store unavailable in HTTP sidecar — /umrs/models/* will return 503"
        )


def _init_mcp_store(app, db_pool):
    """Initialize the MCP server store."""
    try:
        from umrs.api.mcp import set_mcp_store
        from umrs.mcp import ensure_schema as ensure_mcp_schema
        from umrs.mcp.store import MCPServerStore

        mcp_conn = db_pool.getconn()
        ensure_mcp_schema(mcp_conn)
        mcp_conn.commit()
        app.state.umrs_mcp_store_conn = mcp_conn
        set_mcp_store(MCPServerStore(mcp_conn))
        logger.info("MCP server store initialized for HTTP sidecar")
    except Exception:
        logger.warning(
            "MCP store unavailable in HTTP sidecar — /umrs/mcp/* will return 503"
        )


def _init_audit_store(app, db_pool):
    """Initialize the audit store."""
    try:
        from umrs.api.audit_events import set_audit_store
        from umrs.api.audit_store import AuditStore, ensure_audit_schema

        audit_conn = db_pool.getconn()
        ensure_audit_schema(audit_conn)
        audit_conn.commit()
        app.state.umrs_audit_store_conn = audit_conn
        set_audit_store(AuditStore(audit_conn))
        logger.info("Audit store initialized for HTTP sidecar")
    except Exception:
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
        gov_store = GovernanceStore(gov_conn)
        app.state.umrs_governance_store = gov_store
        set_governance_store(gov_store)
        logger.info("Governance store initialized for HTTP sidecar")
    except Exception:
        logger.warning("Governance store unavailable — quota enforcement disabled")


def _init_research_sessions_store(app, db_pool):
    """Initialize the research-session store (ADR-025)."""
    try:
        from umrs.api.research_sessions import set_session_store
        from umrs.research_sessions import ensure_schema
        from umrs.research_sessions.store import SessionStore

        sess_conn = db_pool.getconn()
        ensure_schema(sess_conn)
        sess_conn.commit()
        app.state.umrs_research_sessions_conn = sess_conn
        set_session_store(SessionStore(sess_conn))
        logger.info("Research session store initialized for HTTP sidecar")
    except Exception:
        logger.warning(
            "Research session store unavailable in HTTP sidecar — "
            "/umrs/research/sessions/* will return 503"
        )


def _init_usage_pool(db_pool):
    """Wire the LLM usage recording pool."""
    try:
        from umrs.api.llm import set_usage_db_pool

        set_usage_db_pool(db_pool)
        logger.info("LLM usage recording pool initialized")
    except Exception:
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
    except Exception:
        pass
    zmq_port = os.environ.get("UMRS_CONTROL_PORT")
    if zmq_port:
        try:
            caps["zmq_control_port"] = int(zmq_port)
        except ValueError:
            pass
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
    except Exception:
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
    except Exception:
        return {"status": "unavailable"}


def _check_models(app):
    """Return the model store component status."""
    try:
        from umrs.api.models import get_optional_model_store

        store = get_optional_model_store()
        if store is not None:
            return {"status": "ok"}
        return {"status": "not_configured"}
    except Exception:
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
    except Exception:
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
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from fastapi.responses import Response

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
    except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
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
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:120]}


def _check_backend():
    """Return backend/site-config status for /readiness."""
    try:
        from umrs.orchestrator.site_config import load_site_config

        site = load_site_config()
        return {"status": "ok", "type": site.scheduler.kind}
    except Exception as exc:
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


def _create_app():
    """Create a minimal FastAPI app mounting the Gateway-facing UMRS routers."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from umrs.api.knowledge_v2 import router as knowledge_v2_router
    from umrs.knowledge.events_sse import router as events_sse_router

    from umrs.server.structured_logging import configure_structured_logging

    configure_structured_logging()

    from umrs.api import (
        a2l_router,
        artifact_router,
        bundle_restore_router,
        catalog_router,
        chat_router,
        egress_audit_router,
        endpoints_router,
        files_router,
        intake_uploads_router,
        jobs_router,
        knowledge_graph_router,
        knowledge_lineage_router,
        knowledge_query_router,
        knowledge_router,
        llm_router,
        loops_router,
        mcp_router,
        models_router,
        ops_router,
        pipelines_router,
        policy_router,
        provenance_router,
        reflective_memory_router,
        research_sessions_router,
        scientific_memory_router,
        evidence_router,
        experiment_registry_router,
        rag_router,
        skills_router,
        traces_router,
        usage_router,
        verification_router,
        vision_router,
        workflows_router,
        zmq_sessions_router,
    )

    environment = os.environ.get("GATEWAY_ENVIRONMENT", "development").lower()
    enable_docs = environment in ("development", "dev", "test")

    @asynccontextmanager
    async def _lifespan(application: FastAPI):
        # Tag all LLM metrics with the operational source (site name).
        try:
            from umrs.orchestrator.site_config import load_site_config
            from umrs.metrics import METRICS

            METRICS.set_source(load_site_config().name)
            logger.info("LLM metrics source tagged: %s", METRICS._source)  # noqa: SLF001
        except Exception:
            logger.debug(
                "No site config available — LLM metrics will have empty source"
            )
        from umrs.orchestrator.janitor import start_janitor, stop_janitor
        from umrs.scheduler.service import start_scheduler, stop_scheduler

        start_janitor()
        logger.info("Janitor started via lifespan hook")
        if start_scheduler():
            logger.info("Scheduler service started via lifespan hook")
        try:
            yield
        finally:
            stop_scheduler()
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

    # Correlation ID middleware: extracts X-Request-ID from Gateway (NGINX)
    # or X-Correlation-ID from R4 SDK; makes it available via get_request_id().
    from umrs.server.correlation import CorrelationIDMiddleware

    app.add_middleware(CorrelationIDMiddleware)

    # Structured error responses: all HTTPException responses include
    # error_id and request_id for distributed tracing and automation.
    from fastapi import HTTPException as _HTTPException
    from umrs.api.error_responses import (
        structured_http_exception_handler,
        structured_warning_exception_handler,
    )
    from umrs.shared.warnings import UMRSWarningError

    app.add_exception_handler(_HTTPException, structured_http_exception_handler)
    app.add_exception_handler(UMRSWarningError, structured_warning_exception_handler)

    app.include_router(catalog_router)
    app.include_router(a2l_router)
    app.include_router(artifact_router)
    app.include_router(bundle_restore_router)
    app.include_router(chat_router)
    app.include_router(egress_audit_router)
    app.include_router(endpoints_router)
    app.include_router(files_router)
    app.include_router(intake_uploads_router)
    app.include_router(jobs_router)
    app.include_router(knowledge_router)
    app.include_router(knowledge_graph_router)
    app.include_router(knowledge_lineage_router)
    app.include_router(knowledge_query_router)
    app.include_router(knowledge_v2_router)
    app.include_router(events_sse_router)
    app.include_router(llm_router)
    app.include_router(loops_router)
    app.include_router(mcp_router)
    app.include_router(models_router)
    app.include_router(ops_router)
    app.include_router(pipelines_router)
    app.include_router(policy_router)
    app.include_router(rag_router)
    app.include_router(skills_router)
    app.include_router(verification_router)
    app.include_router(vision_router)
    app.include_router(provenance_router)
    app.include_router(research_sessions_router)
    app.include_router(traces_router)
    app.include_router(usage_router)
    app.include_router(workflows_router)
    app.include_router(zmq_sessions_router)
    app.include_router(web_dashboard_router)
    # Previously-dormant satellite routers (now durable when DB env present).
    app.include_router(scientific_memory_router)
    app.include_router(reflective_memory_router)
    app.include_router(evidence_router)
    app.include_router(experiment_registry_router)

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
        except Exception:  # noqa: BLE001
            logger.warning("%s satellite store init failed", name, exc_info=True)

    db_pool = _build_db_pool(app)
    if db_pool is not None:
        _init_job_store(db_pool)
        _init_catalog_meta_store(db_pool)
        _init_model_store(app, db_pool)
        _init_mcp_store(app, db_pool)
        _init_audit_store(app, db_pool)
        _init_governance_store(app, db_pool)
        _init_research_sessions_store(app, db_pool)
        _init_usage_pool(db_pool)

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

        app = _create_app()
        logger.info("Starting HTTP sidecar on %s:%d", resolved_host, port)

        # Boot-time reconciler: re-resolve endpoints stuck in PROVISIONING/
        # STARTING after a UMRS restart (Task 6 Gap B).  Runs synchronously
        # before uvicorn binds, so no requests are served until reconciliation
        # is complete.  UMRS_RECONCILE_TIMEOUT_SEC controls the per-endpoint
        # timeout (default 300s).  UMRS_BOOT_RECONCILE_ENABLED gates the
        # feature ("true" by default; set "false" to disable).
        if os.environ.get("UMRS_BOOT_RECONCILE_ENABLED", "true").lower() not in (
            "true",
            "1",
            "yes",
        ):
            logger.info("Boot-time reconciler disabled by UMRS_BOOT_RECONCILE_ENABLED")
        else:
            try:
                from ..orchestrator.service import get_orchestration_service

                service = get_orchestration_service()
                timeout_sec = int(os.environ.get("UMRS_RECONCILE_TIMEOUT_SEC", "300"))
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        service.reconcile_stale_endpoints(timeout_sec=timeout_sec)
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

        uvicorn.run(app, host=resolved_host, port=port, log_level="warning")

    thread = threading.Thread(
        target=_run,
        name="umrs-http-sidecar",
        daemon=True,
    )
    thread.start()
    logger.info("HTTP sidecar thread started on port %d", port)
    return thread
