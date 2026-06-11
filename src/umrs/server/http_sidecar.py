"""
HTTP sidecar for the UMRS server.

Serves the endpoint registry REST API alongside the ZMQ server,
allowing the Gateway to discover active endpoints via HTTP.

Usage:
    from umrs.server.http_sidecar import start_http_sidecar

    thread = start_http_sidecar(port=8082)
"""

from __future__ import annotations

import logging
import os
import threading

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


def _create_app():
    """Create a minimal FastAPI app mounting the Gateway-facing UMRS routers."""
    from fastapi import FastAPI

    from umrs.server.structured_logging import configure_structured_logging

    configure_structured_logging()

    from umrs.api import (
        egress_audit_router,
        endpoints_router,
        files_router,
        jobs_router,
        llm_router,
        mcp_router,
        models_router,
        ops_router,
        pipelines_router,
        policy_router,
        provenance_router,
        rag_router,
        skills_router,
        traces_router,
        usage_router,
        vision_router,
        workflows_router,
        zmq_sessions_router,
    )

    environment = os.environ.get("GATEWAY_ENVIRONMENT", "development").lower()
    enable_docs = environment in ("development", "dev", "test")

    app = FastAPI(
        title="UMRS Endpoint Discovery API",
        description="HTTP sidecar for Gateway endpoint discovery, LLM/RAG/MCP proxy",
        version="0.3.0",
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
    )

    # Correlation ID middleware: extracts X-Request-ID from Gateway (NGINX)
    # or X-Correlation-ID from R4 SDK; makes it available via get_request_id().
    from umrs.server.correlation import CorrelationIDMiddleware

    app.add_middleware(CorrelationIDMiddleware)

    # Structured error responses: all HTTPException responses include
    # error_id and request_id for distributed tracing and automation.
    from fastapi import HTTPException as _HTTPException
    from umrs.api.error_responses import structured_http_exception_handler

    app.add_exception_handler(_HTTPException, structured_http_exception_handler)

    app.include_router(egress_audit_router)
    app.include_router(endpoints_router)
    app.include_router(files_router)
    app.include_router(jobs_router)
    app.include_router(llm_router)
    app.include_router(mcp_router)
    app.include_router(models_router)
    app.include_router(ops_router)
    app.include_router(pipelines_router)
    app.include_router(policy_router)
    app.include_router(rag_router)
    app.include_router(skills_router)
    app.include_router(vision_router)
    app.include_router(provenance_router)
    app.include_router(traces_router)
    app.include_router(usage_router)
    app.include_router(workflows_router)
    app.include_router(zmq_sessions_router)

    db_pool = None

    try:
        from umrs.server.job_tracking_handler import build_durable_job_db_pool_from_env

        db_pool = build_durable_job_db_pool_from_env()
        app.state.umrs_db_pool = db_pool
    except Exception:
        logger.warning("Shared DB pool unavailable in HTTP sidecar")

    # Initialize the job store so the jobs router can serve requests.
    if db_pool is not None:
        try:
            from umrs.api.jobs import set_job_store
            from umrs.jobs.store import JobStore

            set_job_store(JobStore(db_pool))
            logger.info("Job store initialized for HTTP sidecar")
        except Exception:
            logger.warning(
                "Job store unavailable in HTTP sidecar — /umrs/jobs/* will return 503"
            )

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
            set_model_store(ModelStore(model_conn, path_validator=PathValidator()))
            logger.info("Model store initialized for HTTP sidecar")
        except Exception:
            logger.warning(
                "Model store unavailable in HTTP sidecar — /umrs/models/* will return 503"
            )

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

        # Usage recording pool — shares the main DB pool with llm proxy
        try:
            from umrs.api.llm import set_usage_db_pool

            set_usage_db_pool(db_pool)
            logger.info("LLM usage recording pool initialized")
        except Exception:
            logger.warning(
                "LLM usage recording unavailable — usage will not be tracked"
            )

    @app.get("/health")
    async def health():
        """Liveness probe — returns ok if the process is alive.

        Also includes non-secret site capabilities for client discovery
        (scheduler kind, delegation mode, site name, ZMQ port).
        """
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
        # Expose ZMQ control port for client auto-discovery
        zmq_port = os.environ.get("UMRS_CONTROL_PORT")
        if zmq_port:
            try:
                caps["zmq_control_port"] = int(zmq_port)
            except ValueError:
                pass
        return {"status": "ok"}

    @app.get("/readiness")
    async def readiness():
        """Readiness probe — checks subsystem availability.

        Returns 200 with component statuses when the system can accept work.
        Returns 503 if critical subsystems are unavailable.
        """
        components: dict[str, str] = {}
        healthy = True

        # Check DB pool
        if hasattr(app.state, "umrs_db_pool") and app.state.umrs_db_pool is not None:
            try:
                conn = app.state.umrs_db_pool.getconn()
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT 1")
                    cur.close()
                finally:
                    app.state.umrs_db_pool.putconn(conn)
                components["database"] = "ok"
            except Exception:
                components["database"] = "unavailable"
                healthy = False
        else:
            components["database"] = "not_configured"
            healthy = False

        # Check job store
        from umrs.api.jobs import get_optional_job_store

        if get_optional_job_store() is not None:
            components["job_store"] = "ok"
        else:
            components["job_store"] = "unavailable"
            healthy = False

        # Check model store
        from umrs.api.models import get_optional_model_store

        if get_optional_model_store() is not None:
            components["model_store"] = "ok"
        else:
            components["model_store"] = "not_configured"

        result = {
            "status": "ready" if healthy else "degraded",
            "components": components,
        }
        if not healthy:
            from fastapi.responses import JSONResponse

            return JSONResponse(content=result, status_code=503)
        return result

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
        uvicorn.run(app, host=resolved_host, port=port, log_level="warning")

    thread = threading.Thread(
        target=_run,
        name="umrs-http-sidecar",
        daemon=True,
    )
    thread.start()
    logger.info("HTTP sidecar thread started on port %d", port)
    return thread
