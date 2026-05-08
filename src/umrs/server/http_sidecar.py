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

    from umrs.api import (
        egress_audit_router,
        endpoints_router,
        jobs_router,
        llm_router,
        mcp_router,
        models_router,
        pipelines_router,
        policy_router,
        rag_router,
        skills_router,
        workflows_router,
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

    app.include_router(egress_audit_router)
    app.include_router(endpoints_router)
    app.include_router(jobs_router)
    app.include_router(llm_router)
    app.include_router(mcp_router)
    app.include_router(models_router)
    app.include_router(pipelines_router)
    app.include_router(policy_router)
    app.include_router(rag_router)
    app.include_router(skills_router)
    app.include_router(workflows_router)

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
            logger.warning("Job store unavailable in HTTP sidecar — /umrs/jobs/* will return 503")

        try:
            from umrs.api.models import set_model_store
            from umrs.model_registry import ensure_schema as ensure_model_registry_schema
            from umrs.model_registry.store import ModelStore, PathValidator

            model_conn = db_pool.getconn()
            ensure_model_registry_schema(model_conn)
            model_conn.commit()
            app.state.umrs_model_store_conn = model_conn
            set_model_store(ModelStore(model_conn, path_validator=PathValidator()))
            logger.info("Model store initialized for HTTP sidecar")
        except Exception:
            logger.warning("Model store unavailable in HTTP sidecar — /umrs/models/* will return 503")

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
            logger.warning("MCP store unavailable in HTTP sidecar — /umrs/mcp/* will return 503")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

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
