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
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8082


def _create_app():
    """Create a minimal FastAPI app mounting endpoint-related routers."""
    from fastapi import FastAPI

    from umrs.api import endpoints_router

    app = FastAPI(
        title="UMRS Endpoint Discovery API",
        description="HTTP sidecar for Gateway endpoint discovery",
        version="0.1.0",
        docs_url="/docs",
    )

    app.include_router(endpoints_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def start_http_sidecar(
    port: Optional[int] = None,
    host: str = "0.0.0.0",
) -> Optional[threading.Thread]:
    """
    Start the HTTP sidecar in a daemon thread.

    Returns the thread, or None if disabled via UMRS_HTTP_SIDECAR_ENABLED=false.
    """
    enabled = os.environ.get("UMRS_HTTP_SIDECAR_ENABLED", "true").lower()
    if enabled not in ("true", "1", "yes"):
        logger.info("HTTP sidecar disabled (UMRS_HTTP_SIDECAR_ENABLED=%s)", enabled)
        return None

    port = port or int(os.environ.get("UMRS_HTTP_PORT", str(_DEFAULT_PORT)))

    def _run():
        import uvicorn

        app = _create_app()
        logger.info("Starting HTTP sidecar on %s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(
        target=_run,
        name="umrs-http-sidecar",
        daemon=True,
    )
    thread.start()
    logger.info("HTTP sidecar thread started on port %d", port)
    return thread
