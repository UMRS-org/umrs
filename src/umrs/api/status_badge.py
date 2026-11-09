"""Public status surface: the service-model badge (ADR-106 amendment, 2026-11-07).

- GET /umrs/status/service-badge — shields.io endpoint JSON for this site's
  service roles (``umrs.orchestrator.service_badge``)

The one anonymous UMRS route outside the probes. Its own ``status`` family
(v1-stable: the public README depends on the shape), not the internal
``ops`` lane. Precedent: ``/health/backend`` and ``/gw/ops/ping``. The route
takes no parameters and reads no request header; it makes at most one
registry read and answers exactly
``{"schemaVersion", "label", "message", "color", "cacheSeconds"}``, built
from integers and fixed words. It sets no ``Cache-Control``: the gateway
micro-caches it for 30 s and adds the public ``max-age`` itself (an upstream
header would override ``proxy_cache_valid``).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/status", tags=["status"])


@router.get(
    "/service-badge",
    summary="Service-model badge (public, anonymous)",
    response_model=None,
)
def service_badge_route() -> Any:
    """How many of this site's service roles are up, as a shields badge.

    # auth: none (public by design, ADR-106 amendment)

    200 with the badge; 503 with the fixed red "down" badge when the
    endpoint registry cannot be read.
    """
    from umrs.orchestrator import endpoint_registry
    from umrs.orchestrator.service_badge import DOWN, service_badge

    try:
        return dict(service_badge(lambda: endpoint_registry.get_registry().list_all()))
    except Exception:  # a registry outage is a red badge, not a stack
        logger.warning("service badge: registry read failed", exc_info=True)
        return JSONResponse(
            status_code=503, content=dict(DOWN), headers={"Cache-Control": "no-store"}
        )
