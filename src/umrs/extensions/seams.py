"""Core seams that consume ExtensionContext registrations (U3 S3).

- :func:`mount_extension_routers`: appends extension routers after the core
  routers; an extension route whose (method, path) a mounted route already
  serves fails boot instead of silently shadowing it.
- :func:`run_extension_startups`: runs ``add_startup`` callbacks once the db
  pool exists. Each one is isolated: a failing startup is logged, like every
  other per-component init in the HTTP sidecar.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from umrs.extensions.context import ExtensionContext, ExtensionError

logger = logging.getLogger(__name__)

RouteKey = tuple[str, str]


def _route_keys(routes: Iterable[Any], prefix: str = "") -> set[RouteKey]:
    """(method, path) for every route, descending into included routers."""
    keys: set[RouteKey] = set()
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:  # FastAPI >= 0.13x keeps includes lazily
            inner = getattr(route.include_context, "prefix", "") or ""
            keys |= _route_keys(included.routes, prefix + inner)
            continue
        path = getattr(route, "path", None)
        for method in getattr(route, "methods", None) or ():
            keys.add((method, prefix + path))
    return keys


def route_keys(app_or_router: Any) -> set[RouteKey]:
    """Public helper: the (method, path) pairs an app or router serves."""
    router = getattr(app_or_router, "router", app_or_router)
    return _route_keys(router.routes)


def mount_extension_routers(app: Any, ctx: ExtensionContext) -> list[str]:
    """Include every registered router after the core ones; return families."""
    served = route_keys(app)
    mounted: list[str] = []
    for reg in ctx.routers:
        keys = route_keys(reg.router)
        clash = sorted(served & keys)
        if clash:
            raise ExtensionError(
                f"extension {reg.extension!r} router {reg.family!r} collides with "
                f"mounted routes: {clash[:5]}"
            )
        app.include_router(reg.router)
        served |= keys
        mounted.append(reg.family)
    return mounted


def run_extension_startups(ctx: ExtensionContext, db_pool: Any) -> None:
    for extension, fn in ctx.startups:
        try:
            fn(db_pool)
        except Exception:
            logger.warning(
                "extension %s startup %s failed",
                extension,
                getattr(fn, "__name__", fn),
                exc_info=True,
            )
