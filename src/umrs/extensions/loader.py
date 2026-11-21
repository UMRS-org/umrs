"""Extension discovery, enablement and registration (ADR-117 §6, U3 S3).

Enablement, first match wins:

1. ``UMRS_EXTENSIONS`` (comma or space separated; ``none`` or empty = none);
2. site.yaml ``extensions: [...]``;
3. otherwise every installed extension (U3 S5, agent decision: the r4 code
   left core, so the transitional ``("r4",)`` default became "what is
   installed"; a UMRS-only deployment installs none and gets none, an R4
   front installs ``umrs-r4`` and keeps its surface without an ops change).

An enabled extension registers through its installed entry point (group
``umrs.extensions``). Installed-but-disabled is allowed; enabled-but-absent
is logged. A module whose ``EXTENSION_API_VERSION`` differs from core's fails
boot.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from umrs.extensions.context import (
    EXTENSION_API_VERSION,
    ExtensionContext,
    ExtensionError,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "umrs.extensions"
ENV = "UMRS_EXTENSIONS"
_NONE = frozenset({"", "none", "-"})


def _split(raw: str) -> tuple[str, ...]:
    names = [n for n in re.split(r"[\s,]+", raw.strip()) if n]
    return () if len(names) == 1 and names[0].lower() in _NONE else tuple(names)


def _site_extensions() -> tuple[str, ...] | None:
    try:
        from umrs.orchestrator.site_config import load_site_config

        configured = load_site_config().extensions
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning(
            "site.yaml unreadable for extensions; using the default", exc_info=True
        )
        return None
    return None if configured is None else tuple(str(n) for n in configured)


def enabled_extensions() -> tuple[str, ...]:
    if ENV in os.environ:
        return _split(os.environ[ENV])
    configured = _site_extensions()
    return tuple(sorted(_installed())) if configured is None else configured


def _installed() -> dict[str, Any]:
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def _check_version(name: str, register: Callable[..., Any]) -> None:
    module = __import__(register.__module__, fromlist=["_"])
    target = getattr(module, "EXTENSION_API_VERSION", None)
    if target != EXTENSION_API_VERSION:
        raise ExtensionError(
            f"extension {name!r} targets ExtensionContext API {target!r}, "
            f"core provides {EXTENSION_API_VERSION}"
        )


def _register_one(ctx: ExtensionContext, name: str, installed: dict[str, Any]) -> bool:
    ep = installed.get(name)
    if ep is None:
        logger.warning("extension %r is enabled but not installed", name)
        return False
    ctx.bind(name)
    register = ep.load()
    _check_version(name, register)
    register(ctx)
    return True


def load_extensions(names: tuple[str, ...] | None = None) -> ExtensionContext:
    """Build a fresh context from the enabled extensions."""
    ctx = ExtensionContext()
    installed = _installed()
    for name in enabled_extensions() if names is None else names:
        if _register_one(ctx, name, installed):
            ctx.active.append(name)
    ctx.bind("core")
    disabled = sorted(set(installed) - set(ctx.active))
    logger.info("UMRS extensions active=%s installed-disabled=%s", ctx.active, disabled)
    return ctx


_CONTEXT: ExtensionContext | None = None


def get_extension_context() -> ExtensionContext:
    """The process-wide context, loaded on first use."""
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = load_extensions()
    return _CONTEXT


def reset_extension_context(ctx: ExtensionContext | None = None) -> None:
    """Tests: drop (or replace) the process-wide context."""
    global _CONTEXT
    _CONTEXT = ctx
