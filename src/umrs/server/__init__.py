"""Server-side UMRS package surface.

Lazy imports to avoid circular dependency chains when compatibility
wrappers (e.g. ``umrs.sql``) resolve through ``umrs.server.*``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from umrs.server.provisioning import ServerCfg as ServerCfg  # noqa: F401


def __getattr__(name: str):
    if name == "ServerCfg":
        from umrs.server.provisioning import ServerCfg
        return ServerCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ServerCfg"]