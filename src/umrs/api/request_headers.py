"""Request headers read under their neutral and R4 names (U3 S6b, ADR-117).

FastAPI dependencies over :mod:`umrs.shared.compat_names`, so a route keeps
its parameter and body while reading ``x-umrs-<name>`` first, then
``x-r4-<name>``. A leaf module: ``identity`` imports ``workflows``, so the
routes cannot depend on ``identity`` for these.
"""

from __future__ import annotations

from fastapi import Request

from umrs.shared.compat_names import header


def session_id_header(request: Request) -> str | None:
    """The caller's companion session id under either name; None when absent."""
    return header(request.headers, "session-id") or None


__all__ = ["session_id_header"]
