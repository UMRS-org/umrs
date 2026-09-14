"""Projection of the r4-contracts error taxonomy onto FastAPI (G-P12-5).

The taxonomy lives in r4.contracts (stdlib-only); UMRS routes raise
``HTTPException(**error.to_http_parts())`` via ``raise_typed``. The detail
body stays byte-identical to the legacy stringly error; the typed code rides
the ``x-platform-error-code`` response header — additive, no API-shape drift.
"""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException
from r4_contracts.errors import PlatformError


def raise_typed(error: PlatformError, *, cause: Exception | None = None) -> NoReturn:
    """Raise the FastAPI exception for a typed platform error."""
    exc = HTTPException(**error.to_http_parts())
    if cause is not None:
        exc.__cause__ = cause
    raise exc


__all__ = ["PlatformError", "raise_typed"]
