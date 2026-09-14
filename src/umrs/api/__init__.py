"""
UMRS REST API module (ADR-053).

Routers are mounted by the server's explicit registry in
`umrs/src/umrs/server/http_sidecar.py`; this package does NOT eagerly
import router modules. `from umrs.api import <name>_router` still resolves
through a lazy `__getattr__` shim for one release — prefer importing the
router module directly.
"""

from __future__ import annotations

import importlib
import warnings

_LAZY_ROUTER_PREFIXES = ()
__all__: tuple[str, ...] = _LAZY_ROUTER_PREFIXES


def __getattr__(name: str):
    if name.endswith("_router"):
        warnings.warn(
            f"import the router module directly; umrs.api no longer "
            f"re-exports {name!r} (ADR-053)",
            DeprecationWarning,
            stacklevel=2,
        )
        module_name = name[: -len("_router")]
        return importlib.import_module(f"umrs.api.{module_name}").router
    raise AttributeError(f"module 'umrs.api' has no attribute {name!r}")
