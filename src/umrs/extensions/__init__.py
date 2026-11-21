"""UMRS extensions (ADR-117 §6): core hooks the umrs-r4 extension plugs into.

Core never imports an extension package; the loader discovers them through the
``umrs.extensions`` entry point group. The API is internal and versioned
(``EXTENSION_API_VERSION``); it is not a public plugin API until a second
extension exists.
"""

from __future__ import annotations

from umrs.extensions.context import (
    EXTENSION_API_VERSION,
    ExtensionContext,
    ExtensionError,
    JobTypeRegistration,
    LoopIterationKind,
    RouterRegistration,
)
from umrs.extensions.loader import (
    enabled_extensions,
    get_extension_context,
    load_extensions,
    reset_extension_context,
)

__all__ = [
    "EXTENSION_API_VERSION",
    "ExtensionContext",
    "ExtensionError",
    "JobTypeRegistration",
    "LoopIterationKind",
    "RouterRegistration",
    "enabled_extensions",
    "get_extension_context",
    "load_extensions",
    "reset_extension_context",
]
