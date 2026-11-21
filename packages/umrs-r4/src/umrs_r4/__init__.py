"""umrs-r4: the R4 runtime hosted by UMRS, as a UMRS extension (ADR-117 §6).

Core ``umrs`` never imports this package; the core loader finds it through the
``umrs.extensions`` entry point and calls :func:`register`.

U3 S2 (first packaging proof): the package holds the MCP verifier adapter,
which needs no hook, so :func:`register` adds nothing yet. U3 S3 lands the
loader and its ``ExtensionContext`` hooks; S4 and S5 move the companion,
remote-session, venue and plan-run code here and register it through them.
"""

from __future__ import annotations

from typing import Any

# The ExtensionContext API version this extension targets (U3 S3 checks it).
EXTENSION_API_VERSION = 1


def register(ext: Any) -> None:
    """Register the r4 extension's parts with the core ``ExtensionContext``.

    Nothing to register in U3 S2; the hooks arrive with the loader in S3.
    """
    del ext
