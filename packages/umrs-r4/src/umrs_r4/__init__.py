"""umrs-r4: the R4 runtime hosted by UMRS, as a UMRS extension (ADR-117 §6).

Core ``umrs`` never imports this package; the core loader finds it through the
``umrs.extensions`` entry point and calls :func:`register` after the r4 parts
still in core (``umrs.extensions.intree_r4``) registered.

Contents:
- U3 S2: the MCP knowledge-verifier adapter (``umrs_r4.verifiers``), no hook.
- U3 S4b: the companion trust ledger (``umrs_r4.trust``): its routes, its
  startup (the pool-backed store), the session-archive participant (trust
  revoke + tombstone, ADR-035 item 8) and the approval listener (D7).
"""

from __future__ import annotations

from typing import Any

# The ExtensionContext API version this extension targets (checked at load).
EXTENSION_API_VERSION = 1


def init_trust_store(db_pool: Any) -> None:
    """The pool-backed trust store (schema ensured by core beforehand)."""
    from umrs_r4.trust.access import set_trust_store
    from umrs_r4.trust.store import TrustAllowlistStore

    # Borrow per call, never a boot lease (http_sidecar._init_mcp_store).
    set_trust_store(TrustAllowlistStore(db_pool))


def register(ext: Any) -> None:
    """Register the r4 extension's parts with the core ``ExtensionContext``."""
    from umrs_r4.api.trust_routes import router as trust_router
    from umrs_r4.trust.approval_listener import mint_session_trust_on_always
    from umrs_r4.trust.archive_participant import revoke_trust_on_archive

    ext.add_router(trust_router, family="companion")
    ext.add_startup(init_trust_store)
    ext.add_session_archive_participant(revoke_trust_on_archive)
    ext.add_approval_listener(mint_session_trust_on_always)
