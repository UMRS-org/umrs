"""The companion trust ledger (wave T3c, ADR-035): the r4 extension's part.

Moved out of core in U3 S4b. The tables stay in core migrations and
``umrs.companion_trust.schema``; the durable approval inbox stays core.
"""

from .store import SessionArchivedError, TrustAllowlistStore, TrustRow

__all__ = ["SessionArchivedError", "TrustAllowlistStore", "TrustRow"]
