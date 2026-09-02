"""Companion trust allowlist — per-user-scope server records (wave T3c).

Trust rows record that ``user_id`` explicitly approves auto-execution of
``tool_name`` inside ``scope`` until ``expires_at``. They are the
server-enforced counterpart of the client-side PermissionManager session
approvals: autopilot/TRUST mode consults these rows so trust decisions
survive process restarts and are visible across clients.

Follows the jobs/schema.py per-component pattern:
``COMPANION_TRUST_SCHEMA_VERSION`` + :func:`ensure_schema` registering the
``companion_trust`` component in ``schema_versions``.
"""

from .schema import COMPANION_TRUST_SCHEMA_VERSION, ensure_schema
from .store import TrustAllowlistStore, TrustRow

__all__ = [
    "COMPANION_TRUST_SCHEMA_VERSION",
    "TrustAllowlistStore",
    "TrustRow",
    "ensure_schema",
]
