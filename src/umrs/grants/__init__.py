"""Asset grants (ADR-041) — version-pinned sharing with tombstone-on-revoke.

Sharing happens by explicit grants, not by copy/re-scope. A grant is a
durable row; revoke tombstones it (never DELETE); fetch-path authorization
consults live grants after the static scope tier denies.
"""

from .store import (
    ASSET_KINDS,
    GRANTEE_KINDS,
    AssetGrant,
    GrantStore,
    grant_authorizes,
    grant_is_live,
)

__all__ = [
    "ASSET_KINDS",
    "GRANTEE_KINDS",
    "AssetGrant",
    "GrantStore",
    "grant_authorizes",
    "grant_is_live",
]
