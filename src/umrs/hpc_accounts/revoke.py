"""Ending a person's site access (ADR-112 B1: ``DELETE /umrs/sites/me/hpc``
and the account cascade's site-account leg).

Revoking an account destroys every transport key the person has at the
venue (the files; the gate can no longer be reached), marks the account
``revoked`` and suspends its project links. A project removal suspends only
that project's link and keeps the account and its key.

Wave C (plan §6 C/D.0 C4): a revoke (and so the account cascade) also ends
every active grant of the person at the venue (``account_revoked``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import transport_keys
from .grants import end_account_grants

ACCOUNT_REVOKED = "account_revoked"


def revoke_account(
    store: Any,
    username: str,
    *,
    reason: str = ACCOUNT_REVOKED,
    root: Path | None = None,
) -> tuple[int, int]:
    """(keys destroyed, links suspended). Key files go even without a row."""
    destroyed = transport_keys.destroy(store.venue, username, root=root)
    if store.get_account(username) is not None:
        store.revoke_account(username)
        end_account_grants(store, username)
    return destroyed, store.suspend_links(username, reason)


def end_site_access(
    store: Any,
    username: str,
    project: str | None,
    *,
    reason: str,
    root: Path | None = None,
) -> tuple[int, int]:
    """The cascade leg: the whole account (``project`` None) or one link."""
    if project:
        return 0, store.suspend_links(username, reason, project_id=project)
    return revoke_account(store, username, reason=reason, root=root)


__all__ = ["ACCOUNT_REVOKED", "end_site_access", "revoke_account"]
