"""Session-policy refusals as typed API errors (U3 S1 item 5, ADR-117).

The admin session-policy routes and the loop start route raised these through
the frontend venue router (``companion_venue.policy_typed``); the venue router
moves to the ``umrs_r4`` extension in U3 S5, so the mapping lives in core.
"""

from __future__ import annotations

from typing import NoReturn

from umrs_protocol.errors import PlatformError

from umrs.session_policy import PolicyError

from .typed_errors import raise_typed


def policy_typed(exc: PolicyError) -> NoReturn:
    """A session-policy refusal as its typed ``sessionpolicy.*`` error."""
    raise_typed(PlatformError.sessionpolicy(exc.code, exc.message, **exc.detail), cause=exc)


__all__ = ["policy_typed"]
