"""Handshake secret accessor (ADR-053, jobs↔api cycle closure).

`umrs.api._verified_headers` previously owned this accessor — jobs' one
back-edge into `umrs.api` (`jobs/cluster_venue.py`). The definition moves to
the foundation layer; the api module re-exports it unchanged so every api
caller is byte-identical (steps: 1.1 hardening).
"""

from __future__ import annotations

import os

from r4_contracts.handshake_hmac import ENV_VAR


def handshake_secret() -> str | None:
    """Shared sign/verify secret from env (empty ⇒ unset)."""
    return os.environ.get(ENV_VAR, "").strip() or None


__all__ = ["ENV_VAR", "handshake_secret"]
