"""Verified caller identity: the public API (U3 S1 item 10, ADR-117).

Routes outside this package (and the ``umrs_r4`` extension after U3 S5) need
the strict-mode identity checks. They were reached through private names:
``workflows._require_verified_workflow_identity``, ``_verified_headers`` and
``_scope_identity``. This module is the one public import point; the objects
are the same.
"""

from __future__ import annotations

from ._scope_identity import request_caller_scope
from ._verified_headers import resolve_caller
from .workflows import require_verified_workflow_identity

__all__ = ["request_caller_scope", "require_verified_workflow_identity", "resolve_caller"]
