"""U3 S1 item 8: the tool-approval inbox routes are core.

``GET /umrs/companion/approvals/pending`` and
``POST /umrs/companion/approvals/{id}/decision`` moved from ``companion`` to
``umrs.api.approvals`` with the same paths, mounted through the companion
router until U3 S4/S5 mounts them directly. The strict identity dependency
is the same object, so dependency overrides keep applying.
"""

from __future__ import annotations

import ast
from pathlib import Path

from umrs.api import approvals, companion, identity

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs" / "api"
PATHS = {
    ("GET", "/umrs/companion/approvals/pending"),
    ("POST", "/umrs/companion/approvals/{approval_id}/decision"),
}


def _routes(router) -> set[tuple[str, str]]:
    return {(m, r.path) for r in router.routes for m in getattr(r, "methods", ())}


def test_companion_router_still_serves_the_approval_paths():
    assert PATHS <= _routes(companion.router)


def test_the_approval_routes_come_from_the_core_module():
    endpoints = {
        r.endpoint
        for r in companion.router.routes
        if getattr(r, "path", "").endswith(
            ("/approvals/pending", "/approvals/{approval_id}/decision")
        )
    }
    assert endpoints == {approvals.list_pending_approvals, approvals.record_approval_decision}


def test_strict_identity_dependency_is_one_object():
    assert companion._strict_headers is identity.strict_user_header


def test_core_approvals_import_companion_only_lazily():
    tree = ast.parse((SRC / "approvals.py").read_text(encoding="utf-8"))
    top_level = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("companion")
    ]
    assert not top_level
