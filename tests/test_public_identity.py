"""U3 S1 item 10: verified caller identity has one public import point.

Routes that will live in the ``umrs_r4`` extension reached into private names
(``workflows._require_verified_workflow_identity``). ``umrs.api.identity``
re-exports the strict identity gate, the gateway-header resolver and the
caller-scope reader; the objects are unchanged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.api import _scope_identity, _verified_headers, identity, workflows

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs" / "api"


def test_identity_reexports_the_same_objects():
    assert identity.request_caller_scope is _scope_identity.request_caller_scope
    assert identity.resolve_caller is _verified_headers.resolve_caller
    assert (
        identity.require_verified_workflow_identity is workflows.require_verified_workflow_identity
    )
    assert (
        workflows._require_verified_workflow_identity
        is workflows.require_verified_workflow_identity
    )


@pytest.mark.parametrize("name", ["companion.py", "loop.py"])
def test_no_private_identity_import_across_modules(name):
    tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
    private = [
        a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for a in n.names
        if a.name == "_require_verified_workflow_identity"
    ]
    assert not private, name
