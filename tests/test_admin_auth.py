"""U3 S1 item 6: the admin credential dependency is core API.

The session-policy and usage admin routes took ``require_session_admin`` from
the venue admin router (``admin_sessions``), which moves to ``umrs_r4`` in
U3 S5. It now lives in ``umrs.api.admin_auth``; ``admin_sessions`` re-exports
it, and the typed refusals are unchanged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.api import admin_auth, admin_sessions

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs" / "api"


def test_admin_sessions_reexports_the_core_dependency():
    assert admin_sessions.require_session_admin is admin_auth.require_session_admin


@pytest.mark.parametrize("name", ["admin_usage.py", "admin_session_policy.py"])
def test_core_admin_routes_take_the_dependency_from_core(name):
    tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
    sources = {
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and any(a.name == "require_session_admin" for a in n.names)
    }
    assert sources == {"admin_auth"}, (name, sources)
