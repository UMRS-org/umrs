"""U3 S5a: the seams between core and the S5 cluster use public core surfaces.

Before the remote-session / venue / plan-run cluster moves to umrs_r4 (S5b):
- core ``admin_session_policy`` no longer reaches the venue module's private
  ``_site_name`` (it uses ``site_config.this_site_name``);
- the cluster's routes take identity helpers from the public
  ``umrs.api.identity``, not the private ``_verified_headers`` /
  ``_scope_identity`` modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

API = Path(__file__).resolve().parents[1] / "src" / "umrs" / "api"
R4_API = (
    Path(__file__).resolve().parents[1] / "packages" / "umrs-r4" / "src" / "umrs_r4" / "api"
)  # the cluster since U3 S5b


def _from_imports(name: str) -> set[str]:
    path = API / name if (API / name).exists() else R4_API / name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        f"{'.' * n.level}{n.module or ''}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    }


def test_core_admin_session_policy_does_not_import_the_venue_routes():
    assert ".companion_venue" not in _from_imports("admin_session_policy.py")


def test_cluster_routes_use_the_public_identity_module():
    for name in ("companion_sessions.py", "companion_venue.py"):
        private = _from_imports(name) & {
            "._verified_headers",
            "._scope_identity",
            "umrs.api._verified_headers",
            "umrs.api._scope_identity",
        }
        assert not private, (name, private)


def test_strict_mode_is_public_and_tracks_the_env(monkeypatch):
    from umrs.api.identity import strict_mode

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    assert strict_mode() is True
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    assert strict_mode() is False
