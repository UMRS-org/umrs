"""U3 S1 item 5: session-policy helpers that core needs live in core.

Core session_policy imported the egress allowlist normalizer from the frontend
venue; core routes raised policy refusals through the venue router and
checked a loop's web choice through the venue service; the venue router read
the site name its own way. All four now live in core.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.api import companion_venue
from umrs.api.session_policy_errors import policy_typed
from umrs.orchestrator import site_config
from umrs.session_policy import WebChoice
from umrs.session_policy.model import normalize_allow_domain
from umrs.session_policy.web_check import web_choice
from umrs.venues import frontend_podman, frontend_service

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs"


def _imported(rel: str) -> set[str]:
    tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
    return {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}


def test_core_session_policy_does_not_import_the_venue():
    for rel in ("session_policy/model.py", "session_policy/web_check.py"):
        assert not {m for m in _imported(rel) if m.startswith("umrs.venues")}, rel


@pytest.mark.parametrize("rel", ["api/admin_session_policy.py", "api/loop.py"])
def test_core_routes_take_policy_typed_from_core(rel):
    tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
    sources = {
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and any(a.name == "policy_typed" for a in n.names)
    }
    assert sources == {"session_policy_errors"}, (rel, sources)


def test_reexports_are_the_core_objects():
    assert frontend_podman.normalize_allow_domain is normalize_allow_domain
    assert companion_venue.policy_typed is policy_typed
    assert frontend_service._web_choice is web_choice


def test_normalize_and_web_choice_keep_their_behaviour():
    assert normalize_allow_domain("  Example.ORG. ") == "example.org"
    assert web_choice(" allowlist ", [" a.org ", "", None]) == WebChoice("allowlist", ("a.org",))


def test_this_site_name_is_soft(monkeypatch):
    from umrs.jobs import store

    def boom() -> str:
        raise ValueError("unreadable site.yaml")

    monkeypatch.setattr(store, "configured_site_name", boom)
    assert site_config.this_site_name() == ""
    monkeypatch.setattr(store, "configured_site_name", lambda: "rccs-front")
    assert site_config.this_site_name() == "rccs-front"
