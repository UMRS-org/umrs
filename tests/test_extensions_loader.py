"""U3 S3a: the core extension loader, its router and startup seams.

The r4 parts still in core (remote sessions, venues, admin sessions, plan-run
venue, the plan-loop host) register as extension ``r4`` through the same
hooks the umrs-r4 package will use; with ``r4`` disabled they are absent.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import APIRouter, FastAPI
from umrs.extensions import (
    EXTENSION_API_VERSION,
    ExtensionError,
    enabled_extensions,
    load_extensions,
    loader,
    reset_extension_context,
)
from umrs.extensions.context import ExtensionContext
from umrs.extensions.seams import mount_extension_routers, route_keys, run_extension_startups

R4_PREFIXES = ("/umrs/companion/sessions", "/umrs/admin/sessions", "/umrs/traces/plan-runs")


@pytest.fixture(autouse=True)
def _fresh_context(monkeypatch):
    monkeypatch.delenv(loader.ENV, raising=False)
    reset_extension_context()
    yield
    reset_extension_context()


# ── enablement ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("none", ()), ("", ()), ("-", ()), ("r4", ("r4",)), ("r4, other  x", ("r4", "other", "x"))],
)
def test_env_wins(monkeypatch, raw, expected):
    monkeypatch.setenv(loader.ENV, raw)
    monkeypatch.setattr(loader, "_site_extensions", lambda: ["never-read"])
    assert enabled_extensions() == expected


def test_site_yaml_list_applies_when_env_unset(monkeypatch):
    monkeypatch.setattr(loader, "_site_extensions", lambda: ())
    assert enabled_extensions() == ()


def test_transitional_default_is_r4(monkeypatch):
    monkeypatch.setattr(loader, "_site_extensions", lambda: None)
    assert enabled_extensions() == ("r4",)


def test_site_config_parses_and_validates_extensions():
    from umrs.orchestrator.site_config import _parse_extensions, _parse_raw

    assert _parse_raw({}).extensions is None
    assert _parse_raw({"extensions": ["r4"]}).extensions == ["r4"]
    assert _parse_raw({"extensions": []}).extensions == []
    for bad in ("r4", [1], [""]):
        with pytest.raises(ValueError, match="extensions"):
            _parse_extensions(bad)


# ── loading ─────────────────────────────────────────────────────────────────


def test_r4_registers_the_in_tree_parts():
    ctx = load_extensions(("r4",))
    assert ctx.active == ["r4"]
    prefixes = {reg.router.prefix for reg in ctx.routers}
    assert prefixes == set(R4_PREFIXES)
    assert {reg.extension for reg in ctx.routers} == {"r4"}
    assert [(ext, fn.__name__) for ext, fn in ctx.startups] == [("r4", "init_loop_plan_host")]


def test_nothing_enabled_registers_nothing():
    ctx = load_extensions(())
    assert (ctx.active, ctx.routers, ctx.startups) == ([], [], [])


def test_enabled_but_absent_is_skipped(caplog):
    ctx = load_extensions(("nope",))
    assert ctx.active == []
    assert "enabled but not installed" in caplog.text


def _fake_entry_point(name: str, register):
    return SimpleNamespace(name=name, load=lambda: register)


def _module_register(version):
    import types

    mod = types.ModuleType("fake_ext_mod")
    mod.EXTENSION_API_VERSION = version
    calls: list[str] = []

    def register(ext):
        calls.append("called")
        ext.add_startup(lambda _pool: None)

    register.__module__ = mod.__name__
    return mod, register, calls


def test_installed_extension_registers_after_its_in_tree_part(monkeypatch):
    mod, register, calls = _module_register(EXTENSION_API_VERSION)
    monkeypatch.setitem(__import__("sys").modules, mod.__name__, mod)
    monkeypatch.setattr(loader, "_installed", lambda: {"r4": _fake_entry_point("r4", register)})
    ctx = load_extensions(("r4",))
    assert calls == ["called"]
    assert [ext for ext, _fn in ctx.startups] == ["r4", "r4"]
    assert ctx.startups[0][1].__name__ == "init_loop_plan_host"


def test_api_version_mismatch_fails(monkeypatch):
    mod, register, calls = _module_register(EXTENSION_API_VERSION + 1)
    monkeypatch.setitem(__import__("sys").modules, mod.__name__, mod)
    monkeypatch.setattr(loader, "_installed", lambda: {"x": _fake_entry_point("x", register)})
    with pytest.raises(ExtensionError, match="targets ExtensionContext API"):
        load_extensions(("x",))
    assert calls == []


def test_duplicate_job_type_and_iteration_kind_fail():
    from umrs.extensions import JobTypeRegistration, LoopIterationKind

    ctx = ExtensionContext()
    reg = JobTypeRegistration(value="t", build_lane=lambda _p: None)
    ctx.add_job_type(reg)
    with pytest.raises(ExtensionError):
        ctx.add_job_type(reg)
    kind = LoopIterationKind("k", lambda _l: False, lambda *_a: None, lambda *_a: None, object)
    ctx.add_loop_iteration_kind(kind)
    with pytest.raises(ExtensionError):
        ctx.add_loop_iteration_kind(kind)


# ── seams ───────────────────────────────────────────────────────────────────


def _router(prefix: str, path: str = "/a/{i}") -> APIRouter:
    router = APIRouter(prefix=prefix)
    router.add_api_route(path, lambda i: i, methods=["GET"])
    return router


def test_mount_appends_and_refuses_collisions():
    app = FastAPI()
    app.include_router(_router("/umrs/core"))
    ctx = ExtensionContext().bind("x")
    ctx.add_router(_router("/umrs/ext"), family="ext")
    assert mount_extension_routers(app, ctx) == ["ext"]
    assert ("GET", "/umrs/ext/a/{i}") in route_keys(app)

    clash = ExtensionContext().bind("y")
    clash.add_router(_router("/umrs/core"), family="core-copy")
    with pytest.raises(ExtensionError, match="collides"):
        mount_extension_routers(app, clash)


def test_startups_are_isolated(caplog):
    seen: list[object] = []
    ctx = ExtensionContext().bind("x")

    def boom(_pool):
        raise RuntimeError("no")

    ctx.add_startup(boom)
    ctx.add_startup(seen.append)
    run_extension_startups(ctx, "pool")
    assert seen == ["pool"]
    assert "startup boom failed" in caplog.text


# ── the real sidecar ────────────────────────────────────────────────────────


def _sidecar(monkeypatch, extensions: str | None):
    if extensions is not None:
        monkeypatch.setenv(loader.ENV, extensions)
    reset_extension_context()
    from umrs.server import http_sidecar

    with patch.object(http_sidecar, "_build_db_pool", lambda _app: None):
        return http_sidecar._create_app()


def _r4_routes():
    """(method, path) -> endpoint for every route the in-tree r4 part registers."""
    ctx = load_extensions(("r4",))
    return {
        (method, route.path): route.endpoint
        for reg in ctx.routers
        for route in reg.router.routes
        for method in route.methods
    }


def test_sidecar_without_r4_serves_no_r4_routes(monkeypatch):
    # Fails on the pre-S3 tree, which mounted these routers unconditionally.
    served = route_keys(_sidecar(monkeypatch, "none"))
    assert not served & set(_r4_routes())


def test_sidecar_default_serves_the_r4_routes(monkeypatch):
    assert set(_r4_routes()) <= route_keys(_sidecar(monkeypatch, None))


def _flat(app):
    """Served routes in match order; lazy includes (FastAPI 0.13x) flattened."""
    table = []
    for route in app.router.routes:
        if hasattr(route, "original_router"):
            table += [
                (c.methods or set(), c.path_regex, c.endpoint) for c in route.effective_candidates()
            ]
        elif hasattr(route, "path_regex"):
            table.append((getattr(route, "methods", None) or set(), route.path_regex, None))
    return table


def _winner(table, method: str, path: str):
    concrete = re.sub(r"\{[^}]+\}", "zz1", path)
    return next((ep for ms, rx, ep in table if method in ms and rx.match(concrete)), None)


def test_every_r4_route_resolves_to_its_own_endpoint(monkeypatch):
    """Mount order is safe: extension routers mount after every core router
    (they used to sit mid-list), and no core route wins any r4 path."""
    table = _flat(_sidecar(monkeypatch, None))
    lost = sorted(k for k, ep in _r4_routes().items() if _winner(table, *k) is not ep)
    assert not lost, lost
