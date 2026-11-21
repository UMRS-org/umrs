"""umrs-r4 packaging proof (U3 S2): the entry point and the moved adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import umrs_r4

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 runners
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def test_entry_point_names_register():
    eps = _project()["entry-points"]["umrs.extensions"]
    assert eps == {"r4": "umrs_r4:register"}
    module, _, attr = eps["r4"].partition(":")
    assert getattr(sys.modules[module], attr) is umrs_r4.register


def _registered():
    from umrs.extensions.context import ExtensionContext

    ctx = ExtensionContext().bind("r4")
    umrs_r4.register(ctx)
    return ctx


def test_register_adds_every_r4_router():
    """U3 S4b + S5: the extension mounts every route the r4 runtime adds."""
    assert {reg.router.prefix for reg in _registered().routers} == {
        "/umrs/companion",
        "/umrs/companion/sessions",
        "/umrs/admin/sessions",
        "/umrs/traces",
        "/umrs/traces/plan-runs",
    }


def test_register_adds_the_lanes_kinds_startups_and_listeners():
    from umrs_r4.trust.approval_listener import mint_session_trust_on_always
    from umrs_r4.trust.archive_participant import revoke_trust_on_archive

    ctx = _registered()
    assert list(ctx.job_types) == ["frontend_session"]
    assert list(ctx.loop_iteration_kinds) == ["plan"]
    startups = [fn for _ext, fn in ctx.startups]
    assert startups == [umrs_r4.init_trust_store, umrs_r4.init_loop_plan_host]
    assert ctx.session_archive_participants == [revoke_trust_on_archive]
    assert ctx.approval_listeners == [mint_session_trust_on_always]
    assert umrs_r4.EXTENSION_API_VERSION == 1


def test_lockstep_version_with_core():
    core_toml = PYPROJECT.parents[2] / "src" / "pyproject.toml"
    core = tomllib.loads(core_toml.read_text(encoding="utf-8"))["project"]
    assert _project()["version"] == core["version"]


def test_adapter_registers_into_core_registry(tmp_path: Path):
    from umrs.knowledge.verifiers import VERIFIER_REGISTRY
    from umrs_r4.verifiers import MCPVerifierConfig, register_mcp_verifier

    config = MCPVerifierConfig(tool_name="s2_probe", config_path=tmp_path)
    verifier = register_mcp_verifier(config)
    try:
        assert VERIFIER_REGISTRY[verifier.name] is verifier
    finally:
        VERIFIER_REGISTRY.pop(verifier.name, None)


def test_core_no_longer_reexports_the_adapter():
    import umrs.knowledge.verifiers as core_verifiers

    for name in ("MCPVerifier", "MCPVerifierConfig", "register_mcp_verifier"):
        assert not hasattr(core_verifiers, name)
        assert name not in core_verifiers.__all__
