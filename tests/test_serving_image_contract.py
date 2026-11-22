"""U3 S6f (ADR-117 D10): core UMRS names no R4 serving image or command.

The egress guard and the custom MCP runner run a site-declared image and
command (site.yaml network.egress_proxy_image/_command and
runtime.mcp_custom_image/_command, exported as EGRESS_GUARD_IMAGE/_COMMAND and
MCP_CUSTOM_IMAGE/_COMMAND). An undeclared one refuses by name, in the
provision guard and again in the wrapper. These pins read only this umrs
checkout, so they run in UMRS-org's standalone CI too.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from umrs.orchestrator import site_config as sc
from umrs.orchestrator.endpoint_mounts import (
    MCP_CUSTOM_RUNNER_REFUSAL,
    build_mount_environment,
)
from umrs.orchestrator.site_config import NetworkConfig, RuntimeConfig, SiteConfig

UMRS = Path(__file__).resolve().parents[1]
MINI_RIVAULT = UMRS / "src" / "umrs" / "mini-rivault-llm"
SERVE_MODEL = MINI_RIVAULT / "serve-model.yaml"
SERVE_MODEL_SLURM = MINI_RIVAULT / "serve-model.slurm"
SITE_CONFIG_PY = UMRS / "src" / "umrs" / "orchestrator" / "site_config.py"
MCP_MOUNTS = {"mcp_servers": [{"source": "/home/alice/tools", "target": "/mcp-servers/tools"}]}
_DECLARED = {
    "EGRESS_GUARD_IMAGE": "registry.example.org/egress:1",
    "EGRESS_GUARD_COMMAND": "egress-guard",
    "MCP_CUSTOM_IMAGE": "registry.example.org/mcp-runner:1",
    "MCP_CUSTOM_COMMAND": "mcp-runner",
}


@pytest.mark.parametrize("path", [SERVE_MODEL, SERVE_MODEL_SLURM, SITE_CONFIG_PY])
def test_core_serving_files_name_no_r4_mcp(path):
    text = path.read_text(encoding="utf-8")
    assert "r4-mcp" not in text
    assert "r4_mcp" not in text


def test_site_config_defaults_declare_no_serving_image():
    assert NetworkConfig().egress_proxy_image == ""
    assert NetworkConfig().egress_proxy_command == ""
    assert RuntimeConfig().mcp_custom_image == ""
    assert RuntimeConfig().mcp_custom_command == ""


def test_dead_mcp_vision_service_is_gone():
    services = yaml.safe_load(SERVE_MODEL.read_text())["services"]
    assert "mcp-vision" not in services
    assert "mcp-vision" not in (MINI_RIVAULT / "mcpo_fs" / "config.json").read_text()


def test_compose_runs_the_declared_images_and_commands():
    services = yaml.safe_load(SERVE_MODEL.read_text())["services"]
    guard, custom = services["egress-guard"], services["mcp-custom"]
    assert guard["image"] == "${EGRESS_GUARD_IMAGE}"
    assert guard["command"].startswith("${EGRESS_GUARD_COMMAND} --port ")
    assert custom["image"] == "${MCP_CUSTOM_IMAGE}"
    assert custom["command"].startswith("${MCP_CUSTOM_COMMAND} --config-dir /mcp-servers ")


def test_site_yaml_commands_flow_to_env(tmp_path, monkeypatch):
    for name in _DECLARED:
        monkeypatch.delenv(name, raising=False)
    site = tmp_path / "site.yaml"
    site.write_text(
        "name: t\n"
        "runtime:\n  mcp_custom_image: reg/runner:1\n  mcp_custom_command: run-mcp\n"
        "network:\n  egress_proxy_image: reg/egress:1\n  egress_proxy_command: run-guard\n"
    )
    env = sc.load_site_config(str(site)).to_env()
    assert env["EGRESS_GUARD_IMAGE"] == "reg/egress:1"
    assert env["EGRESS_GUARD_COMMAND"] == "run-guard"
    assert env["MCP_CUSTOM_IMAGE"] == "reg/runner:1"
    assert env["MCP_CUSTOM_COMMAND"] == "run-mcp"


def test_undeclared_values_export_nothing():
    env = SiteConfig().to_env()
    for name in _DECLARED:
        assert name not in env


@pytest.fixture
def declared(monkeypatch):
    for name, value in _DECLARED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "umrs.orchestrator.site_config.get_site_config", lambda _path=None: SiteConfig()
    )


def test_mcp_mounts_carry_the_declared_runner(declared):
    env, _ = build_mount_environment(tier="project", mounts=MCP_MOUNTS)
    for name, value in _DECLARED.items():
        assert env[name] == value


@pytest.mark.parametrize(
    ("missing", "what"), [("MCP_CUSTOM_IMAGE", "image"), ("MCP_CUSTOM_COMMAND", "command")]
)
def test_mcp_mounts_without_a_runner_declaration_refuse_by_name(
    declared, monkeypatch, missing, what
):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError) as excinfo:
        build_mount_environment(tier="project", mounts=MCP_MOUNTS)
    assert str(excinfo.value) == MCP_CUSTOM_RUNNER_REFUSAL.format(what=what, env=missing[11:])
    assert f"runtime.mcp_custom_{what}" in str(excinfo.value)


def test_egress_domains_alone_need_no_mcp_runner(declared, monkeypatch):
    monkeypatch.delenv("MCP_CUSTOM_IMAGE")
    env, _ = build_mount_environment(tier="project", egress_domains=["api.example.com"])
    assert "MCP_CUSTOM_IMAGE" not in env
    assert env["EGRESS_GUARD_COMMAND"] == "egress-guard"


# ── the wrapper's refusal block, executed ───────────────────────────────────


def _refusal_block() -> str:
    text = SERVE_MODEL_SLURM.read_text()
    start = text.index("# BEGIN serving-image refusals")
    return text[start : text.index("# END serving-image refusals", start)]


def _run_refusals(profiles: str, **env: str) -> subprocess.CompletedProcess:
    script = "rivault_ensure_image() { return 0; }\n" + _refusal_block() + "echo LAUNCH\n"
    base = {"PATH": "/usr/bin:/bin", "RIVAULT_PROFILES": profiles}
    return subprocess.run(
        ["bash", "-c", script], env={**base, **env}, capture_output=True, text=True, check=False
    )


def test_wrapper_launches_when_everything_is_declared():
    proc = _run_refusals("harness egress custom-mcp", **_DECLARED)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LAUNCH" in proc.stdout


@pytest.mark.parametrize("missing", ["EGRESS_GUARD_IMAGE", "EGRESS_GUARD_COMMAND"])
def test_wrapper_refuses_egress_without_declared_image_or_command(missing):
    env = {k: v for k, v in _DECLARED.items() if k != missing}
    proc = _run_refusals("harness egress", **env)
    assert proc.returncode == 1
    assert "FATAL: egress profile requested but this site declares no egress-guard" in proc.stdout
    assert "LAUNCH" not in proc.stdout


@pytest.mark.parametrize("missing", ["MCP_CUSTOM_IMAGE", "MCP_CUSTOM_COMMAND"])
def test_wrapper_refuses_custom_mcp_without_declared_runner(missing):
    env = {k: v for k, v in _DECLARED.items() if k != missing}
    proc = _run_refusals("harness egress custom-mcp", **env)
    assert proc.returncode == 1
    assert "FATAL: custom-mcp profile requested but this site declares no custom MCP" in proc.stdout


def test_wrapper_without_egress_profiles_needs_no_declaration():
    proc = _run_refusals("harness vllm")
    assert proc.returncode == 0
    assert "LAUNCH" in proc.stdout
