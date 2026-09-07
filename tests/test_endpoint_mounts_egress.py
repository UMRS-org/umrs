"""ADR-040 D4: provisions that need the egress profile are refused unless the
site declares an egress proxy image (env EGRESS_GUARD_IMAGE or
site.yaml network.egress_proxy_image)."""

from __future__ import annotations

import pytest
from umrs.orchestrator.endpoint_mounts import (
    EGRESS_PROXY_REFUSAL,
    build_mount_environment,
    egress_proxy_image,
    egress_reasons,
)
from umrs.orchestrator.site_config import SiteConfig

MCP_MOUNTS = {"mcp_servers": [{"source": "/home/alice/tools", "target": "/mcp-servers/tools"}]}


@pytest.fixture
def no_site_proxy(monkeypatch):
    monkeypatch.delenv("EGRESS_GUARD_IMAGE", raising=False)
    monkeypatch.setattr(
        "umrs.orchestrator.site_config.get_site_config", lambda _path=None: SiteConfig()
    )


def test_egress_reasons_names_both_triggers():
    assert egress_reasons(mcp_mounts=False, egress_domains=[]) == []
    assert egress_reasons(mcp_mounts=True, egress_domains=["a.example.com", "b.org"]) == [
        "custom MCP mounts",
        "egress_domains=a.example.com,b.org",
    ]


def test_no_proxy_declared_reads_as_empty(no_site_proxy):
    assert egress_proxy_image() == ""


def test_egress_domains_refused_without_declared_proxy_image(no_site_proxy):
    with pytest.raises(ValueError) as excinfo:
        build_mount_environment(tier="project", egress_domains=["api.example.com"])
    message = str(excinfo.value)
    assert message == EGRESS_PROXY_REFUSAL.format(why="egress_domains=api.example.com")
    assert "refusing per ADR-040 D4" in message


def test_custom_mcp_mounts_refused_without_declared_proxy_image(no_site_proxy):
    with pytest.raises(ValueError, match="custom MCP mounts"):
        build_mount_environment(tier="project", mounts=MCP_MOUNTS)


def test_provision_without_egress_needs_no_proxy(no_site_proxy):
    env, metadata = build_mount_environment(
        tier="project", mounts={"project_data": {"source": "/project/data", "target": "/project"}}
    )
    assert "EGRESS_GUARD_IMAGE" not in env
    assert "egress" not in env.get("RIVAULT_EXTRA_PROFILES", "")
    assert metadata["tier"] == "project"


def test_declared_image_travels_with_the_job_env(monkeypatch):
    monkeypatch.setenv("EGRESS_GUARD_IMAGE", "localhost/r4-mcp:test")
    env, metadata = build_mount_environment(
        tier="project", mounts=MCP_MOUNTS, egress_domains=["api.example.com"]
    )
    assert env["EGRESS_GUARD_IMAGE"] == "localhost/r4-mcp:test"
    assert env["RIVAULT_EXTRA_PROFILES"] == "custom-mcp egress"
    assert env["EGRESS_DOMAIN_ALLOWLIST"] == "api.example.com"
    assert metadata["egress_domains"] == ["api.example.com"]


def test_site_yaml_declaration_is_honoured(monkeypatch):
    monkeypatch.delenv("EGRESS_GUARD_IMAGE", raising=False)
    declared = SiteConfig()
    declared.network.egress_proxy_image = "registry.example.org/r4-mcp@sha256:abc"
    monkeypatch.setattr(
        "umrs.orchestrator.site_config.get_site_config", lambda _path=None: declared
    )
    env, _ = build_mount_environment(tier="project", egress_domains=["api.example.com"])
    assert env["EGRESS_GUARD_IMAGE"] == "registry.example.org/r4-mcp@sha256:abc"
