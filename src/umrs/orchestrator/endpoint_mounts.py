"""Endpoint mount translation for mini-RiVault provisioning."""

from __future__ import annotations

import json
from typing import Any

MCP_ROOT = "/mcp-servers"
PROJECT_ROOT = "/project"
RAG_ROOT = "/rag-index"


def build_mount_environment(
    *,
    tier: str = "user",
    mounts: dict[str, Any] | None = None,
    egress_domains: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Translate endpoint mount intent into launch env and registry metadata."""
    normalized_mounts = normalize_mounts(mounts)
    domains = [str(domain).strip() for domain in (egress_domains or []) if str(domain).strip()]
    env: dict[str, str] = {"RIVAULT_TIER": tier}
    metadata: dict[str, Any] = {"tier": tier}
    extra_profiles: list[str] = []

    mcp_mounts = normalized_mounts.get("mcp_servers", [])
    if mcp_mounts:
        extra_profiles.extend(["custom-mcp", "egress"])
        env.update(_mcp_mount_env(mcp_mounts))

    project_data = normalized_mounts.get("project_data")
    rag_index = normalized_mounts.get("rag_index")
    if project_data:
        extra_profiles.append("rag")
        env["PROJECT_DATA_PATH"] = project_data["source"]
    if rag_index:
        extra_profiles.append("rag")
        env["RAG_INDEX_PATH"] = rag_index["source"]

    if domains:
        extra_profiles.append("egress")
        env["EGRESS_DOMAIN_ALLOWLIST"] = ",".join(domains)
        metadata["egress_domains"] = domains

    if extra_profiles:
        env["RIVAULT_EXTRA_PROFILES"] = " ".join(_dedupe(extra_profiles))
    if normalized_mounts:
        metadata["mounts"] = normalized_mounts
    return env, metadata


def normalize_mounts(mounts: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize accepted endpoint mount shapes to plain dictionaries."""
    if not mounts:
        return {}

    normalized: dict[str, Any] = {}
    mcp_servers = mounts.get("mcp_servers") or []
    if isinstance(mcp_servers, list):
        normalized_mcp = [
            _normalize_mount(item, root=MCP_ROOT)
            for item in mcp_servers
            if isinstance(item, dict)
        ]
        if normalized_mcp:
            normalized["mcp_servers"] = normalized_mcp

    project_data = mounts.get("project_data")
    if isinstance(project_data, dict):
        normalized["project_data"] = _normalize_mount(project_data, root=PROJECT_ROOT)

    rag_index = mounts.get("rag_index")
    if isinstance(rag_index, dict):
        normalized["rag_index"] = _normalize_mount(rag_index, root=RAG_ROOT)

    return normalized


def _normalize_mount(raw: dict[str, Any], *, root: str) -> dict[str, str]:
    source = str(raw.get("source", "")).strip()
    target = str(raw.get("target", root)).strip() or root
    if not source:
        raise ValueError("mount source is required")
    if not _target_under(target, root):
        raise ValueError(f"mount target must be under {root}: {target}")
    return {"source": source, "target": target}


def _mcp_mount_env(mounts: list[dict[str, str]]) -> dict[str, str]:
    if len(mounts) == 1 and mounts[0]["target"].rstrip("/") == MCP_ROOT:
        return {"MCP_SERVERS_PATH": mounts[0]["source"]}
    return {"RIVAULT_MCP_MOUNTS_JSON": json.dumps(mounts, separators=(",", ":"))}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _target_under(target: str, root: str) -> bool:
    normalized = target.rstrip("/") or "/"
    return normalized == root or normalized.startswith(root + "/")
