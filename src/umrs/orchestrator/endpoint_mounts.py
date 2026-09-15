"""Endpoint mount translation for mini-RiVault provisioning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MCP_ROOT = "/mcp-servers"
PROJECT_ROOT = "/project"
RAG_ROOT = "/rag-index"
DEFAULT_ALLOWED_SOURCE_PREFIXES = (
    "/data",
    "/home",
    "/lvs0",
    "/models",
    "/project",
    "/projects",
    "/scratch",
    "/shared",
    "/Users",
    "/work",
)
DEFAULT_DENIED_SOURCE_PREFIXES = (
    "/dev",
    "/etc",
    "/private/etc",
    "/proc",
    "/root",
    "/run",
    "/sys",
    "/var/run",
)


def build_mount_environment(
    *,
    tier: str = "user",
    mounts: dict[str, Any] | None = None,
    egress_domains: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Translate endpoint mount intent into launch env and registry metadata."""
    normalized_mounts = normalize_mounts(mounts)
    domains = [
        str(domain).strip() for domain in (egress_domains or []) if str(domain).strip()
    ]
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
    source = validate_mount_source(raw.get("source", ""))
    target = str(raw.get("target", root)).strip() or root
    if not _target_under(target, root):
        raise ValueError(f"mount target must be under {root}: {target}")
    return {"source": source, "target": target}


def validate_mount_source(source: Any) -> str:
    """Return a normalized host source path if it is safe to bind mount."""
    resolved = _resolved_mount_source(source)
    _reject_denied_mount_source(resolved)
    _require_allowed_mount_source(resolved)
    return str(resolved)


def _resolved_mount_source(source: Any) -> Path:
    raw = str(source).strip()
    if not raw:
        raise ValueError("mount source is required")
    if "\x00" in raw or "://" in raw:
        raise ValueError("mount source must be a local absolute path")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("mount source must be an absolute host path")
    return path.resolve(strict=False)


def _reject_denied_mount_source(resolved: Path) -> None:
    denied_prefixes = _source_prefixes(
        "UMRS_ENDPOINT_DENIED_MOUNT_PREFIXES",
        DEFAULT_DENIED_SOURCE_PREFIXES,
    )
    if _path_under_any(resolved, denied_prefixes):
        raise ValueError(f"mount source is denied by policy: {resolved}")


def _require_allowed_mount_source(resolved: Path) -> None:
    allowed_prefixes = _source_prefixes(
        "UMRS_ENDPOINT_ALLOWED_MOUNT_PREFIXES",
        DEFAULT_ALLOWED_SOURCE_PREFIXES,
    )
    if allowed_prefixes and not _path_under_any(resolved, allowed_prefixes):
        allowed = ", ".join(str(prefix) for prefix in allowed_prefixes)
        raise ValueError(f"mount source must be under an allowed prefix ({allowed})")


def _source_prefixes(env_name: str, defaults: tuple[str, ...]) -> list[Path]:
    raw = os.environ.get(env_name)
    values = (
        [part.strip() for part in raw.split(",")] if raw is not None else list(defaults)
    )
    return [Path(value).expanduser().resolve(strict=False) for value in values if value]


def _path_under_any(path: Path, prefixes: list[Path]) -> bool:
    return any(_path_under(path, prefix) for prefix in prefixes)


def _path_under(path: Path, prefix: Path) -> bool:
    return path == prefix or prefix in path.parents


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
