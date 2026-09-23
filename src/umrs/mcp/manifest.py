"""hatch.json manifest validation — the mcp-trusted start executor's input contract.

The manifest format EXISTS (S8 contract pin; rich examples: the fugaku
and rikyu-hpc packages carry the ADR-061 capabilities block verbatim —
per-server network allowlists, filesystem mounts, auth, dangerous-tool
gating). This module is the SERVER-SIDE validator the MCP_SERVER start
executor consumes when the mcp-trusted sandbox lands: it checks the
manifest's shape and refuses by name — never approximating a malformed
manifest into a launch.
"""

from __future__ import annotations

from typing import Any

_VALID_TRANSPORTS = frozenset({"stdio", "http"})
# "low" ships on real manifests (rikyu-docs): the registry's own vocabulary.
_VALID_RISK = frozenset({"low", "standard", "privileged"})


class ManifestError(ValueError):
    """Named refusal: the manifest is malformed (never approximated)."""


def validate_manifest(data: Any) -> dict[str, Any]:
    """Validate one hatch.json document; return the normalized view.

    Required: name, version, tools (non-empty list of strings).
    capabilities: optional but shape-checked when present — transport,
    risk, network.outbound (list of strings), filesystem.readonly (list),
    approval.dangerous_tools (list, MUST be a subset of tools),
    requires/auth free-form (named surfaces, not interpreted here).
    """
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    name = str(data.get("name") or "").strip()
    if not name:
        raise ManifestError("manifest requires a non-empty name")
    version = str(data.get("version") or "").strip()
    if not version:
        raise ManifestError(f"manifest {name!r} requires a version")
    tools = _string_list(data.get("tools"), f"manifest {name!r} tools")
    if not tools:
        raise ManifestError(f"manifest {name!r} requires at least one tool")

    caps_raw = data.get("capabilities") or {}
    if not isinstance(caps_raw, dict):
        raise ManifestError(f"manifest {name!r} capabilities must be an object")
    caps: dict[str, Any] = {}
    transport = str(caps_raw.get("transport") or "stdio").strip()
    if transport not in _VALID_TRANSPORTS:
        raise ManifestError(
            f"manifest {name!r} transport {transport!r} invalid (one of {sorted(_VALID_TRANSPORTS)})"
        )
    caps["transport"] = transport
    risk = str(caps_raw.get("risk") or "standard").strip()
    if risk not in _VALID_RISK:
        raise ManifestError(
            f"manifest {name!r} risk {risk!r} invalid (one of {sorted(_VALID_RISK)})"
        )
    caps["risk"] = risk
    network_raw = caps_raw.get("network") or {}
    if not isinstance(network_raw, dict):
        raise ManifestError(f"manifest {name!r} capabilities.network must be an object")
    caps["egress_allowlist"] = _string_list(
        network_raw.get("outbound"), f"manifest {name!r} capabilities.network.outbound"
    )
    fs_raw = caps_raw.get("filesystem") or {}
    if not isinstance(fs_raw, dict):
        raise ManifestError(
            f"manifest {name!r} capabilities.filesystem must be an object"
        )
    caps["readonly_mounts"] = _string_list(
        fs_raw.get("readonly"), f"manifest {name!r} capabilities.filesystem.readonly"
    )
    approval_raw = caps_raw.get("approval") or {}
    if not isinstance(approval_raw, dict):
        raise ManifestError(
            f"manifest {name!r} capabilities.approval must be an object"
        )
    dangerous = _string_list(
        approval_raw.get("dangerous_tools"),
        f"manifest {name!r} capabilities.approval.dangerous_tools",
    )
    unknown = sorted(set(dangerous) - set(tools))
    if unknown:
        raise ManifestError(
            f"manifest {name!r} approval.dangerous_tools names tools absent from tools: {unknown}"
        )
    caps["dangerous_tools"] = dangerous
    if caps_raw.get("requires") is not None:
        caps["requires"] = _string_list(
            caps_raw.get("requires"), f"manifest {name!r} capabilities.requires"
        )
    else:
        caps["requires"] = []

    return {"name": name, "version": version, "tools": tools, "capabilities": caps}


def _string_list(raw: Any, label: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(
        isinstance(x, str) and x.strip() for x in raw
    ):
        raise ManifestError(f"{label} must be a list of non-empty strings")
    return [str(x).strip() for x in raw]
