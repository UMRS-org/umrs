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
    caps_raw = _object(data.get("capabilities"), f"manifest {name!r} capabilities")
    caps: dict[str, Any] = {
        "transport": _choice(
            caps_raw.get("transport"), "stdio", _VALID_TRANSPORTS, name, "transport"
        ),
        "risk": _choice(caps_raw.get("risk"), "standard", _VALID_RISK, name, "risk"),
    }
    network = _object(
        caps_raw.get("network"), f"manifest {name!r} capabilities.network"
    )
    caps["egress_allowlist"] = _string_list(
        network.get("outbound"), f"manifest {name!r} capabilities.network.outbound"
    )
    fs = _object(
        caps_raw.get("filesystem"), f"manifest {name!r} capabilities.filesystem"
    )
    caps["readonly_mounts"] = _string_list(
        fs.get("readonly"), f"manifest {name!r} capabilities.filesystem.readonly"
    )
    caps["dangerous_tools"] = _dangerous_tools(caps_raw, tools, name)
    requires = caps_raw.get("requires")
    caps["requires"] = (
        _string_list(requires, f"manifest {name!r} capabilities.requires")
        if requires is not None
        else []
    )
    return {"name": name, "version": version, "tools": tools, "capabilities": caps}


def _object(value: Any, label: str) -> dict[str, Any]:
    """An optional object field ({} when absent); a non-object is refused."""
    value = value or {}
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    return value


def _choice(value: Any, default: str, valid: Any, name: str, field: str) -> str:
    chosen = str(value or default).strip()
    if chosen not in valid:
        raise ManifestError(
            f"manifest {name!r} {field} {chosen!r} invalid (one of {sorted(valid)})"
        )
    return chosen


def _dangerous_tools(
    caps_raw: dict[str, Any], tools: list[str], name: str
) -> list[str]:
    """approval.dangerous_tools — must name only declared tools."""
    approval = _object(
        caps_raw.get("approval"), f"manifest {name!r} capabilities.approval"
    )
    dangerous = _string_list(
        approval.get("dangerous_tools"),
        f"manifest {name!r} capabilities.approval.dangerous_tools",
    )
    unknown = sorted(set(dangerous) - set(tools))
    if unknown:
        raise ManifestError(
            f"manifest {name!r} approval.dangerous_tools names tools absent from tools: {unknown}"
        )
    return dangerous


def _string_list(raw: Any, label: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(
        isinstance(x, str) and x.strip() for x in raw
    ):
        raise ManifestError(f"{label} must be a list of non-empty strings")
    return [str(x).strip() for x in raw]
