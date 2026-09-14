"""MCP-based knowledge verifier adapter.

Wraps a project-registered MCP tool (declared with KNOWLEDGE_VERIFY
capability) as a Verifier so it can run inside verify jobs alongside
the existing cheap and model-backed verifiers.

Security invariants:
- MCP tool MUST declare KNOWLEDGE_VERIFY in config.json tool_capabilities
- MCP tools run in sandboxed containers (egress_guard + read_only)
- Findings are tagged with provenance "mcp:<tool_name>"
- Capability check fails fast — missing capability = no execution
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Finding, VerifierBase

logger = logging.getLogger(__name__)


@dataclass
class MCPVerifierConfig:
    """Configuration for an MCP-based verifier."""

    tool_name: str
    config_path: Path  # Path to config.json or directory containing it
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def server_name(self) -> str:
        """Derive a human-readable server name from config path."""
        config = self.config_path
        if config.is_file():
            config = config.parent
        name = config.name
        return f"mcp-{name}" if name else "mcp-unknown"

    @property
    def provenance(self) -> str:
        return f"mcp:{self.tool_name}"


class MCPVerifier(VerifierBase):
    """Run a single MCP tool as a knowledge verifier.

    The MCP tool must declare KNOWLEDGE_VERIFY capability in its
    config.json.  The verifier reads the manifest, validates the
    capability, then invokes the tool within the standard MCP sandbox.
    """

    name: str = "mcp_verifier"  # Set per-instance in __init__
    description: str = "MCP-based knowledge verifier"
    severity: str = "warning"

    def __init__(self, config: MCPVerifierConfig) -> None:
        self.config = config
        self.name = f"mcp_{self.config.tool_name}"

    # -- capability check ------------------------------------------------

    def validate_config(self) -> list[str]:
        """Validate that the MCP manifest declares KNOWLEDGE_VERIFY.

        Returns a list of error strings; empty means OK.
        """
        errors: list[str] = []

        try:
            manifests = self._load_manifests()
        except Exception as exc:
            errors.append(f"Failed to load MCP manifest: {exc}")
            return errors

        if not manifests:
            errors.append("No MCP manifest found")
            return errors

        manifest = manifests[0]
        tool_caps = manifest.tool_capabilities.get(self.config.tool_name)
        if tool_caps is None:
            errors.append(
                f"Tool '{self.config.tool_name}' not declared in manifest tools"
            )
            return errors

        # Lazy import to avoid umrs→r4_mcp circular import at module load.
        from r4_mcp.security import ToolCapability

        required_cap = ToolCapability.KNOWLEDGE_VERIFY
        if (tool_caps & required_cap) == ToolCapability.NONE:
            errors.append(
                f"Tool '{self.config.tool_name}' missing KNOWLEDGE_VERIFY capability"
            )

        return errors

    # -- manifest loading ------------------------------------------------

    def _load_manifests(self) -> list:
        """Load CustomMCPManifest from config_path."""
        from r4_mcp.runners.custom_mcp import load_manifests

        return load_manifests(str(self.config.config_path))

    # -- verifier protocol -----------------------------------------------

    async def run(
        self,
        knowledge_store,
        project_id: str,
    ) -> list[Finding]:
        """Run the MCP tool and convert its output to findings."""
        from r4_mcp.security import ToolCapability

        # Validate capability first
        errors = self.validate_config()
        if errors:
            return [
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity="info",
                    rule="capability_check_failed",
                    message="MCP capability validation failed",
                )
            ]

        # Load manifest and get required capability
        manifests = self._load_manifests()
        manifest = manifests[0]
        required_cap = manifest.tool_capabilities.get(
            self.config.tool_name, ToolCapability.NONE
        )

        # Check KNOWLEDGE_VERIFY capability (redundant guard — validate_config already checked)
        if (required_cap & ToolCapability.KNOWLEDGE_VERIFY) == ToolCapability.NONE:
            return [
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity="info",
                    rule="missing_knowledge_verify_capability",
                    message=(
                        f"Tool '{self.config.tool_name}' does not declare"
                        f" KNOWLEDGE_VERIFY capability — rejected"
                    ),
                )
            ]

        # Build the MCP server with this single manifest
        from r4_mcp.runners.custom_mcp import CustomMCPServer
        from r4_mcp.security import CapabilityEnforcer, SecurityContext

        enforcer = CapabilityEnforcer()
        ctx = SecurityContext(
            user_id=project_id,
            session_id=f"verify-{self.name}",
            granted_capabilities=ToolCapability.STANDARD_USER,
        )
        server = CustomMCPServer(manifests, enforcer=enforcer, context=ctx)

        # Invoke the MCP tool
        try:
            result = await asyncio.wait_for(
                server.handle_call_tool(
                    {"name": self.config.tool_name, "arguments": self.config.arguments}
                ),
                timeout=manifest.sandbox.max_runtime_sec * 2,
            )
        except asyncio.TimeoutError:
            return [
                Finding(
                    verifier=self.name,
                    artifact_id=None,
                    severity="warning",
                    rule="mcp_tool_timeout",
                    message=f"MCP tool '{self.config.tool_name}' timed out",
                )
            ]

        # Parse result — MCP returns content blocks with text
        return self._extract_findings(result, self.name)

    def _extract_findings(
        self, result: dict[str, Any], verifier_name: str
    ) -> list[Finding]:
        """Extract findings from an MCP tool result."""
        findings: list[Finding] = []

        if result.get("isError"):
            content = self._get_content(result)
            findings.append(
                Finding(
                    verifier=verifier_name,
                    artifact_id=None,
                    severity="warning",
                    rule="mcp_error",
                    message=f"MCP tool error: {content}",
                )
            )
            return findings

        content = self._get_content(result)
        if not content:
            return findings

        # Try to parse structured findings (JSON)
        parsed = self._parse_structured(content)
        if parsed:
            for item in parsed:
                findings.append(
                    Finding(
                        verifier=verifier_name,
                        artifact_id=item.get("artifact_id"),
                        severity=item.get("severity", "warning"),
                        rule=item.get("rule", "mcp_finding"),
                        message=item.get("message", content),
                    )
                )
            return findings

        # Fallback: single finding from raw text
        findings.append(
            Finding(
                verifier=verifier_name,
                artifact_id=None,
                severity="warning",
                rule="mcp_finding",
                message=content[:500],
            )
        )

        return findings

    def _get_content(self, result: dict[str, Any]) -> str:
        """Extract text content from MCP tool result."""
        try:
            content_list = result.get("content", [])
            if isinstance(content_list, list):
                for block in content_list:
                    if isinstance(block, dict) and "text" in block:
                        return str(block["text"])
            # Empty or no text blocks — return empty string
            return ""
        except (TypeError, ValueError):
            return str(result)

    def _parse_structured(self, content: str) -> list[dict[str, Any]]:
        """Try to parse structured findings from tool output."""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "findings" in data:
                return data["findings"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try finding a JSON array embedded in text
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass

        return []


# -- Registry helper -------------------------------------------------------


def register_mcp_verifier(config: MCPVerifierConfig) -> MCPVerifier:
    """Create and register an MCP verifier in the VERIFIER_REGISTRY.

    Usage:
        config = MCPVerifierConfig(
            tool_name="my_knowledge_checker",
            config_path=Path("/mnt/mcp-servers/my-server"),
            arguments={"check": "terminology"},
        )
        register_mcp_verifier(config)
    """
    from . import register_verifier

    verifier = MCPVerifier(config=config)
    register_verifier(verifier)
    return verifier
