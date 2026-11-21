"""Knowledge verifiers the r4 extension adds to core's VERIFIER_REGISTRY."""

from __future__ import annotations

from .mcp_adapter import MCPVerifier, MCPVerifierConfig, register_mcp_verifier

__all__ = ["MCPVerifier", "MCPVerifierConfig", "register_mcp_verifier"]
