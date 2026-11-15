"""Pre-flight security scanners for recipes and skill bundles.

Pure-stdlib (uses only `re`). Mirrors `umrs.security.scanner` so clients
can run the same checks locally before submitting. Corrected 2026-10-19
(cleanup C-3b(iv)): the previous sentence claimed UMRS re-runs the scan at
registration time as the authoritative gate; no umrs ``src`` code calls
``RecipeScanner``/``SkillScanner`` today, so the client-side scan is the
only one that runs (D-61 inventory; wiring the server-side scan is feature
work, not cleanup).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "SeverityLevel",
    "ScanFinding",
    "ScanResult",
    "RecipeScanner",
    "SkillScanner",
]


class SeverityLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ScanFinding:
    """A single finding from the scanner."""

    severity: SeverityLevel
    category: str
    message: str
    location: str = ""  # e.g. "step[2].command"
    suggestion: str = ""


@dataclass
class ScanResult:
    """Result of scanning a recipe or skill."""

    target_name: str
    findings: list[ScanFinding] = field(default_factory=list)
    scanned: bool = True

    @property
    def has_errors(self) -> bool:
        return any(
            f.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL) for f in self.findings
        )

    @property
    def error_count(self) -> int:
        return sum(
            1 for f in self.findings if f.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SeverityLevel.WARNING)


# Dangerous patterns
_SHELL_INJECTION_PATTERNS = [
    (r"\$\(", "Command substitution ($(...)) detected"),
    (r"`[^`]+`", "Backtick command substitution detected"),
    (r";\s*(rm|dd|mkfs|shutdown|reboot|kill)\b", "Dangerous command chained with semicolon"),
    (r"\|\s*(sh|bash|zsh|eval)\b", "Pipe to shell interpreter"),
    (r">\s*/dev/sd", "Write to block device"),
    (r"rm\s+-rf?\s+/(?!\w)", "Recursive delete from root"),
    (r"chmod\s+777", "World-writable permissions"),
    (r"curl\s+.*\|\s*(sh|bash)", "Pipe from curl to shell"),
    (r"wget\s+.*\|\s*(sh|bash)", "Pipe from wget to shell"),
]

_ENV_SUSPICIOUS_PATTERNS = [
    (r"\$\{?(AWS_SECRET|GITHUB_TOKEN|API_KEY|PASSWORD|SECRET)", "Potential secret in env var"),
]

_PATH_TRAVERSAL_PATTERNS = [
    (r"\.\./\.\.", "Path traversal pattern detected"),
    (r"/etc/passwd", "Access to /etc/passwd"),
    (r"/etc/shadow", "Access to /etc/shadow"),
]

_NETWORK_PATTERNS = [
    (r"\bcurl\b", "External network access (curl)"),
    (r"\bwget\b", "External network access (wget)"),
    (r"\bnc\b", "Netcat usage detected"),
    (r"\bssh\b", "SSH usage detected"),
]


class RecipeScanner:
    """Scan recipe definitions for security issues."""

    def scan_recipe(self, recipe_data: dict[str, Any]) -> ScanResult:
        """Scan a recipe definition (parsed YAML dict)."""
        name = recipe_data.get("name", "<unknown>")
        result = ScanResult(target_name=name)

        for i, step in enumerate(recipe_data.get("steps", [])):
            self._scan_step(step, i, result)

        return result

    def _scan_step(self, step: dict[str, Any], index: int, result: ScanResult) -> None:
        step_type = step.get("type", "shell")
        location = f"step[{index}]"

        if step_type == "shell":
            self._scan_shell_command(step.get("command", ""), f"{location}.command", result)

        if step_type == "prompt":
            self._scan_prompt(step.get("prompt", ""), f"{location}.prompt", result)

        for key, value in step.get("args", {}).items():
            if isinstance(value, str):
                self._scan_value(value, f"{location}.args.{key}", result)

        for j, sub in enumerate(step.get("steps", [])):
            self._scan_step(sub, j, result)

    def _scan_shell_command(self, command: str, location: str, result: ScanResult) -> None:
        for pattern, msg in _SHELL_INJECTION_PATTERNS:
            if re.search(pattern, command):
                result.findings.append(
                    ScanFinding(
                        severity=SeverityLevel.ERROR,
                        category="shell_injection",
                        message=msg,
                        location=location,
                        suggestion="Review command for injection risks",
                    )
                )

        for pattern, msg in _ENV_SUSPICIOUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                result.findings.append(
                    ScanFinding(
                        severity=SeverityLevel.WARNING,
                        category="secrets_exposure",
                        message=msg,
                        location=location,
                        suggestion="Use UMRS credential store instead",
                    )
                )

        for pattern, msg in _PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, command):
                result.findings.append(
                    ScanFinding(
                        severity=SeverityLevel.ERROR,
                        category="path_traversal",
                        message=msg,
                        location=location,
                    )
                )

    def _scan_prompt(self, prompt: str, location: str, result: ScanResult) -> None:
        injection_markers = [
            "ignore previous instructions",
            "disregard the above",
            "you are now",
            "forget everything",
        ]
        lower = prompt.lower()
        for marker in injection_markers:
            if marker in lower:
                result.findings.append(
                    ScanFinding(
                        severity=SeverityLevel.WARNING,
                        category="prompt_injection",
                        message=f"Potential prompt injection: '{marker}'",
                        location=location,
                    )
                )

    def _scan_value(self, value: str, location: str, result: ScanResult) -> None:
        for pattern, msg in _PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value):
                result.findings.append(
                    ScanFinding(
                        severity=SeverityLevel.WARNING,
                        category="path_traversal",
                        message=msg,
                        location=location,
                    )
                )


class SkillScanner:
    """Scan skill bundles for security issues."""

    def scan_skill(self, config: dict[str, Any], skill_md: str = "") -> ScanResult:
        """Scan a skill `config.json` and `SKILL.md` text."""
        name = config.get("name", "<unknown>")
        result = ScanResult(target_name=name)

        mcp = config.get("mcp_server", {})
        if mcp:
            self._scan_mcp_config(mcp, result)

        egress = config.get("egress", {})
        if egress:
            domains = egress.get("domains", [])
            if "*" in domains:
                result.findings.append(
                    ScanFinding(
                        severity=SeverityLevel.ERROR,
                        category="egress",
                        message="Wildcard egress domain (*) is not allowed",
                        location="config.egress.domains",
                        suggestion="Specify exact domains",
                    )
                )

        for pattern, msg in _NETWORK_PATTERNS:
            if re.search(pattern, skill_md):
                result.findings.append(
                    ScanFinding(
                        severity=SeverityLevel.INFO,
                        category="network",
                        message=f"Skill instructions reference: {msg}",
                        location="SKILL.md",
                    )
                )

        return result

    def _scan_mcp_config(self, mcp: dict[str, Any], result: ScanResult) -> None:
        entry = mcp.get("entry_point", "")
        if entry:
            for pattern, msg in _SHELL_INJECTION_PATTERNS:
                if re.search(pattern, entry):
                    result.findings.append(
                        ScanFinding(
                            severity=SeverityLevel.ERROR,
                            category="shell_injection",
                            message=f"MCP entry_point: {msg}",
                            location="config.mcp_server.entry_point",
                        )
                    )

        sandbox = mcp.get("sandbox", {})
        max_mem = sandbox.get("max_memory_mb", 512)
        if max_mem > 4096:
            result.findings.append(
                ScanFinding(
                    severity=SeverityLevel.WARNING,
                    category="resource",
                    message=f"MCP server requests {max_mem}MB memory (>4GB)",
                    location="config.mcp_server.sandbox.max_memory_mb",
                )
            )
