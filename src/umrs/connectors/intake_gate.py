"""IntakeGate — the single policy gate for connector-ingested documents (I2).

One admission-control point for every document flowing through the platform
via connectors — replacing per-connector ad-hoc checks and guaranteeing that
no path silently bypasses it. Two layers:

1. **Size/format limits** — per-document content ceiling + per-sync doc cap
   (a single runaway connection or prompt-injection funnel is bounded).
2. **Secret-scan** — block documents containing *likely* credentials (AWS
   keys, private keys, GitHub/Slack tokens, api-key/password assignments,
   signed JWTs).

Gate decisions are recorded by the runner (blocked docs land in the DLQ
with the matched pattern name, scan summary persisted as
``checkpoint_json.last_scan``) — operator-visible and replayable, matching
the P2 contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import Document

# ── Secret-scan patterns ───────────────────────────────────────────────
# Each tuple: (pattern_name, compiled regex). Tight signature chosen so the
# false-positive rate stays low on real documents (research code/data).

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "aws_secret_key",
        re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    ),
    (
        "private_key_pem",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY( BLOCK)?-----"),
    ),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("github_token", re.compile(r"\bgh[posru]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "api_key_assignment",
        re.compile(
            r"api[_-]?key['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}",
            re.IGNORECASE,
        ),
    ),
    (
        "password_assignment",
        re.compile(r"password['\"]?\s*[:=]\s*['\"]?[^\s'\"]{8,}", re.IGNORECASE),
    ),
    (
        "generic_jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b"
        ),
    ),
)

# Document extensions the gate refuses outright (binary formats that the
# intake path cannot parse — ingest skips them identically to parse errors).
_DEFAULT_BLOCKED_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".ico",
    ".svg",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".dmg",
    ".iso",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".pptx",
    ".odp",
    ".exe",
    ".dll",
    ".so",
    ".o",
    ".a",
    ".dylib",
    ".bin",
    ".dat",
)


@dataclass
class IntakePolicy:
    """Policy configuration for the intake gate."""

    max_content_bytes: int = 10 * 1024 * 1024  # 10 MiB per document
    max_docs_per_run: int = 1000  # hard cap per sync run
    block_secrets: bool = True
    block_unparseable_formats: bool = True
    blocked_extensions: tuple[str, ...] = _DEFAULT_BLOCKED_EXTENSIONS
    custom_patterns: tuple[tuple[str, str], ...] = ()

    def scan_patterns(self) -> tuple[tuple[str, re.Pattern], ...]:
        if not self.custom_patterns:
            return _SECRET_PATTERNS
        custom = tuple((name, re.compile(raw)) for name, raw in self.custom_patterns)
        return _SECRET_PATTERNS + custom


@dataclass
class GateDecision:
    """Admission decision for one document."""

    action: str  # "allow" | "block"
    reason: str = ""
    pattern: str | None = None


@dataclass
class ScanReport:
    """Aggregate findings across one sync run."""

    blocked: int = 0
    patterns: dict[str, int] = field(default_factory=dict)

    def register(self, pattern: str) -> None:
        self.blocked += 1
        self.patterns[pattern] = self.patterns.get(pattern, 0) + 1


class IntakeGate:
    """Single admission gate for the intake path (I2)."""

    def __init__(self, policy: IntakePolicy | None = None) -> None:
        self._policy = policy or IntakePolicy()
        self._patterns = self._policy.scan_patterns()

    def inspect(self, document: Document, *, doc_index: int = 0) -> GateDecision:
        """Decide whether a connector-fetched document is ingestible."""
        policy = self._policy

        if doc_index >= policy.max_docs_per_run:
            return GateDecision(
                action="block",
                reason=f"sync run exceeds max_docs_per_run={policy.max_docs_per_run}",
                pattern="max_docs_per_run",
            )

        ext = _extension(document)
        if policy.block_unparseable_formats and ext in policy.blocked_extensions:
            return GateDecision(
                action="block",
                reason=f"unsupported document format {ext}",
                pattern="unsupported_format",
            )

        if (
            len(document.content.encode("utf-8", errors="ignore"))
            > policy.max_content_bytes
        ):
            return GateDecision(
                action="block",
                reason=f"content exceeds max_content_bytes={policy.max_content_bytes}",
                pattern="max_content_bytes",
            )

        if policy.block_secrets:
            for name, pattern in self._patterns:
                if pattern.search(document.content):
                    return GateDecision(
                        action="block",
                        reason=f"secret-like content matched {name}",
                        pattern=name,
                    )

        return GateDecision(action="allow")


def _extension(document: Document) -> str:
    """Document extension from source_id, fall back to content_type."""
    source = (document.source_id or "").lower()
    dot = source.rfind(".")
    if dot > source.rfind("/") and dot != -1:
        return source[dot:]
    content_type = (document.content_type or "").lower()
    return f".{content_type.split('/')[-1]}" if "/" in content_type else ""


__all__ = ["GateDecision", "IntakeGate", "IntakePolicy", "ScanReport"]
