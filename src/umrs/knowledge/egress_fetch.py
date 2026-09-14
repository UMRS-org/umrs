"""Egress fetch + content scanning for KNOWLEDGE_INGEST remote sources.

Supported kinds: ``http_url``, ``git_repo``.

Policy:
  - Fail-closed: empty ``allow_domains`` blocks all remote fetches.
  - Domain allowlist enforced per-request.
  - Content scanned against PII/credential patterns before storage.

Stdlib-only: uses ``urllib.request``.

Environment variables:
  EGRESS_TIMEOUT_SEC  – per-request timeout, default 30
  EGRESS_MAX_BYTES    – max content size, default 10MB (10 * 1024 * 1024)
"""

from __future__ import annotations

import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_EGRESS_TIMEOUT_SEC: int = int(os.environ.get("EGRESS_TIMEOUT_SEC", "30"))

_EGRESS_MAX_BYTES: int = int(os.environ.get("EGRESS_MAX_BYTES", str(10 * 1024 * 1024)))

# Default block patterns (PII, credentials, secrets).
_DEFAULT_BLOCK_PATTERNS: list[str] = [
    r"(?i)(password|passwd|pwd)\s*[:=]",
    r"(?i)(secret|secret_key)\s*[:=]",
    r"(?i)(api_key|apikey)\s*[:=]",
    r"(?i)(token|auth_token|access_token)\s*[:=]",
    r"(?i)(private_key)\s*[:=]",
    r"(?i)(aws_secret_access_key)\s*[:=]",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Result of content scanning."""

    passed: bool
    blocked_reason: str = ""
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class FetchResult:
    """Result of an egress fetch."""

    url: str
    content: bytes | None
    status_code: int | None
    size_bytes: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Content scanning
# ---------------------------------------------------------------------------


def scan_content(
    content: bytes,
    block_patterns: list[str] | None = None,
) -> ScanResult:
    """Scan ``content`` for PII / credential patterns.

    Returns ``ScanResult(passed=True)`` when clean, or
    ``ScanResult(passed=False, blocked_reason=...)`` when blocked.

    Patterns are matched against the content decoded as UTF-8 (with fallback
    to latin-1 so that bytes-based scans never crash).
    """
    patterns = list(block_patterns) if block_patterns else list(_DEFAULT_BLOCK_PATTERNS)
    matched: list[str] = []

    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    # Limit scan to first 500 KB to avoid O(n) cost on large payloads.
    scan_window = text[:500_000]

    for pat in patterns:
        try:
            if re.search(pat, scan_window):
                matched.append(pat)
        except re.error:
            continue

    if matched:
        pattern_summary = "; ".join(matched[:3])
        return ScanResult(
            passed=False,
            blocked_reason=f"content blocked: {len(matched)} pattern(s) matched — {pattern_summary}",
            matched_patterns=matched,
        )

    # Additional heuristic: check for email-like addresses in non-text content.
    if len(matched) == 0 and len(content) < 100_000:
        emails = re.findall(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", scan_window
        )
        # If more than 5 unique emails in a short window, flag it.
        unique_emails = set(emails)
        if len(unique_emails) > 5:
            return ScanResult(
                passed=False,
                blocked_reason=f"content blocked: {len(unique_emails)} unique email addresses detected",
                matched_patterns=["email_density"],
            )

    return ScanResult(passed=True)


# ---------------------------------------------------------------------------
# URL fetching
# ---------------------------------------------------------------------------


def _extract_domain(url: str) -> str:
    """Extract the host domain from a URL."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return parsed.hostname or ""


def fetch_url(
    url: str,
    allow_domains: list[str] | None = None,
    timeout: int | None = None,
    block_patterns: list[str] | None = None,
) -> FetchResult:
    """Fetch content from ``url`` with domain allowlist and content scan.

    Parameters:
        url: The URL to fetch.
        allow_domains: Domain allowlist (fail-closed when empty/None).
        timeout: Request timeout in seconds (default from env).
        block_patterns: Content scan patterns (default PII/credential set).

    Returns:
        FetchResult with content (or error) and scan status.
    """
    timeout = timeout or _EGRESS_TIMEOUT_SEC

    # Domain validation
    domain = _extract_domain(url)
    if not domain:
        return FetchResult(
            url=url, content=None, status_code=None, error="invalid URL: no domain"
        )

    domains = list(allow_domains) if allow_domains else []
    if not domains:
        return FetchResult(
            url=url,
            content=None,
            status_code=None,
            error="egress blocked: allow_domains is empty (fail-closed)",
        )

    if domain not in domains:
        return FetchResult(
            url=url,
            content=None,
            status_code=None,
            error=f"egress blocked: domain {domain!r} not in allowlist {domains}",
        )

    # Fetch
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            url, headers={"User-Agent": "UMRS-KnowledgeIngest/1.0"}
        )
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        raw = resp.read()
    except urllib.error.HTTPError as exc:
        return FetchResult(
            url=url,
            content=None,
            status_code=exc.code,
            error=f"HTTP {exc.code}: {exc.reason}",
        )
    except urllib.error.URLError as exc:
        return FetchResult(
            url=url,
            content=None,
            status_code=None,
            error=f"URL error: {exc.reason}",
        )
    except Exception as exc:
        return FetchResult(
            url=url,
            content=None,
            status_code=None,
            error=f"fetch failed: {exc}",
        )

    size = len(raw)

    # Size limit
    if size > _EGRESS_MAX_BYTES:
        return FetchResult(
            url=url,
            content=None,
            status_code=resp.code if hasattr(resp, "code") else None,
            error=f"content too large: {size} bytes (max {_EGRESS_MAX_BYTES})",
        )

    # Content scan
    scan = scan_content(raw, block_patterns=block_patterns)
    if not scan.passed:
        return FetchResult(
            url=url,
            content=None,
            status_code=resp.code if hasattr(resp, "code") else None,
            size_bytes=size,
            error=f"content blocked: {scan.blocked_reason}",
        )

    return FetchResult(
        url=url,
        content=raw,
        status_code=resp.code if hasattr(resp, "code") else None,
        size_bytes=size,
    )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
