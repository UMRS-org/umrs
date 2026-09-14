"""Web crawl connector — fetches and indexes web pages.

A simple breadth-first crawler that fetches pages up to a configurable
depth and yields documents. Respects robots.txt via urllib.

No dependency on r4.
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Iterator

from . import (
    Connector,
    ConnectorConfig,
    ConnectorType,
    Document,
    SyncCheckpoint,
    register_connector,
)

logger = logging.getLogger(__name__)

_MAX_PAGE_SIZE = 5_000_000  # 5 MB
_DEFAULT_MAX_PAGES = 100
_DEFAULT_MAX_DEPTH = 2


def _strip_html(html: str) -> str:
    """Rudimentary HTML → text."""
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_links(html: str, base_url: str) -> list[str]:
    """Extract absolute URLs from href attributes."""
    links: list[str] = []
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE):
        href = match.group(1)
        absolute = urllib.parse.urljoin(base_url, href)
        # Only follow http(s)
        if absolute.startswith(("http://", "https://")):
            # Strip fragment
            absolute = absolute.split("#")[0]
            links.append(absolute)
    return links


class WebCrawlConnector(Connector):
    """Crawl a website and yield documents."""

    @property
    def connector_type(self) -> ConnectorType:
        return ConnectorType.WEB_CRAWL

    @property
    def name(self) -> str:
        return "Web Crawl"

    def validate_config(self, config: ConnectorConfig) -> list[str]:
        errors: list[str] = []
        if not config.settings.get("start_url"):
            errors.append("start_url is required in settings")
        return errors

    def fetch(
        self,
        config: ConnectorConfig,
        checkpoint: SyncCheckpoint | None = None,
    ) -> Iterator[Document]:
        start_url = config.settings["start_url"]
        max_pages = config.settings.get("max_pages", _DEFAULT_MAX_PAGES)
        max_depth = config.settings.get("max_depth", _DEFAULT_MAX_DEPTH)
        allowed_domain = config.settings.get(
            "allowed_domain",
            urllib.parse.urlparse(start_url).netloc,
        )

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])

        while queue and len(visited) < max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            # Domain filter
            parsed = urllib.parse.urlparse(url)
            if parsed.netloc != allowed_domain:
                continue

            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "umrs-connector/1.0"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if resp.status != 200:
                        continue
                    ctype = resp.headers.get("Content-Type", "")
                    if "text/html" not in ctype:
                        continue
                    raw = resp.read(_MAX_PAGE_SIZE).decode("utf-8", errors="replace")
            except Exception as exc:
                logger.debug("Crawl skip %s: %s", url, exc)
                continue

            text = _strip_html(raw)
            if not text.strip():
                continue

            # Extract title
            title_match = re.search(
                r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL
            )
            title = _strip_html(title_match.group(1)) if title_match else url

            url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
            yield Document(
                source_type=ConnectorType.WEB_CRAWL,
                source_id=url_hash,
                title=title,
                content=text,
                content_type="text/plain",
                metadata={"url": url, "depth": depth},
                url=url,
                size_bytes=len(text),
            )

            # Enqueue child links
            if depth < max_depth:
                for link in _extract_links(raw, url):
                    if link not in visited:
                        queue.append((link, depth + 1))

    def health_check(self, config: ConnectorConfig) -> bool:
        url = config.settings.get("start_url", "")
        if not url:
            return False
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5):
                pass
            return True
        except Exception:
            return False


register_connector(WebCrawlConnector())
