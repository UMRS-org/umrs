"""UMRS Confluence connector — fetches pages from Atlassian Confluence.

No dependency on r4.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from . import (
    Connector,
    ConnectorConfig,
    ConnectorType,
    Document,
    DocumentStatus,
    SyncCheckpoint,
    register_connector,
)

logger = logging.getLogger(__name__)


class ConfluenceConnector(Connector):
    """Fetches pages from Atlassian Confluence Cloud or Server.

    Required config settings:
      - ``base_url``: Confluence base URL (e.g. ``https://mysite.atlassian.net/wiki``)
      - ``space_key``: Space to sync (optional, syncs all if empty)

    Required credentials:
      - ``email``: Atlassian account email
      - ``api_token``: Atlassian API token
    """

    @property
    def connector_type(self) -> ConnectorType:
        return ConnectorType.CONFLUENCE

    @property
    def name(self) -> str:
        return "Confluence"

    def validate_config(self, config: ConnectorConfig) -> List[str]:
        errors: List[str] = []
        if not config.settings.get("base_url"):
            errors.append("base_url is required")
        if not config.credentials.get("email"):
            errors.append("email credential is required")
        if not config.credentials.get("api_token"):
            errors.append("api_token credential is required")
        return errors

    def fetch(
        self,
        config: ConnectorConfig,
        checkpoint: Optional[SyncCheckpoint] = None,
    ) -> Iterator[Document]:
        base_url = config.settings["base_url"].rstrip("/")
        space_key = config.settings.get("space_key", "")
        email = config.credentials["email"]
        api_token = config.credentials["api_token"]

        start = 0
        limit = 25
        cursor = checkpoint.cursor if checkpoint else None
        if cursor:
            try:
                start = int(cursor)
            except (ValueError, TypeError):
                start = 0

        while True:
            url = f"{base_url}/rest/api/content?start={start}&limit={limit}"
            url += "&expand=body.storage,version,space"
            if space_key:
                url += f"&spaceKey={urllib.parse.quote(space_key)}"

            try:
                data = self._api_get(url, email, api_token)
            except Exception as exc:
                logger.error("Confluence API error: %s", exc)
                break

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                page_id = page.get("id", "")
                title = page.get("title", "")
                body = page.get("body", {}).get("storage", {}).get("value", "")
                space = page.get("space", {}).get("key", "")

                doc = Document(
                    source_type=ConnectorType.CONFLUENCE,
                    source_id=f"confluence:{page_id}",
                    title=title,
                    content=body,
                    content_type="text/html",
                    metadata={
                        "space": space,
                        "page_id": page_id,
                        "version": page.get("version", {}).get("number", 0),
                    },
                    url=f"{base_url}/pages/{page_id}",
                    status=DocumentStatus.FETCHED,
                )
                yield doc

            # Pagination
            size = data.get("size", 0)
            if size < limit:
                break
            start += limit

    def health_check(self, config: ConnectorConfig) -> bool:
        base_url = config.settings.get("base_url", "").rstrip("/")
        email = config.credentials.get("email", "")
        api_token = config.credentials.get("api_token", "")
        if not base_url or not email or not api_token:
            return False
        try:
            self._api_get(f"{base_url}/rest/api/space?limit=1", email, api_token)
            return True
        except Exception:
            return False

    @staticmethod
    def _api_get(url: str, email: str, api_token: str) -> Dict[str, Any]:
        """Make an authenticated GET request to Confluence REST API."""
        import base64

        auth = base64.b64encode(f"{email}:{api_token}".encode()).decode()
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Basic {auth}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode())


register_connector(ConfluenceConnector())
