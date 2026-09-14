"""UMRS Notion connector — fetches pages from a Notion workspace.

No dependency on r4.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Iterator
from typing import Any

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

_NOTION_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


class NotionConnector(Connector):
    """Fetches pages from a Notion workspace.

    Required credentials:
      - ``api_token``: Notion internal integration token

    Optional settings:
      - ``database_id``: Restrict sync to a specific database
      - ``filter``: JSON filter object for database queries
    """

    @property
    def connector_type(self) -> ConnectorType:
        return ConnectorType.NOTION

    @property
    def name(self) -> str:
        return "Notion"

    def validate_config(self, config: ConnectorConfig) -> list[str]:
        errors: list[str] = []
        if not config.credentials.get("api_token"):
            errors.append("api_token credential is required")
        return errors

    def fetch(
        self,
        config: ConnectorConfig,
        checkpoint: SyncCheckpoint | None = None,
    ) -> Iterator[Document]:
        token = config.credentials["api_token"]
        database_id = config.settings.get("database_id")

        if database_id:
            yield from self._fetch_database(token, database_id, config, checkpoint)
        else:
            yield from self._search_pages(token, config, checkpoint)

    def _search_pages(
        self,
        token: str,
        config: ConnectorConfig,
        checkpoint: SyncCheckpoint | None,
    ) -> Iterator[Document]:
        """Search all pages accessible to the integration."""
        cursor = checkpoint.cursor if checkpoint else None
        body: dict[str, Any] = {"filter": {"value": "page", "property": "object"}}
        if cursor:
            body["start_cursor"] = cursor

        while True:
            data = self._api_post(f"{_NOTION_API}/search", token, body)
            results = data.get("results", [])
            for page in results:
                doc = self._page_to_doc(page)
                if doc:
                    yield doc

            if not data.get("has_more"):
                break
            body["start_cursor"] = data.get("next_cursor", "")
            if not body["start_cursor"]:
                break

    def _fetch_database(
        self,
        token: str,
        database_id: str,
        config: ConnectorConfig,
        checkpoint: SyncCheckpoint | None,
    ) -> Iterator[Document]:
        """Query a specific Notion database for pages."""
        cursor = checkpoint.cursor if checkpoint else None
        body: dict[str, Any] = {}
        if cursor:
            body["start_cursor"] = cursor

        filter_obj = config.settings.get("filter")
        if filter_obj:
            body["filter"] = filter_obj

        while True:
            url = f"{_NOTION_API}/databases/{database_id}/query"
            data = self._api_post(url, token, body)
            results = data.get("results", [])
            for page in results:
                doc = self._page_to_doc(page)
                if doc:
                    yield doc

            if not data.get("has_more"):
                break
            body["start_cursor"] = data.get("next_cursor", "")
            if not body["start_cursor"]:
                break

    def _page_to_doc(self, page: dict[str, Any]) -> Document | None:
        """Convert a Notion page object to a Document."""
        page_id = page.get("id", "")
        url = page.get("url", "")

        # Extract title from properties
        title = ""
        props = page.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                title = "".join(t.get("plain_text", "") for t in title_parts)
                break

        if not title:
            title = page_id

        return Document(
            source_type=ConnectorType.NOTION,
            source_id=f"notion:{page_id}",
            title=title,
            content="",  # Full content requires block-level fetch
            content_type="text/plain",
            metadata={
                "page_id": page_id,
                "created_time": page.get("created_time", ""),
                "last_edited_time": page.get("last_edited_time", ""),
            },
            url=url,
            status=DocumentStatus.FETCHED,
        )

    def health_check(self, config: ConnectorConfig) -> bool:
        token = config.credentials.get("api_token", "")
        if not token:
            return False
        try:
            self._api_post(
                f"{_NOTION_API}/search",
                token,
                {"page_size": 1},
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _api_post(
        url: str,
        token: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Make an authenticated POST request to Notion API."""
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Notion-Version": _NOTION_VERSION,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())


register_connector(NotionConnector())
