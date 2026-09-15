"""UMRS Slack connector — fetches messages from Slack channels.

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

_SLACK_API = "https://slack.com/api"


class SlackConnector(Connector):
    """Fetches messages from Slack channels for RAG indexing.

    Required credentials:
      - ``bot_token``: Slack Bot User OAuth Token (xoxb-…)

    Optional settings:
      - ``channels``: List of channel IDs or names to sync
      - ``include_threads``: Whether to fetch thread replies (default: true)
      - ``max_messages``: Max messages per channel per sync (default: 1000)
    """

    @property
    def connector_type(self) -> ConnectorType:
        return ConnectorType.SLACK

    @property
    def name(self) -> str:
        return "Slack"

    def validate_config(self, config: ConnectorConfig) -> List[str]:
        errors: List[str] = []
        if not config.credentials.get("bot_token"):
            errors.append("bot_token credential is required")
        return errors

    def fetch(
        self,
        config: ConnectorConfig,
        checkpoint: Optional[SyncCheckpoint] = None,
    ) -> Iterator[Document]:
        token = config.credentials["bot_token"]
        channels = config.settings.get("channels", [])
        include_threads = config.settings.get("include_threads", True)
        max_messages = config.settings.get("max_messages", 1000)

        # If no channels specified, list all joined channels
        if not channels:
            channels = self._list_channels(token)

        for channel_id in channels:
            yield from self._fetch_channel(
                token, channel_id, include_threads, max_messages, checkpoint
            )

    def _list_channels(self, token: str) -> List[str]:
        """List channels the bot is a member of."""
        data = self._api_get(
            f"{_SLACK_API}/conversations.list",
            token,
            {"types": "public_channel,private_channel", "limit": "200"},
        )
        return [ch["id"] for ch in data.get("channels", []) if ch.get("is_member")]

    def _fetch_channel(
        self,
        token: str,
        channel_id: str,
        include_threads: bool,
        max_messages: int,
        checkpoint: Optional[SyncCheckpoint],
    ) -> Iterator[Document]:
        """Fetch messages from a single channel."""
        params: Dict[str, str] = {"channel": channel_id, "limit": "100"}

        # Incremental: use oldest from checkpoint
        if checkpoint and checkpoint.cursor:
            params["oldest"] = checkpoint.cursor

        fetched = 0
        while fetched < max_messages:
            data = self._api_get(f"{_SLACK_API}/conversations.history", token, params)
            messages = data.get("messages", [])
            if not messages:
                break

            for msg in messages:
                if fetched >= max_messages:
                    break

                user = msg.get("user", "unknown")
                text = msg.get("text", "")
                ts = msg.get("ts", "")

                doc = Document(
                    source_type=ConnectorType.SLACK,
                    source_id=f"slack:{channel_id}:{ts}",
                    title=f"Slack message by {user}",
                    content=text,
                    content_type="text/plain",
                    metadata={
                        "channel_id": channel_id,
                        "user": user,
                        "ts": ts,
                        "thread_ts": msg.get("thread_ts", ""),
                        "reply_count": msg.get("reply_count", 0),
                    },
                    status=DocumentStatus.FETCHED,
                )
                yield doc
                fetched += 1

                # Fetch thread replies
                thread_ts = msg.get("thread_ts")
                if include_threads and thread_ts and msg.get("reply_count", 0) > 0:
                    yield from self._fetch_thread(token, channel_id, thread_ts)

            if not data.get("has_more"):
                break
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
            params["cursor"] = cursor

    def _fetch_thread(
        self,
        token: str,
        channel_id: str,
        thread_ts: str,
    ) -> Iterator[Document]:
        """Fetch replies in a message thread."""
        data = self._api_get(
            f"{_SLACK_API}/conversations.replies",
            token,
            {"channel": channel_id, "ts": thread_ts, "limit": "100"},
        )
        # Skip the first message (parent, already yielded)
        for msg in data.get("messages", [])[1:]:
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            yield Document(
                source_type=ConnectorType.SLACK,
                source_id=f"slack:{channel_id}:{ts}",
                title=f"Slack reply by {user}",
                content=text,
                content_type="text/plain",
                metadata={
                    "channel_id": channel_id,
                    "user": user,
                    "ts": ts,
                    "thread_ts": thread_ts,
                },
                status=DocumentStatus.FETCHED,
            )

    def health_check(self, config: ConnectorConfig) -> bool:
        token = config.credentials.get("bot_token", "")
        if not token:
            return False
        try:
            data = self._api_get(f"{_SLACK_API}/auth.test", token, {})
            return data.get("ok", False)
        except Exception:
            return False

    @staticmethod
    def _api_get(
        url: str,
        token: str,
        params: Dict[str, str],
    ) -> Dict[str, Any]:
        """Make an authenticated GET request to Slack API."""
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode())


register_connector(SlackConnector())
