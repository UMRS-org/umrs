"""UMRS Google Drive connector — fetches documents from Google Drive.

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

_DRIVE_API = "https://www.googleapis.com/drive/v3"


class GoogleDriveConnector(Connector):
    """Fetches documents from Google Drive.

    Required credentials:
      - ``access_token``: OAuth2 access token (or service account token)

    Optional settings:
      - ``folder_id``: Restrict sync to a specific folder
      - ``mime_types``: List of MIME types to include
      - ``include_shared``: Include shared drives (default: false)
    """

    @property
    def connector_type(self) -> ConnectorType:
        return ConnectorType.GOOGLE_DRIVE

    @property
    def name(self) -> str:
        return "Google Drive"

    def validate_config(self, config: ConnectorConfig) -> List[str]:
        errors: List[str] = []
        if not config.credentials.get("access_token"):
            errors.append("access_token credential is required")
        return errors

    def fetch(
        self,
        config: ConnectorConfig,
        checkpoint: Optional[SyncCheckpoint] = None,
    ) -> Iterator[Document]:
        token = config.credentials["access_token"]
        folder_id = config.settings.get("folder_id", "")
        mime_types = config.settings.get("mime_types", [])
        include_shared = config.settings.get("include_shared", False)

        # Build query
        query_parts: List[str] = ["trashed=false"]
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        if mime_types:
            mime_clauses = " or ".join(f"mimeType='{m}'" for m in mime_types)
            query_parts.append(f"({mime_clauses})")

        query = " and ".join(query_parts)

        page_token = checkpoint.cursor if checkpoint else None

        while True:
            params: Dict[str, str] = {
                "q": query,
                "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,size,webViewLink)",
                "pageSize": "100",
            }
            if page_token:
                params["pageToken"] = page_token
            if include_shared:
                params["includeItemsFromAllDrives"] = "true"
                params["supportsAllDrives"] = "true"

            data = self._api_get(f"{_DRIVE_API}/files", token, params)
            files = data.get("files", [])

            for f in files:
                doc = self._file_to_doc(f, token)
                if doc:
                    yield doc

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    def _file_to_doc(
        self,
        file_info: Dict[str, Any],
        token: str,
    ) -> Optional[Document]:
        """Convert a Drive file entry to a Document."""
        file_id = file_info.get("id", "")
        name = file_info.get("name", "")
        mime_type = file_info.get("mimeType", "application/octet-stream")
        size = int(file_info.get("size", 0))
        url = file_info.get("webViewLink", "")

        # Google Docs/Sheets/Slides → export as text
        export_mime = self._export_mime(mime_type)
        content = ""

        if export_mime:
            try:
                content = self._export_file(file_id, export_mime, token)
            except Exception as exc:
                logger.warning("Failed to export %s: %s", name, exc)
        elif mime_type.startswith("text/") and size < 10_000_000:
            try:
                content = self._download_file(file_id, token)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", name, exc)

        return Document(
            source_type=ConnectorType.GOOGLE_DRIVE,
            source_id=f"gdrive:{file_id}",
            title=name,
            content=content,
            content_type=export_mime or mime_type,
            metadata={
                "file_id": file_id,
                "mime_type": mime_type,
                "modified_time": file_info.get("modifiedTime", ""),
            },
            url=url,
            size_bytes=size,
            status=DocumentStatus.FETCHED,
        )

    @staticmethod
    def _export_mime(mime_type: str) -> str:
        """Map Google Workspace MIME types to export formats."""
        mapping = {
            "application/vnd.google-apps.document": "text/plain",
            "application/vnd.google-apps.spreadsheet": "text/csv",
            "application/vnd.google-apps.presentation": "text/plain",
        }
        return mapping.get(mime_type, "")

    def _export_file(self, file_id: str, mime_type: str, token: str) -> str:
        """Export a Google Workspace file to the specified MIME type."""
        params = {"mimeType": mime_type}
        url = f"{_DRIVE_API}/files/{file_id}/export"
        return self._api_get_text(url, token, params)

    def _download_file(self, file_id: str, token: str) -> str:
        """Download a regular file's content."""
        url = f"{_DRIVE_API}/files/{file_id}"
        return self._api_get_text(url, token, {"alt": "media"})

    def health_check(self, config: ConnectorConfig) -> bool:
        token = config.credentials.get("access_token", "")
        if not token:
            return False
        try:
            self._api_get(
                f"{_DRIVE_API}/about",
                token,
                {"fields": "user"},
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _api_get(
        url: str,
        token: str,
        params: Dict[str, str],
    ) -> Dict[str, Any]:
        """Authenticated GET returning JSON."""
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

    @staticmethod
    def _api_get_text(
        url: str,
        token: str,
        params: Dict[str, str],
    ) -> str:
        """Authenticated GET returning plain text."""
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            return resp.read().decode(errors="replace")


register_connector(GoogleDriveConnector())
