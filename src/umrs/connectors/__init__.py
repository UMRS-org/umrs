"""UMRS Data Connector Framework.

Provides the abstract interface and registry for ingesting data from
external sources (GitHub, Confluence, Notion, Slack, Google Drive, etc.)
into the UMRS RAG pipeline.

Each connector is a standalone class that:
1. Polls or receives webhooks from a source
2. Emits documents as durable UMRS jobs (docfetching → docprocessing)
3. Tracks sync state (last sync time, cursor/checkpoint)

**UMRS is independently workable without r4.** This module has zero
dependencies on r4. r4 may provide CLI wrappers (``r4 connector …``)
but all connector logic lives here.

Architecture inspired by Onyx's connector framework.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data models ─────────────────────────────────────────────────────────────


class ConnectorType(Enum):
    """Source type for a connector."""

    FILESYSTEM = "filesystem"
    GITHUB = "github"
    CONFLUENCE = "confluence"
    NOTION = "notion"
    SLACK = "slack"
    GOOGLE_DRIVE = "google_drive"
    S3 = "s3"
    WEB_CRAWL = "web_crawl"
    CUSTOM = "custom"


class DocumentStatus(Enum):
    PENDING = "pending"
    FETCHED = "fetched"
    PROCESSED = "processed"
    INDEXED = "indexed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ConnectorConfig:
    """Configuration for a connector instance."""

    connector_type: ConnectorType
    name: str
    credentials: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    schedule_cron: str | None = None  # e.g. "0 */6 * * *"
    project_id: str | None = None
    owner_id: str | None = None
    enabled: bool = True


@dataclass
class Document:
    """A document fetched from a connector source."""

    doc_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    source_type: ConnectorType = ConnectorType.CUSTOM
    source_id: str = ""  # Unique ID within the source (e.g. file path, page ID)
    title: str = ""
    content: str = ""
    content_type: str = "text/plain"  # MIME type
    metadata: dict[str, Any] = field(default_factory=dict)
    url: str | None = None
    last_modified: datetime | None = None
    size_bytes: int = 0
    status: DocumentStatus = DocumentStatus.PENDING

    @property
    def is_binary(self) -> bool:
        return not self.content_type.startswith("text/")


@dataclass
class SyncCheckpoint:
    """Tracks sync progress for incremental fetching."""

    connector_name: str
    last_sync: datetime | None = None
    cursor: str | None = None  # Opaque cursor from the source API
    docs_synced: int = 0
    errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorResult:
    """Result from a connector sync run."""

    documents: list[Document] = field(default_factory=list)
    checkpoint: SyncCheckpoint | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def doc_count(self) -> int:
        return len(self.documents)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ── Connector ABC ───────────────────────────────────────────────────────────


class Connector(ABC):
    """Abstract base class for data source connectors.

    Implementations must be self-contained (no r4 imports). They are
    registered with the :class:`ConnectorRegistry` and invoked by the
    UMRS scheduler as durable jobs.
    """

    @property
    @abstractmethod
    def connector_type(self) -> ConnectorType:
        """The source type this connector handles."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable connector name."""

    @abstractmethod
    def validate_config(self, config: ConnectorConfig) -> list[str]:
        """Validate config. Returns list of errors (empty = valid)."""

    @abstractmethod
    def fetch(
        self,
        config: ConnectorConfig,
        checkpoint: SyncCheckpoint | None = None,
    ) -> Iterator[Document]:
        """Fetch documents from the source, yielding one at a time.

        Supports incremental sync via *checkpoint*. If checkpoint is
        ``None``, perform a full sync.
        """

    def health_check(self, config: ConnectorConfig) -> bool:
        """Test connectivity to the source. Default: always True."""
        return True


# ── Connector Registry ──────────────────────────────────────────────────────


class ConnectorRegistry:
    """Registry of available connector implementations.

    Connectors register themselves at import time. The UMRS scheduler
    and API layer look up connectors by type.
    """

    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}

    def register(self, connector: Connector) -> None:
        key = connector.connector_type.value
        if key in self._connectors:
            logger.warning("Overwriting connector registration for %s", key)
        self._connectors[key] = connector

    def get(self, connector_type: str) -> Connector | None:
        return self._connectors.get(connector_type)

    def list_types(self) -> list[str]:
        return list(self._connectors.keys())

    def __contains__(self, connector_type: object) -> bool:
        return isinstance(connector_type, str) and connector_type in self._connectors

    def __iter__(self) -> Iterator[Connector]:
        return iter(self._connectors.values())


# Global registry
_registry = ConnectorRegistry()


def get_connector_registry() -> ConnectorRegistry:
    """Return the global connector registry."""
    return _registry


def register_connector(connector: Connector) -> None:
    """Register a connector with the global registry."""
    _registry.register(connector)
