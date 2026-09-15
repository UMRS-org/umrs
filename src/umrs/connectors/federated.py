"""Federated connector — search without indexing.

For data sources that cannot be fully indexed (NDA, PHI, ephemeral),
this module provides a "search-through" interface that queries the
source API in real time and returns results without persisting them.

No dependency on r4.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FederatedResult:
    """A single result from a federated search."""

    source_name: str
    title: str
    snippet: str
    url: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FederatedSource(ABC):
    """Abstract base for a federated (search-only) source."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name."""

    @abstractmethod
    def search(
        self,
        query: str,
        credentials: Dict[str, str],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[FederatedResult]:
        """Search the source without indexing any data."""

    def available(self, credentials: Dict[str, str]) -> bool:
        """Check if the source is reachable."""
        return True


class FederatedSearcher:
    """Fan-out search across multiple federated sources.

    Results are merged and ranked by score, then returned.
    No data is persisted — this is purely a query-time operation.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, FederatedSource] = {}

    def register(self, source: FederatedSource) -> None:
        self._sources[source.name] = source

    def list_sources(self) -> List[str]:
        return list(self._sources.keys())

    def search(
        self,
        query: str,
        credentials: Dict[str, Dict[str, str]],
        sources: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[FederatedResult]:
        """Search across registered sources.

        Args:
            query: The search query.
            credentials: Mapping of source_name → credentials dict.
            sources: Subset of sources to search. None = all.
            limit: Max results per source.
        """
        target_sources = sources or list(self._sources.keys())
        all_results: List[FederatedResult] = []

        for src_name in target_sources:
            source = self._sources.get(src_name)
            if source is None:
                logger.warning("Unknown federated source: %s", src_name)
                continue
            creds = credentials.get(src_name, {})
            try:
                results = source.search(query, creds, limit=limit)
                all_results.extend(results)
            except Exception as exc:
                logger.warning("Federated search failed for %s: %s", src_name, exc)

        # Sort by score descending
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results


# Global instance
_searcher = FederatedSearcher()


def get_federated_searcher() -> FederatedSearcher:
    return _searcher


def register_federated_source(source: FederatedSource) -> None:
    _searcher.register(source)
