"""External knowledge integration — search and ingest from external sources.

Phase 8 of the Scientific Knowledge Build-up roadmap.

This module integrates external knowledge sources:
  - arXiv — preprint papers
  - Semantic Scholar — academic paper metadata
  - CrossRef — DOI metadata
  - PubMed — biomedical literature
  - Google Scholar — metadata (where appropriate)
  - General web search

The agent should:
  - search
  - summarize
  - compare
  - store references
  - update UMRS

Acceptance: External knowledge becomes first-class project knowledge.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── External Source Types ──────────────────────────────────────────────────


class ExternalSourceType(str, Enum):
    """Types of external knowledge sources."""

    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    PUBMED = "pubmed"
    GOOGLE_SCHOLAR = "google_scholar"
    WEB = "web"
    GITHUB = "github"


class SearchStatus(str, Enum):
    """Status of an external search."""

    SUCCESS = "success"
    NO_RESULTS = "no_results"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"


# ── Search Result Models ───────────────────────────────────────────────────


@dataclass
class ExternalReference:
    """A reference to an external knowledge source."""

    source_type: ExternalSourceType
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pubmed_id: str = ""
    url: str = ""
    published_date: str = ""
    keywords: list[str] = field(default_factory=list)
    citations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pubmed_id": self.pubmed_id,
            "url": self.url,
            "published_date": self.published_date,
            "keywords": self.keywords,
            "citations": self.citations,
            "metadata": self.metadata,
        }


@dataclass
class ExternalSearchResult:
    """Result of an external knowledge search."""

    query: str
    source_type: ExternalSourceType
    status: SearchStatus = SearchStatus.SUCCESS
    references: list[ExternalReference] = field(default_factory=list)
    total_results: int = 0
    error_message: str = ""
    search_duration_sec: float = 0.0
    searched_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "source_type": self.source_type.value,
            "status": self.status.value,
            "references": [r.to_dict() for r in self.references],
            "total_results": self.total_results,
            "error_message": self.error_message,
            "search_duration_sec": self.search_duration_sec,
            "searched_at": self.searched_at,
        }


# ── External Knowledge Sources ────────────────────────────────────────────


class ArxivSource:
    """arXiv preprint server source."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> ExternalSearchResult:
        """Search arXiv for papers."""
        start_time = datetime.now(timezone.utc)
        result = ExternalSearchResult(
            query=query,
            source_type=ExternalSourceType.ARXIV,
            searched_at=start_time.isoformat(),
        )

        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": "descending",
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            response = self._fetch_url(url)
            if response:
                references = self._parse_arxiv_response(response)
                result.references = references
                result.total_results = len(references)
                result.status = SearchStatus.SUCCESS if references else SearchStatus.NO_RESULTS
            else:
                result.status = SearchStatus.ERROR
                result.error_message = "Failed to fetch from arXiv"
        except Exception as exc:
            result.status = SearchStatus.ERROR
            result.error_message = str(exc)

        result.search_duration_sec = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        return result

    def _fetch_url(self, url: str, timeout: int = 30) -> str | None:
        """Fetch URL content."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "R4-Knowledge/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _parse_arxiv_response(self, response: str) -> list[ExternalReference]:
        """Parse arXiv API Atom XML response."""
        references: list[ExternalReference] = []

        # Parse XML using regex (avoid xml.etree for simplicity)
        entry_pattern = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
        for entry_match in entry_pattern.finditer(response):
            entry = entry_match.group(1)

            ref = ExternalReference(source_type=ExternalSourceType.ARXIV)

            # Extract title
            title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            if title_match:
                ref.title = title_match.group(1).strip()

            # Extract arxiv ID
            id_match = re.search(r"<id>(http://arxiv.org/abs/)?(.+?)</id>", entry, re.DOTALL)
            if id_match:
                ref.arxiv_id = id_match.group(2).strip()
                ref.url = f"http://arxiv.org/abs/{ref.arxiv_id}"

            # Extract authors
            author_pattern = re.compile(r"<author>.*?<name>(.*?)</name>.*?</author>", re.DOTALL)
            for author_match in author_pattern.finditer(entry):
                ref.authors.append(author_match.group(1).strip())

            # Extract abstract
            summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            if summary_match:
                ref.abstract = summary_match.group(1).strip()

            # Extract published date
            published_match = re.search(r"<published>(.*?)</published>", entry, re.DOTALL)
            if published_match:
                ref.published_date = published_match.group(1).strip()

            if ref.title:
                references.append(ref)

        return references


class SemanticScholarSource:
    """Semantic Scholar academic paper source."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
    ) -> ExternalSearchResult:
        """Search Semantic Scholar for papers."""
        start_time = datetime.now(timezone.utc)
        result = ExternalSearchResult(
            query=query,
            source_type=ExternalSourceType.SEMANTIC_SCHOLAR,
            searched_at=start_time.isoformat(),
        )

        try:
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,abstract,year,citationCount,externalIds",
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            response = self._fetch_url(url)
            if response:
                data = json.loads(response)
                references = self._parse_s2_response(data)
                result.references = references
                result.total_results = len(references)
                result.status = SearchStatus.SUCCESS if references else SearchStatus.NO_RESULTS
            else:
                result.status = SearchStatus.ERROR
                result.error_message = "Failed to fetch from Semantic Scholar"
        except Exception as exc:
            result.status = SearchStatus.ERROR
            result.error_message = str(exc)

        result.search_duration_sec = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        return result

    def _fetch_url(self, url: str, timeout: int = 30) -> str | None:
        """Fetch URL content."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "R4-Knowledge/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _parse_s2_response(self, data: dict) -> list[ExternalReference]:
        """Parse Semantic Scholar API JSON response."""
        references: list[ExternalReference] = []

        for paper in data.get("data", []):
            ref = ExternalReference(source_type=ExternalSourceType.SEMANTIC_SCHOLAR)
            ref.title = paper.get("title", "")
            ref.abstract = paper.get("abstract", "") or ""

            # Authors
            for author in paper.get("authors", []):
                ref.authors.append(author.get("name", ""))

            # Year
            year = paper.get("year")
            if year:
                ref.published_date = str(year)

            # Citation count
            ref.citations = paper.get("citationCount", 0) or 0

            # External IDs (DOI, arXiv, PubMed)
            ext_ids = paper.get("externalIds", {}) or {}
            ref.doi = ext_ids.get("DOI", "")
            ref.arxiv_id = ext_ids.get("ArXiv", "")
            ref.pubmed_id = ext_ids.get("PubMed", "")

            # URL
            if ref.doi:
                ref.url = f"https://doi.org/{ref.doi}"
            elif ref.arxiv_id:
                ref.url = f"http://arxiv.org/abs/{ref.arxiv_id}"

            if ref.title:
                references.append(ref)

        return references


class CrossRefSource:
    """CrossRef DOI metadata source."""

    BASE_URL = "https://api.crossref.org/works"

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
    ) -> ExternalSearchResult:
        """Search CrossRef for papers by query."""
        start_time = datetime.now(timezone.utc)
        result = ExternalSearchResult(
            query=query,
            source_type=ExternalSourceType.CROSSREF,
            searched_at=start_time.isoformat(),
        )

        try:
            params = {
                "query": query,
                "rows": max_results,
                "select": "DOI,title,author,abstract,published,reference",
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            response = self._fetch_url(url)
            if response:
                data = json.loads(response)
                references = self._parse_crossref_response(data)
                result.references = references
                result.total_results = len(references)
                result.status = SearchStatus.SUCCESS if references else SearchStatus.NO_RESULTS
            else:
                result.status = SearchStatus.ERROR
                result.error_message = "Failed to fetch from CrossRef"
        except Exception as exc:
            result.status = SearchStatus.ERROR
            result.error_message = str(exc)

        result.search_duration_sec = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        return result

    def _fetch_url(self, url: str, timeout: int = 30) -> str | None:
        """Fetch URL content."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "R4-Knowledge/1.0 (mailto:research@r4.io)",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _parse_crossref_response(self, data: dict) -> list[ExternalReference]:
        """Parse CrossRef API JSON response."""
        references: list[ExternalReference] = []

        for item in data.get("message", {}).get("items", []):
            ref = ExternalReference(source_type=ExternalSourceType.CROSSREF)
            ref.doi = item.get("DOI", "")

            # Title
            titles = item.get("title", [])
            if titles:
                ref.title = titles[0]

            # Authors
            for author in item.get("author", []):
                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                if name:
                    ref.authors.append(name)

            # Abstract
            ref.abstract = item.get("abstract", "") or ""

            # Published date
            published = item.get("published", {})
            if published:
                parts = [
                    published.get("date-parts", [[]])[0],
                ]
                if parts[0]:
                    ref.published_date = "-".join(str(p) for p in parts[0])

            # URL
            if ref.doi:
                ref.url = f"https://doi.org/{ref.doi}"

            if ref.title:
                references.append(ref)

        return references


# ── External Knowledge Searcher ───────────────────────────────────────────


class ExternalKnowledgeSearcher:
    """Search and ingest external knowledge from multiple sources.

    Usage:
        searcher = ExternalKnowledgeSearcher()
        results = searcher.search_all("transformer attention mechanism")
    """

    def __init__(self) -> None:
        self._sources: dict[ExternalSourceType, Any] = {
            ExternalSourceType.ARXIV: ArxivSource(),
            ExternalSourceType.SEMANTIC_SCHOLAR: SemanticScholarSource(),
            ExternalSourceType.CROSSREF: CrossRefSource(),
        }

    def search(
        self,
        query: str,
        source_type: ExternalSourceType,
        *,
        max_results: int = 10,
    ) -> ExternalSearchResult:
        """Search a single external source."""
        source = self._sources.get(source_type)
        if not source:
            return ExternalSearchResult(
                query=query,
                source_type=source_type,
                status=SearchStatus.ERROR,
                error_message=f"Source {source_type} not available",
            )
        return source.search(query, max_results=max_results)

    def search_all(
        self,
        query: str,
        *,
        max_results_per_source: int = 5,
        sources: list[ExternalSourceType] | None = None,
    ) -> list[ExternalSearchResult]:
        """Search all configured external sources."""
        if sources is None:
            sources = list(self._sources.keys())

        results: list[ExternalSearchResult] = []
        for source_type in sources:
            result = self.search(
                query,
                source_type,
                max_results=max_results_per_source,
            )
            results.append(result)
        return results

    def compare(
        self,
        query: str,
        *,
        max_results_per_source: int = 5,
    ) -> dict[str, Any]:
        """Search and compare results across sources."""
        results = self.search_all(
            query,
            max_results_per_source=max_results_per_source,
        )

        # Aggregate all references
        all_refs: list[ExternalReference] = []
        for result in results:
            all_refs.extend(result.references)

        # Deduplicate by title
        seen_titles: set[str] = set()
        unique_refs: list[ExternalReference] = []
        for ref in all_refs:
            title_lower = ref.title.lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_refs.append(ref)

        return {
            "query": query,
            "total_unique_references": len(unique_refs),
            "results_by_source": [r.to_dict() for r in results],
            "unique_references": [r.to_dict() for r in unique_refs],
        }


# ── Convenience Functions ──────────────────────────────────────────────────


def search_arxiv(query: str, max_results: int = 10) -> ExternalSearchResult:
    """Search arXiv for papers (convenience function)."""
    source = ArxivSource()
    return source.search(query, max_results=max_results)


def search_semantic_scholar(
    query: str,
    max_results: int = 10,
) -> ExternalSearchResult:
    """Search Semantic Scholar for papers (convenience function)."""
    source = SemanticScholarSource()
    return source.search(query, max_results=max_results)


def search_crossref(query: str, max_results: int = 10) -> ExternalSearchResult:
    """Search CrossRef for papers (convenience function)."""
    source = CrossRefSource()
    return source.search(query, max_results=max_results)


def search_external(
    query: str,
    *,
    sources: list[ExternalSourceType] | None = None,
    max_results_per_source: int = 5,
) -> list[ExternalSearchResult]:
    """Search multiple external sources (convenience function)."""
    searcher = ExternalKnowledgeSearcher()
    return searcher.search_all(
        query,
        max_results_per_source=max_results_per_source,
        sources=sources,
    )
