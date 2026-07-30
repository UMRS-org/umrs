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
    CINII = "cinii"
    JSTAGE = "jstage"
    WEKO3 = "weko3"
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
                result.status = (
                    SearchStatus.SUCCESS if references else SearchStatus.NO_RESULTS
                )
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
            id_match = re.search(
                r"<id>(http://arxiv.org/abs/)?(.+?)</id>", entry, re.DOTALL
            )
            if id_match:
                ref.arxiv_id = id_match.group(2).strip()
                ref.url = f"http://arxiv.org/abs/{ref.arxiv_id}"

            # Extract authors
            author_pattern = re.compile(
                r"<author>.*?<name>(.*?)</name>.*?</author>", re.DOTALL
            )
            for author_match in author_pattern.finditer(entry):
                ref.authors.append(author_match.group(1).strip())

            # Extract abstract
            summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            if summary_match:
                ref.abstract = summary_match.group(1).strip()

            # Extract published date
            published_match = re.search(
                r"<published>(.*?)</published>", entry, re.DOTALL
            )
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
                result.status = (
                    SearchStatus.SUCCESS if references else SearchStatus.NO_RESULTS
                )
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

    @staticmethod
    def _parse_s2_paper(paper: dict) -> ExternalReference:
        """Build one reference from a single Semantic Scholar record."""
        ref = ExternalReference(source_type=ExternalSourceType.SEMANTIC_SCHOLAR)
        ref.title = paper.get("title", "")
        ref.abstract = paper.get("abstract", "") or ""
        ref.authors = [a.get("name", "") for a in paper.get("authors", [])]

        year = paper.get("year")
        if year:
            ref.published_date = str(year)
        ref.citations = paper.get("citationCount", 0) or 0

        ext_ids = paper.get("externalIds", {}) or {}
        ref.doi = ext_ids.get("DOI", "")
        ref.arxiv_id = ext_ids.get("ArXiv", "")
        ref.pubmed_id = ext_ids.get("PubMed", "")

        if ref.doi:
            ref.url = f"https://doi.org/{ref.doi}"
        elif ref.arxiv_id:
            ref.url = f"http://arxiv.org/abs/{ref.arxiv_id}"
        return ref

    def _parse_s2_response(self, data: dict) -> list[ExternalReference]:
        """Parse Semantic Scholar API JSON response."""
        references = [self._parse_s2_paper(p) for p in data.get("data", [])]
        return [r for r in references if r.title]


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
                result.status = (
                    SearchStatus.SUCCESS if references else SearchStatus.NO_RESULTS
                )
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

    @staticmethod
    def _parse_crossref_item(item: dict) -> ExternalReference:
        """Build one reference from a single CrossRef work item."""
        ref = ExternalReference(source_type=ExternalSourceType.CROSSREF)
        ref.doi = item.get("DOI", "")

        titles = item.get("title", [])
        if titles:
            ref.title = titles[0]

        for author in item.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                ref.authors.append(name)

        ref.abstract = item.get("abstract", "") or ""

        published = item.get("published", {})
        if published:
            parts = published.get("date-parts", [[]])[0]
            if parts:
                ref.published_date = "-".join(str(p) for p in parts)

        if ref.doi:
            ref.url = f"https://doi.org/{ref.doi}"
        return ref

    def _parse_crossref_response(self, data: dict) -> list[ExternalReference]:
        """Parse CrossRef API JSON response."""
        refs = [
            self._parse_crossref_item(i)
            for i in data.get("message", {}).get("items", [])
        ]
        return [r for r in refs if r.title]


class CiNiiSource:
    """CiNii Research (NII) — Japan's researcher/publisher-integrated index.

    Uses the CiNii Research REST API (`/v1/rsearch`) which returns Atom-style
    XML over JPCOAR metadata. No API key is required for read-only search.
    """

    BASE_URL = "https://cir.nii.ac.jp/opensearch/all"

    def search(self, query: str, *, max_results: int = 10) -> ExternalSearchResult:
        """Search CiNii for papers in Japanese or English metadata."""
        start_time = datetime.now(timezone.utc)
        result = ExternalSearchResult(
            query=query,
            source_type=ExternalSourceType.CINII,
            searched_at=start_time.isoformat(),
        )
        try:
            params = {
                "q": query,
                "count": max_results,
                "format": "json",
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            response = self._fetch_url(url)
            if response:
                data = json.loads(response)
                references = self._parse_response(data)
                result.references = references
                result.total_results = len(references)
                result.status = (
                    SearchStatus.SUCCESS if references else SearchStatus.NO_RESULTS
                )
            else:
                result.status = SearchStatus.ERROR
                result.error_message = "Failed to fetch from CiNii"
        except Exception as exc:
            result.status = SearchStatus.ERROR
            result.error_message = str(exc)
        result.search_duration_sec = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        return result

    def _fetch_url(self, url: str, timeout: int = 30) -> str | None:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "R4-Knowledge/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    @staticmethod
    def _authors_from_item(item: dict) -> list[str]:
        """Normalize the variable creator shape into a list of names."""
        creator = item.get("creator")
        if isinstance(creator, list):
            return [c.get("name", "") for c in creator if isinstance(c, dict)]
        if isinstance(creator, dict):
            name = creator.get("name", "")
            return [name] if name else []
        return []

    def _parse_one_item(self, item: dict) -> ExternalReference:
        """Build one reference from a CiNii metadata item."""
        ref = ExternalReference(source_type=ExternalSourceType.CINII)
        ref.title = item.get("title", "").strip()
        ref.authors = self._authors_from_item(item)
        ref.abstract = (item.get("description") or "").strip()
        ref.published_date = str(item.get("pubDate") or "").strip()
        link = item.get("link") or {}
        if isinstance(link, dict):
            ref.url = link.get("@id", "") or link.get("item", "")
        return ref

    def _parse_response(self, data: dict) -> list[ExternalReference]:
        refs = [
            self._parse_one_item(i)
            for i in data.get("@graph", [{}])[0].get("items", [])
        ]
        return [r for r in refs if r.title]


class JStageSource:
    """J-STAGE (JST) — Japanese Scientific and Technological Aggregator, Electronic.

    Uses the `/material/search` OpenSearch endpoint which returns JROS XML
    with the OpenSearch 1.1 envelope.
    """

    BASE_URL = "https://www.jstage.jst.go.jp/browseapi/mat/search"

    def search(self, query: str, *, max_results: int = 10) -> ExternalSearchResult:
        """Search J-STAGE for research articles."""
        start_time = datetime.now(timezone.utc)
        result = ExternalSearchResult(
            query=query,
            source_type=ExternalSourceType.JSTAGE,
            searched_at=start_time.isoformat(),
        )
        try:
            params = {
                "service": "1",
                "kwd": query,
                "count": max_results,
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            response = self._fetch_url(url)
            if response:
                references = self._parse_response(response)
                result.references = references
                result.total_results = len(references)
                result.status = (
                    SearchStatus.SUCCESS if references else SearchStatus.NO_RESULTS
                )
            else:
                result.status = SearchStatus.ERROR
                result.error_message = "Failed to fetch from J-STAGE"
        except Exception as exc:
            result.status = SearchStatus.ERROR
            result.error_message = str(exc)
        result.search_duration_sec = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        return result

    def _fetch_url(self, url: str, timeout: int = 30) -> str | None:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "R4-Knowledge/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _parse_one_entry(self, entry: str) -> ExternalReference:
        """Build one reference from a JROS `<entry>` block."""
        ref = ExternalReference(source_type=ExternalSourceType.JSTAGE)

        title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        if title_match:
            ref.title = title_match.group(1).strip()
        author_pattern = re.compile(
            r"<\w*:?creator\b[^>]*>(.*?)</\w*:?creator>", re.DOTALL
        )
        ref.authors = [m.group(1).strip() for m in author_pattern.finditer(entry)]
        desc_match = re.search(
            r"<\w*:?description\b[^>]*>(.*?)</\w*:?description>", entry, re.DOTALL
        )
        if desc_match:
            ref.abstract = desc_match.group(1).strip()
        pub_match = re.search(
            r"<\w*:?publicationDate\b[^>]*>(.*?)</\w*:?publicationDate>", entry
        )
        if pub_match:
            ref.published_date = pub_match.group(1).strip()
        link_match = re.search(r'<link href="([^"]+)"', entry)
        if link_match:
            ref.url = link_match.group(1)
        return ref

    def _parse_response(self, xml_text: str) -> list[ExternalReference]:
        entry_pattern = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
        refs = [
            self._parse_one_entry(m.group(1)) for m in entry_pattern.finditer(xml_text)
        ]
        return [r for r in refs if r.title]


class WEKO3Source:
    """WEKO3 (NII Official Repository software) — per-university OAI-PMH.

    Queries via OAI-PMH `ListRecords&metadataPrefix=oai_dc` so every
    institution using stock WEKO3 metadata is reachable the same way.
    """

    def search(
        self,
        query: str,
        *,
        base_url: str = "https://repository.nii.ac.jp/oai",
        max_results: int = 10,
        timeout: int = 30,
    ) -> ExternalSearchResult:
        """Search WEKO3 OAI-PMH endpoint for repository records."""
        start_time = datetime.now(timezone.utc)
        result = ExternalSearchResult(
            query=query,
            source_type=ExternalSourceType.WEKO3,
            searched_at=start_time.isoformat(),
        )
        try:
            params = {
                "verb": "ListRecords",
                "metadataPrefix": "oai_dc",
                "set": "weko3",
            }
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            response = self._fetch_url(url, timeout=timeout)
            if response:
                references = self._parse_oai_dc(response, query)
                result.references = references[:max_results]
                result.total_results = len(result.references)
                result.status = (
                    SearchStatus.SUCCESS
                    if result.references
                    else SearchStatus.NO_RESULTS
                )
            else:
                result.status = SearchStatus.ERROR
                result.error_message = "Failed to fetch from WEKO3 endpoint"
        except Exception as exc:
            result.status = SearchStatus.ERROR
            result.error_message = str(exc)
        result.search_duration_sec = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        return result

    def _fetch_url(self, url: str, timeout: int = 30) -> str | None:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "R4-Knowledge/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _parse_oai_dc_record(self, record: str) -> ExternalReference:
        """Build one reference from a single ``<record>`` block."""
        ref = ExternalReference(source_type=ExternalSourceType.WEKO3)
        title_m = re.search(r"<dc:title>(.*?)</dc:title>", record, re.DOTALL)
        if title_m:
            ref.title = title_m.group(1).strip()
        creator_pattern = re.compile(
            r"<\w*:?creator\b[^>]*>(.*?)</\w*:?creator>", re.DOTALL
        )
        ref.authors = [c.group(1).strip() for c in creator_pattern.finditer(record)]
        desc_m = re.search(
            r"<\w*:?description\b[^>]*>(.*?)</\w*:?description>", record, re.DOTALL
        )
        if desc_m:
            ref.abstract = desc_m.group(1).strip()
        date_m = re.search(r"<dc:date>(.*?)</dc:date>", record)
        if date_m:
            ref.published_date = date_m.group(1).strip()
        ident_m = re.search(r"<dc:identifier>(http[^<]+)</dc:identifier>", record)
        if ident_m:
            ref.url = ident_m.group(1).strip()
        return ref

    def _parse_oai_dc(self, xml_text: str, query: str) -> list[ExternalReference]:
        needle = query.lower()
        record_pattern = re.compile(r"<record>.*?</record>", re.DOTALL)
        refs = [
            self._parse_oai_dc_record(m.group(0))
            for m in record_pattern.finditer(xml_text)
        ]
        return [
            r for r in refs if r.title and (not needle or needle in r.title.lower())
        ]


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
            ExternalSourceType.CINII: CiNiiSource(),
            ExternalSourceType.JSTAGE: JStageSource(),
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


def search_cinii(query: str, max_results: int = 10) -> ExternalSearchResult:
    """Search CiNii Research (NII) for papers (convenience function)."""
    return CiNiiSource().search(query, max_results=max_results)


def search_jstage(query: str, max_results: int = 10) -> ExternalSearchResult:
    """Search J-STAGE for articles (convenience function)."""
    return JStageSource().search(query, max_results=max_results)


def search_weko3(
    query: str,
    max_results: int = 10,
    *,
    base_url: str = "https://repository.nii.ac.jp/oai",
) -> ExternalSearchResult:
    """Search a WEKO3 OAI-PMH endpoint for records (convenience function)."""
    return WEKO3Source().search(query, base_url=base_url, max_results=max_results)


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
