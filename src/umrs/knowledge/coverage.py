"""Knowledge coverage dashboard — assess knowledge base completeness.

Phase 7 of the Scientific Knowledge Build-up roadmap.

This module implements a coverage dashboard that shows:
  - Papers, Code, Notebooks, Experiments, Datasets, Embeddings
  - Missing metadata, Unknown files
  - Coverage score, Confidence score
  - Last indexing time
  - Incomplete parsing, Broken references, Unknown formats

The dashboard helps the user understand exactly how complete the
knowledge base is before starting autonomous research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .extract import coverage_fields_for_type

logger = logging.getLogger(__name__)


# ── Coverage Status ────────────────────────────────────────────────────────


class CoverageStatus(str, Enum):
    """Status of a knowledge artifact in the coverage dashboard."""

    COMPLETE = "complete"  # All required metadata present
    PARTIAL = "partial"  # Some metadata missing
    MISSING = "missing"  # Artifact not found
    ERROR = "error"  # Parsing/extraction failed
    UNKNOWN = "unknown"  # Unknown format or type


class ArtifactType(str, Enum):
    """Types of knowledge artifacts tracked by the dashboard."""

    PAPER = "paper"
    CODE = "code"
    NOTEBOOK = "notebook"
    EXPERIMENT = "experiment"
    DATASET = "dataset"
    EMBEDDING = "embedding"
    DOCUMENT = "document"
    IMAGE = "image"
    TEXT = "text"
    STRUCTURED = "structured"
    UNKNOWN = "unknown"


# ── Coverage Models ────────────────────────────────────────────────────────


@dataclass
class ArtifactCoverage:
    """Coverage information for a single artifact."""

    artifact_id: str
    artifact_type: ArtifactType
    filename: str = ""
    status: CoverageStatus = CoverageStatus.UNKNOWN

    # Metadata completeness (0.0–1.0)
    metadata_completeness: float = 0.0

    # Extraction quality (0.0–1.0)
    extraction_confidence: float = 0.0

    # What's missing
    missing_fields: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)

    # Error information
    error_message: str = ""
    error_type: str = ""

    # Indexing status
    indexed: bool = False
    indexed_at: str = ""
    embedding_generated: bool = False

    # Content metrics
    content_size_bytes: int = 0
    extracted_text_length: int = 0
    chunk_count: int = 0

    # References
    references_count: int = 0
    broken_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "filename": self.filename,
            "status": self.status.value,
            "metadata_completeness": self.metadata_completeness,
            "extraction_confidence": self.extraction_confidence,
            "missing_fields": self.missing_fields,
            "missing_metadata": self.missing_metadata,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "indexed": self.indexed,
            "indexed_at": self.indexed_at,
            "embedding_generated": self.embedding_generated,
            "content_size_bytes": self.content_size_bytes,
            "extracted_text_length": self.extracted_text_length,
            "chunk_count": self.chunk_count,
            "references_count": self.references_count,
            "broken_references": self.broken_references,
        }


@dataclass
class CoverageDashboard:
    """Complete coverage dashboard for a project.

    The dashboard aggregates coverage information across all artifact types
    and provides summary scores for knowledge completeness and confidence.
    """

    # Per-artifact coverage
    artifacts: list[ArtifactCoverage] = field(default_factory=list)

    # Summary by artifact type
    type_counts: dict[str, int] = field(default_factory=dict)
    type_complete: dict[str, int] = field(default_factory=dict)
    type_partial: dict[str, int] = field(default_factory=dict)
    type_missing: dict[str, int] = field(default_factory=dict)
    type_error: dict[str, int] = field(default_factory=dict)

    # Overall scores
    coverage_score: float = 0.0  # % of artifacts with complete metadata
    confidence_score: float = 0.0  # Average extraction confidence
    completeness_score: float = 0.0  # Average metadata completeness

    # Issues
    missing_artifacts: list[str] = field(default_factory=list)
    broken_references: list[str] = field(default_factory=list)
    unknown_formats: list[str] = field(default_factory=list)
    incomplete_parsing: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)

    # Indexing status
    total_indexed: int = 0
    total_embeddings: int = 0
    last_indexing_time: str = ""
    last_indexing_duration_sec: float = 0.0

    # Totals
    total_artifacts: int = 0
    total_size_bytes: int = 0
    total_chunks: int = 0
    total_references: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [a.to_dict() for a in self.artifacts],
            "type_counts": self.type_counts,
            "type_complete": self.type_complete,
            "type_partial": self.type_partial,
            "type_missing": self.type_missing,
            "type_error": self.type_error,
            "coverage_score": self.coverage_score,
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score,
            "missing_artifacts": self.missing_artifacts,
            "broken_references": self.broken_references,
            "unknown_formats": self.unknown_formats,
            "incomplete_parsing": self.incomplete_parsing,
            "missing_metadata": self.missing_metadata,
            "total_indexed": self.total_indexed,
            "total_embeddings": self.total_embeddings,
            "last_indexing_time": self.last_indexing_time,
            "last_indexing_duration_sec": self.last_indexing_duration_sec,
            "total_artifacts": self.total_artifacts,
            "total_size_bytes": self.total_size_bytes,
            "total_chunks": self.total_chunks,
            "total_references": self.total_references,
        }


# ── Coverage Calculator ───────────────────────────────────────────────────


class CoverageCalculator:
    """Calculate knowledge coverage for a project.

    Usage:
        calculator = CoverageCalculator()
        dashboard = calculator.calculate(artifacts)
    """

    def calculate(
        self,
        artifacts: list[dict[str, Any]],
    ) -> CoverageDashboard:
        """Calculate coverage dashboard for a list of artifacts.

        Args:
            artifacts: A list of artifact dictionaries with at least:
                - artifact_id: The artifact's id
                - artifact_type: The artifact's type
                - filename: The artifact's filename
                - extracted_knowledge: The extracted knowledge (optional)
                - metadata: Additional metadata (optional)

        Returns:
            A ``CoverageDashboard`` with coverage information.
        """
        dashboard = CoverageDashboard()

        for artifact in artifacts:
            coverage = self._assess_artifact(artifact)
            dashboard.artifacts.append(coverage)
            self._update_totals(dashboard, coverage)
            self._update_type_counts(dashboard, coverage)
            self._update_issues(dashboard, coverage)

        self._calculate_scores(dashboard)
        self._update_indexing_status(dashboard)

        return dashboard

    def _assess_artifact(
        self,
        artifact: dict[str, Any],
    ) -> ArtifactCoverage:
        """Assess coverage for a single artifact."""
        artifact_id = artifact.get("artifact_id", "")
        artifact_type_str = artifact.get("artifact_type", "unknown")
        filename = artifact.get("filename", "")

        try:
            artifact_type = ArtifactType(artifact_type_str)
        except ValueError:
            artifact_type = ArtifactType.UNKNOWN

        coverage = ArtifactCoverage(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
        )

        # Extract knowledge if available
        knowledge = artifact.get("extracted_knowledge", {})
        if isinstance(knowledge, dict):
            coverage.extraction_confidence = knowledge.get("extraction_confidence", 0.0)
            coverage.missing_fields = self._find_missing_fields(
                knowledge,
                artifact_type_str,
            )
        elif knowledge is not None:
            coverage.extraction_confidence = getattr(
                knowledge, "extraction_confidence", 0.0
            )
            coverage.missing_fields = self._find_missing_fields(
                knowledge,
                artifact_type_str,
            )

        # Check metadata
        metadata = artifact.get("metadata", {})
        coverage.missing_metadata = self._find_missing_metadata(
            artifact_type,
            metadata,
            knowledge,
        )

        # Calculate completeness
        coverage.metadata_completeness = self._calculate_completeness(
            coverage.missing_metadata,
            coverage.missing_fields,
        )

        # Determine status
        coverage.status = self._determine_status(coverage)

        # Content metrics
        coverage.content_size_bytes = artifact.get("size_bytes", 0)
        coverage.extracted_text_length = len(
            knowledge.get("extracted_text", "") if isinstance(knowledge, dict) else ""
        )
        coverage.chunk_count = artifact.get("chunk_count", 0)

        # References
        coverage.references_count = artifact.get("references_count", 0)
        coverage.broken_references = artifact.get("broken_references", [])

        # Indexing status
        coverage.indexed = artifact.get("indexed", False)
        coverage.indexed_at = artifact.get("indexed_at", "")
        coverage.embedding_generated = artifact.get("embedding_generated", False)

        return coverage

    def _find_missing_fields(
        self,
        knowledge: Any,
        artifact_type: str = "paper",
    ) -> list[str]:
        """Find missing type-relevant fields in extracted knowledge."""
        missing: list[str] = []
        key_fields = coverage_fields_for_type(artifact_type)

        if isinstance(knowledge, dict):
            for field_name in key_fields:
                value = knowledge.get(field_name)
                if not value:
                    missing.append(field_name)
        elif knowledge is not None:
            for field_name in key_fields:
                value = getattr(knowledge, field_name, None)
                if not value:
                    missing.append(field_name)

        return missing

    def _find_missing_metadata(
        self,
        artifact_type: ArtifactType,
        metadata: dict[str, Any],
        knowledge: Any,
    ) -> list[str]:
        """Find missing metadata for an artifact type."""
        missing: list[str] = []

        # Common metadata fields
        common_fields = ["title", "author", "created_at", "description"]
        for field_name in common_fields:
            if not metadata.get(field_name):
                missing.append(field_name)

        # Type-specific fields
        if artifact_type == ArtifactType.PAPER:
            specific_fields = ["doi", "abstract", "keywords"]
        elif artifact_type == ArtifactType.CODE:
            specific_fields = ["language", "version", "license"]
        elif artifact_type == ArtifactType.NOTEBOOK:
            specific_fields = ["kernel", "language", "cell_count"]
        elif artifact_type == ArtifactType.DATASET:
            specific_fields = ["format", "schema", "size", "source"]
        else:
            specific_fields = []

        for field_name in specific_fields:
            if not metadata.get(field_name):
                missing.append(field_name)

        return missing

    def _calculate_completeness(
        self,
        missing_metadata: list[str],
        missing_fields: list[str],
    ) -> float:
        """Calculate metadata completeness score (0.0–1.0)."""
        total = len(missing_metadata) + len(missing_fields)
        if total == 0:
            return 1.0
        # Weight: missing fields are more important than missing metadata
        max_possible = 10  # Maximum fields we check
        return max(0.0, 1.0 - (total / max_possible))

    def _determine_status(self, coverage: ArtifactCoverage) -> CoverageStatus:
        """Determine the overall status of an artifact."""
        if coverage.artifact_type == ArtifactType.UNKNOWN:
            return CoverageStatus.UNKNOWN

        if coverage.metadata_completeness >= 0.8:
            return CoverageStatus.COMPLETE
        elif coverage.metadata_completeness >= 0.4:
            return CoverageStatus.PARTIAL
        else:
            return CoverageStatus.MISSING

    def _update_totals(
        self,
        dashboard: CoverageDashboard,
        coverage: ArtifactCoverage,
    ) -> None:
        """Update total counts."""
        dashboard.total_artifacts += 1
        dashboard.total_size_bytes += coverage.content_size_bytes
        dashboard.total_chunks += coverage.chunk_count
        dashboard.total_references += coverage.references_count

    def _update_type_counts(
        self,
        dashboard: CoverageDashboard,
        coverage: ArtifactCoverage,
    ) -> None:
        """Update type-specific counts."""
        type_key = coverage.artifact_type.value

        dashboard.type_counts[type_key] = dashboard.type_counts.get(type_key, 0) + 1

        if coverage.status == CoverageStatus.COMPLETE:
            dashboard.type_complete[type_key] = (
                dashboard.type_complete.get(type_key, 0) + 1
            )
        elif coverage.status == CoverageStatus.PARTIAL:
            dashboard.type_partial[type_key] = (
                dashboard.type_partial.get(type_key, 0) + 1
            )
        elif coverage.status == CoverageStatus.MISSING:
            dashboard.type_missing[type_key] = (
                dashboard.type_missing.get(type_key, 0) + 1
            )
        elif coverage.status == CoverageStatus.ERROR:
            dashboard.type_error[type_key] = dashboard.type_error.get(type_key, 0) + 1

    def _update_issues(
        self,
        dashboard: CoverageDashboard,
        coverage: ArtifactCoverage,
    ) -> None:
        """Update issue lists."""
        if coverage.status == CoverageStatus.UNKNOWN:
            dashboard.unknown_formats.append(coverage.artifact_id)
        if coverage.status == CoverageStatus.ERROR:
            dashboard.incomplete_parsing.append(coverage.artifact_id)
        if coverage.broken_references:
            dashboard.broken_references.extend(coverage.broken_references)
        if coverage.missing_metadata:
            dashboard.missing_metadata.append(coverage.artifact_id)

    def _calculate_scores(self, dashboard: CoverageDashboard) -> None:
        """Calculate overall coverage and confidence scores."""
        if not dashboard.artifacts:
            return

        # Coverage score: % of artifacts with COMPLETE status
        complete_count = sum(
            1 for a in dashboard.artifacts if a.status == CoverageStatus.COMPLETE
        )
        dashboard.coverage_score = complete_count / len(dashboard.artifacts)

        # Confidence score: average extraction confidence
        confidences = [a.extraction_confidence for a in dashboard.artifacts]
        dashboard.confidence_score = sum(confidences) / len(confidences)

        # Completeness score: average metadata completeness
        completenesses = [a.metadata_completeness for a in dashboard.artifacts]
        dashboard.completeness_score = sum(completenesses) / len(completenesses)

    def _update_indexing_status(self, dashboard: CoverageDashboard) -> None:
        """Update indexing status totals."""
        for artifact in dashboard.artifacts:
            if artifact.indexed:
                dashboard.total_indexed += 1
            if artifact.embedding_generated:
                dashboard.total_embeddings += 1
            if artifact.indexed_at:
                # Track the most recent indexing time
                if (
                    not dashboard.last_indexing_time
                    or artifact.indexed_at > dashboard.last_indexing_time
                ):
                    dashboard.last_indexing_time = artifact.indexed_at


# ── Convenience Function ────────────────────────────────────────────────────


def calculate_coverage(
    artifacts: list[dict[str, Any]],
) -> CoverageDashboard:
    """Calculate coverage dashboard for a list of artifacts (convenience function).

    Args:
        artifacts: A list of artifact dictionaries

    Returns:
        A ``CoverageDashboard`` with coverage information.
    """
    calculator = CoverageCalculator()
    return calculator.calculate(artifacts)
