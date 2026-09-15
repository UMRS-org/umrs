"""Tests for the knowledge coverage dashboard module (Phase 7)."""

from __future__ import annotations

from umrs.knowledge.coverage import (
    ArtifactCoverage,
    ArtifactType,
    CoverageCalculator,
    CoverageDashboard,
    CoverageStatus,
    calculate_coverage,
)


# ── Model Tests ────────────────────────────────────────────────────────────


def test_coverage_status_enum():
    """Test CoverageStatus enum values."""
    assert CoverageStatus.COMPLETE == "complete"
    assert CoverageStatus.PARTIAL == "partial"
    assert CoverageStatus.MISSING == "missing"
    assert CoverageStatus.ERROR == "error"
    assert CoverageStatus.UNKNOWN == "unknown"


def test_artifact_type_enum():
    """Test ArtifactType enum values."""
    assert ArtifactType.PAPER == "paper"
    assert ArtifactType.CODE == "code"
    assert ArtifactType.NOTEBOOK == "notebook"
    assert ArtifactType.DATASET == "dataset"
    assert ArtifactType.EXPERIMENT == "experiment"
    assert ArtifactType.EMBEDDING == "embedding"
    assert ArtifactType.UNKNOWN == "unknown"


def test_artifact_coverage_defaults():
    """Test ArtifactCoverage defaults."""
    coverage = ArtifactCoverage(
        artifact_id="paper-001",
        artifact_type=ArtifactType.PAPER,
    )

    assert coverage.artifact_id == "paper-001"
    assert coverage.artifact_type == ArtifactType.PAPER
    assert coverage.status == CoverageStatus.UNKNOWN
    assert coverage.metadata_completeness == 0.0
    assert coverage.extraction_confidence == 0.0
    assert coverage.missing_fields == []
    assert coverage.missing_metadata == []
    assert coverage.indexed is False
    assert coverage.embedding_generated is False


def test_artifact_coverage_to_dict():
    """Test ArtifactCoverage to_dict."""
    coverage = ArtifactCoverage(
        artifact_id="paper-001",
        artifact_type=ArtifactType.PAPER,
        filename="paper.pdf",
        status=CoverageStatus.COMPLETE,
        metadata_completeness=0.9,
        extraction_confidence=0.8,
    )

    data = coverage.to_dict()
    assert data["artifact_id"] == "paper-001"
    assert data["artifact_type"] == "paper"
    assert data["status"] == "complete"
    assert data["metadata_completeness"] == 0.9
    assert data["extraction_confidence"] == 0.8


def test_coverage_dashboard_defaults():
    """Test CoverageDashboard defaults."""
    dashboard = CoverageDashboard()

    assert dashboard.artifacts == []
    assert dashboard.type_counts == {}
    assert dashboard.coverage_score == 0.0
    assert dashboard.confidence_score == 0.0
    assert dashboard.total_artifacts == 0
    assert dashboard.total_indexed == 0
    assert dashboard.total_embeddings == 0


def test_coverage_dashboard_to_dict():
    """Test CoverageDashboard to_dict."""
    dashboard = CoverageDashboard()
    dashboard.total_artifacts = 10
    dashboard.coverage_score = 0.8
    dashboard.confidence_score = 0.75

    data = dashboard.to_dict()
    assert data["total_artifacts"] == 10
    assert data["coverage_score"] == 0.8
    assert data["confidence_score"] == 0.75


# ── CoverageCalculator Tests ───────────────────────────────────────────────


def test_calculator_init():
    """Test CoverageCalculator initialization."""
    calculator = CoverageCalculator()
    assert calculator is not None


def test_calculator_empty():
    """Test calculating coverage for empty list."""
    calculator = CoverageCalculator()
    dashboard = calculator.calculate([])

    assert dashboard.total_artifacts == 0
    assert dashboard.coverage_score == 0.0
    assert dashboard.confidence_score == 0.0


def test_calculator_single_artifact():
    """Test calculating coverage for a single artifact."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
            "metadata": {
                "title": "Test Paper",
                "author": "Test Author",
                "created_at": "2026-01-01",
                "description": "A test paper",
                "doi": "10.1234/test",
                "abstract": "Test abstract",
                "keywords": ["test", "paper"],
            },
            "extracted_knowledge": {
                "objective": "Test the system",
                "methodology": "We use method X",
                "conclusions": ["It works"],
                "limitations": ["Small dataset"],
                "assumptions": ["Data is clean"],
                "equations": ["E=mc^2"],
                "datasets": ["MNIST"],
                "experiments": ["Exp 1"],
                "extraction_confidence": 0.9,
            },
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert dashboard.total_artifacts == 1
    assert dashboard.type_counts["paper"] == 1
    assert dashboard.coverage_score > 0.0
    assert dashboard.confidence_score > 0.0


def test_calculator_multiple_types():
    """Test calculating coverage for multiple artifact types."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
        },
        {
            "artifact_id": "code-001",
            "artifact_type": "code",
            "filename": "model.py",
        },
        {
            "artifact_id": "notebook-001",
            "artifact_type": "notebook",
            "filename": "experiment.ipynb",
        },
        {
            "artifact_id": "dataset-001",
            "artifact_type": "dataset",
            "filename": "data.csv",
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert dashboard.total_artifacts == 4
    assert dashboard.type_counts["paper"] == 1
    assert dashboard.type_counts["code"] == 1
    assert dashboard.type_counts["notebook"] == 1
    assert dashboard.type_counts["dataset"] == 1


def test_calculator_unknown_type():
    """Test calculating coverage for unknown artifact type."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "unknown-001",
            "artifact_type": "unknown",
            "filename": "unknown.xyz",
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert dashboard.total_artifacts == 1
    assert dashboard.type_counts["unknown"] == 1
    assert len(dashboard.unknown_formats) == 1


def test_calculator_missing_fields():
    """Test that missing fields are detected."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
            "extracted_knowledge": {
                "objective": "Test",
                # Missing: methodology, conclusions, limitations, etc.
                "extraction_confidence": 0.5,
            },
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert dashboard.total_artifacts == 1
    # Should have missing fields detected
    assert len(dashboard.artifacts[0].missing_fields) > 0


def test_calculator_missing_metadata():
    """Test that missing metadata is detected."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
            "metadata": {
                # Missing: title, author, created_at, description, doi, etc.
            },
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert dashboard.total_artifacts == 1
    # Should have missing metadata detected
    assert len(dashboard.artifacts[0].missing_metadata) > 0
    assert len(dashboard.missing_metadata) == 1


def test_calculator_indexing_status():
    """Test that indexing status is tracked."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
            "indexed": True,
            "indexed_at": "2026-06-30T10:00:00",
            "embedding_generated": True,
        },
        {
            "artifact_id": "paper-002",
            "artifact_type": "paper",
            "filename": "paper2.pdf",
            "indexed": False,
            "indexed_at": "",
            "embedding_generated": False,
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert dashboard.total_indexed == 1
    assert dashboard.total_embeddings == 1
    assert dashboard.last_indexing_time == "2026-06-30T10:00:00"


def test_calculator_scores():
    """Test that coverage and confidence scores are calculated."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
            "extracted_knowledge": {
                "extraction_confidence": 0.8,
                "objective": "Test",
                "methodology": "Method X",
                "conclusions": ["Works"],
            },
            "metadata": {
                "title": "Test",
                "author": "Author",
                "created_at": "2026-01-01",
                "description": "Desc",
                "doi": "10.1234",
                "abstract": "Abstract",
                "keywords": ["test"],
            },
        },
        {
            "artifact_id": "paper-002",
            "artifact_type": "paper",
            "filename": "paper2.pdf",
            "extracted_knowledge": {
                "extraction_confidence": 0.6,
            },
            "metadata": {},
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert 0.0 <= dashboard.coverage_score <= 1.0
    assert 0.0 <= dashboard.confidence_score <= 1.0
    assert 0.0 <= dashboard.completeness_score <= 1.0


def test_calculator_broken_references():
    """Test that broken references are tracked."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
            "broken_references": ["ref-001", "ref-002"],
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert len(dashboard.broken_references) == 2


# ── Convenience Function Tests ─────────────────────────────────────────────


def test_calculate_coverage_convenience_function():
    """Test the calculate_coverage convenience function."""
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
        },
    ]
    dashboard = calculate_coverage(artifacts)

    assert isinstance(dashboard, CoverageDashboard)
    assert dashboard.total_artifacts == 1


def test_determine_status_low_completeness_is_missing():
    """Artifacts with very low completeness should be MISSING, not PARTIAL."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "paper.pdf",
            "extracted_knowledge": {},
            "metadata": {},
        },
    ]
    dashboard = calculator.calculate(artifacts)

    assert dashboard.artifacts[0].status == CoverageStatus.MISSING
    assert dashboard.type_missing.get("paper", 0) == 1


def test_code_missing_fields_are_type_aware():
    """Code coverage should not require paper-only fields like objective."""
    calculator = CoverageCalculator()
    artifacts = [
        {
            "artifact_id": "code-001",
            "artifact_type": "code",
            "filename": "model.py",
            "extracted_knowledge": {
                "modules": ["model"],
                "apis": ["TransformerModel"],
                "algorithms": ["train"],
                "dependencies": ["torch"],
                "parameters": ["input_dim"],
                "kernels": [],
                "execution_flow": "has_main_block",
            },
        },
    ]
    dashboard = calculator.calculate(artifacts)

    missing = dashboard.artifacts[0].missing_fields
    assert "objective" not in missing
    assert "kernels" in missing
