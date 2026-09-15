"""Tests for the research readiness gate module (Phase 10)."""

from __future__ import annotations

from umrs.knowledge.readiness import (
    ConfidenceLevel,
    IssueSeverity,
    ReadinessIssue,
    ReadinessLevel,
    ResearchReadinessGate,
    ResearchReadinessReport,
    evaluate_readiness,
)


# ── Enum Tests ─────────────────────────────────────────────────────────────


def test_readiness_level_enum():
    """Test ReadinessLevel enum values."""
    assert ReadinessLevel.READY == "ready"
    assert ReadinessLevel.PARTIALLY_READY == "partially_ready"
    assert ReadinessLevel.NOT_READY == "not_ready"
    assert ReadinessLevel.BLOCKED == "blocked"


def test_confidence_level_enum():
    """Test ConfidenceLevel enum values."""
    assert ConfidenceLevel.HIGH == "high"
    assert ConfidenceLevel.MEDIUM == "medium"
    assert ConfidenceLevel.LOW == "low"
    assert ConfidenceLevel.UNKNOWN == "unknown"


def test_issue_severity_enum():
    """Test IssueSeverity enum values."""
    assert IssueSeverity.CRITICAL == "critical"
    assert IssueSeverity.WARNING == "warning"
    assert IssueSeverity.INFO == "info"


# ── Model Tests ────────────────────────────────────────────────────────────


def test_readiness_issue_defaults():
    """Test ReadinessIssue defaults."""
    issue = ReadinessIssue()

    assert issue.issue_id == ""
    assert issue.severity == IssueSeverity.INFO
    assert issue.category == ""
    assert issue.description == ""
    assert issue.affected_artifacts == []
    assert issue.recommendation == ""


def test_readiness_issue_to_dict():
    """Test ReadinessIssue to_dict."""
    issue = ReadinessIssue(
        issue_id="issue-001",
        severity=IssueSeverity.CRITICAL,
        category="missing_dataset",
        description="Dataset X is missing",
        affected_artifacts=["ds-001"],
        recommendation="Upload dataset X",
    )

    data = issue.to_dict()
    assert data["issue_id"] == "issue-001"
    assert data["severity"] == "critical"
    assert data["category"] == "missing_dataset"
    assert data["description"] == "Dataset X is missing"
    assert data["affected_artifacts"] == ["ds-001"]
    assert data["recommendation"] == "Upload dataset X"


def test_research_readiness_report_defaults():
    """Test ResearchReadinessReport defaults."""
    report = ResearchReadinessReport()

    assert report.project_id == ""
    assert report.knowledge_completeness == 0.0
    assert report.knowledge_confidence == 0.0
    assert report.readiness_level == ReadinessLevel.NOT_READY
    assert report.confidence_level == ConfidenceLevel.UNKNOWN
    assert report.issues == []
    assert report.critical_issue_count == 0
    assert report.warning_issue_count == 0
    assert report.info_issue_count == 0
    assert report.user_approved is False


def test_research_readiness_report_to_dict():
    """Test ResearchReadinessReport to_dict."""
    report = ResearchReadinessReport(
        project_id="proj-001",
        knowledge_completeness=0.9,
        knowledge_confidence=0.85,
        readiness_level=ReadinessLevel.READY,
        confidence_level=ConfidenceLevel.HIGH,
    )

    data = report.to_dict()
    assert data["project_id"] == "proj-001"
    assert data["knowledge_completeness"] == 0.9
    assert data["knowledge_confidence"] == 0.85
    assert data["readiness_level"] == "ready"
    assert data["confidence_level"] == "high"


# ── ResearchReadinessGate Tests ────────────────────────────────────────────


def test_gate_init():
    """Test ResearchReadinessGate initialization."""
    gate = ResearchReadinessGate()
    assert gate is not None


def test_gate_evaluate_empty_dashboard():
    """Test evaluating with an empty dashboard (blocked)."""
    gate = ResearchReadinessGate()
    dashboard = type("Dashboard", (), {"to_dict": lambda self: {}})()
    report = gate.evaluate("proj-001", dashboard)

    assert report.project_id == "proj-001"
    assert report.readiness_level == ReadinessLevel.BLOCKED
    assert report.recommendation == "Blocked — no knowledge base found"


def test_gate_evaluate_ready():
    """Test evaluating a ready project."""
    gate = ResearchReadinessGate()
    dashboard_data = {
        "coverage_score": 0.9,
        "completeness_score": 0.85,
        "confidence_score": 0.9,
        "type_counts": {"paper": 5, "code": 3, "notebook": 2},
        "type_complete": {"paper": 4, "code": 2, "notebook": 2},
        "type_partial": {"paper": 1, "code": 1},
        "type_missing": {},
        "missing_datasets": [],
        "missing_references": [],
        "missing_implementations": [],
        "unknown_assumptions": [],
        "incomplete_parsing": [],
        "broken_references": [],
        "unknown_formats": [],
    }
    dashboard = type("Dashboard", (), {"to_dict": lambda self: dashboard_data})()
    report = gate.evaluate("proj-001", dashboard)

    assert report.knowledge_completeness == 0.85
    assert report.knowledge_confidence == 0.9
    assert report.readiness_level == ReadinessLevel.READY
    assert report.confidence_level == ConfidenceLevel.HIGH
    assert report.recommendation == "Ready for autonomous research"
    assert len(report.next_steps) == 3


def test_gate_evaluate_partially_ready():
    """Test evaluating a partially ready project."""
    gate = ResearchReadinessGate()
    dashboard_data = {
        "coverage_score": 0.6,
        "completeness_score": 0.6,
        "confidence_score": 0.6,
        "type_counts": {"paper": 3, "code": 2},
        "type_complete": {"paper": 2, "code": 1},
        "type_partial": {"paper": 1, "code": 1},
        "type_missing": {},
        "missing_datasets": ["ds-001"],
        "missing_references": [],
        "missing_implementations": [],
        "unknown_assumptions": [],
        "incomplete_parsing": [],
        "broken_references": [],
        "unknown_formats": [],
    }
    dashboard = type("Dashboard", (), {"to_dict": lambda self: dashboard_data})()
    report = gate.evaluate("proj-001", dashboard)

    assert report.readiness_level == ReadinessLevel.PARTIALLY_READY
    assert report.confidence_level == ConfidenceLevel.MEDIUM
    assert report.recommendation == "Partially ready — proceed with caution"
    assert len(report.issues) >= 1
    assert len(report.next_steps) >= 4


def test_gate_evaluate_not_ready():
    """Test evaluating a not ready project."""
    gate = ResearchReadinessGate()
    dashboard_data = {
        "coverage_score": 0.2,
        "completeness_score": 0.2,
        "confidence_score": 0.2,
        "type_counts": {"paper": 1},
        "type_complete": {"paper": 0},
        "type_partial": {"paper": 1},
        "type_missing": {},
        "missing_datasets": ["ds-001"],
        "missing_references": ["ref-001"],
        "missing_implementations": ["impl-001"],
        "unknown_assumptions": ["assump-001"],
        "incomplete_parsing": ["parse-001"],
        "broken_references": ["broken-001"],
        "unknown_formats": ["unknown-001"],
    }
    dashboard = type("Dashboard", (), {"to_dict": lambda self: dashboard_data})()
    report = gate.evaluate("proj-001", dashboard)

    assert report.readiness_level == ReadinessLevel.NOT_READY
    assert report.confidence_level == ConfidenceLevel.LOW
    assert report.recommendation == "Not ready — knowledge base needs improvement"
    assert report.critical_issue_count >= 1  # unknown_assumptions is critical
    assert report.warning_issue_count >= 5  # missing_datasets, missing_references, etc.


def test_gate_evaluate_critical_issues():
    """Test that critical issues make the project not ready."""
    gate = ResearchReadinessGate()
    dashboard_data = {
        "coverage_score": 0.95,
        "completeness_score": 0.95,
        "confidence_score": 0.95,
        "type_counts": {"paper": 5},
        "type_complete": {"paper": 5},
        "type_partial": {},
        "type_missing": {},
        "missing_datasets": [],
        "missing_references": [],
        "missing_implementations": [],
        "unknown_assumptions": ["assump-001"],
        "incomplete_parsing": [],
        "broken_references": [],
        "unknown_formats": [],
    }
    dashboard = type("Dashboard", (), {"to_dict": lambda self: dashboard_data})()
    report = gate.evaluate("proj-001", dashboard)

    assert report.readiness_level == ReadinessLevel.NOT_READY
    assert report.critical_issue_count >= 1


def test_gate_evaluate_confidence_levels():
    """Test that confidence levels are correctly determined."""
    gate = ResearchReadinessGate()

    # High confidence
    dashboard_data = {
        "coverage_score": 0.9,
        "completeness_score": 0.9,
        "confidence_score": 0.9,
        "type_counts": {"paper": 5},
        "type_complete": {"paper": 5},
        "type_partial": {},
        "type_missing": {},
        "missing_datasets": [],
        "missing_references": [],
        "missing_implementations": [],
        "unknown_assumptions": [],
        "incomplete_parsing": [],
        "broken_references": [],
        "unknown_formats": [],
    }
    dashboard = type("Dashboard", (), {"to_dict": lambda self: dashboard_data})()
    report = gate.evaluate("proj-001", dashboard)
    assert report.confidence_level == ConfidenceLevel.HIGH

    # Medium confidence
    dashboard_data["confidence_score"] = 0.6
    dashboard = type("Dashboard", (), {"to_dict": lambda self: dashboard_data})()
    report = gate.evaluate("proj-001", dashboard)
    assert report.confidence_level == ConfidenceLevel.MEDIUM

    # Low confidence
    dashboard_data["confidence_score"] = 0.3
    dashboard = type("Dashboard", (), {"to_dict": lambda self: dashboard_data})()
    report = gate.evaluate("proj-001", dashboard)
    assert report.confidence_level == ConfidenceLevel.LOW


# ── Convenience Function Tests ─────────────────────────────────────────────


def test_evaluate_readiness_convenience_function():
    """Test the evaluate_readiness convenience function."""
    dashboard_data = {
        "coverage_score": 0.9,
        "completeness_score": 0.9,
        "confidence_score": 0.9,
        "type_counts": {"paper": 5},
        "type_complete": {"paper": 5},
        "type_partial": {},
        "type_missing": {},
        "missing_datasets": [],
        "missing_references": [],
        "missing_implementations": [],
        "unknown_assumptions": [],
        "incomplete_parsing": [],
        "broken_references": [],
        "unknown_formats": [],
    }
    dashboard = type("Dashboard", (), {"to_dict": lambda self: dashboard_data})()
    report = evaluate_readiness("proj-001", dashboard)

    assert isinstance(report, ResearchReadinessReport)
    assert report.readiness_level == ReadinessLevel.READY
