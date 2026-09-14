"""Tests for the experiment_registry subsystem (Phase 1, Step 2.4)."""

from __future__ import annotations

import pytest
from umrs.experiment_registry import (
    Experiment,
    ExperimentNotFoundError,
    ExperimentResult,
    ExperimentStatus,
    InMemoryExperimentRegistry,
    InvalidStatusTransitionError,
)

# ── Enum Tests ────────────────────────────────────────────────────────────


def test_experiment_status_enum():
    """Test ExperimentStatus enum values."""
    assert ExperimentStatus.PENDING == "pending"
    assert ExperimentStatus.DESIGNED == "designed"
    assert ExperimentStatus.SUBMITTED == "submitted"
    assert ExperimentStatus.RUNNING == "running"
    assert ExperimentStatus.COMPLETED == "completed"
    assert ExperimentStatus.FAILED == "failed"
    assert ExperimentStatus.CANCELLED == "cancelled"


# ── Experiment Tests ─────────────────────────────────────────────────────


def test_experiment_defaults():
    """Test Experiment defaults."""
    exp = Experiment()

    assert exp.experiment_id != ""  # Auto-generated
    assert exp.project_id == ""
    assert exp.hypothesis_id == ""
    assert exp.protocol == {}
    assert exp.parameters == {}
    assert exp.validation_method == "statistical_test"
    assert exp.status == ExperimentStatus.PENDING
    assert exp.results == []
    assert exp.created_at != ""
    assert exp.updated_at != ""
    assert exp.created_by == ""
    assert exp.submitted_at is None
    assert exp.completed_at is None
    assert exp.notes == ""


def test_experiment_to_dict():
    """Test Experiment to_dict."""
    exp = Experiment(
        experiment_id="exp-001",
        project_id="proj-001",
        hypothesis_id="hyp-001",
        protocol={"steps": ["step1", "step2"]},
        parameters={"lr": 0.001, "batch_size": 32},
        validation_method="cross_validation",
        status=ExperimentStatus.RUNNING,
        notes="Test experiment",
    )

    data = exp.to_dict()

    assert data["experiment_id"] == "exp-001"
    assert data["project_id"] == "proj-001"
    assert data["hypothesis_id"] == "hyp-001"
    assert data["protocol"] == {"steps": ["step1", "step2"]}
    assert data["parameters"] == {"lr": 0.001, "batch_size": 32}
    assert data["validation_method"] == "cross_validation"
    assert data["status"] == "running"
    assert data["notes"] == "Test experiment"


def test_experiment_from_dict():
    """Test Experiment from_dict."""
    data = {
        "experiment_id": "exp-001",
        "project_id": "proj-001",
        "hypothesis_id": "hyp-001",
        "protocol": {"steps": ["step1"]},
        "parameters": {"lr": 0.001},
        "validation_method": "hold_out",
        "status": "completed",
        "results": [],
        "created_at": "2026-06-30T10:00:00Z",
        "updated_at": "2026-06-30T11:00:00Z",
        "created_by": "user-001",
        "submitted_at": "2026-06-30T10:30:00Z",
        "completed_at": "2026-06-30T11:00:00Z",
        "notes": "Test",
    }

    exp = Experiment.from_dict(data)

    assert exp.experiment_id == "exp-001"
    assert exp.project_id == "proj-001"
    assert exp.hypothesis_id == "hyp-001"
    assert exp.protocol == {"steps": ["step1"]}
    assert exp.parameters == {"lr": 0.001}
    assert exp.validation_method == "hold_out"
    assert exp.status == ExperimentStatus.COMPLETED
    assert exp.created_by == "user-001"
    assert exp.submitted_at == "2026-06-30T10:30:00Z"
    assert exp.completed_at == "2026-06-30T11:00:00Z"
    assert exp.notes == "Test"


def test_experiment_round_trip():
    """Test that to_dict -> from_dict -> to_dict produces the same result."""
    exp = Experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
        protocol={"steps": ["step1"]},
        parameters={"lr": 0.001},
    )

    data1 = exp.to_dict()
    exp2 = Experiment.from_dict(data1)
    data2 = exp2.to_dict()

    assert data1 == data2


# ── ExperimentResult Tests ────────────────────────────────────────────────


def test_experiment_result_defaults():
    """Test ExperimentResult defaults."""
    result = ExperimentResult()

    assert result.result_id != ""  # Auto-generated
    assert result.experiment_id == ""
    assert result.metrics == {}
    assert result.artifacts == []
    assert result.logs == ""
    assert result.status == "completed"
    assert result.created_at != ""
    assert result.created_by == ""


def test_experiment_result_to_dict():
    """Test ExperimentResult to_dict."""
    result = ExperimentResult(
        result_id="res-001",
        experiment_id="exp-001",
        metrics={"accuracy": 0.95, "loss": 0.05},
        artifacts=["s3://bucket/result.json"],
        logs="Training started...\nTraining complete.",
        status="completed",
    )

    data = result.to_dict()

    assert data["result_id"] == "res-001"
    assert data["experiment_id"] == "exp-001"
    assert data["metrics"] == {"accuracy": 0.95, "loss": 0.05}
    assert data["artifacts"] == ["s3://bucket/result.json"]
    assert data["logs"] == "Training started...\nTraining complete."
    assert data["status"] == "completed"


def test_experiment_result_from_dict():
    """Test ExperimentResult from_dict."""
    data = {
        "result_id": "res-001",
        "experiment_id": "exp-001",
        "metrics": {"accuracy": 0.95},
        "artifacts": ["s3://bucket/result.json"],
        "logs": "Training complete.",
        "status": "completed",
        "created_at": "2026-06-30T11:00:00Z",
        "created_by": "user-001",
    }

    result = ExperimentResult.from_dict(data)

    assert result.result_id == "res-001"
    assert result.experiment_id == "exp-001"
    assert result.metrics == {"accuracy": 0.95}
    assert result.artifacts == ["s3://bucket/result.json"]
    assert result.logs == "Training complete."
    assert result.status == "completed"
    assert result.created_by == "user-001"


# ── InMemoryExperimentRegistry Tests ──────────────────────────────────────


def test_registry_create_and_get():
    """Test creating and getting experiments."""
    registry = InMemoryExperimentRegistry()
    exp_id = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
        protocol={"steps": ["step1", "step2"]},
        parameters={"lr": 0.001, "batch_size": 32},
        validation_method="cross_validation",
        created_by="user-001",
        notes="Test experiment",
    )

    exp = registry.get_experiment(exp_id)
    assert exp is not None
    assert exp.project_id == "proj-001"
    assert exp.hypothesis_id == "hyp-001"
    assert exp.protocol == {"steps": ["step1", "step2"]}
    assert exp.parameters == {"lr": 0.001, "batch_size": 32}
    assert exp.validation_method == "cross_validation"
    assert exp.status == ExperimentStatus.PENDING
    assert exp.created_by == "user-001"
    assert exp.notes == "Test experiment"


def test_registry_get_nonexistent():
    """Test getting a non-existent experiment returns None."""
    registry = InMemoryExperimentRegistry()
    assert registry.get_experiment("nonexistent") is None


def test_registry_update_status_valid():
    """Test updating experiment status with valid transitions."""
    registry = InMemoryExperimentRegistry()
    exp_id = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )

    # PENDING -> DESIGNED
    assert registry.update_experiment_status(exp_id, ExperimentStatus.DESIGNED) is True
    assert registry.get_experiment(exp_id).status == ExperimentStatus.DESIGNED

    # DESIGNED -> SUBMITTED
    assert registry.update_experiment_status(exp_id, ExperimentStatus.SUBMITTED) is True
    assert registry.get_experiment(exp_id).status == ExperimentStatus.SUBMITTED
    assert registry.get_experiment(exp_id).submitted_at is not None

    # SUBMITTED -> RUNNING
    assert registry.update_experiment_status(exp_id, ExperimentStatus.RUNNING) is True
    assert registry.get_experiment(exp_id).status == ExperimentStatus.RUNNING

    # RUNNING -> COMPLETED
    assert registry.update_experiment_status(exp_id, ExperimentStatus.COMPLETED) is True
    assert registry.get_experiment(exp_id).status == ExperimentStatus.COMPLETED
    assert registry.get_experiment(exp_id).completed_at is not None


def test_registry_update_status_invalid():
    """Test that invalid status transitions raise an error."""
    registry = InMemoryExperimentRegistry()
    exp_id = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )

    # PENDING -> RUNNING (invalid, must go through DESIGNED and SUBMITTED)
    with pytest.raises(InvalidStatusTransitionError):
        registry.update_experiment_status(exp_id, ExperimentStatus.RUNNING)


def test_registry_update_status_nonexistent():
    """Test that updating a non-existent experiment returns False."""
    registry = InMemoryExperimentRegistry()
    assert registry.update_experiment_status("nonexistent", ExperimentStatus.DESIGNED) is False


def test_registry_add_result():
    """Test adding a result to an experiment."""
    registry = InMemoryExperimentRegistry()
    exp_id = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )

    result_id = registry.add_experiment_result(
        exp_id,
        metrics={"accuracy": 0.95, "loss": 0.05},
        artifacts=["s3://bucket/result.json"],
        logs="Training complete.",
        status="completed",
        created_by="user-001",
    )

    assert result_id != ""

    exp = registry.get_experiment(exp_id)
    assert len(exp.results) == 1
    assert exp.results[0].metrics == {"accuracy": 0.95, "loss": 0.05}
    assert exp.results[0].artifacts == ["s3://bucket/result.json"]
    assert exp.results[0].logs == "Training complete."
    assert exp.results[0].status == "completed"
    assert exp.results[0].created_by == "user-001"


def test_registry_add_result_nonexistent():
    """Test that adding a result to a non-existent experiment raises an error."""
    registry = InMemoryExperimentRegistry()
    with pytest.raises(ExperimentNotFoundError):
        registry.add_experiment_result("nonexistent", metrics={})


def test_registry_add_multiple_results():
    """Test adding multiple results to an experiment."""
    registry = InMemoryExperimentRegistry()
    exp_id = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )

    registry.add_experiment_result(exp_id, metrics={"accuracy": 0.90}, status="completed")
    registry.add_experiment_result(exp_id, metrics={"accuracy": 0.95}, status="completed")

    exp = registry.get_experiment(exp_id)
    assert len(exp.results) == 2
    assert exp.results[0].metrics == {"accuracy": 0.90}
    assert exp.results[1].metrics == {"accuracy": 0.95}


def test_registry_list_experiments():
    """Test listing experiments with filters."""
    registry = InMemoryExperimentRegistry()
    registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )
    registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-002",
    )
    registry.create_experiment(
        project_id="proj-002",
        hypothesis_id="hyp-003",
    )

    # All experiments
    all_exps = registry.list_experiments()
    assert len(all_exps) == 3

    # Filter by project
    proj1_exps = registry.list_experiments(project_id="proj-001")
    assert len(proj1_exps) == 2

    # Filter by hypothesis
    hyp1_exps = registry.list_experiments(hypothesis_id="hyp-001")
    assert len(hyp1_exps) == 1

    # Filter by status (all should be PENDING)
    pending_exps = registry.list_experiments(status=ExperimentStatus.PENDING)
    assert len(pending_exps) == 3


def test_registry_list_experiments_limit():
    """Test that limit is respected."""
    registry = InMemoryExperimentRegistry()
    for i in range(10):
        registry.create_experiment(
            project_id=f"proj-{i:03d}",
            hypothesis_id=f"hyp-{i:03d}",
        )

    limited = registry.list_experiments(limit=5)
    assert len(limited) == 5


def test_registry_get_experiment_history():
    """Test getting experiment history for a project."""
    registry = InMemoryExperimentRegistry()
    # Create experiments with different created_at timestamps
    exp1 = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )
    exp2 = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-002",
    )
    exp3 = registry.create_experiment(
        project_id="proj-002",
        hypothesis_id="hyp-003",
    )

    history = registry.get_experiment_history("proj-001")

    # Should be sorted by created_at ascending (oldest first)
    assert len(history) == 2
    assert history[0].experiment_id == exp1
    assert history[1].experiment_id == exp2


def test_registry_cancel_experiment():
    """Test cancelling an experiment."""
    registry = InMemoryExperimentRegistry()
    exp_id = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )

    # PENDING -> CANCELLED
    assert registry.update_experiment_status(exp_id, ExperimentStatus.CANCELLED) is True
    assert registry.get_experiment(exp_id).status == ExperimentStatus.CANCELLED

    # CANCELLED is terminal — cannot transition to anything else
    with pytest.raises(InvalidStatusTransitionError):
        registry.update_experiment_status(exp_id, ExperimentStatus.RUNNING)


def test_registry_fail_experiment():
    """Test failing an experiment."""
    registry = InMemoryExperimentRegistry()
    exp_id = registry.create_experiment(
        project_id="proj-001",
        hypothesis_id="hyp-001",
    )

    # PENDING -> DESIGNED -> SUBMITTED -> RUNNING -> FAILED
    registry.update_experiment_status(exp_id, ExperimentStatus.DESIGNED)
    registry.update_experiment_status(exp_id, ExperimentStatus.SUBMITTED)
    registry.update_experiment_status(exp_id, ExperimentStatus.RUNNING)
    registry.update_experiment_status(exp_id, ExperimentStatus.FAILED)

    assert registry.get_experiment(exp_id).status == ExperimentStatus.FAILED
    assert registry.get_experiment(exp_id).completed_at is not None

    # FAILED is terminal
    with pytest.raises(InvalidStatusTransitionError):
        registry.update_experiment_status(exp_id, ExperimentStatus.RUNNING)
