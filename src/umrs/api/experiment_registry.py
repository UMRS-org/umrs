"""UMRS API routes for the experiment registry (aligned with K1 satellite).

Durable store auto-resolves via
``umrs.experiment_registry.build_default_store()``. Postgres config comes
through the shared pool for any `pg|auto` backend.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from umrs.experiment_registry import (
    ExperimentStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/experiment-registry", tags=["experiment-registry"])

_registry_store: Any = None


def set_registry(registry) -> None:
    global _registry_store
    _registry_store = registry


def _get_registry():
    if _registry_store is None:
        raise_typed(
            PlatformError.experiments(
                "unavailable",
                "Experiment registry not initialized"
            ),
        )
    return _registry_store


class CreateExperimentRequest(BaseModel):
    project_id: str
    hypothesis_id: str
    protocol: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    validation_method: str = "statistical_test"
    created_by: str = ""
    notes: str = ""


class UpdateStatusRequest(BaseModel):
    status: str


class AddResultRequest(BaseModel):
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    logs: str = ""
    status: str = "completed"
    created_by: str = ""


@router.post("", summary="Create experiment")
def create_experiment(req: CreateExperimentRequest):
    experiment_id = _get_registry().create_experiment(
        project_id=req.project_id,
        hypothesis_id=req.hypothesis_id,
        protocol=req.protocol,
        parameters=req.parameters,
        validation_method=req.validation_method,
        created_by=req.created_by,
        notes=req.notes,
    )
    return {"experiment_id": experiment_id}


@router.get("/{experiment_id}", summary="Single experiment with results")
def get_experiment(experiment_id: str):
    exp = _get_registry().get_experiment(experiment_id)
    if exp is None:
        raise_typed(
            PlatformError.experiments(
                "not_found",
                f"Experiment {experiment_id} not found"
            ),
        )
    return exp.to_dict()


@router.get("", summary="List experiments with optional filters")
def list_experiments(
    project_id: str | None = Query(None),
    hypothesis_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    status_obj = ExperimentStatus(status) if status else None
    entries = _get_registry().list_experiments(
        project_id=project_id,
        hypothesis_id=hypothesis_id,
        status=status_obj,
        limit=limit,
    )
    return {"experiments": [e.to_dict() for e in entries]}


@router.put("/{experiment_id}/status", summary="Transition experiment status")
def update_status(experiment_id: str, req: UpdateStatusRequest):
    try:
        ok = _get_registry().update_experiment_status(
            experiment_id, ExperimentStatus(req.status.lower())
        )
    except Exception as exc:  # noqa: BLE001 — maps the lifecycle validator
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not ok:
        raise_typed(
            PlatformError.experiments(
                "not_found",
                f"Experiment {experiment_id} not found"
            ),
        )
    return _get_registry().get_experiment(experiment_id).to_dict()


@router.post("/{experiment_id}/results", summary="Attach result to experiment")
def add_result(experiment_id: str, req: AddResultRequest):
    result_id = _get_registry().add_experiment_result(
        experiment_id,
        metrics=req.metrics,
        artifacts=req.artifacts,
        logs=req.logs,
        status=req.status,
        created_by=req.created_by,
    )
    return {"result_id": result_id}


@router.get("/history/{project_id}", summary="Full experiment history for a project")
def experiment_history(project_id: str):
    entries = _get_registry().get_experiment_history(project_id)
    return {"experiments": [e.to_dict() for e in entries]}
