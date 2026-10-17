"""UMRS API routes for the experiment registry (aligned with K1 satellite).

Durable store auto-resolves via
``umrs.experiment_registry.build_default_store()``. Postgres config comes
through the shared pool for any `pg|auto` backend.
"""

from __future__ import annotations

import logging
from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from umrs.api._scope_identity import request_caller_scope
from umrs.api.typed_errors import PlatformError, raise_typed
from umrs.experiment_registry import (
    ExperimentStatus,
)
from umrs.knowledge.access_scope import (
    CallerScope,
    project_readable,
    project_writable,
    writable_project_for,
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
                "unavailable", "Experiment registry not initialized"
            ),
        )
    return _registry_store


# ---------------------------------------------------------------------------
# Caller scope (H-1, 2026-10-17): every route took no identity, so any caller
# created experiments in any project, listed any project's history and changed
# the status or results of any experiment id. Writes go only into projects the
# caller may write, reads only cover projects the caller may read, and another
# project's experiment reads as not found. Dev (mvp) anonymous callers keep
# the old open behaviour (``CallerScope.anonymous``).
# ---------------------------------------------------------------------------


def _forbidden(project_id: str) -> NoReturn:
    raise_typed(
        PlatformError.verified(
            "forbidden", f"project '{project_id}' is outside the caller's scope"
        ),
    )


def _not_found(experiment_id: str) -> NoReturn:
    raise_typed(
        PlatformError.experiments("not_found", f"Experiment {experiment_id} not found"),
    )


def _readable_filter(scope: CallerScope, requested: str | None) -> str | None:
    """Project filter for a read: the caller's own project when none is named;
    a project the caller may not read is refused (admins: any, None = all)."""
    if scope.is_admin or scope.anonymous:
        return requested
    if not requested:
        return scope.project_id
    if not project_readable(requested, scope):
        _forbidden(requested)
    return requested


def _experiment_for(request: Request, experiment_id: str, *, write: bool) -> Any:
    """The experiment, when the caller may read (or write) its project."""
    scope = request_caller_scope(request)
    exp = _get_registry().get_experiment(experiment_id)
    allowed = project_writable if write else project_readable
    if exp is None or not allowed(exp.project_id, scope):
        _not_found(experiment_id)
    return exp


def _status_filter(status: str | None) -> ExperimentStatus | None:
    """The list filter's status; an unknown value is a typed 400
    (``experiments.bad_request``), not a 500."""
    if not status:
        return None
    try:
        return ExperimentStatus(status)
    except ValueError as exc:
        raise_typed(
            PlatformError.experiments(
                "bad_request", f"unknown experiment status '{status}'"
            ),
            cause=exc,
        )


def _author(scope: CallerScope, declared: str) -> str:
    """The verified user is the author; a body ``created_by`` only counts for
    anonymous dev callers."""
    return scope.user_id or declared


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
def create_experiment(req: CreateExperimentRequest, request: Request):
    scope = request_caller_scope(request)
    project_id = writable_project_for(req.project_id, scope)
    if project_id is None:
        _forbidden(req.project_id)
    experiment_id = _get_registry().create_experiment(
        project_id=project_id,
        hypothesis_id=req.hypothesis_id,
        protocol=req.protocol,
        parameters=req.parameters,
        validation_method=req.validation_method,
        created_by=_author(scope, req.created_by),
        notes=req.notes,
    )
    return {"experiment_id": experiment_id}


@router.get("/{experiment_id}", summary="Single experiment with results")
def get_experiment(experiment_id: str, request: Request):
    return _experiment_for(request, experiment_id, write=False).to_dict()


@router.get("", summary="List experiments with optional filters")
def list_experiments(
    request: Request,
    project_id: str | None = Query(None),
    hypothesis_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    # The caller is resolved before any query value is parsed, so an
    # anonymous strict caller gets 401, never a parse error.
    project_id = _readable_filter(request_caller_scope(request), project_id)
    status_obj = _status_filter(status)
    entries = _get_registry().list_experiments(
        project_id=project_id,
        hypothesis_id=hypothesis_id,
        status=status_obj,
        limit=limit,
    )
    return {"experiments": [e.to_dict() for e in entries]}


@router.put("/{experiment_id}/status", summary="Transition experiment status")
def update_status(experiment_id: str, req: UpdateStatusRequest, request: Request):
    _experiment_for(request, experiment_id, write=True)
    try:
        ok = _get_registry().update_experiment_status(
            experiment_id, ExperimentStatus(req.status.lower())
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not ok:
        raise_typed(
            PlatformError.experiments(
                "not_found", f"Experiment {experiment_id} not found"
            ),
        )
    return _get_registry().get_experiment(experiment_id).to_dict()


@router.post("/{experiment_id}/results", summary="Attach result to experiment")
def add_result(experiment_id: str, req: AddResultRequest, request: Request):
    _experiment_for(request, experiment_id, write=True)
    result_id = _get_registry().add_experiment_result(
        experiment_id,
        metrics=req.metrics,
        artifacts=req.artifacts,
        logs=req.logs,
        status=req.status,
        created_by=_author(request_caller_scope(request), req.created_by),
    )
    return {"result_id": result_id}


@router.get("/history/{project_id}", summary="Full experiment history for a project")
def experiment_history(project_id: str, request: Request):
    scope = request_caller_scope(request)
    if not project_readable(project_id, scope):
        _forbidden(project_id)
    entries = _get_registry().get_experiment_history(project_id)
    return {"experiments": [e.to_dict() for e in entries]}
