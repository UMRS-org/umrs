"""``GET /umrs/workflows/executions`` behind the route (ADR-105 D-6, S6a).

Visibility is decided here, record by record, with the same ACL as
``workflow_status`` (``can_access_execution``): a non-admin is pinned to
their own records and to their verified project; records with
``unverified`` provenance stay invisible (fail-closed). Only visible
records reach the pure shaping in ``workflow_execution_list``; a cursor
only carries a position and is bound to the caller and filters, so it
never widens what a caller sees.

Source of truth: the in-process execution registry (executions submitted
since this UMRS started), as the ZMQ ``WORKFLOW_EXECUTIONS`` list. Durable
job rows are listed by ``job_list``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from ..jobs.read_summary import fold_summary
from ..server.workflow_execution_registry import ExecutionRecord, get_execution_registry
from .jobs import CallerIdentity
from .read_cursor import decode_cursor, encode_cursor, scope_key
from .read_fields import FieldsError, parse_fields, project
from .read_shaping_models import ReadSummary, WorkflowExecutionListResponse
from .typed_errors import PlatformError, raise_typed
from .workflow_auth import caller_from_identity, can_access_execution
from .workflow_execution_list import (
    EXECUTION_FIELDS,
    ExecutionPosition,
    execution_row,
    page_executions,
    position_from_fields,
    position_to_fields,
    summary_groups,
)

_KEY_FIELDS = ("execution_id",)


@dataclass(frozen=True)
class ExecutionListQuery:
    """The route's inputs (``owner``: an admin's owner filter)."""

    status: str | None = None
    owner: str | None = None
    limit: int = 50
    cursor: str | None = None
    fields: str | None = None
    summary: bool = False


def _bad_request(message: str, exc: Exception) -> NoReturn:
    raise_typed(PlatformError.workflows("bad_request", message), cause=exc)


def _status_filter(status: str | None) -> str | None:
    return (status or "").strip().lower() or None


def visible_executions(
    identity: CallerIdentity, owner: str | None, status: str | None
) -> list[ExecutionRecord]:
    """The records ``identity`` may read, filtered by status.

    A non-admin's owner is always themselves (``owner`` ignored); an admin
    may name one owner or leave it unset for every record.
    """
    user_id, project_id, is_admin, provenance = caller_from_identity(identity)
    owner_filter = (owner or None) if is_admin else user_id
    wanted = _status_filter(status)
    return [
        record
        for record in get_execution_registry().list(user_id=owner_filter)
        if can_access_execution(
            user_id,
            project_id,
            is_admin=is_admin,
            caller_project_provenance=provenance,
            record=record,
        )
        and (wanted is None or record.status.lower() == wanted)
    ]


def _scope(identity: CallerIdentity, query: ExecutionListQuery) -> str:
    return scope_key(
        {
            "op": "workflow_executions",
            "caller": identity.user_id,
            "caller_project": identity.project_id,
            "admin": "1" if identity.is_admin else "0",
            "owner": (query.owner or None) if identity.is_admin else identity.user_id,
            "status": _status_filter(query.status),
        }
    )


def _after(cursor: str, scope: str) -> ExecutionPosition:
    try:
        return position_from_fields(decode_cursor(cursor, scope))
    except ValueError as exc:  # CursorError is a ValueError
        _bad_request(str(exc), exc)


def _projection(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    try:
        return parse_fields(raw, EXECUTION_FIELDS, _KEY_FIELDS)
    except FieldsError as exc:
        _bad_request(str(exc), exc)


def list_executions(
    identity: CallerIdentity, query: ExecutionListQuery
) -> WorkflowExecutionListResponse:
    """The whole read: visibility, then summary or one shaped page."""
    records = visible_executions(identity, query.owner, query.status)
    if query.summary:
        summary = ReadSummary(**fold_summary(summary_groups(records)))
        return WorkflowExecutionListResponse(status="ok", count=0, summary=summary)
    after = _after(query.cursor, _scope(identity, query)) if query.cursor else None
    fields = _projection(query.fields)
    page, next_position = page_executions(records, after=after, limit=query.limit)
    rows = [execution_row(r) for r in page]
    if fields is not None:
        rows = [project(r, fields) for r in rows]
    next_cursor = (
        encode_cursor(position_to_fields(next_position), _scope(identity, query))
        if next_position is not None
        else None
    )
    return WorkflowExecutionListResponse(
        status="ok", count=len(rows), executions=rows, next_cursor=next_cursor
    )
