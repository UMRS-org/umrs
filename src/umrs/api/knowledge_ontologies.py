"""UMRS API routes for the knowledge ontology registry (ADR-116 D-5).

Routes (mounted next to the knowledge graph routes):

  POST /umrs/knowledge/ontologies                         — register (LinkML YAML or ontology JSON)
  GET  /umrs/knowledge/ontologies                         — list (built-ins included)
  GET  /umrs/knowledge/ontologies/{ontology_id}/{version} — one ontology with its source text
  POST /umrs/knowledge/ontologies/{ontology_id}/{version}/retire — tombstone (owner or admin)

Registration and retirement are human decisions: a request marked
``x-r4-actor: agent`` gets 403 ``ontology.forbidden`` (ADR-084 D-3). A
project-scoped ontology needs write access to the caller's project; a global
one needs an admin. Errors keep the knowledge routes' envelope: the HTTP
status, ``{"detail": <message>}`` and the code on the
``x-platform-error-code`` header (``ontology.invalid`` 400,
``ontology.forbidden`` 403, ``ontology.not_found`` 404, ``ontology.conflict``
409).

Each project, and the global scope, has its own ``id@version`` namespace
(0139): ``ontology.conflict`` means the caller's own scope already has that
version live. Reads and bindings use the caller's project record first,
then the global one; another project's record is never shown, bound or
resolved. Retiring a version blocks new bindings only: graphs already bound
to it keep extracting with it until they are bound to another version.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Literal, NoReturn

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ValidationError

from umrs.knowledge.access_scope import CallerScope, project_readable, project_writable
from umrs.knowledge.linkml_import import (
    MAX_SOURCE_BYTES,
    OntologySourceError,
    check_limits,
    linkml_to_ontology,
    validate_coordinate,
)
from umrs.knowledge.ontology import (
    BUILTIN_ONTOLOGY_IDS,
    Ontology,
    builtin_ontology,
    normalize_term,
    ontology_from_dict,
)
from umrs.knowledge.ontology_registry import (
    OntologyCoordinateConflict,
    OntologyRecord,
    OntologyRegistryStore,
    source_sha256,
)

from ._knowledge_mutation_guard import AGENT_ACTOR, request_actor
from ._scope_identity import get_caller_scope
from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/knowledge/ontologies", tags=["knowledge-ontologies"])

# The JSON body wraps the source: escaping newlines, quotes and non-ASCII
# characters can double its size, plus the framing. The source limit
# (MAX_SOURCE_BYTES, checked on the source text) is the real one.
MAX_BODY_BYTES = 2 * MAX_SOURCE_BYTES + 4096
_ERROR_STATUS = {"invalid": 400, "forbidden": 403, "not_found": 404, "conflict": 409}
_UNIQUE_VIOLATION = "23505"


def ontology_error(
    name: str, message: str, *, http_status: int | None = None
) -> NoReturn:
    """Raise ``ontology.<name>`` in the knowledge routes' error envelope."""
    raise_typed(
        PlatformError(
            code=f"ontology.{name}",
            message=message,
            http_status=http_status or _ERROR_STATUS[name],
        )
    )


def _refuse_agent(request: Request) -> None:
    """Ontology registration and retirement are human decisions."""
    if request_actor(request) == AGENT_ACTOR:
        ontology_error(
            "forbidden",
            "Ontology registration and retirement are human decisions; "
            "agents may not register or retire ontologies.",
        )


@contextmanager
def open_ontology_registry(request: Request) -> Iterator[OntologyRegistryStore]:
    """A registry store on a connection from the request's sync pool."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None or not hasattr(pool, "getconn"):
        raise_typed(
            PlatformError.knowledgegraphs("unavailable", "Database unavailable")
        )
    conn = pool.getconn()
    try:
        yield OntologyRegistryStore(conn)
    finally:
        pool.putconn(conn)


class RegisterOntologyRequest(BaseModel):
    source: str
    format: Literal["linkml", "json"] = "linkml"
    scope: Literal["project", "global"] = "project"


async def _bounded_body(request: Request) -> RegisterOntologyRequest:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        ontology_error("invalid", f"request body is larger than {MAX_BODY_BYTES} bytes")
    try:
        return RegisterOntologyRequest.model_validate(json.loads(raw or b"null"))
    except (ValueError, ValidationError) as exc:
        ontology_error("invalid", f"request body is not a valid registration: {exc}")


def _caller_scope(identity: CallerIdentity) -> CallerScope:
    return CallerScope(
        user_id=identity.user_id or "",
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )


def _registration_project(scope: str, identity: CallerIdentity) -> str | None:
    """The project a registration is stored under (None = global)."""
    if scope == "global":
        if not identity.is_admin:
            ontology_error("forbidden", "Registering a global ontology needs an admin.")
        return None
    project_id = identity.project_id or ""
    if not project_writable(project_id, _caller_scope(identity)):
        ontology_error(
            "forbidden",
            f"Registering an ontology in project {project_id!r} needs write access.",
        )
    return project_id


def _json_ontology(source: str) -> Ontology:
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise OntologySourceError(
            f"JSON source is larger than {MAX_SOURCE_BYTES} bytes"
        )
    raw = json.loads(source)
    if not isinstance(raw, dict):
        raise OntologySourceError("JSON ontology source must be an object")
    ontology = replace(ontology_from_dict(raw), source_format="json")
    validate_coordinate(ontology.ontology_id, ontology.version)
    terms = ontology.entity_types | ontology.relation_types
    odd = sorted(t for t in terms if normalize_term(t) != t)
    if odd:
        raise OntologySourceError(
            f"class and relation names must be lowercase snake_case: {odd}"
        )
    return check_limits(ontology)


def _parse_source(source: str, source_format: str) -> Ontology:
    """The registered ontology, or ``ontology.invalid``. Blocking (YAML and
    JSON parsing of up to MAX_SOURCE_BYTES): the route runs it in a thread."""
    try:
        if source_format == "json":
            ontology = _json_ontology(source)
        else:
            ontology = linkml_to_ontology(source)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        ontology_error("invalid", str(exc))
    except RecursionError:
        ontology_error("invalid", "ontology source is nested too deeply")
    if ontology.ontology_id in BUILTIN_ONTOLOGY_IDS:
        ontology_error(
            "conflict", f"{ontology.ontology_id!r} is a built-in ontology id"
        )
    return ontology


def _register(
    store: OntologyRegistryStore,
    ontology: Ontology,
    body: RegisterOntologyRequest,
    owner_id: str,
    project_id: str | None,
) -> OntologyRecord:
    try:
        return store.register(
            ontology,
            owner_id=owner_id,
            project_id=project_id,
            source_format=body.format,
            source_text=body.source,
        )
    except OntologyCoordinateConflict as exc:
        ontology_error("conflict", str(exc))
    except Exception as exc:
        if getattr(exc, "pgcode", None) == _UNIQUE_VIOLATION:
            ontology_error(
                "conflict", f"{ontology.ontology_id}@{ontology.version} is already live"
            )
        raise


@router.post("", status_code=201)
async def register_ontology(
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Register an ontology from LinkML YAML or the ontology JSON (humans only).
    # auth: verified
    """
    _refuse_agent(request)
    body = await _bounded_body(request)
    project_id = _registration_project(body.scope, identity)
    ontology = await asyncio.to_thread(_parse_source, body.source, body.format)
    with open_ontology_registry(request) as store:
        record = _register(store, ontology, body, identity.user_id or "", project_id)
    return {"ontology": ontology.to_dict(), "record": record.summary()}


# ── reads ────────────────────────────────────────────────────────────────


def _builtin_source(ontology_id: str) -> str | None:
    if ontology_id != "r4-science":
        return None
    from umrs.knowledge.r4_science_schema import R4_SCIENCE_LINKML

    return R4_SCIENCE_LINKML


def _counts(ontology: Ontology) -> dict[str, int]:
    return {
        "n_classes": len(ontology.entity_types),
        "n_relations": len(ontology.relation_types),
    }


def _builtin_row(ontology_id: str) -> dict[str, Any]:
    ontology = builtin_ontology(ontology_id)
    assert ontology is not None
    return {
        "record_id": None,
        "ontology_id": ontology.ontology_id,
        "version": ontology.version,
        "description": ontology.description,
        "project_id": None,
        "owner_id": "system",
        "status": "active",
        "created_at": None,
        "source_sha256": source_sha256(_builtin_source(ontology_id)),
        "source_format": "builtin",
        **_counts(ontology),
        "builtin": True,
    }


def _record_row(record: OntologyRecord) -> dict[str, Any]:
    return {**record.summary(), **_counts(record.ontology), "builtin": False}


def _record_visible(record: OntologyRecord, scope: CallerScope) -> bool:
    return project_readable(record.project_id or "", scope)


@router.get("")
async def list_ontologies(
    request: Request,
    include_retired: bool = Query(False),
    scope: CallerScope = Depends(get_caller_scope),
) -> dict[str, Any]:
    """Built-in ontologies, then the registered ones the caller may read."""
    with open_ontology_registry(request) as store:
        records = store.list(include_retired=include_retired)
    rows = [_builtin_row(ontology_id) for ontology_id in BUILTIN_ONTOLOGY_IDS]
    rows.extend(_record_row(r) for r in records if _record_visible(r, scope))
    return {"ontologies": rows}


def _builtin_or_none(ontology_id: str, version: str) -> Ontology | None:
    try:
        return builtin_ontology(ontology_id, version)
    except ValueError as exc:
        ontology_error("not_found", str(exc))


@router.get("/{ontology_id}/{version}")
async def show_ontology(
    ontology_id: str,
    version: str,
    request: Request,
    scope: CallerScope = Depends(get_caller_scope),
) -> dict[str, Any]:
    """One live ontology (the caller's project record, else the global one);
    registered ones carry their source text."""
    builtin = _builtin_or_none(ontology_id, version)
    if builtin is not None:
        return {
            "ontology": builtin.to_dict(),
            "record": None,
            "source_format": "builtin",
            "source_text": _builtin_source(ontology_id),
        }
    with open_ontology_registry(request) as store:
        record = store.live_coordinate(ontology_id, version, scope.project_id or None)
    if record is None or not _record_visible(record, scope):
        ontology_error(
            "not_found", f"ontology {ontology_id}@{version} is not registered"
        )
    return {
        "ontology": record.ontology.to_dict(),
        "record": record.summary(),
        "source_format": record.source_format,
        "source_text": record.source_text,
    }


@router.post("/{ontology_id}/{version}/retire")
async def retire_ontology(
    ontology_id: str,
    version: str,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Tombstone a registered ontology (its owner or an admin; humans only).

    The record is the caller's project one, else the global one. Retiring
    blocks new bindings only: graphs already bound to this version keep
    extracting with it (``OntologyRegistryStore.resolve_coordinate``).
    # auth: verified
    """
    _refuse_agent(request)
    if ontology_id in BUILTIN_ONTOLOGY_IDS:
        ontology_error(
            "forbidden", f"{ontology_id!r} is built in and cannot be retired"
        )
    project_id = identity.project_id or None
    with open_ontology_registry(request) as store:
        record = store.live_coordinate(ontology_id, version, project_id)
        if record is None:
            ontology_error(
                "not_found", f"ontology {ontology_id}@{version} is not registered"
            )
        if not (identity.is_admin or record.owner_id == identity.user_id):
            ontology_error(
                "forbidden", "Only the ontology's owner or an admin may retire it."
            )
        store.retire(ontology_id, version, project_id)
    return {"retired": True}


# ── graph binding (used by api/knowledge_graphs.py) ──────────────────────


def pinned_ontology_version(
    request: Request, ontology_id: str, version: str | None, project_id: str | None
) -> str:
    """Validate a graph's ontology coordinate; the version to store.

    A built-in id pins its version; a registered id without a version pins
    the latest live version visible to the graph's project (its own records
    and global ones). Unknown, retired or other-project ontologies are 400
    ``ontology.not_found``: retiring blocks new bindings, while graphs bound
    before keep resolving the retired version.
    """
    try:
        builtin = builtin_ontology(ontology_id, version)
    except ValueError as exc:
        ontology_error("not_found", str(exc), http_status=400)
    if builtin is not None:
        return builtin.version
    scope = project_id or None
    with open_ontology_registry(request) as store:
        pinned = version or store.latest_live_version(ontology_id, scope)
        record = store.live_coordinate(ontology_id, pinned, scope) if pinned else None
    if record is None or record.project_id not in (None, scope):
        ontology_error(
            "not_found",
            f"ontology {ontology_id}@{version or 'latest'} is not registered for this graph",
            http_status=400,
        )
    return str(pinned)


__all__ = [
    "MAX_BODY_BYTES",
    "ontology_error",
    "open_ontology_registry",
    "pinned_ontology_version",
    "router",
]
