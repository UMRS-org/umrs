"""
Workflow Registry Handler — bridges ZMQ protocol to workflow registry operations.

Provides handle_workflow_command() for the server to call from its ZMQ loop.
Uses a JSON file catalog for lightweight workflow registration (mirrors the
model_registry_handler.py pattern).

Supported commands (via TaskCmd):
  LIST_WORKFLOWS  — list all registered workflows (with optional scope/user filter)

Also provides register_workflow() for internal use (auto-registration on upload).

Catalog file: <resolved UMRS config dir>/workflow_catalog.json
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from umrs.shared.access_scopes import ResourceCaller, access_admits
from umrs.shared.config import get_resolved_umrs_subdir
from umrs.shared.workflow_execution_access import VERIFIED_PROJECT_SPEC_KEY

if TYPE_CHECKING:
    from umrs.server.job_tracking_handler import DurableJobCommandBackend

logger = logging.getLogger(__name__)

_CATALOG_LOCK = threading.Lock()

# ADR-080: circle membership resolver hook for access-admit decisions.
# The default resolves through the UMRS default pool (0119 authority,
# ADR-086); tests replace it via ``set_workflow_circles_resolver``.
_circles_resolver_hook: Callable[[str], list[str]] | None = None
_circles_resolver_ready = False
_circles_resolver_lock = threading.Lock()


def set_workflow_circles_resolver(resolver: Callable[[str], list[str]] | None) -> None:
    """Tests / boot code replace the circle resolver (None = default pool)."""
    global _circles_resolver_hook, _circles_resolver_ready
    with _circles_resolver_lock:
        _circles_resolver_hook = resolver
        _circles_resolver_ready = True


def _default_circles_resolver() -> Callable[[str], list[str]]:
    def resolve(user_id: str) -> list[str]:
        from umrs.shared.pg import get_default_sync_pool, pg_configured_from_env

        if not pg_configured_from_env():
            return []
        from umrs.shared.circle_membership import circle_refs_for_user

        pool = get_default_sync_pool()
        conn = pool.getconn()
        try:
            return circle_refs_for_user(conn, user_id)
        finally:
            pool.putconn(conn)

    return resolve


def _resolve_circles(user_id: str) -> list[str]:
    """Active circle refs for an access decision; fail-closed ([]) on error."""
    global _circles_resolver_ready, _circles_resolver_hook
    with _circles_resolver_lock:
        if not _circles_resolver_ready:
            _circles_resolver_hook = _default_circles_resolver()
            _circles_resolver_ready = True
        resolver = _circles_resolver_hook
    if resolver is None or not user_id:
        return []
    try:
        return resolver(user_id)
    except Exception:
        logger.warning(
            "workflow circle resolution failed for %s", user_id, exc_info=True
        )
        return []


def _catalog_path() -> str:
    """Return path to the JSON workflow catalog file.

    ``UMRS_WORKFLOW_STATE_DIR`` (set in umrs-front compose to a bind-mounted
    dir) relocates the catalog off container tmpfs so registrations survive
    restarts; default preserves the legacy config-dir location.
    """
    state_dir = os.environ.get("UMRS_WORKFLOW_STATE_DIR", "").strip()
    if state_dir:
        return os.path.join(state_dir, "workflow_catalog.json")
    return get_resolved_umrs_subdir("workflow_catalog.json")


def _load_catalog() -> dict[str, Any]:
    """Load workflow catalog from JSON file."""
    path = _catalog_path()
    if not os.path.exists(path):
        return {"workflows": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_catalog(catalog: dict[str, Any]) -> None:
    """Save workflow catalog to JSON file.

    All four umrs-front cluster servers share one state dir; writes go
    tmp-then-os.replace so the four process writers never truncate each
    other's concurrent read.
    """
    path = _catalog_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def register_workflow(
    workflow_name: str,
    version: str,
    user_id: str,
    source_path: str,
    workflow_json: dict[str, Any],
    scope: str = "user",
    description: str | None = None,
) -> dict[str, Any]:
    """Register a workflow in the catalog.

    Called internally by the upload handler (auto-registration) or
    potentially via a future REGISTER_WORKFLOW command.

    Args:
        workflow_name: Unique name for the workflow.
        version: Version string (e.g. "1.0.0", "0.1").
        user_id: Client/user who owns this workflow.
        source_path: Absolute path to the extracted workflow directory.
        workflow_json: Parsed contents of workflow.json.
        scope: "user" (private) or "system" (shared).
        description: Optional human-readable description.

    Returns:
        Dict with "status" ("ok" or "error") and details.
    """
    # Extract step metadata from workflow_json
    steps_meta: list[dict[str, str]] = []
    for step in workflow_json.get("steps", []):
        steps_meta.append(
            {
                "name": step.get("name", ""),
                "entrypoint": step.get("entrypoint", ""),
            }
        )

    entry = {
        "name": workflow_name,
        "version": version,
        "user_id": user_id,
        "source_path": source_path,
        "scope": scope,
        "access": [],
        "description": description or workflow_json.get("description", ""),
        "steps": steps_meta,
        "requirements": workflow_json.get("requirements", []),
        "registered_at": datetime.now().isoformat(),
    }

    with _CATALOG_LOCK:
        catalog = _load_catalog()
        # Check for duplicate (name, version, scope, user_id)
        for existing in catalog["workflows"]:
            if (
                existing["name"] == workflow_name
                and existing.get("version") == version
                and existing.get("scope") == scope
                and existing.get("user_id") == user_id
            ):
                # Update in place (re-upload overwrites; an ADR-080 access
                # list set on the entry survives the re-upload)
                access = existing.get("access")
                existing.update(entry)
                if access:
                    existing["access"] = access
                _save_catalog(catalog)
                logger.info(
                    "Updated workflow registration: %s v%s (user=%s, scope=%s)",
                    workflow_name,
                    version,
                    user_id,
                    scope,
                )
                return {
                    "status": "ok",
                    "message": f"Workflow '{workflow_name}' v{version} updated",
                    "workflow": entry,
                }
        catalog["workflows"].append(entry)
        _save_catalog(catalog)

    logger.info(
        "Registered workflow: %s v%s (user=%s, scope=%s, %d steps)",
        workflow_name,
        version,
        user_id,
        scope,
        len(steps_meta),
    )
    return {
        "status": "ok",
        "message": f"Workflow '{workflow_name}' v{version} registered",
        "workflow": entry,
    }


def unregister_workflow(
    workflow_name: str,
    *,
    user_id: str,
    scope: str | None = None,
) -> dict[str, Any]:
    """Remove a workflow registration from the catalog.

    Deletes exactly the entries that match ``(name, user_id, scope?)`` —
    ownership is decided by the caller (REST layer checks admin OR owner;
    system scope requires admin), this function enforces the match.

    Returns ``{"status": "ok", "removed": n}``; ``removed == 0`` when no
    matching registration exists (caller maps to 404).
    """
    with _CATALOG_LOCK:
        catalog = _load_catalog()
        keep: list[dict[str, Any]] = []
        removed = 0
        for entry in catalog.get("workflows", []):
            if (
                entry.get("name") == workflow_name
                and entry.get("user_id") == user_id
                and (scope is None or entry.get("scope", "user") == scope)
            ):
                removed += 1
                continue
            keep.append(entry)
        if removed:
            catalog["workflows"] = keep
            _save_catalog(catalog)

    if removed:
        logger.info(
            "Unregistered workflow: %s (user=%s, scope=%s, removed=%d)",
            workflow_name,
            user_id,
            scope or "*",
            removed,
        )
    return {"status": "ok", "removed": removed}


def _handle_list_workflows(client_id: str, payload_str: str) -> str:
    """List registered workflows, optionally filtered by scope or user.

    Payload JSON (all optional):
      - scope: "user" | "system" — filter by scope
      - user_id: str — filter by owner
      - name: str — filter by workflow name (substring match)
      - include_shared_for: str — also append entries whose ADR-080 access
        list admits this caller (marked ``shared: true``)
      - verified_project: str — the caller's gateway-verified project (for
        project-scoped access entries)
      - include_system: bool — also append the platform's ``system``-scope
        workflows (every user may run them; an owner filter used to hide them)
    """
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    with _CATALOG_LOCK:
        catalog = _load_catalog()
    workflows = _filtered_workflows(catalog.get("workflows", []), payload)
    _append_shared_workflows(workflows, catalog.get("workflows", []), payload)
    _append_system_workflows(workflows, catalog.get("workflows", []), payload)

    return json.dumps(
        {
            "status": "ok",
            "count": len(workflows),
            "workflows": workflows,
        },
        default=str,
    )


def _filtered_workflows(
    workflows: list[dict[str, Any]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Scope/user/name filters straight from the list payload."""
    scope_f = payload.get("scope")
    user_f = payload.get("user_id")
    name_f = (payload.get("name") or "").lower()
    return [
        w
        for w in workflows
        if (not scope_f or w.get("scope") == scope_f)
        and (not user_f or w.get("user_id") == user_f)
        and (not name_f or name_f in w.get("name", "").lower())
    ]


def _shared_candidate(entry: dict[str, Any], seen: set, name_f: str) -> bool:
    """May this catalog entry even be offered as a shared row?"""
    key = (entry.get("name"), entry.get("version"), entry.get("user_id"))
    return (
        key not in seen
        and entry.get("scope") != "system"
        and (not name_f or name_f in entry.get("name", "").lower())
    )


def _append_shared_workflows(
    workflows: list[dict[str, Any]],
    catalog_entries: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    """ADR-080: append entries whose access list admits the payload's
    caller, marked ``shared: true`` (never silently)."""
    include_for = (
        payload.get("include_shared_for") if isinstance(payload, dict) else None
    )
    if not include_for:
        return
    caller = ResourceCaller(
        str(include_for),
        str(payload.get("verified_project") or ""),
        circles_resolver=lambda: _resolve_circles(str(include_for)),
    )
    name_f = (payload.get("name") or "").lower()
    seen = {(w.get("name"), w.get("version"), w.get("user_id")) for w in workflows}
    for entry in catalog_entries:
        if not _shared_candidate(entry, seen, name_f):
            continue
        if access_admits(entry.get("access"), caller):
            marked = dict(entry)
            marked["shared"] = True
            workflows.append(marked)
            seen.add((entry.get("name"), entry.get("version"), entry.get("user_id")))


def _append_system_workflows(
    workflows: list[dict[str, Any]],
    catalog_entries: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    """Append ``system``-scope entries when the payload asks (a caller's own
    list is owner-filtered, which dropped the platform workflows every user
    may run — review 2026-10-04). The name filter still applies."""
    if not (isinstance(payload, dict) and payload.get("include_system")):
        return
    name_f = (payload.get("name") or "").lower()
    seen = {(w.get("name"), w.get("version"), w.get("user_id")) for w in workflows}
    for entry in catalog_entries:
        if _system_candidate(entry, seen, name_f):
            workflows.append(entry)
            seen.add((entry.get("name"), entry.get("version"), entry.get("user_id")))


def _system_candidate(entry: dict[str, Any], seen: set, name_f: str) -> bool:
    """A system-scope entry not listed yet that matches the name filter."""
    key = (entry.get("name"), entry.get("version"), entry.get("user_id"))
    return (
        entry.get("scope") == "system"
        and key not in seen
        and (not name_f or name_f in entry.get("name", "").lower())
    )


def _handle_unregister_workflow(client_id: str, payload_str: str) -> str:
    """ZMQ command path mirroring the REST DELETE.

    Payload JSON:
      - name (required): workflow name
      - scope (optional): only delete entries with this scope

    Authorization (admin/owner) is enforced by the authenticated caller
    identity that reaches this handler — same trust as REGISTER/UPLOAD.
    """
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"status": "error", "message": f"invalid JSON: {exc}"})

    name = str(payload.get("name", "")).strip()
    if not name:
        return json.dumps({"status": "error", "message": "name is required"})
    result = unregister_workflow(
        name,
        user_id=client_id,
        scope=payload.get("scope") if isinstance(payload, dict) else None,
    )
    if result["removed"] == 0:
        return json.dumps(
            {"status": "error", "message": f"Workflow '{name}' not found"}
        )
    return json.dumps(result)


def _workflow_identity_provenance(
    params: dict[str, Any],
    *,
    identity_verified: bool = False,
) -> str:
    from umrs.api.workflow_auth import strict_auth_mode

    project_id = _workflow_project_id(params)
    if project_id == "default":
        return "unverified"
    if strict_auth_mode():
        return "gateway_verified" if identity_verified else "unverified"
    return "mvp_claim"


def _workflow_project_id(params: Any) -> str:
    if isinstance(params, dict) and isinstance(params.get("project_id"), str):
        project_id = params["project_id"].strip()
        if project_id:
            return project_id
    return "default"


def _verified_caller_project(params: Any, identity_verified: bool) -> str | None:
    """ADR-080: project-scoped sharing admits only a gateway-verified
    project (never an unverified claim, never the ``default`` bucket)."""
    if not identity_verified:
        return None
    project = _workflow_project_id(params)
    return None if project == "default" else project


def _workflow_run_context(
    workflow_name: str,
    client_id: str,
    source_path: str | None = None,
    *,
    caller_project: str | None = None,
) -> tuple[str, dict[str, Any], int]:
    if source_path is None:
        # A durable job without a usable source_path: resolved like a
        # trigger (ADR-074 D-8, ADR-080), never another owner's same-named
        # workflow. *caller_project* is the job's own project only when the
        # server recorded it as gateway-verified (:func:`durable_job_verified_project`).
        entry = resolve_workflow_for_caller(
            workflow_name, client_id, caller_project=caller_project
        )
        if entry is None:
            raise ValueError(f"Workflow '{workflow_name}' not found in registry")
        source_path = entry.get("source_path", "")

    if not source_path or not os.path.isdir(source_path):
        raise ValueError(f"Workflow source path not found: {source_path}")

    wf_json_path = os.path.join(source_path, "workflow.json")
    if not os.path.isfile(wf_json_path):
        raise ValueError(f"workflow.json missing at {source_path}")

    try:
        with open(wf_json_path, encoding="utf-8") as handle:
            workflow_json = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Failed to load workflow.json: {exc}") from exc

    return source_path, workflow_json, len(workflow_json.get("steps", []))


def _resolve_workflow_catalog_entry(
    workflow_name: str,
    client_id: str,
) -> dict[str, Any] | None:
    """Resolve a workflow by caller-owned entry, then system, then any-name.

    Scopes ("user"/"system") are advisory — durable-jobs run under varying
    runtime identities (upload client, scheduler-side execution engine,
    deployment umbrella). A registered workflow shall be resolvable by
    everyone; otherwise different legacies of upload user vs execution user
    resurface as spurious "not found in registry" errors deep in the chain.

    INTERNAL ONLY (delete's owner check). A run — a caller-facing trigger,
    describe, or a durable job without a usable ``source_path`` — resolves
    through :func:`resolve_workflow_for_caller` (ADR-074 D-8): the any-name
    fallback here would run another owner's code.
    """
    with _CATALOG_LOCK:
        catalog = _load_catalog()

    owned: dict[str, Any] | None = None
    system: dict[str, Any] | None = None
    any_name: dict[str, Any] | None = None
    for workflow_entry in catalog.get("workflows", []):
        if workflow_entry.get("name") != workflow_name:
            continue
        if any_name is None:
            any_name = workflow_entry
        if workflow_entry.get("user_id") == client_id:
            owned = workflow_entry
            break
        if workflow_entry.get("scope") == "system" and system is None:
            system = workflow_entry

    return owned or system or any_name


def _classify_workflow_entries(
    entries: list[dict[str, Any]],
    workflow_name: str,
    client_id: str,
    caller: ResourceCaller,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Scan same-named entries → (own, system, shared, first-foreign)."""
    own: dict[str, Any] | None = None
    system: dict[str, Any] | None = None
    shared: dict[str, Any] | None = None
    foreign: dict[str, Any] | None = None
    for entry in entries:
        if entry.get("name") != workflow_name:
            continue
        if entry.get("user_id") == client_id:
            own = entry
            break  # the caller's own entry always wins
        if entry.get("scope") == "system":
            system = system or entry
        elif shared is None and access_admits(entry.get("access"), caller):
            shared = entry
        elif foreign is None:
            foreign = entry
    return own, system, shared, foreign


def resolve_workflow_for_caller(
    workflow_name: str,
    client_id: str,
    *,
    is_admin: bool = False,
    caller_project: str | None = None,
) -> dict[str, Any] | None:
    """The workflow a caller may trigger or describe (ADR-074 D-8, ADR-080).

    Order: the caller's own entry, then a ``system`` entry (shared), then an
    entry whose ADR-080 access list admits the caller (user/project/circle
    scopes; ``system`` entries there equal a system workflow); another
    owner's entry otherwise only for admins — never a blind cross-owner name
    fallback, which let a normal user run another owner's code against their
    own buckets. Circle membership resolves lazily (only a circle-scoped
    candidate triggers the lookup); ``project:`` scopes admit only a
    verified ``caller_project``.
    """
    with _CATALOG_LOCK:
        catalog = _load_catalog()

    caller = ResourceCaller(
        client_id,
        caller_project or "",
        is_admin=is_admin,
        circles_resolver=lambda: _resolve_circles(client_id),
    )
    own, system, shared, foreign = _classify_workflow_entries(
        catalog.get("workflows", []), workflow_name, client_id, caller
    )
    return own or system or shared or (foreign if is_admin else None)


def workflow_access_entries(workflow_name: str, owner_id: str) -> list[dict[str, Any]]:
    """Catalog entries owned by *owner_id* under *workflow_name* (all
    versions/scopes) — the rows one ``access`` update touches."""
    with _CATALOG_LOCK:
        catalog = _load_catalog()
    return [
        entry
        for entry in catalog.get("workflows", [])
        if entry.get("name") == workflow_name and entry.get("user_id") == owner_id
    ]


def set_workflow_access(
    workflow_name: str, owner_id: str, access: list[str]
) -> list[dict[str, Any]]:
    """Replace the ADR-080 access list on every (name, owner) entry
    (validated upstream). Returns the updated entries ([] when none)."""
    with _CATALOG_LOCK:
        catalog = _load_catalog()
        updated: list[dict[str, Any]] = []
        for entry in catalog.get("workflows", []):
            if entry.get("name") == workflow_name and entry.get("user_id") == owner_id:
                entry["access"] = list(access)
                updated.append(entry)
        if updated:
            _save_catalog(catalog)
    return updated


def _caller_run_context(
    workflow_name: str,
    client_id: str,
    *,
    is_admin: bool,
    caller_project: str | None = None,
) -> tuple[str, dict[str, Any], int]:
    """:func:`_workflow_run_context` for a caller-facing trigger: owner-scoped
    resolution; the resolved ``source_path`` is what the run (and a durable
    job's spec) carries from here on."""
    entry = resolve_workflow_for_caller(
        workflow_name, client_id, is_admin=is_admin, caller_project=caller_project
    )
    if entry is None:
        raise ValueError(f"Workflow '{workflow_name}' not found in registry")
    return _workflow_run_context(
        workflow_name, client_id, source_path=str(entry.get("source_path") or "")
    )


def _take_over_prior_run(prior: Any, record: Any) -> None:
    """A durable re-dispatch reuses the execution id: the record it replaces
    (read through to Postgres after a restart) may name step jobs still on
    the cluster, and they move to the new record. A dispatch whose durable
    job the user cancelled meanwhile runs nothing, and its launch ends those
    jobs (peer review F2). The cancel handlers cancel the durable job before
    the registry record, and this check follows *record*'s registration: a
    dispatch racing a cancel either finds the job cancelled here, or the
    registry cancel that follows finds *record*
    (``WorkflowExecutionRegistry.cancel``). A requeued job (an admin retry,
    a reproduce) runs again: only the durable job's status counts, never an
    earlier record's cancel."""
    from umrs.server.step_reaper import carry_step_jobs, durable_job_cancelled

    if prior is not None:
        carry_step_jobs(prior, record)
    if durable_job_cancelled(record.durable_job_id):
        logger.info(
            "workflow execution %s was cancelled; its dispatch runs nothing",
            record.execution_id,
        )
        record.update(cancel_requested=True)


def _launch_workflow_from_catalog(
    *,
    workflow_name: str,
    client_id: str,
    params: dict[str, Any],
    execution_id: str | None = None,
    durable_job_id: str | None = None,
    source_path: str | None = None,
    identity_verified: bool = False,
    caller_project: str | None = None,
):
    source_path, workflow_json, step_count = _workflow_run_context(
        workflow_name,
        client_id,
        source_path=source_path,
        caller_project=caller_project,
    )

    logger.info(
        "Running workflow '%s' for user '%s' with %d steps (params=%s, durable_job_id=%s)",
        workflow_name,
        client_id,
        step_count,
        params,
        durable_job_id,
    )

    from umrs.server.step_router import build_registry_backed_router
    from umrs.server.workflow_execution_registry import get_execution_registry
    from umrs.server.workflow_runner import run_workflow_record

    registry = get_execution_registry()
    # A second dispatch of a run still running here (a scheduler requeue after
    # a lost submit acknowledgement, a boot reconcile) attaches to it: two live
    # runs of one execution id would cancel each other's step jobs and race on
    # the record (peer review 2026-09-26).
    live = registry.running_here(execution_id) if execution_id else None
    if live is not None:
        logger.info(
            "workflow execution %s is still running in this process; the dispatch attaches to it",
            execution_id,
        )
        return live, source_path, step_count
    prior = registry.get(execution_id) if execution_id else None
    record = registry.create(
        workflow_name=workflow_name,
        user_id=client_id,
        params=params,
        execution_id=execution_id,
        durable_job_id=durable_job_id,
        identity_provenance=_workflow_identity_provenance(
            params,
            identity_verified=identity_verified,
        ),
    )
    _take_over_prior_run(prior, record)

    project_id = _workflow_project_id(params)
    router = build_registry_backed_router(project_id=project_id)

    def _runner(record_to_run):
        run_workflow_record(
            record_to_run,
            source_path=source_path,
            workflow_json=workflow_json,
            router=router,
            project_id=project_id,
            user_id=client_id,
        )

    registry.launch(record, _runner)
    return record, source_path, step_count


def _submit_durable_workflow_run(
    *,
    client_id: str,
    workflow_name: str,
    params: dict[str, Any],
    execution_id: str,
    source_path: str,
    step_count: int,
    job_backend: DurableJobCommandBackend,
    verified_project: str | None = None,
) -> str:
    spec: dict[str, Any] = {
        "workflow_name": workflow_name,
        "execution_id": execution_id,
        "params": params,
        "source_path": source_path,
        # Auto-ingest (TASK-035): completed workflow outputs flow
        # into the knowledge pipeline. "step" "*" marks run-level
        # provenance (not a single named step).
        "ingest_results": True,
        "workflow": workflow_name,
        "step": "*",
    }
    if verified_project:
        spec[VERIFIED_PROJECT_SPEC_KEY] = verified_project
    raw = job_backend.submit_workflow_run(
        client_id,
        json.dumps(
            {
                "project_id": _workflow_project_id(params),
                "job_type": "workflow_run",
                "spec": spec,
                "max_attempts": 4,
                "restart_policy": "on_crash",
            }
        ),
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return json.dumps(
            {"status": "error", "message": f"Invalid durable job response: {exc}"}
        )

    if data.get("status") == "error":
        return raw

    job = data.get("job") if isinstance(data.get("job"), dict) else {}
    durable_job_id = job.get("job_id")
    return json.dumps(
        {
            "status": "ok",
            "action": "run-workflow",
            "workflow_name": workflow_name,
            "source_path": source_path,
            "steps": step_count,
            "params": params,
            "execution_id": execution_id,
            "durable_job_id": durable_job_id,
            "durable_job": job,
            "message": (
                f"Workflow '{workflow_name}' execution queued as durable job "
                f"{durable_job_id or '<unknown>'} ({step_count} steps)"
            ),
        },
        default=str,
    )


def _handle_run_workflow(
    client_id: str,
    payload_str: str,
    job_backend: DurableJobCommandBackend | None = None,
    *,
    trusted_identity: bool = False,
    caller_is_admin: bool = False,
) -> str:
    """Trigger execution of a registered workflow.

    Payload JSON:
      - workflow_name: str — name of the workflow to run
      - params: dict — additional runtime parameters (e.g. bucket alias)

    Resolution is owner-scoped (ADR-074 D-8): the caller's own workflow or a
    ``system`` one; another owner's only when ``caller_is_admin``.
    """
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "Invalid JSON payload"})

    workflow_name = payload.get("workflow_name", "")
    params = payload.get("params", {})

    if not workflow_name:
        return json.dumps({"status": "error", "message": "workflow_name is required"})
    if not isinstance(params, dict):
        return json.dumps(
            {"status": "error", "message": "workflow params must be an object"}
        )

    identity_verified = trusted_identity and bool(payload.get("_identity_verified"))
    caller_project = _verified_caller_project(params, identity_verified)

    try:
        source_path, _, step_count = _caller_run_context(
            workflow_name,
            client_id,
            is_admin=caller_is_admin,
            caller_project=caller_project,
        )
    except ValueError as exc:
        return json.dumps({"status": "error", "message": str(exc)})

    if job_backend is not None:
        execution_id = f"wf-{uuid.uuid4().hex[:12]}"
        return _submit_durable_workflow_run(
            client_id=client_id,
            workflow_name=workflow_name,
            params=params,
            execution_id=execution_id,
            source_path=source_path,
            step_count=step_count,
            job_backend=job_backend,
            verified_project=caller_project,
        )

    record, source_path, step_count = _launch_workflow_from_catalog(
        workflow_name=workflow_name,
        client_id=client_id,
        params=params,
        source_path=source_path,
        identity_verified=identity_verified,
    )

    return json.dumps(
        {
            "status": "ok",
            "action": "run-workflow",
            "workflow_name": workflow_name,
            "source_path": source_path,
            "steps": step_count,
            "params": params,
            "execution_id": record.execution_id,
            "durable_job_id": record.durable_job_id,
            "message": f"Workflow '{workflow_name}' execution accepted ({step_count} steps)",
        },
        default=str,
    )


def _drop_untrusted_identity_claim(payload_str: str, trusted_identity: bool) -> str:
    """*payload_str* without ``_identity_verified`` unless the caller is the
    server's own verified path (ADR-097 D-7, 2026-10-22). The status family
    (``zmq_caller_from_payload``) read the flag from any payload, so a ZMQ
    client could assert gateway-verified provenance for its project claim."""
    if trusted_identity or "_identity_verified" not in (payload_str or ""):
        return payload_str
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        return payload_str
    if not isinstance(payload, dict):
        return payload_str
    payload.pop("_identity_verified", None)
    return json.dumps(payload)


def handle_workflow_command(
    cmd: str,
    client_id: str,
    payload_str: str,
    job_backend: DurableJobCommandBackend | None = None,
    *,
    trusted_identity: bool = False,
    caller_is_admin: bool = False,
) -> str:
    """Dispatch a workflow registry command and return a JSON response string.

    Called from the server main loop when a LIST_WORKFLOWS command is received.

    ``trusted_identity`` must be set ONLY by server-side callers that already
    verified identity (the HTTP trigger route after HMAC signature validation).
    For ZMQ-exposed commands the payload's ``_identity_verified`` flag is a
    client-supplied claim and is ignored unless ``trusted_identity`` is set
    (wave-4 audit fix). ``caller_is_admin`` likewise comes only from the
    server's own verification (the gateway-signed admin flag, the ZMQ auth
    session's role), never from the payload.
    """
    from umrs.server.correlation import correlation_scope

    payload_str = _drop_untrusted_identity_claim(payload_str, trusted_identity)
    try:
        _payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        _payload = {}

    with correlation_scope(
        (_payload or {}).get("_request_id") or (_payload or {}).get("_correlation_id")
    ):
        return _dispatch_workflow_command(
            cmd, client_id, payload_str, job_backend, trusted_identity, caller_is_admin
        )


def _dispatch_workflow_command(
    cmd,
    client_id,
    payload_str,
    job_backend,
    trusted_identity,
    caller_is_admin=False,
):  # pragma: no cover - factored from handle_workflow_command body
    """Command dispatch under a fresh correlation scope (Q-05/G-001)."""
    from umrs.core.TaskCmdStatus import TaskCmd

    try:
        if cmd == TaskCmd.LIST_WORKFLOWS:
            return _handle_list_workflows(client_id, payload_str)
        elif cmd == TaskCmd.RUN_WORKFLOW:
            return _handle_run_workflow(
                client_id,
                payload_str,
                job_backend=job_backend,
                trusted_identity=trusted_identity,
                caller_is_admin=caller_is_admin,
            )
        elif cmd == TaskCmd.WORKFLOW_STATUS:
            return _handle_workflow_status(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_EXECUTIONS:
            return _handle_workflow_executions(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_RESULTS:
            return _handle_workflow_results(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_CANCEL:
            return _handle_workflow_cancel(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_ERRORS:
            return _handle_workflow_errors(client_id, payload_str)
        elif cmd == TaskCmd.WORKFLOW_LOGS:
            return _handle_workflow_logs(client_id, payload_str)
        elif cmd == TaskCmd.UNREGISTER_WORKFLOW:
            return _handle_unregister_workflow(client_id, payload_str)
        else:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Unknown workflow command: {cmd}",
                }
            )
    except Exception as e:
        logger.exception("Workflow command '%s' failed", cmd)
        return json.dumps({"status": "error", "message": str(e)})


def _parse_execution_id(payload_str: str) -> tuple[str | None, dict[str, Any]]:
    """Parse execution_id from payload. Returns (execution_id, payload)."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        return None, {}
    if not isinstance(payload, dict):
        return None, {}
    return payload.get("execution_id"), payload


def _zmq_execution_access(
    client_id: str,
    payload: dict[str, Any],
    execution_id: str,
) -> tuple[Any | None, Any | None, str | None]:
    """Resolve execution context and return an access-denied JSON string when needed."""
    from umrs.api.workflow_auth import (
        check_zmq_execution_access,
        resolve_execution_context,
        zmq_caller_from_payload,
    )
    from umrs.server.step_reaper import durable_job_store

    # The sidecar's job store, else the durable runtime's: the job is found
    # also where only the runtime dispatches the runs.
    store = durable_job_store()
    record, job = resolve_execution_context(execution_id, store=store)
    if record is None and job is None:
        return None, None, None
    caller = zmq_caller_from_payload(client_id, payload)
    denied = check_zmq_execution_access(caller, record=record, job=job)
    if denied is not None:
        return record, job, denied
    return record, job, None


def _handle_workflow_executions(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_EXECUTIONS — list workflow execution ids.

    Source of truth: the in-process execution registry — executions
    submitted since UMRS start. Durable jobs owned by the scheduler
    (job-store rows) are listed separately under ``umrs job list``.

    Visibility uses the same per-record ACL as WORKFLOW_STATUS
    (can_access_execution): records with ``unverified`` provenance are
    invisible even to their owner (fail-closed); records carrying a
    ``project_id`` show only to callers whose payload claims the same
    project — pass the same ``project_id`` the status path receives.
    The ZMQ rail has no admin promotion today, so admins on the HTTP rail
    should use the REST layer instead.

    Payload JSON (all optional):
      - user_id: str  — owner filter (admins only; regular users are always
        pinned to their own scope)
      - status: str   — exact status filter (submitted|running|succeeded|failed|cancelled)
      - project_id: str — caller project context (visibility depends on it)
      - limit: int    — max rows, newest first (default 50, max 500)
    """
    try:
        payload = json.loads(payload_str) if payload_str and payload_str.strip() else {}
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        return json.dumps({"status": "error", "message": "invalid JSON payload"})

    from umrs.api.workflow_auth import can_access_execution, zmq_caller_from_payload
    from umrs.server.workflow_execution_registry import get_execution_registry

    caller = zmq_caller_from_payload(client_id, payload)
    limit = payload.get("limit")
    if not isinstance(limit, int) or limit < 1 or limit > 500:
        limit = 50
    status_filter = payload.get("status")
    if isinstance(status_filter, str):
        status_filter = status_filter.strip().lower() or None
    else:
        status_filter = None
    # Regular users are pinned to their own scope. Admins may ask for a
    # specific owner or leave it unset for a system-wide listing.
    user_filter = payload.get("user_id")
    if not caller.is_admin:
        user_filter = caller.user_id

    registry = get_execution_registry()
    records = registry.list(
        user_id=user_filter if isinstance(user_filter, str) and user_filter else None
    )

    visible = []
    for record in records:
        if not can_access_execution(
            caller.user_id,
            caller.project_id,
            is_admin=caller.is_admin,
            caller_project_provenance=caller.project_provenance,
            record=record,
        ):
            continue
        if status_filter and record.status.lower() != status_filter:
            continue
        steps = list(record.steps.values())
        done = sum(1 for s in steps if s.status in ("succeeded", "failed", "skipped"))
        visible.append(
            {
                "execution_id": record.execution_id,
                "workflow_name": record.workflow_name,
                "user_id": record.user_id,
                "status": record.status,
                "progress": record.progress,
                "submitted_at": record.submitted_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "error": record.error,
                "steps_done": done,
                "steps_total": len(steps),
                "durable_job_id": record.durable_job_id,
            }
        )
        if len(visible) >= limit:
            break

    return json.dumps(
        {"status": "ok", "count": len(visible), "executions": visible},
        default=str,
    )


def _handle_workflow_status(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_STATUS — show execution status for a workflow run."""
    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None and job is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )
    if record is not None:
        return json.dumps(
            {
                "status": "ok",
                "execution": record.to_dict(),
            },
            default=str,
        )
    from umrs.api.workflow_auth import execution_payload_from_job

    return json.dumps(
        {
            "status": "ok",
            "execution": execution_payload_from_job(job),
        },
        default=str,
    )


def _handle_workflow_results(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_RESULTS — fetch results for a completed workflow run."""
    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None and job is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )
    if record is None:
        from umrs.api.workflow_auth import execution_payload_from_job

        return json.dumps(
            {
                "status": "ok",
                "execution": execution_payload_from_job(job),
                "results": {"steps": {}, "final": None},
                "artifacts": [],
            },
            default=str,
        )

    return json.dumps(
        {
            "status": "ok",
            "execution": record.to_dict(),
            "results": record.results,
            "artifacts": record.artifacts,
        },
        default=str,
    )


def _record_durable_job(store: Any, record: Any) -> Any:
    """The durable job *record* names (the execution-id lookup scans only the
    newest workflow_run jobs); None when there is none."""
    job_id = getattr(record, "durable_job_id", None)
    if store is None or not job_id:
        return None
    from umrs.jobs.store import JobNotFoundError

    try:
        return store.get_job(job_id)
    except JobNotFoundError:
        return None


def _reread_job(store: Any, job: Any) -> Any:
    """*job* as the store has it now, for the reply; the loaded one when there
    is no job or store, or the store cannot say."""
    if job is None or store is None:
        return job
    try:
        return store.get_job(job.job_id)
    except Exception:  # noqa: BLE001 — only the reply's status is at stake
        logger.warning("durable job %s unreadable after its cancel", job.job_id)
        return job


def _handle_workflow_cancel(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_CANCEL — cancel a running workflow execution."""
    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None and job is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )

    from umrs.jobs.workflow_run import cancel_workflow_execution
    from umrs.server.step_reaper import durable_job_store

    store = durable_job_store()
    if job is None:
        job = _record_durable_job(store, record)
    # The durable job, then the registry record whatever the lookup above
    # found (peer review F2); the reply's re-read comes last.
    record = cancel_workflow_execution(execution_id, store=store, job=job)
    job = _reread_job(store, job)

    if record is None and job is not None:
        from umrs.api.workflow_auth import execution_payload_from_job

        return json.dumps(
            {
                "status": "ok",
                "message": f"Execution '{execution_id}' cancelled",
                "execution": execution_payload_from_job(job),
            },
            default=str,
        )

    if record is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )

    return json.dumps(
        {
            "status": "ok",
            "message": f"Execution '{execution_id}' cancelled",
            "execution": record.to_dict(),
        },
        default=str,
    )


def _handle_workflow_errors(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_ERRORS — show error logs for a failed workflow execution.

    This command surfaces error logs for failed Attempts, including:
    - Workflow-level error message
    - Per-step error details for failed steps
    - Error log entries (if available)
    """

    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    tail = 50
    if isinstance(payload, dict):
        tail_val = payload.get("tail")
        if isinstance(tail_val, int) and tail_val > 0:
            tail = tail_val

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )

    # Collect error logs from the execution record
    error_logs: list[dict[str, Any]] = []

    # Workflow-level error
    if record.error:
        error_logs.append(
            {
                "timestamp": datetime.fromtimestamp(
                    record.finished_at or record.submitted_at
                ).isoformat(),
                "level": "ERROR",
                "source": "workflow",
                "message": record.error,
            }
        )

    # Per-step errors
    for step_name, step in record.steps.items():
        if step.status == "failed" and step.error:
            error_logs.append(
                {
                    "timestamp": datetime.fromtimestamp(
                        step.finished_at or step.started_at or record.submitted_at
                    ).isoformat(),
                    "level": "ERROR",
                    "source": f"step:{step_name}",
                    "message": step.error,
                }
            )

    # Limit to tail
    error_logs = error_logs[-tail:]

    return json.dumps(
        {
            "status": "ok",
            "execution": record.to_dict(),
            "error_logs": error_logs,
            "count": len(error_logs),
        },
        default=str,
    )


def _handle_workflow_logs(client_id: str, payload_str: str) -> str:
    """Handle WORKFLOW_LOGS — stream execution logs for a workflow run.

    Returns a structured log timeline including:
    - Workflow lifecycle events (submitted, started, finished)
    - Per-step events (started, routing, succeeded/failed)
    - Workflow-level error
    - Per-step errors

    The CLI can poll this endpoint with ``--follow`` to stream updates
    until the workflow reaches a terminal state.
    """
    execution_id, payload = _parse_execution_id(payload_str)
    if not execution_id:
        return json.dumps({"status": "error", "message": "execution_id is required"})

    tail = 200
    if isinstance(payload, dict):
        tail_val = payload.get("tail")
        if isinstance(tail_val, int) and tail_val > 0:
            tail = tail_val

    record, job, denied = _zmq_execution_access(client_id, payload, execution_id)
    if denied is not None:
        return denied
    if record is None and job is None:
        return json.dumps(
            {
                "status": "error",
                "message": (f"No workflow execution exists with ID '{execution_id}'."),
            }
        )
    if record is None:
        from umrs.api.workflow_auth import execution_payload_from_job

        return json.dumps(
            {
                "status": "ok",
                "execution": execution_payload_from_job(job),
                "logs": [],
                "count": 0,
            },
            default=str,
        )

    # Build log timeline
    log_entries: list[dict[str, Any]] = []

    # Workflow lifecycle
    log_entries.append(
        {
            "timestamp": datetime.fromtimestamp(record.submitted_at).isoformat(),
            "level": "INFO",
            "source": "workflow",
            "message": f"Workflow '{record.workflow_name}' submitted",
        }
    )
    if record.started_at:
        log_entries.append(
            {
                "timestamp": datetime.fromtimestamp(record.started_at).isoformat(),
                "level": "INFO",
                "source": "workflow",
                "message": "Workflow execution started",
            }
        )
    if record.finished_at:
        log_entries.append(
            {
                "timestamp": datetime.fromtimestamp(record.finished_at).isoformat(),
                "level": "INFO" if record.status == "succeeded" else "ERROR",
                "source": "workflow",
                "message": f"Workflow finished with status '{record.status}'",
            }
        )

    # Per-step events
    for step_name, step in record.steps.items():
        if step.started_at:
            log_entries.append(
                {
                    "timestamp": datetime.fromtimestamp(step.started_at).isoformat(),
                    "level": "INFO",
                    "source": f"step:{step_name}",
                    "message": f"Step '{step_name}' started",
                }
            )
        if step.finished_at:
            level = "ERROR" if step.status == "failed" else "INFO"
            log_entries.append(
                {
                    "timestamp": datetime.fromtimestamp(step.finished_at).isoformat(),
                    "level": level,
                    "source": f"step:{step_name}",
                    "message": (
                        f"Step '{step_name}' {step.status}"
                        + (f": {step.error}" if step.error else "")
                    ),
                }
            )

    # Sort by timestamp
    log_entries.sort(key=lambda e: e.get("timestamp", ""))

    # Limit to tail
    log_entries = log_entries[-tail:]

    return json.dumps(
        {
            "status": "ok",
            "execution": record.to_dict(),
            "logs": log_entries,
            "count": len(log_entries),
        },
        default=str,
    )


# ADR-053 (1.1/1.2): register the catalog launcher behind
# shared.workflow_execution_access so jobs/workflow_run never imports
# umrs.server.workflow_registry_handler again.
# The wrapper resolves the module-global name at every CALL, so a test
# patching `umrs.server.workflow_registry_handler._launch_workflow_from_catalog`
# keeps the old lazy-import semantics.
from umrs.shared.workflow_execution_access import (
    register_workflow_handler,
)


def _workflow_launch_factory(**kwargs):
    return _launch_workflow_from_catalog(**kwargs)


register_workflow_handler(_workflow_launch_factory)
