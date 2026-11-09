"""W4: catalog/ingest event → workflow trigger dispatch.

Workflows declare subscriptions at the top level of ``workflow.json``::

    "triggers": [
      {"on": "ingest", "project": "proj-a", "bucket": "uploads", "prefix": "images/"},
      {"on": "ingest", "bucket": "b", "prefix": "videos/", "stem_key_only": true}
    ]

Fields other than ``on`` are optional matchers or behavior flags; absent
(or ``"*"``) matcher values mean "match any". ``stem_key_only: true``
adds ``only_videos = <uploaded stem>`` to the run params so file-scoped
workflows run per uploaded file instead of rescanning the bucket.
``batch_quiet_seconds: N`` instead collects stems during a rolling
quiet window and fires ONE run with ``only_videos = `` a comma list of
all collected stems (multi-file uploads become ONE batch job; pending
batches live only in-process — a server restart drops them).
Cron subscriptions are handled separately by the SchedulerService
(W4 cron half) — this module covers event dispatch.

One (owner, name) fires once per event (ADR-113 D-7): the version its owner
pinned (``trigger_version``), else the default version, with that version's
own ``triggers``; a name adopted under another name fires nothing.

Philosophy mirrors the connector triggers: dispatch is best-effort and
never raises — a broken workflow subscription must not break the
post-upload ingestion path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_INGEST_EVENT = "ingest"


def _wildcard_or_equal(value: Any, actual: str) -> bool:
    """Absent/``"*"`` subscription values match anything."""
    return not value or value == "*" or str(value) == actual


def _trigger_matches(
    trigger: dict[str, Any], *, project_id: str, bucket: str, s3_key: str
) -> bool:
    """Whether one trigger subscription matches an ingest event."""
    if trigger.get("on") != _INGEST_EVENT:
        return False
    if not _wildcard_or_equal(trigger.get("project"), project_id):
        return False
    if not _wildcard_or_equal(trigger.get("bucket"), bucket):
        return False
    prefix = trigger.get("prefix")
    return not prefix or s3_key.startswith(str(prefix))


def _entry_triggers(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the declared trigger list from an entry's workflow.json."""
    source_path = entry.get("source_path", "")
    workflow_path = os.path.join(source_path, "workflow.json")
    if not source_path or not os.path.isfile(workflow_path):
        return []
    try:
        with open(workflow_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        logger.warning("Failed reading triggers from %s", workflow_path, exc_info=True)
        return []
    triggers = data.get("triggers") or []
    if not isinstance(triggers, list):
        return []
    return [t for t in triggers if isinstance(t, dict)]


def _execution_id_from_response(raw: str) -> str | None:
    """Extract the execution id from a workflow command response."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    execution = data.get("execution")
    if isinstance(execution, dict) and execution.get("execution_id"):
        return str(execution["execution_id"])
    if data.get("execution_id"):
        return str(data["execution_id"])
    return None


def _owner_event_project(client_id: str, project_id: str | None) -> str | None:
    """The ingest event's project when the workflow owner's user record lists
    it, else None. The upload route recorded the project for the uploader;
    the run executes the owner's workflow, so the project counts for the HPC
    lane rule (ADR-112 A.3.4) only when the owner belongs to it too (an
    unreadable record fails closed; an admin owner is admitted anyway)."""
    from umrs.shared.hpc_user_work_access import user_in_project

    return project_id if user_in_project(client_id, project_id) else None


def _run_triggered(
    workflow_name: str,
    client_id: str,
    params: dict[str, Any],
    version: str | None = None,
) -> str:
    """Launch one ingest-triggered run of *workflow_name* (the *version*
    whose triggers fired, ADR-113 D-7) for its owner."""
    from umrs.server.workflow_registry_handler import run_ingest_workflow

    payload = {"workflow_name": workflow_name, "version": version, "params": params}
    return run_ingest_workflow(
        client_id,
        json.dumps(payload),
        event_project=_owner_event_project(client_id, params.get("project_id")),
    )


# ── batch quiet-window collection (single run per batch) ────────────────────

_BATCH_LOCK = threading.Lock()
_BATCHES: dict[tuple, dict[str, Any]] = {}
_BATCH_CAP = int(os.environ.get("UMRS_INGEST_BATCH_CAP", "200"))


def _collect_batch_fire(
    workflow_name: str, client_id: str, trigger: dict[str, Any], stems: list[str],
) -> None:
    """Timer expiry: launch ONE run with only_videos=<comma-joined stems>."""
    params = dict(trigger.get("_params") or {})
    params["only_videos"] = ",".join(sorted(stems))
    params["_trigger"] = {
        k: v for k, v in trigger.items() if k not in ("_params", "_version")
    }
    try:
        raw = _run_triggered(workflow_name, client_id, params, trigger.get("_version"))
        execution_id = _execution_id_from_response(raw)
        logger.info(
            "Ingest batch fired: workflow=%s execution=%s files=%d",
            workflow_name, execution_id or "<unresolved>", len(stems),
        )
    except Exception:
        logger.exception("Ingest batch dispatch failed for %s", workflow_name)


def _queue_batch(
    workflow_name: str, client_id: str, trigger: dict[str, Any], stem: str,
) -> bool:
    """Collect the stem into the rolling batch and (re)arm the quiet timer.
    Returns True when the call was absorbed into a batch (no immediate fire).
    """
    quiet = float(trigger.get("batch_quiet_seconds") or 0)
    if quiet <= 0:
        return False
    key = (workflow_name, str(trigger.get("bucket") or ""))
    with _BATCH_LOCK:
        b = _BATCHES.setdefault(
            key, {"stems": set(), "timer": None, "client_id": client_id,
                  "trigger": trigger}
        )
        b["stems"].add(stem)
        b["client_id"] = client_id
        old = b["timer"]
        if old is not None:
            old.cancel()
        if len(b["stems"]) >= _BATCH_CAP:
            _collect_batch_fire(workflow_name, client_id, trigger,
                                sorted(b["stems"]))
            _BATCHES[key] = {"stems": set(), "timer": None, "client_id": client_id,
                             "trigger": trigger}
            return True
        t = threading.Timer(
            quiet, _collect_batch_fire,
            args=(workflow_name, client_id, trigger, sorted(b["stems"])),
        )
        t.daemon = True
        b["timer"] = t
        t.start()
    logger.info(
        "Ingest batch queued: workflow=%s bucket=%s stems=%d quiet=%ss",
        workflow_name, trigger.get("bucket"), len(b["stems"]), quiet,
    )
    return True


def _dispatch_entry(
    entry: dict[str, Any], *, project_id: str, bucket: str, s3_key: str, owner: str
) -> list[str]:
    """Fire ``on: ingest`` triggers declared by one workflow entry.

    Optional matcher extensions: ``stem_key_only: true`` scopes the launch to
    the uploaded stem; ``batch_quiet_seconds: N`` collects stems inside a
    rolling quiet window and fires ONE run with only_videos list (batch upload
    = one job), declared per trigger.
    """
    from pathlib import Path

    workflow_name = entry.get("name")
    if not workflow_name:
        return []
    client_id = entry.get("user_id") or owner
    version = entry.get("version")
    execution_ids: list[str] = []
    for trigger in _entry_triggers(entry):
        if not _trigger_matches(
            trigger, project_id=project_id, bucket=bucket, s3_key=s3_key
        ):
            continue
        stem = Path(s3_key).stem
        base = {
            "project_id": project_id,
            "bucket": bucket,
            "ingest_s3_key": s3_key,
        }
        if _queue_batch(
            workflow_name, client_id,
            {**trigger, "_params": dict(base), "_version": version}, stem,
        ):
            execution_ids.append("batch-queued")
            continue
        params = {
            **base,
            "_trigger": dict(trigger),
        }
        if trigger.get("stem_key_only"):
            params["only_videos"] = stem
        raw = _run_triggered(workflow_name, client_id, params, version)
        execution_id = _execution_id_from_response(raw)
        if execution_id:
            execution_ids.append(execution_id)
        logger.info(
            "Ingest trigger fired: workflow=%s execution=%s s3_key=%s",
            workflow_name,
            execution_id or "<unresolved>",
            s3_key,
        )
    return execution_ids


def _pinned_row(rows: list[dict[str, Any]], available: list[dict[str, Any]]):
    """The available row of the owner's ``trigger_version`` pin, or None."""
    pins = {str(r["trigger_version"]) for r in rows if r.get("trigger_version")}
    return next((r for r in available if str(r.get("version")) in pins), None)


def _trigger_target(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The one row of (owner, name) whose triggers fire (ADR-113 D-7): the
    owner's ``trigger_version`` pin when that version has code, else the
    default version; none once the name was adopted under another name."""
    from umrs.server.workflow_versions import default_entry, entry_available

    available = [r for r in rows if entry_available(r)]
    if not available or any(r.get("triggers_disabled") for r in rows):
        return None
    return _pinned_row(rows, available) or default_entry(available)


def trigger_targets(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (owner, name): before ADR-113 every registered version
    fired its own run for one event."""
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for entry in entries:
        if isinstance(entry, dict):
            groups.setdefault((entry.get("user_id"), entry.get("name")), []).append(entry)
    targets = (_trigger_target(rows) for rows in groups.values())
    return [t for t in targets if t is not None]


def dispatch_catalog_event(
    *, project_id: str, bucket: str, s3_key: str, owner: str
) -> list[str]:
    """W4: fan an ingest event out to every subscribed workflow.

    Returns execution ids of launched runs. Never raises.
    """
    from umrs.server.workflow_registry_handler import _CATALOG_LOCK, _load_catalog

    execution_ids: list[str] = []
    try:
        with _CATALOG_LOCK:
            catalog = _load_catalog()
    except Exception:
        logger.warning(
            "Workflow trigger dispatch skipped: catalog unavailable", exc_info=True
        )
        return execution_ids

    for entry in trigger_targets(catalog.get("workflows", [])):
        try:
            execution_ids.extend(
                _dispatch_entry(
                    entry,
                    project_id=project_id,
                    bucket=bucket,
                    s3_key=s3_key,
                    owner=owner,
                )
            )
        except Exception:
            logger.exception(
                "Workflow trigger dispatch failed for entry %s",
                entry.get("name", "<unnamed>"),
            )
    return execution_ids
