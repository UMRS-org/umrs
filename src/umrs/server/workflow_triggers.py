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


# ── batch quiet-window collection (single run per batch) ────────────────────

_BATCH_LOCK = threading.Lock()
_BATCHES: dict[tuple, dict[str, Any]] = {}
_BATCH_CAP = int(os.environ.get("UMRS_INGEST_BATCH_CAP", "200"))


def _collect_batch_fire(
    workflow_name: str, client_id: str, trigger: dict[str, Any], stems: list[str],
) -> None:
    """Timer expiry: launch ONE run with only_videos=<comma-joined stems>."""
    from umrs.core.TaskCmdStatus import TaskCmd
    from umrs.server.workflow_registry_handler import handle_workflow_command

    params = dict(trigger.get("_params") or {})
    params["only_videos"] = ",".join(sorted(stems))
    params["_trigger"] = {k: v for k, v in trigger.items() if k != "_params"}
    try:
        raw = handle_workflow_command(
            TaskCmd.RUN_WORKFLOW, client_id,
            json.dumps({"workflow_name": workflow_name, "params": params,
                        "_identity_verified": False}),
        )
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

    from umrs.core.TaskCmdStatus import TaskCmd
    from umrs.server.workflow_registry_handler import handle_workflow_command

    workflow_name = entry.get("name")
    if not workflow_name:
        return []
    client_id = entry.get("user_id") or owner
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
            {**trigger, "_params": dict(base)}, stem,
        ):
            execution_ids.append("batch-queued")
            continue
        params = {
            **base,
            "_trigger": dict(trigger),
        }
        if trigger.get("stem_key_only"):
            params["only_videos"] = stem
        raw = handle_workflow_command(
            TaskCmd.RUN_WORKFLOW,
            client_id,
            json.dumps(
                {
                    "workflow_name": workflow_name,
                    "params": params,
                    "_identity_verified": False,
                }
            ),
        )
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

    for entry in catalog.get("workflows", []):
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
