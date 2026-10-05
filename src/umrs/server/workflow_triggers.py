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


def _dispatch_entry(
    entry: dict[str, Any], *, project_id:str, bucket: str, s3_key: str, owner: str
) -> list[str]:
    """Fire ``on: ingest`` triggers declared by one workflow entry.

    Optional matcher extension: ``stem_key_only: true`` scopes the launched
    run to the uploaded file's stem (``only_videos``) so multi-file uploads
    fan out as per-file runs instead of rescanning the whole bucket.
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
        params = {
            "project_id": project_id,
            "bucket": bucket,
            "ingest_s3_key": s3_key,
            "_trigger": dict(trigger),
        }
        if trigger.get("stem_key_only"):
            params["only_videos"] = Path(s3_key).stem
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
