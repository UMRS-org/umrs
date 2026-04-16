"""BentoML task-queue connector — helpers for the native client."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import bentoml

from umrs.vision_service.bentoml_client import submit_and_wait  # noqa: F401 — re-export

logger = logging.getLogger(__name__)


def submit_batch(client: bentoml.SyncHTTPClient, endpoint: str,
                 items: List[Dict[str, Any]],
                 timeout: float = 300.0,
                 poll_interval: float = 2.0) -> List[Dict[str, Any]]:
    """Submit multiple tasks and wait for all of them.

    Args:
        client: ``bentoml.SyncHTTPClient`` connected to the service.
        endpoint: Task endpoint name (e.g. ``"infer"``).
        items: List of kwarg dicts, one per task.
        timeout: Maximum seconds to wait per task.
        poll_interval: Seconds between status polls.

    Returns:
        List of result dicts in the same order as *items*.
    """
    endpoint_obj = getattr(client, endpoint)
    tasks = []
    for kwargs in items:
        task = endpoint_obj.submit(**kwargs)
        tasks.append(task)
        logger.info("Submitted %s task %s", endpoint, task.id)

    results = []
    for task in tasks:
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Task {task.id} did not complete within {timeout}s"
                )
            status = task.get_status()
            if status.value == "success":
                raw = task.get()
                if isinstance(raw, bytes):
                    import pickle
                    results.append(pickle.loads(raw))  # noqa: S301
                else:
                    results.append(raw)
                break
            elif status.value in ("failure", "cancelled"):
                raise RuntimeError(f"Task {task.id} {status.value}")
            time.sleep(poll_interval)
    return results
