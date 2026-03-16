"""BentoML task-queue connector — re-exports the existing client with helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from umrs.bentoml_client import BentoMLTaskClient  # noqa: F401 — re-export

logger = logging.getLogger(__name__)


def submit_and_wait(client: BentoMLTaskClient, endpoint: str,
                    timeout: float = 300.0, poll_interval: float = 2.0,
                    **kwargs) -> Dict[str, Any]:
    """Submit a task and block until it completes.

    Convenience wrapper around ``client.submit()`` +
    ``client.wait_for_result()``.
    """
    task_id = client.submit(endpoint, **kwargs)
    logger.info("Submitted %s task %s", endpoint, task_id)
    return client.wait_for_result(
        endpoint, task_id,
        poll_interval=poll_interval,
        timeout=timeout,
    )


def submit_batch(client: BentoMLTaskClient, endpoint: str,
                 items: List[Dict[str, Any]],
                 timeout: float = 300.0,
                 poll_interval: float = 2.0) -> List[Dict[str, Any]]:
    """Submit multiple tasks and wait for all of them.

    Args:
        items: List of kwarg dicts, one per task.

    Returns:
        List of result dicts in the same order as *items*.
    """
    task_ids = []
    for kwargs in items:
        tid = client.submit(endpoint, **kwargs)
        task_ids.append(tid)
        logger.info("Submitted %s task %s", endpoint, tid)

    results = []
    for tid in task_ids:
        result = client.wait_for_result(
            endpoint, tid,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        results.append(result)
    return results
