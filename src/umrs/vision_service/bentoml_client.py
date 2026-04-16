"""BentoML task queue helpers for UMRS vision services.

Thin convenience layer on top of ``bentoml.SyncHTTPClient`` for the
submit → poll → get workflow used throughout UMRS.
"""

import pickle
import time
import logging

import bentoml

logger = logging.getLogger(__name__)


def submit_and_wait(client: bentoml.SyncHTTPClient, endpoint_name: str,
                    poll_interval: float = 2.0, timeout: float = 300.0,
                    **kwargs) -> dict:
    """Submit a task to a @bentoml.task endpoint and block until done.

    Uses BentoML's native task object API::

        task = client.<endpoint>.submit(**kwargs)
        task.get_status()  →  status.value in {"success", "failure", ...}
        task.get()         →  result dict

    Args:
        client: ``bentoml.SyncHTTPClient`` connected to the service.
        endpoint_name: Name of the task endpoint (e.g. ``"infer"``).
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait for completion.
        **kwargs: Arguments forwarded to the task function.

    Returns:
        Task result (dict).

    Raises:
        TimeoutError: If the task doesn't complete within *timeout*.
        RuntimeError: If the task fails or is cancelled.
    """
    endpoint = getattr(client, endpoint_name)
    task = endpoint.submit(**kwargs)
    task_id = task.id
    logger.info("Task submitted: %s -> %s", endpoint_name, task_id)

    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise TimeoutError(
                f"Task {task_id} did not complete within {timeout}s"
            )

        status = task.get_status()
        logger.debug("Task %s status: %s (%.1fs elapsed)",
                      task_id, status.value, elapsed)

        if status.value == "success":
            raw = task.get()
            # Service returns pickle-serialised bytes; deserialise back
            # to the original result dict.  Falls back to raw value
            # when it is already a dict (e.g. older service).
            if isinstance(raw, bytes):
                return pickle.loads(raw)  # noqa: S301 — trusted internal service
            return raw
        elif status.value == "failure":
            raise RuntimeError(f"Task {task_id} failed")
        elif status.value == "cancelled":
            raise RuntimeError(f"Task {task_id} was cancelled")

        time.sleep(poll_interval)
