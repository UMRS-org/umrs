"""BentoML task-queue connector — async-first helpers for the native client."""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from typing import Any

import bentoml

from umrs.vision_service.bentoml_client import (  # noqa: F401 — re-export
    async_submit_and_wait,
    submit_and_wait,
)

logger = logging.getLogger(__name__)


async def async_submit_batch(
    client: bentoml.AsyncHTTPClient,
    endpoint: str,
    items: list[dict[str, Any]],
    timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> list[dict[str, Any]]:
    """Submit multiple tasks and await all results.

    Args:
        client: An *already-opened* ``bentoml.AsyncHTTPClient``.
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
        task = await endpoint_obj.submit(**kwargs)
        tasks.append(task)
        logger.info("Submitted %s task %s", endpoint, task.id)

    results: list[dict[str, Any]] = []
    for task in tasks:
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise TimeoutError(f"Task {task.id} did not complete within {timeout}s")
            status = await task.get_status()
            if status.value in {"success", "completed"}:
                raw = await task.get()
                if isinstance(raw, bytes):
                    results.append(pickle.loads(raw))
                else:
                    results.append(raw)
                break
            elif status.value in {"failure", "failed", "cancelled", "canceled"}:
                raise RuntimeError(f"Task {task.id} {status.value}")
            await asyncio.sleep(poll_interval)
    return results


def submit_batch(
    service_url: str,
    endpoint: str,
    items: list[dict[str, Any]],
    *,
    token: str | None = None,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> list[dict[str, Any]]:
    """Synchronous wrapper — submit multiple tasks and block until all complete.

    Creates a short-lived ``AsyncHTTPClient`` internally.
    """

    async def _run() -> list[dict[str, Any]]:
        async with bentoml.AsyncHTTPClient(service_url, token=token) as client:
            return await async_submit_batch(
                client,
                endpoint,
                items,
                timeout=timeout,
                poll_interval=poll_interval,
            )

    return asyncio.run(_run())
