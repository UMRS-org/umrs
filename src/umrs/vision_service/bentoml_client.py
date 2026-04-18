"""BentoML task-queue helpers for UMRS vision services.

Provides **async-first** submit → poll → get helpers using
``bentoml.AsyncHTTPClient``.  Thin sync wrappers (``submit_and_wait``,
``submit_vision_and_wait``) are kept for callers that cannot manage their
own event loop — they create a short-lived ``AsyncHTTPClient`` internally
via ``asyncio.run()``.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from typing import Any

import bentoml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------

async def async_submit_and_wait(
    client: bentoml.AsyncHTTPClient,
    endpoint_name: str,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
    **kwargs: Any,
) -> dict:
    """Submit a task to a ``@bentoml.task`` endpoint and ``await`` completion.

    Args:
        client: An *already-opened* ``bentoml.AsyncHTTPClient``.
        endpoint_name: Task endpoint (e.g. ``"infer"``).
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait.
        **kwargs: Forwarded to the endpoint's ``submit()``.

    Returns:
        Deserialised task result (dict).
    """
    endpoint = getattr(client, endpoint_name)
    task = await endpoint.submit(**kwargs)
    task_id = task.id
    logger.info("Task submitted: %s -> %s", endpoint_name, task_id)

    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise TimeoutError(
                f"Task {task_id} did not complete within {timeout}s"
            )

        status = await task.get_status()
        logger.debug("Task %s status: %s (%.1fs elapsed)",
                     task_id, status.value, elapsed)

        if status.value in {"success", "completed"}:
            raw = await task.get()
            if isinstance(raw, bytes):
                return pickle.loads(raw)  # noqa: S301 — trusted internal service
            return raw
        if status.value in {"failure", "failed"}:
            raise RuntimeError(f"Task {task_id} failed")
        if status.value in {"cancelled", "canceled"}:
            raise RuntimeError(f"Task {task_id} was cancelled")

        await asyncio.sleep(poll_interval)


async def async_submit_vision(
    client: bentoml.AsyncHTTPClient,
    model_type: str,
    image_url: str,
    inference_config: dict | None = None,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
) -> dict:
    """Submit a vision inference task to the generic ``/infer`` endpoint.

    This is a thin wrapper around :func:`async_submit_and_wait` that maps
    the standard UMRS vision parameters.
    """
    config = dict(inference_config or {})
    return await async_submit_and_wait(
        client,
        "infer",
        poll_interval=poll_interval,
        timeout=timeout,
        model_type=model_type,
        inference_config=config,
        image_url=image_url,
    )


# ---------------------------------------------------------------------------
# Sync convenience wrappers  (create a per-call AsyncHTTPClient)
# ---------------------------------------------------------------------------

def submit_and_wait(
    service_url: str,
    endpoint_name: str,
    *,
    token: str | None = None,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
    **kwargs: Any,
) -> dict:
    """Synchronous wrapper — submit a task and block until complete.

    Creates a short-lived ``AsyncHTTPClient`` internally so callers do not
    need to manage an event loop.

    Args:
        service_url: Base URL of the BentoML service (e.g.
                     ``"http://localhost:3001"``).
        endpoint_name: Task endpoint name (e.g. ``"infer"``).
        token: Optional auth token.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait.
        **kwargs: Forwarded to the endpoint's ``submit()``.
    """

    async def _run() -> dict:
        async with bentoml.AsyncHTTPClient(service_url, token=token) as client:
            return await async_submit_and_wait(
                client, endpoint_name,
                poll_interval=poll_interval, timeout=timeout, **kwargs,
            )

    return asyncio.run(_run())


def submit_vision_and_wait(
    service_url: str,
    model_type: str,
    image_url: str,
    *,
    token: str | None = None,
    inference_config: dict | None = None,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
) -> dict:
    """Synchronous wrapper — submit a vision task via ``/infer`` and block.

    Creates a short-lived ``AsyncHTTPClient`` internally.
    """

    async def _run() -> dict:
        async with bentoml.AsyncHTTPClient(service_url, token=token) as client:
            return await async_submit_vision(
                client, model_type, image_url,
                inference_config=inference_config,
                poll_interval=poll_interval, timeout=timeout,
            )

    return asyncio.run(_run())
