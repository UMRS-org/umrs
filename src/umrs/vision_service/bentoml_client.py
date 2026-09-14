"""BentoML task-queue helpers for UMRS vision services.

Provides **async-first** submit → poll → get helpers using
``bentoml.AsyncHTTPClient``.  Thin sync wrappers (``submit_and_wait``,
``submit_vision_and_wait``) are kept for callers that cannot manage their
own event loop — they create a short-lived ``AsyncHTTPClient`` internally
via ``asyncio.run()``.

R-16 (Phase 2k):
- ``async_submit_and_wait`` now uses exponential backoff with jitter on the
  status poll loop instead of a fixed ``poll_interval``.
- ``pickle.loads`` is gated behind ``BENTOML_TRUSTED`` (default true) so
  deployments that expose the BentoML port through a tunnel can opt out of
  the RCE surface.
- Failure paths surface the BentoML status detail when the client provides
  one, instead of a bare "Task X failed".
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import random
import time
from typing import Any

logger = logging.getLogger(__name__)


def _bentoml_trusted() -> bool:
    """Whether the BentoML response is trusted enough for ``pickle.loads``.

    R-16: defaults to True for backward compatibility (current deployment
    keeps BentoML on the private network), but operators in less-trusted
    topologies can set ``BENTOML_TRUSTED=0`` to refuse pickled bytes.
    """
    return os.environ.get("BENTOML_TRUSTED", "1").lower() in {"1", "true", "yes", "on"}


def _backoff_sleep_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter, bounded by ``cap``."""
    raw = min(cap, base * (2**attempt))
    return random.uniform(0, raw)


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
    attempt = 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

        status = await task.get_status()
        logger.debug(
            "Task %s status: %s (%.1fs elapsed)", task_id, status.value, elapsed
        )

        if status.value in {"success", "completed"}:
            raw = await task.get()
            if isinstance(raw, bytes):
                if not _bentoml_trusted():
                    raise RuntimeError(
                        f"Refusing to unpickle BentoML task {task_id} response: "
                        "BENTOML_TRUSTED=0"
                    )
                return pickle.loads(raw)
            return raw
        if status.value in {"failure", "failed"}:
            # R-16: surface BentoML's failure detail when available.
            detail = ""
            for attr in ("error", "exception", "message"):
                getter = getattr(task, attr, None)
                if callable(getter):
                    try:
                        value = (
                            await getter()
                            if asyncio.iscoroutinefunction(getter)
                            else getter()
                        )
                        if value:
                            detail = str(value)
                            break
                    except Exception:
                        continue
            raise RuntimeError(
                f"Task {task_id} failed" + (f": {detail}" if detail else "")
            )
        if status.value in {"cancelled", "canceled"}:
            raise RuntimeError(f"Task {task_id} was cancelled")

        # R-16: exponential backoff with jitter, capped at ``poll_interval * 5``
        # so very slow tasks don't spam the status endpoint thousands of times.
        sleep_for = _backoff_sleep_seconds(attempt, poll_interval, poll_interval * 5)
        await asyncio.sleep(sleep_for)
        attempt += 1


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

    import bentoml

    async def _run() -> dict:
        async with bentoml.AsyncHTTPClient(service_url, token=token) as client:
            return await async_submit_and_wait(
                client,
                endpoint_name,
                poll_interval=poll_interval,
                timeout=timeout,
                **kwargs,
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

    import bentoml

    async def _run() -> dict:
        async with bentoml.AsyncHTTPClient(service_url, token=token) as client:
            return await async_submit_vision(
                client,
                model_type,
                image_url,
                inference_config=inference_config,
                poll_interval=poll_interval,
                timeout=timeout,
            )

    return asyncio.run(_run())
