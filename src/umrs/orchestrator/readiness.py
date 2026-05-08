"""
Readiness Polling Module.

UMRS readiness polling is the AUTHORITATIVE source of endpoint status.
Backends only provide discovery coordinates (host/port).

This module polls LiteLLM `/v1/models` endpoint with exponential backoff
to verify that a Mini-RiVault instance is ready to serve requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ReadinessState(str, Enum):
    """Readiness check states."""

    UNKNOWN = "unknown"  # Not yet checked
    POLLING = "polling"  # Currently polling
    READY = "ready"  # Endpoint is ready
    TIMEOUT = "timeout"  # Timed out waiting for ready
    FAILED = "failed"  # Failed (e.g., connection error)
    CANCELLED = "cancelled"  # Polling was cancelled


@dataclass
class ReadinessResult:
    """Result of a readiness check."""

    state: ReadinessState
    job_id: str
    host: str
    port: int
    attempts: int = 0
    elapsed_sec: float = 0.0
    models: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def is_ready(self) -> bool:
        """Check if endpoint is ready."""
        return self.state == ReadinessState.READY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "job_id": self.job_id,
            "host": self.host,
            "port": self.port,
            "attempts": self.attempts,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "models": self.models,
            "error": self.error,
        }


@dataclass
class ReadinessConfig:
    """Configuration for readiness polling."""

    # Timeout settings
    timeout_sec: int = 300  # Max time to wait for readiness (5 min)
    connect_timeout_sec: float = 5.0  # Connection timeout per request
    read_timeout_sec: float = 10.0  # Read timeout per request

    # Backoff settings (exponential with jitter)
    initial_delay_sec: float = 1.0  # Initial delay between polls
    max_delay_sec: float = 10.0  # Maximum delay between polls
    backoff_multiplier: float = 2.0  # Backoff multiplier
    jitter_factor: float = 0.1  # Random jitter (±10%)

    # Validation settings
    require_models: bool = True  # Require at least one model in response
    expected_model: str | None = None  # Specific model name to check
    auth_token: str | None = None  # Bearer token for protected LiteLLM endpoints
    probe_read_timeout_sec: float = 30.0  # Read timeout for inference probe

    @classmethod
    def from_env(cls) -> ReadinessConfig:
        """Create config from environment variables."""
        import os

        try:
            from .site_config import get_site_config

            get_site_config().inject_env(overwrite=False)
        except Exception:
            pass

        return cls(
            timeout_sec=int(os.getenv("RIVAULT_READINESS_TIMEOUT_SEC", "300")),
            connect_timeout_sec=float(
                os.getenv("RIVAULT_READINESS_CONNECT_TIMEOUT_SEC", "5.0")
            ),
            read_timeout_sec=float(
                os.getenv("RIVAULT_READINESS_READ_TIMEOUT_SEC", "10.0")
            ),
            initial_delay_sec=float(
                os.getenv("RIVAULT_READINESS_INITIAL_DELAY_SEC", "1.0")
            ),
            max_delay_sec=float(os.getenv("RIVAULT_READINESS_MAX_DELAY_SEC", "10.0")),
            backoff_multiplier=float(
                os.getenv("RIVAULT_READINESS_BACKOFF_MULTIPLIER", "2.0")
            ),
            expected_model=os.getenv("RIVAULT_EXPECTED_MODEL"),
            auth_token=os.getenv("LITELLM_MASTER_KEY") or None,
            probe_read_timeout_sec=float(
                os.getenv("RIVAULT_READINESS_PROBE_READ_TIMEOUT_SEC", "30.0")
            ),
        )


class ReadinessPoller:
    """
    Polls LiteLLM /v1/models endpoint to check readiness.

    This is the authoritative source of endpoint readiness status.
    Backends (Slurm, K8s, Compose) only provide discovery coordinates.

    Usage:
        poller = ReadinessPoller()
        result = await poller.wait_for_ready("job-123", "compute-01", 11434)
        if result.is_ready:
            print(f"Ready! Models: {result.models}")
    """

    def __init__(self, config: ReadinessConfig | None = None):
        self.config = config or ReadinessConfig.from_env()
        self._client: httpx.AsyncClient | None = None
        self._active_polls: dict[str, asyncio.Task] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        ``trust_env=False`` is critical: proxy environment variables
        (http_proxy, https_proxy, no_proxy) can prevent direct connections
        to Slurm compute nodes.  The readiness poller always connects
        directly to the target host:port.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout_sec,
                    read=self.config.read_timeout_sec,
                    write=5.0,
                    pool=5.0,
                ),
                trust_env=False,
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        import random

        delay = self.config.initial_delay_sec * (
            self.config.backoff_multiplier**attempt
        )
        delay = min(delay, self.config.max_delay_sec)

        # Add jitter
        jitter = delay * self.config.jitter_factor
        delay += random.uniform(-jitter, jitter)

        return max(0.1, delay)  # Minimum 100ms

    def _truncate_error_detail(self, detail: str) -> str:
        """Keep probe errors readable in logs and API surfaces."""
        cleaned = " ".join(detail.split())
        if len(cleaned) <= 240:
            return cleaned
        return f"{cleaned[:237]}..."

    async def _check_once(
        self,
        host: str,
        port: int,
        *,
        probe_model: str | None = None,
        probe_kind: str = "models",
    ) -> tuple[bool, list[str], str]:
        """
        Single readiness check.

        Returns:
            Tuple of (is_ready, models_list, error_message)
        """
        url = f"http://{host}:{port}/v1/models"
        client = await self._get_client()
        headers = None
        if self.config.auth_token:
            headers = {"Authorization": f"Bearer {self.config.auth_token}"}

        try:
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                return False, [], f"HTTP {response.status_code}"

            data = response.json()
            models = []

            # Parse models from response
            if "data" in data:
                for model in data["data"]:
                    model_id = model.get("id") or model.get("model")
                    if model_id:
                        models.append(model_id)

            # Validation
            if self.config.require_models and not models:
                return False, models, "No models in response"

            if self.config.expected_model and self.config.expected_model not in models:
                return (
                    False,
                    models,
                    f"Expected model '{self.config.expected_model}' not found",
                )

            if probe_kind == "chat":
                target_model = probe_model or self.config.expected_model or (models[0] if models else None)
                if not target_model:
                    return False, models, "Chat probe requires a model name"

                try:
                    probe_response = await client.post(
                        f"http://{host}:{port}/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": target_model,
                            "messages": [{"role": "user", "content": "ready"}],
                            "max_tokens": 1,
                            "temperature": 0.0,
                        },
                        timeout=httpx.Timeout(
                            connect=self.config.connect_timeout_sec,
                            read=self.config.probe_read_timeout_sec,
                            write=5.0,
                            pool=5.0,
                        ),
                    )
                except httpx.ConnectError as e:
                    return False, models, f"Chat probe connection error: {e}"
                except httpx.TimeoutException:
                    return False, models, "Chat probe timeout"
                except Exception as e:
                    return False, models, f"Chat probe error: {e}"

                if probe_response.status_code != 200:
                    return (
                        False,
                        models,
                        "Chat probe HTTP "
                        f"{probe_response.status_code}: "
                        f"{self._truncate_error_detail(probe_response.text)}",
                    )

            return True, models, ""

        except httpx.ConnectError as e:
            return False, [], f"Connection error: {e}"
        except httpx.TimeoutException:
            return False, [], "Request timeout"
        except Exception as e:
            return False, [], f"Error: {e}"

    async def wait_for_ready(
        self,
        job_id: str,
        host: str,
        port: int,
        probe_model: str | None = None,
        probe_kind: str = "models",
        on_progress: Callable[[ReadinessResult], None] | None = None,
    ) -> ReadinessResult:
        """
        Wait for endpoint to become ready.

        Args:
            job_id: Job identifier for logging.
            host: Hostname or IP address.
            port: LiteLLM port.
            on_progress: Optional callback for progress updates.

        Returns:
            ReadinessResult with final state.
        """
        start_time = time.monotonic()
        attempt = 0

        result = ReadinessResult(
            state=ReadinessState.POLLING,
            job_id=job_id,
            host=host,
            port=port,
        )

        logger.info(
            f"Starting readiness polling for {job_id} at {host}:{port} "
            f"(timeout={self.config.timeout_sec}s, auth={'yes' if self.config.auth_token else 'no'})"
        )

        last_error: str | None = None

        while True:
            elapsed = time.monotonic() - start_time

            # Check timeout
            if elapsed >= self.config.timeout_sec:
                result.state = ReadinessState.TIMEOUT
                result.elapsed_sec = elapsed
                result.error = f"Timeout after {self.config.timeout_sec}s"
                logger.warning(f"Readiness timeout for {job_id}: {result.error}")
                return result

            # Perform check
            attempt += 1
            result.attempts = attempt

            is_ready, models, error = await self._check_once(
                host,
                port,
                probe_model=probe_model,
                probe_kind=probe_kind,
            )

            if is_ready:
                result.state = ReadinessState.READY
                result.models = models
                result.elapsed_sec = time.monotonic() - start_time
                logger.info(
                    f"Endpoint {job_id} ready after {attempt} attempts "
                    f"({result.elapsed_sec:.1f}s). Models: {models}"
                )
                return result

            # Log progress
            if attempt == 1 or attempt % 5 == 0 or error != last_error:
                result.elapsed_sec = time.monotonic() - start_time
                logger.info(
                    "Readiness pending for %s at %s:%s after attempt %s (%.1fs): %s",
                    job_id,
                    host,
                    port,
                    attempt,
                    result.elapsed_sec,
                    error or "not ready",
                )
                last_error = error

            # Progress callback
            if on_progress:
                result.elapsed_sec = time.monotonic() - start_time
                result.error = error
                on_progress(result)

            # Calculate delay and wait
            delay = self._calculate_delay(attempt)
            remaining = self.config.timeout_sec - elapsed
            delay = min(delay, remaining)

            if delay > 0:
                await asyncio.sleep(delay)

    async def check_ready(
        self,
        host: str,
        port: int,
        *,
        probe_model: str | None = None,
        probe_kind: str = "models",
    ) -> bool:
        """
        Quick readiness check (single attempt).

        Args:
            host: Hostname or IP address.
            port: LiteLLM port.

        Returns:
            True if ready, False otherwise.
        """
        is_ready, _, _ = await self._check_once(
            host,
            port,
            probe_model=probe_model,
            probe_kind=probe_kind,
        )
        return is_ready

    async def check_ready_detail(
        self,
        host: str,
        port: int,
        *,
        probe_model: str | None = None,
        probe_kind: str = "models",
    ) -> tuple[bool, list[str], str]:
        """Return a single readiness check with models and error detail."""
        return await self._check_once(
            host,
            port,
            probe_model=probe_model,
            probe_kind=probe_kind,
        )

    def cancel_poll(self, job_id: str) -> bool:
        """
        Cancel an active polling task.

        Returns:
            True if cancelled, False if not found.
        """
        task = self._active_polls.get(job_id)
        if task and not task.done():
            task.cancel()
            del self._active_polls[job_id]
            logger.info(f"Cancelled readiness polling for {job_id}")
            return True
        return False


# Global singleton
_poller: ReadinessPoller | None = None


def get_readiness_poller() -> ReadinessPoller:
    """Get the global readiness poller singleton."""
    global _poller
    if _poller is None:
        _poller = ReadinessPoller()
    return _poller
