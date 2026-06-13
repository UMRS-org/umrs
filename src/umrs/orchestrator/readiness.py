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


@dataclass(frozen=True)
class _ReadinessTarget:
    job_id: str
    host: str
    port: int
    probe_model: str | None
    probe_kind: str
    probe_path: str | None
    on_progress: Callable[[ReadinessResult], None] | None


@dataclass
class _ReadinessPoll:
    target: _ReadinessTarget
    start_time: float
    result: ReadinessResult
    attempt: int = 0
    last_error: str | None = None


@dataclass
class ReadinessConfig:
    """Configuration for readiness polling."""

    # Timeout settings
    timeout_sec: int = 300  # Max time to wait for readiness (5 min)
    connect_timeout_sec: float = 5.0  # Connection timeout per request
    read_timeout_sec: float = 10.0  # Read timeout per request

    # Backoff settings (exponential with jitter)
    # R-FastStart: cap at 5 s (was 10 s) so the poller catches a ready
    # endpoint within one additional poll cycle instead of two.
    initial_delay_sec: float = 1.0  # Initial delay between polls
    max_delay_sec: float = 5.0  # Maximum delay between polls
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


def _probe_url(
    host: str,
    port: int,
    probe_kind: str,
    probe_path: str | None,
) -> tuple[str, str]:
    normalized_probe = "http" if probe_kind == "bentoml" else probe_kind
    if normalized_probe == "http":
        path = _normalized_probe_path(probe_path)
        return normalized_probe, f"http://{host}:{port}{path}"
    return normalized_probe, f"http://{host}:{port}/v1/models"


def _normalized_probe_path(probe_path: str | None) -> str:
    path = probe_path or "/readyz"
    return path if path.startswith("/") else f"/{path}"


def _models_from_response(data: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for model in data.get("data") or []:
        model_id = model.get("id") or model.get("model")
        if model_id:
            models.append(model_id)
    return models


def _chat_probe_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "ready"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }


def _chat_probe_target(
    probe_model: str | None,
    expected_model: str | None,
    models: list[str],
) -> str | None:
    return probe_model or expected_model or (models[0] if models else None)


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
        probe_path: str | None = None,
    ) -> tuple[bool, bool, list[str], str]:
        """
        Single readiness check.

        Returns:
            Tuple of (is_ready, is_degraded, models_list, error_message).

            ``is_degraded`` is True when the endpoint responds (HTTP 200)
            but fails model validation or chat probe checks — the endpoint
            is reachable but unhealthy.  This is distinct from being
            entirely unresponsive, which yields (False, False, [], error).
        """
        normalized_probe, url = _probe_url(host, port, probe_kind, probe_path)
        client = await self._get_client()
        headers = self._auth_headers()

        try:
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                return False, False, [], f"HTTP {response.status_code}"

            if normalized_probe == "http":
                return True, False, [], ""

            models = _models_from_response(response.json())
            validation_error = self._model_validation_error(models)
            if validation_error:
                # Endpoint reachable but model mismatch or missing.
                # This is DEGRADED — not dead, not ready.
                return False, True, models, validation_error

            if normalized_probe == "chat":
                chat_ok, chat_models, chat_error = await self._check_chat_probe(
                    client, host, port, headers, probe_model, models
                )
                if not chat_ok:
                    # Chat probe failed but endpoint responded to /v1/models.
                    # Degraded: running but inference not fully available.
                    return False, True, chat_models, chat_error
                return True, False, chat_models, ""

            return True, False, models, ""

        except httpx.ConnectError as e:
            return False, False, [], f"Connection error: {e}"
        except httpx.TimeoutException:
            return False, False, [], "Request timeout"
        except Exception as e:
            return False, False, [], f"Error: {e}"

    def _auth_headers(self) -> dict[str, str] | None:
        if self.config.auth_token:
            return {"Authorization": f"Bearer {self.config.auth_token}"}
        return None

    def _model_validation_error(self, models: list[str]) -> str | None:
        if self.config.require_models and not models:
            return "No models in response"
        if self.config.expected_model and self.config.expected_model not in models:
            return f"Expected model '{self.config.expected_model}' not found"
        return None

    async def _check_chat_probe(
        self,
        client: httpx.AsyncClient,
        host: str,
        port: int,
        headers: dict[str, str] | None,
        probe_model: str | None,
        models: list[str],
    ) -> tuple[bool, list[str], str]:
        target_model = _chat_probe_target(
            probe_model, self.config.expected_model, models
        )
        if not target_model:
            return False, models, "Chat probe requires a model name"

        probe_response, probe_error = await self._post_chat_probe(
            client, host, port, headers, target_model
        )
        if probe_error:
            return False, models, probe_error

        http_error = self._chat_probe_http_error(probe_response)
        if http_error:
            return False, models, http_error
        return True, models, ""

    async def _post_chat_probe(
        self,
        client: httpx.AsyncClient,
        host: str,
        port: int,
        headers: dict[str, str] | None,
        target_model: str,
    ) -> tuple[httpx.Response | None, str | None]:
        try:
            probe_response = await client.post(
                f"http://{host}:{port}/v1/chat/completions",
                headers=headers,
                json=_chat_probe_payload(target_model),
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout_sec,
                    read=self.config.probe_read_timeout_sec,
                    write=5.0,
                    pool=5.0,
                ),
            )
        except httpx.ConnectError as e:
            return None, f"Chat probe connection error: {e}"
        except httpx.TimeoutException:
            return None, "Chat probe timeout"
        except Exception as e:
            return None, f"Chat probe error: {e}"
        return probe_response, None

    def _chat_probe_http_error(
        self, probe_response: httpx.Response | None
    ) -> str | None:
        if probe_response is None or probe_response.status_code == 200:
            return None
        return (
            "Chat probe HTTP "
            f"{probe_response.status_code}: "
            f"{self._truncate_error_detail(probe_response.text)}"
        )

    def _start_result(self, job_id: str, host: str, port: int) -> ReadinessResult:
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
        return result

    async def wait_for_ready(
        self,
        job_id: str,
        host: str,
        port: int,
        probe_model: str | None = None,
        probe_kind: str = "models",
        probe_path: str | None = None,
        on_progress: Callable[[ReadinessResult], None] | None = None,
    ) -> ReadinessResult:
        """Wait for endpoint to become ready and return the final state."""
        poll = self._start_poll(
            _ReadinessTarget(
                job_id, host, port, probe_model, probe_kind, probe_path, on_progress
            )
        )

        while True:
            ready_result = await self._poll_ready_once(poll)
            if ready_result is not None:
                return ready_result

    def _start_poll(self, target: _ReadinessTarget) -> _ReadinessPoll:
        return _ReadinessPoll(
            target=target,
            start_time=time.monotonic(),
            result=self._start_result(target.job_id, target.host, target.port),
        )

    async def _poll_ready_once(self, poll: _ReadinessPoll) -> ReadinessResult | None:
        elapsed = time.monotonic() - poll.start_time

        timeout_result = self._timeout_result(poll.result, poll.target.job_id, elapsed)
        if timeout_result is not None:
            return timeout_result

        poll.attempt += 1
        poll.result.attempts = poll.attempt
        is_ready, _is_degraded, models, error = await self._check_once(
            poll.target.host,
            poll.target.port,
            probe_model=poll.target.probe_model,
            probe_kind=poll.target.probe_kind,
            probe_path=poll.target.probe_path,
        )

        ready_result = self._ready_result(
            poll.result,
            poll.target.job_id,
            poll.attempt,
            models,
            is_ready,
            poll.start_time,
        )
        if ready_result is not None:
            return ready_result

        poll.last_error = self._record_pending(poll, error)
        await self._sleep_before_next_attempt(poll.attempt, elapsed)
        return None

    async def _sleep_before_next_attempt(self, attempt: int, elapsed: float) -> None:
        delay = self._calculate_delay(attempt)
        remaining = self.config.timeout_sec - elapsed
        delay = min(delay, remaining)
        if delay > 0:
            await asyncio.sleep(delay)

    def _timeout_result(
        self,
        result: ReadinessResult,
        job_id: str,
        elapsed: float,
    ) -> ReadinessResult | None:
        if elapsed < self.config.timeout_sec:
            return None
        result.state = ReadinessState.TIMEOUT
        result.elapsed_sec = elapsed
        result.error = f"Timeout after {self.config.timeout_sec}s"
        logger.warning(f"Readiness timeout for {job_id}: {result.error}")
        return result

    def _ready_result(
        self,
        result: ReadinessResult,
        job_id: str,
        attempt: int,
        models: list[str],
        is_ready: bool,
        start_time: float,
    ) -> ReadinessResult | None:
        if not is_ready:
            return None
        result.state = ReadinessState.READY
        result.models = models
        result.elapsed_sec = time.monotonic() - start_time
        logger.info(
            f"Endpoint {job_id} ready after {attempt} attempts "
            f"({result.elapsed_sec:.1f}s). Models: {models}"
        )
        return result

    def _record_pending(self, poll: _ReadinessPoll, error: str) -> str | None:
        if poll.attempt == 1 or poll.attempt % 5 == 0 or error != poll.last_error:
            poll.result.elapsed_sec = time.monotonic() - poll.start_time
            logger.info(
                "Readiness pending for %s at %s:%s after attempt %s (%.1fs): %s",
                poll.target.job_id,
                poll.target.host,
                poll.target.port,
                poll.attempt,
                poll.result.elapsed_sec,
                error or "not ready",
            )
            poll.last_error = error
        if poll.target.on_progress:
            poll.result.elapsed_sec = time.monotonic() - poll.start_time
            poll.result.error = error
            poll.target.on_progress(poll.result)
        return poll.last_error

    async def check_ready(
        self,
        host: str,
        port: int,
        *,
        probe_model: str | None = None,
        probe_kind: str = "models",
        probe_path: str | None = None,
    ) -> bool:
        """
        Quick readiness check (single attempt).

        Args:
            host: Hostname or IP address.
            port: LiteLLM port.

        Returns:
            True if ready, False otherwise.
        """
        is_ready, _, _, _ = await self._check_once(
            host,
            port,
            probe_model=probe_model,
            probe_kind=probe_kind,
            probe_path=probe_path,
        )
        return is_ready

    async def check_ready_detail(
        self,
        host: str,
        port: int,
        *,
        probe_model: str | None = None,
        probe_kind: str = "models",
        probe_path: str | None = None,
    ) -> tuple[bool, bool, list[str], str]:
        """Return a detailed readiness check: (is_ready, is_degraded, models_list, error)."""
        return await self._check_once(
            host,
            port,
            probe_model=probe_model,
            probe_kind=probe_kind,
            probe_path=probe_path,
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
