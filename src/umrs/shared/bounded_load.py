"""Bounded model loading: per-attempt deadline, retry with backoff, cooldown.

Issue #635 (2026-10-27): boot provisioning loads each modality's embedding
model before the HTTP sidecar binds ``/health``. On a cold cache a stalled
Hugging Face transfer hung that load for 9 minutes, the compose healthcheck
expired first, and ``start.sh --server`` failed. :func:`load_bounded` caps
how long one model can hold a boot:

* each attempt runs in a daemon thread and is abandoned after ``timeout_s``
  (Python cannot kill a thread; an abandoned one keeps going in the
  background, and if it finishes it only fills the cache);
* at most ``attempts`` attempts, ``backoff_s * 2**(n-1)`` seconds apart;
* after the last failure the key cools down for ``cooldown_s``: further
  loads of the same model fail at once, so N users sharing one model cost
  one bound per boot, not N.

The worst case for one model is :meth:`LoadPolicy.worst_case_s`; the
umrs-front compose healthcheck ``start_period`` is sized from it
(``tests/test_boot_model_cache_wiring.py`` in the parent repo pins that).
Env overrides: ``UMRS_MODEL_LOAD_TIMEOUT_S``, ``UMRS_MODEL_LOAD_ATTEMPTS``,
``UMRS_MODEL_LOAD_BACKOFF_S``, ``UMRS_MODEL_LOAD_COOLDOWN_S``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

TIMEOUT_ENV = "UMRS_MODEL_LOAD_TIMEOUT_S"
ATTEMPTS_ENV = "UMRS_MODEL_LOAD_ATTEMPTS"
BACKOFF_ENV = "UMRS_MODEL_LOAD_BACKOFF_S"
COOLDOWN_ENV = "UMRS_MODEL_LOAD_COOLDOWN_S"


class ModelLoadError(RuntimeError):
    """A model did not load within its bound (or is cooling down)."""


class ModelLoadTimeout(TimeoutError):
    """One attempt outlived its deadline."""


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return max(minimum, float(raw)) if raw else default
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class LoadPolicy:
    """How long and how often one model load may be tried."""

    timeout_s: float = 90.0
    attempts: int = 3
    backoff_s: float = 5.0
    cooldown_s: float = 600.0

    @classmethod
    def from_env(cls) -> LoadPolicy:
        base = cls()
        return cls(
            timeout_s=_env_float(TIMEOUT_ENV, base.timeout_s, 1.0),
            attempts=int(_env_float(ATTEMPTS_ENV, base.attempts, 1.0)),
            backoff_s=_env_float(BACKOFF_ENV, base.backoff_s, 0.0),
            cooldown_s=_env_float(COOLDOWN_ENV, base.cooldown_s, 0.0),
        )

    def backoff_after(self, attempt: int) -> float:
        return self.backoff_s * 2 ** (attempt - 1)

    def worst_case_s(self) -> float:
        """Longest one model can hold the caller: every attempt times out."""
        pauses = sum(self.backoff_after(n) for n in range(1, self.attempts))
        return self.attempts * self.timeout_s + pauses


def call_with_deadline(fn: Callable[[], T], timeout_s: float, label: str) -> T:
    """Run ``fn`` in a daemon thread; raise :class:`ModelLoadTimeout` past the deadline."""
    outcome: dict[str, object] = {}

    def _run() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised in the caller's thread
            outcome["error"] = exc

    worker = threading.Thread(target=_run, name=f"model-load:{label}", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise ModelLoadTimeout(f"{label}: no result after {timeout_s:.0f}s (attempt abandoned)")
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["value"]  # type: ignore[return-value]


_cooling_until: dict[str, float] = {}
_cooling_lock = threading.Lock()


def reset_cooldowns() -> None:
    """Forget every recorded failure (tests; an operator reload may too)."""
    with _cooling_lock:
        _cooling_until.clear()


def _refuse_while_cooling(key: str, now: float) -> None:
    with _cooling_lock:
        until = _cooling_until.get(key, 0.0)
    if now < until:
        raise ModelLoadError(
            f"{key}: failed to load {until - now:.0f}s before the cooldown ends; not retrying yet"
        )


def _attempt_all(key: str, fn: Callable[[], T], policy: LoadPolicy,
                 sleep: Callable[[float], None]) -> T:
    last: BaseException | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return call_with_deadline(fn, policy.timeout_s, key)
        except Exception as exc:  # noqa: BLE001 — every failure counts against the bound
            last = exc
            logger.warning("model load %s: attempt %d/%d failed: %s",
                           key, attempt, policy.attempts, exc)
            if attempt < policy.attempts:
                sleep(policy.backoff_after(attempt))
    raise ModelLoadError(f"{key}: not loaded after {policy.attempts} attempts: {last}") from last


def load_bounded(key: str, fn: Callable[[], T], policy: LoadPolicy | None = None, *,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> T:
    """Load ``key`` via ``fn`` within ``policy``'s bound, or raise :class:`ModelLoadError`."""
    policy = policy or LoadPolicy.from_env()
    _refuse_while_cooling(key, clock())
    try:
        value = _attempt_all(key, fn, policy, sleep)
    except ModelLoadError:
        with _cooling_lock:
            _cooling_until[key] = clock() + policy.cooldown_s
        raise
    with _cooling_lock:
        _cooling_until.pop(key, None)
    return value
