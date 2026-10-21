"""When a built-in step may answer with a mock instead of failing.

A model call that fails fails its step. Before 2026-10-21 ``LLMStep``,
``VisionStep`` and ``AudioStep`` answered a failed call (no ``api_base``, a
refused connection, an HTTP error) with a mock and the step succeeded, so a
production pipeline reported a result nobody computed (ops-watch #520: every
LLM step of a server-side workflow on umrs-server-ai4s got
``[mock LLM response]`` and the run read as succeeded).

Mocks stay available for tests and local development, by explicit opt-in:

- ``UMRS_LLM_MOCK`` / ``UMRS_VISION_MOCK`` / ``UMRS_AUDIO_MOCK`` ``=1``: the
  step never calls a model and always answers with its mock (unchanged);
- ``UMRS_STEP_MOCK_FALLBACK=1``: a failed call falls back to the mock, which
  says so (``_mock`` / ``mock: true`` and the error). Never on the staged HPC
  lane, which has no network (ADR-074 D-5): there a failed call always fails.
"""

from __future__ import annotations

import os

from umrs.pipeline_sdk.staged_s3 import is_staged

MOCK_FALLBACK_ENV = "UMRS_STEP_MOCK_FALLBACK"

_TRUE = frozenset({"1", "true", "yes", "on"})


def mock_fallback_allowed() -> bool:
    """True when a failed model call may answer with a mock (opt-in, never staged)."""
    if is_staged():
        return False
    return os.environ.get(MOCK_FALLBACK_ENV, "").strip().lower() in _TRUE


def no_mock_reason(step: str, error: object) -> str:
    """The message a step fails with when its model call failed and no mock
    is allowed: names the cause and the opt-in, so the operator can act."""
    return (
        f"{step}: the model call failed ({error}); no mock answer is returned "
        f"(set {MOCK_FALLBACK_ENV}=1 only in development or tests)"
    )


__all__ = ["MOCK_FALLBACK_ENV", "mock_fallback_allowed", "no_mock_reason"]
