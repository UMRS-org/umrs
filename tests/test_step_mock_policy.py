"""A failed model call fails a built-in step; a mock answer is opt-in only.

ops-watch #520 (2026-10-21): on umrs-server-ai4s every ``LLMStep`` of a
server-side workflow logged "LLMStep requires api_base ... returning mock
response" and the run read as succeeded, carrying ``[mock LLM response]``.
"""

from __future__ import annotations

import pytest
from umrs.pipeline_sdk.mock_policy import MOCK_FALLBACK_ENV
from umrs.pipeline_sdk.step import StepContext

_MODEL_ENV = ("OPENAI_BASE_URL", "R4_GATEWAY_URL", "OPENAI_API_KEY")


@pytest.fixture
def no_model_env(monkeypatch):
    """No api_base anywhere, no always-mock switch, not staged, no opt-in."""
    for name in (
        *_MODEL_ENV,
        "UMRS_LLM_MOCK",
        "UMRS_VISION_MOCK",
        "UMRS_AUDIO_MOCK",
        "UMRS_STEP_STAGED",
        MOCK_FALLBACK_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _ctx() -> StepContext:
    return StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output")


def _llm(config: dict | None = None):
    from umrs.pipeline_sdk.templates.llm_step import LLMStep

    step = LLMStep()
    step.setup({"prompt_template": "hello", **(config or {})})
    return step


def test_llm_step_without_api_base_fails_by_name(no_model_env):
    with pytest.raises(RuntimeError, match=r"requires api_base.*no mock answer is returned"):
        _llm().run(_ctx())


def test_llm_step_with_an_unreachable_model_fails(no_model_env):
    step = _llm({"api_base": "http://127.0.0.1:9/v1", "timeout_sec": 2})
    with pytest.raises(RuntimeError, match=MOCK_FALLBACK_ENV):
        step.run(_ctx())


@pytest.mark.parametrize("value", ["1", "true"])
def test_llm_step_mock_fallback_is_an_explicit_opt_in(no_model_env, value):
    no_model_env.setenv(MOCK_FALLBACK_ENV, value)
    extra = _llm({"output_schema": {"type": "object"}}).run(_ctx()).extra
    assert '"_mock": true' in extra["response_text"]
    assert "requires api_base" in extra["response_text"]  # the mock says why


def test_the_opt_in_never_applies_on_the_staged_lane(no_model_env):
    no_model_env.setenv(MOCK_FALLBACK_ENV, "1")
    no_model_env.setenv("UMRS_STEP_STAGED", "1")
    with pytest.raises(RuntimeError, match="staged HPC lane"):
        _llm().run(_ctx())


def test_the_always_mock_switch_is_unchanged(no_model_env):
    no_model_env.setenv("UMRS_LLM_MOCK", "1")
    assert _llm().run(_ctx()).extra["response_text"].startswith("[mock LLM response]")


def test_vision_step_without_api_base_fails(no_model_env):
    from umrs.pipeline_sdk.templates import vision_step

    step = vision_step.VisionStep()
    step.setup({"mode": "caption"})
    image = vision_step._ImageRef(bytes_data=b"\x89PNG")
    with pytest.raises(RuntimeError, match=r"VisionStep: .*no mock answer is returned"):
        step._invoke(None, image)
    no_model_env.setenv(MOCK_FALLBACK_ENV, "1")
    assert step._invoke(None, image).meta["mock"] is True


def test_audio_step_without_whisper_url_fails(no_model_env):
    from umrs.pipeline_sdk.templates.audio_step import AudioStep

    step = AudioStep()
    step.setup({})
    with pytest.raises(RuntimeError, match=r"requires whisper_url.*no mock answer"):
        step._transcribe(b"RIFF", "a.wav")
    no_model_env.setenv(MOCK_FALLBACK_ENV, "1")
    assert step._transcribe(b"RIFF", "a.wav")["mock"] is True
