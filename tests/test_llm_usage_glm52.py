"""Tests for LLM usage extraction with thinking models (GLM-5.2).

Verifies that ``_parse_usage_dict`` correctly extracts:
- ``reasoning_tokens`` from ``completion_tokens_details``
- ``audio_tokens`` from ``completion_tokens_details`` / ``prompt_tokens_details``
- ``accepted/rejected_prediction_tokens`` (speculative decoding)
- ``cached_read_tokens`` from ``prompt_tokens_details.cached_tokens``
- ``cached_write_tokens`` from ``cache_creation_input_tokens``

For thinking models like GLM-5.2, ``completion_tokens`` INCLUDES
``reasoning_tokens``, so both values must be recorded for accurate
billing.
"""

from __future__ import annotations

import json

from umrs.api.llm_usage import (
    UsageRecord,
    _parse_usage_dict,
    extract_usage_oneshot,
    extract_usage_streaming,
    fill_usage,
)


class TestParseUsageDict:
    """Test ``_parse_usage_dict`` for various provider formats."""

    def test_glm52_thinking_model(self):
        """GLM-5.2 usage block with reasoning tokens.

        GLM-5.2 is a thinking model served by vLLM. The usage block
        looks like:
        ```json
        {
          "prompt_tokens": 10,
          "total_tokens": 100,
          "completion_tokens": 90,
          "completion_tokens_details": {
            "reasoning_tokens": 50
          },
          "prompt_tokens_details": {
            "cached_tokens": 5
          }
        }
        ```
        """
        usage = {
            "prompt_tokens": 10,
            "total_tokens": 100,
            "completion_tokens": 90,
            "completion_tokens_details": {
                "reasoning_tokens": 50,
            },
            "prompt_tokens_details": {
                "cached_tokens": 5,
            },
        }

        result = _parse_usage_dict(usage)

        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 90
        assert result["total_tokens"] == 100
        # Reasoning tokens extracted from completion_tokens_details
        assert result["reasoning_tokens"] == 50
        # Visible response tokens = completion - reasoning = 40
        visible = result["completion_tokens"] - result["reasoning_tokens"]
        assert visible == 40
        # Cached read tokens extracted from prompt_tokens_details
        assert result["cached_read_tokens"] == 5

    def test_deepseek_v4_pro_thinking_model(self):
        """DeepSeek-V4-Pro usage block with reasoning tokens.

        DeepSeek-V4-Pro is a thinking model that reports
        ``reasoning_tokens`` in ``completion_tokens_details``.
        """
        usage = {
            "prompt_tokens": 15,
            "total_tokens": 115,
            "completion_tokens": 100,
            "completion_tokens_details": {
                "reasoning_tokens": 60,
            },
        }

        result = _parse_usage_dict(usage)

        assert result["completion_tokens"] == 100
        assert result["reasoning_tokens"] == 60
        assert result["cached_read_tokens"] == 0

    def test_standard_openai_format(self):
        """Standard OpenAI usage block without reasoning tokens."""
        usage = {
            "prompt_tokens": 10,
            "total_tokens": 30,
            "completion_tokens": 20,
        }

        result = _parse_usage_dict(usage)

        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 20
        assert result["total_tokens"] == 30
        assert result["reasoning_tokens"] == 0
        assert result["audio_tokens"] == 0

    def test_anthropic_format(self):
        """Anthropic-style usage block with cache tokens."""
        usage = {
            "prompt_tokens": 100,
            "total_tokens": 200,
            "completion_tokens": 100,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 20,
        }

        result = _parse_usage_dict(usage)

        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 100
        assert result["cached_read_tokens"] == 50
        assert result["cached_write_tokens"] == 20

    def test_multimodal_audio_tokens(self):
        """Multimodal model with audio tokens.

        Some models (e.g., GLM-4-Voice) report audio tokens in
        ``completion_tokens_details.audio_tokens``.
        """
        usage = {
            "prompt_tokens": 10,
            "total_tokens": 50,
            "completion_tokens": 40,
            "completion_tokens_details": {
                "audio_tokens": 20,
            },
        }

        result = _parse_usage_dict(usage)

        assert result["audio_tokens"] == 20

    def test_vllm_null_sub_dicts(self):
        """vLLM may return ``null`` for ``prompt_tokens_details``.

        This should not cause a ``TypeError`` — the code coerces
        ``None`` to ``{}`` before indexing.
        """
        usage = {
            "prompt_tokens": 10,
            "total_tokens": 30,
            "completion_tokens": 20,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }

        result = _parse_usage_dict(usage)

        assert result["reasoning_tokens"] == 0
        assert result["cached_read_tokens"] == 0
        assert result["audio_tokens"] == 0

    def test_vllm_speculative_decoding(self):
        """vLLM speculative decoding stats.

        vLLM with ``speculative_config`` reports accepted/rejected
        prediction tokens.
        """
        usage = {
            "prompt_tokens": 10,
            "total_tokens": 50,
            "completion_tokens": 40,
            "accepted_prediction_tokens": 30,
            "rejected_prediction_tokens": 10,
        }

        result = _parse_usage_dict(usage)

        assert result["accepted_prediction_tokens"] == 30
        assert result["rejected_prediction_tokens"] == 10

    def test_anthropic_cache_creation(self):
        """Anthropic cache creation tokens."""
        usage = {
            "prompt_tokens": 100,
            "total_tokens": 200,
            "completion_tokens": 100,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 30,
        }

        result = _parse_usage_dict(usage)

        assert result["cached_read_tokens"] == 30
        assert result["cached_write_tokens"] == 50

    def test_glm52_full_usage_with_cache(self):
        """Full GLM-5.2 usage block with cache tokens.

        GLM-5.2 (zai-org/GLM-5.2) is a thinking model that uses
        ``completion_tokens_details.reasoning_tokens`` for thinking
        tokens and ``prompt_tokens_details.cached_tokens`` for cache
        hits.
        """
        usage = {
            "prompt_tokens": 1000,
            "total_tokens": 5000,
            "completion_tokens": 4000,
            "completion_tokens_details": {
                "reasoning_tokens": 3000,
            },
            "prompt_tokens_details": {
                "cached_tokens": 800,
            },
        }

        result = _parse_usage_dict(usage)

        assert result["prompt_tokens"] == 1000
        assert result["completion_tokens"] == 4000
        assert result["total_tokens"] == 5000
        # Reasoning tokens: 3000 (subset of completion_tokens)
        assert result["reasoning_tokens"] == 3000
        # Visible response tokens: 4000 - 3000 = 1000
        visible = result["completion_tokens"] - result["reasoning_tokens"]
        assert visible == 1000
        # Cache hit tokens: 800 (subset of prompt_tokens)
        assert result["cached_read_tokens"] == 800
        # Non-cached prompt tokens: 1000 - 800 = 200
        non_cached = result["prompt_tokens"] - result["cached_read_tokens"]
        assert non_cached == 200


class TestExtractUsageOneshot:
    """Test ``extract_usage_oneshot`` for oneshot (non-streaming) responses."""

    def test_glm52_oneshot_response(self):
        """Extract usage from a GLM-5.2 oneshot response."""
        response_body = json.dumps({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "model": "zai-org/GLM-5.2",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 100,
                "completion_tokens": 90,
                "completion_tokens_details": {
                    "reasoning_tokens": 50,
                },
                "prompt_tokens_details": {
                    "cached_tokens": 5,
                },
            },
        }).encode()

        result = extract_usage_oneshot(response_body)

        assert result is not None
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 90
        assert result["reasoning_tokens"] == 50
        assert result["cached_read_tokens"] == 5

    def test_error_response_returns_none(self):
        """Error responses should return ``None``."""
        response_body = json.dumps({"error": "Internal error"}).encode()

        result = extract_usage_oneshot(response_body)

        assert result is None

    def test_model_list_response_returns_none(self):
        """Model list responses should return ``None``."""
        response_body = json.dumps({
            "object": "list",
            "data": [{"id": "model-1"}],
        }).encode()

        result = extract_usage_oneshot(response_body)

        assert result is None


class TestExtractUsageStreaming:
    """Test ``extract_usage_streaming`` for SSE streaming responses."""

    def test_glm52_streaming_response(self):
        """Extract usage from a GLM-5.2 streaming response.

        The final SSE chunk (before ``[DONE]``) should contain the
        ``usage`` block with reasoning tokens.
        """
        chunks = [
            b'data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b'data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" there"}}]}\n\n',
            b'data: {"id":"chatcmpl-abc","choices":[],"usage":{"prompt_tokens":10,"total_tokens":100,"completion_tokens":90,"completion_tokens_details":{"reasoning_tokens":50},"prompt_tokens_details":{"cached_tokens":5}}}\n\n',
            b'data: [DONE]\n\n',
        ]

        result = extract_usage_streaming(chunks)

        assert result is not None
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 90
        assert result["reasoning_tokens"] == 50
        assert result["cached_read_tokens"] == 5

    def test_streaming_no_usage_chunk(self):
        """If no usage chunk is present, return ``None``."""
        chunks = [
            b'data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        result = extract_usage_streaming(chunks)

        assert result is None

    def test_streaming_with_reasoning_content(self):
        """Streaming response with reasoning content (GLM-5.2 thinking model).

        GLM-5.2 streams ``reasoning_content`` separately from ``content``
        in delta chunks. The final usage block should still capture
        reasoning_tokens correctly.
        """
        chunks = [
            b'data: {"id":"chatcmpl-abc","choices":[{"delta":{"reasoning_content":"Let me think..."}}]}\n\n',
            b'data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"The answer is 42."}}]}\n\n',
            b'data: {"id":"chatcmpl-abc","choices":[],"usage":{"prompt_tokens":10,"total_tokens":100,"completion_tokens":90,"completion_tokens_details":{"reasoning_tokens":60}}}\n\n',
            b'data: [DONE]\n\n',
        ]

        result = extract_usage_streaming(chunks)

        assert result is not None
        assert result["reasoning_tokens"] == 60
        # Visible response tokens: 90 - 60 = 30
        visible = result["completion_tokens"] - result["reasoning_tokens"]
        assert visible == 30


class TestFillUsage:
    """Test ``fill_usage`` correctly populates ``UsageRecord``."""

    def test_fill_usage_with_reasoning_tokens(self):
        """``fill_usage`` should copy reasoning_tokens into the record."""
        record = UsageRecord(
            user_id="user-1",
            project_id="project-1",
            model="glm-5.2",
        )
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 90,
            "total_tokens": 100,
            "cached_read_tokens": 5,
            "cached_write_tokens": 0,
            "reasoning_tokens": 50,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }

        fill_usage(record, usage)

        assert record.prompt_tokens == 10
        assert record.completion_tokens == 90
        assert record.total_tokens == 100
        assert record.cached_read_tokens == 5
        assert record.cached_write_tokens == 0
        assert record.reasoning_tokens == 50
        assert record.audio_tokens == 0
        assert record.accepted_prediction_tokens == 0
        assert record.rejected_prediction_tokens == 0

    def test_fill_usage_defaults_to_zero(self):
        """``fill_usage`` should default missing values to 0."""
        record = UsageRecord(
            user_id="user-1",
            project_id="project-1",
            model="qwen3-8b",
        )
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cached_read_tokens": 0,
            "cached_write_tokens": 0,
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }

        fill_usage(record, usage)

        assert record.reasoning_tokens == 0
        assert record.audio_tokens == 0


class TestUsageRecordDefaults:
    """Test ``UsageRecord`` dataclass defaults."""

    def test_usage_record_defaults(self):
        """New fields should default to 0."""
        record = UsageRecord(
            user_id="user-1",
            project_id="project-1",
            model="glm-5.2",
        )

        assert record.prompt_tokens == 0
        assert record.completion_tokens == 0
        assert record.total_tokens == 0
        assert record.cached_read_tokens == 0
        assert record.cached_write_tokens == 0
        assert record.reasoning_tokens == 0
        assert record.audio_tokens == 0
        assert record.accepted_prediction_tokens == 0
        assert record.rejected_prediction_tokens == 0
