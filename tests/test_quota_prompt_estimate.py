"""ADR-099 D-4, Amendment 3 (RS-1 round-3 review, finding N1): an image in
a cut stream's request is not charged as bytes / 4.

- the prompt estimate of a request counts text bytes / 4 and each base64
  run of ``BLOB_RUN`` characters or more (an inline image, a file) at most
  ``BLOB_TOKENS``; before, a 1.5 MB PNG data URI read as ~500k tokens and
  every interrupted turn that still carried it was charged that again;
- a run split across request chunks is still one run; short base64-looking
  words and text keep bytes / 4;
- through the proxy: a stream cut before its usage chunk, and one that ends
  without usage, charge an image request a few thousand tokens, not ~500k.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import pytest
import test_quota_proxy as tqp
from test_quota_metering_estimate import CONTENT, EXTRA, PROMPT, USAGE, _two_chunks
from test_quota_proxy import _app
from test_quota_proxy_metering import _disconnecting_call, _post
from umrs.quotas.usage_tap import BLOB_RUN, BLOB_TOKENS, PromptMeter, UsageTap

PNG = base64.b64encode(os.urandom(1_125_000)).decode()  # 1.5 MB of base64
TEXT = "Describe this picture in one sentence, please. " * 10


def _image_request(n_images: int = 1) -> dict:
    image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}
    content = [{"type": "text", "text": TEXT}] + [image] * n_images
    return {"model": "gpt-x", "stream": True, "messages": [{"role": "user", "content": content}]}


def _fed(body: bytes, step: int = 0) -> int:
    meter = PromptMeter()
    step = step or len(body)
    for i in range(0, len(body), step):
        meter.feed(body[i : i + step])
    return meter.tokens()


def _text_tokens(body: bytes) -> int:
    return -(-len(body) // 4)


def _one_row(pool):
    deadline = time.monotonic() + 5
    while not pool.inserts() and time.monotonic() < deadline:
        time.sleep(0.01)
    (row,) = pool.inserts()
    return row


class TestPromptMeter:
    def test_an_image_costs_a_fixed_amount_not_its_bytes(self):
        body = json.dumps(_image_request()).encode()
        tokens = _fed(body)
        assert tokens <= BLOB_TOKENS + _text_tokens(json.dumps(_image_request(0)).encode()) + 32
        assert tokens >= BLOB_TOKENS
        assert len(body) // 4 > 80 * tokens  # the old estimate: ~375k

    def test_each_image_counts(self):
        one = _fed(json.dumps(_image_request(1)).encode())
        two = _fed(json.dumps(_image_request(2)).encode())
        assert BLOB_TOKENS <= two - one <= BLOB_TOKENS + 32  # + the image's JSON wrapper

    @pytest.mark.parametrize("step", [1, 7, 1000, 65536])
    def test_a_run_split_across_chunks_is_one_run(self, step):
        body = json.dumps(_image_request()).encode()[:200_000]
        assert _fed(body, step) == _fed(body)

    def test_text_keeps_bytes_over_four(self):
        body = json.dumps({"messages": [{"content": TEXT * 50}]}).encode()
        assert _fed(body) == _text_tokens(body)

    def test_a_short_run_is_text(self):
        word = b"a" * (BLOB_RUN - 1)
        assert _fed(word) == _text_tokens(word)
        assert _fed(word + b"a") == _text_tokens(word + b"a")  # at the threshold
        uri = b"data:image/png;base64,"
        assert _fed(uri + b"a" * 100 * BLOB_RUN) == _text_tokens(uri) + BLOB_TOKENS

    def test_the_tap_takes_the_meters_prompt(self):
        tap = UsageTap(provider_kind="openai", streaming=True)
        tap.feed(CONTENT % b"hello")
        usage, estimated = tap.metered(request_bytes=2_000_000, generation=True, prompt_tokens=42)
        assert estimated and usage["prompt_tokens"] == 42


class TestThroughTheProxy:
    def test_a_cut_stream_with_an_image_is_not_charged_its_bytes(self, monkeypatch):
        monkeypatch.setattr(tqp, "_stream_response", _two_chunks(CONTENT % b"A", USAGE))
        app, pool, _broker, _calls = _app(monkeypatch, content_type="text/event-stream")
        asyncio.run(_disconnecting_call(app, _image_request()))
        row = _one_row(pool)
        assert json.loads(row[EXTRA])["usage"] == "estimated"
        assert BLOB_TOKENS <= row[PROMPT] <= BLOB_TOKENS + 1000  # before: ~375k

    def test_a_stream_ending_without_usage_with_an_image(self, monkeypatch):
        body = CONTENT % b"hi" + b"data: [DONE]\n\n"
        app, pool, _broker, _calls = _app(monkeypatch, body=body, content_type="text/event-stream")
        assert _post(app, _image_request()).status_code == 200
        row = _one_row(pool)
        assert BLOB_TOKENS <= row[PROMPT] <= BLOB_TOKENS + 1000


# ── Amendment 4 (RS-1 round-4 review, R4-1) ─────────────────────────────────
# ``json.dumps`` escapes every non-ASCII character as ``\uXXXX``, and every
# character of an escape was in the run alphabet: a Japanese message (no
# spaces, its punctuation escaped too) read as ONE run, charged at most
# BLOB_TOKENS (~4-5x under). Words joined by ``_`` or ``-`` did the same, so a
# cut stream could be under-charged on purpose. Only a data URI's run is a
# blob now, and an escape ends a run.

JA = "研究データの解析結果について、詳細に説明してください。" * 800  # 21,600 characters
JOINED = "_".join(["word"] * 40_000)  # 200 KB


def _text_request(text: str) -> dict:
    return {"model": "gpt-x", "stream": True, "messages": [{"role": "user", "content": text}]}


class TestAmendment4:
    def test_escaped_japanese_is_text(self):
        body = json.dumps(_text_request(JA)).encode()  # ensure_ascii: \uXXXX escapes
        assert _fed(body) == _text_tokens(body)  # before: 4,126
        utf8 = json.dumps(_text_request(JA), ensure_ascii=False).encode()
        assert _fed(utf8) == _text_tokens(utf8)

    @pytest.mark.parametrize("joiner", ["_", "-", "+", "/"])
    def test_joined_words_are_text(self, joiner):
        body = json.dumps(_text_request(JOINED.replace("_", joiner))).encode()
        assert _fed(body) == _text_tokens(body)  # before ("_", "-", ...): 4,126

    def test_base64_looking_text_without_a_data_uri_is_text(self):
        body = json.dumps(_text_request(PNG[:200_000])).encode()
        assert _fed(body) == _text_tokens(body)

    @pytest.mark.parametrize("step", [1, 3, 8, 9])
    def test_the_data_uri_marker_may_span_chunks(self, step):
        body = json.dumps(_image_request()).encode()[:200_000]
        assert _fed(body, step) == _fed(body) < _text_tokens(body) // 10

    def test_an_escaped_slash_stays_in_the_run(self):
        """A client whose JSON writes ``\\/`` still has one image run."""
        body = json.dumps(_image_request()).encode().replace(b"/", b"\\/")
        assert _fed(body) < BLOB_TOKENS + _text_tokens(json.dumps(_image_request(0)).encode()) + 64

    @pytest.mark.parametrize(
        "shape",
        [
            # Anthropic /v1/messages
            lambda b64: {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            }
                        ],
                    }
                ]
            },
            # Google generateContent
            lambda b64: {
                "contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": b64}}]}]
            },
            # OpenAI input_audio
            lambda b64: {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": {"format": "wav", "data": b64}}
                        ],
                    }
                ]
            },
        ],
        ids=["anthropic", "google", "input_audio"],
    )
    @pytest.mark.parametrize("separators", [(",", ":"), (", ", ": ")], ids=["compact", "spaced"])
    def test_a_json_data_blob_is_charged_as_one_image(self, shape, separators):
        """Final review F1 (2026-10-23): formats without a data URI were
        charged bytes / 4 again (~367k for a 1.1 MB PNG) after Amendment 4."""
        body = json.dumps(shape(PNG[:400_000]), separators=separators).encode()
        assert _fed(body) <= BLOB_TOKENS + 256
        assert _fed(body, 7) == _fed(body)  # the JSON marker may span chunks

    def test_a_cut_site_stream_with_a_japanese_prompt(self, monkeypatch):
        """Through the proxy: the re-encoded body (``include_usage`` added)
        is ``json.dumps``'d with its escapes."""
        monkeypatch.setattr(tqp, "_stream_response", _two_chunks(CONTENT % b"A", USAGE))
        app, pool, _broker, _calls = _app(monkeypatch, content_type="text/event-stream")
        asyncio.run(_disconnecting_call(app, _text_request(JA)))
        row = _one_row(pool)
        assert json.loads(row[EXTRA])["usage"] == "estimated"
        assert row[PROMPT] >= len(JA)  # before: ~4,100 for ~15-22k real tokens
