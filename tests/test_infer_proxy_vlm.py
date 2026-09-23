"""D-10: the VLM path — image inputs ride the LLM proxy verbatim.

The proxy is a credential-injecting byte-stream forwarder; multimodal
chat bodies (OpenAI-compatible image_url content parts) must reach the
upstream untouched — URL references and base64 data: URIs alike.
"""

from __future__ import annotations

import json

from tests.test_infer_proxy import _app, _request


def _image_chat_body() -> bytes:
    return json.dumps(
        {
            "model": "vlm-8b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this figure?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/fig.png"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aGVsbG8gdmlzaW9u"},
                        },
                    ],
                }
            ],
            "max_tokens": 512,
        }
    ).encode("utf-8")


def test_multimodal_chat_body_forwards_verbatim(monkeypatch):
    app, seen, _broker = _app(monkeypatch)
    body = _image_chat_body()
    resp = _request(
        app,
        "POST",
        "/s/sess-1/v1/chat/completions",
        content=body,
        headers={
            "authorization": "Bearer tok-session",
            "content-type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert seen["body"] == body  # byte-identical — both image forms untouched


def test_multimodal_parts_survive_json_roundtrip(monkeypatch):
    app, seen, _broker = _app(monkeypatch)
    body = _image_chat_body()
    _request(
        app,
        "POST",
        "/s/sess-1/v1/chat/completions",
        content=body,
        headers={
            "authorization": "Bearer tok-session",
            "content-type": "application/json",
        },
    )
    forwarded = json.loads(seen["body"])
    parts = forwarded["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[1]["image_url"]["url"] == "https://example.com/fig.png"
    assert parts[2]["image_url"]["url"].startswith("data:image/png;base64,")
