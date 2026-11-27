"""Opaque list cursors (ADR-105 D-6 S6a, task a105-s6a-02)."""

from __future__ import annotations

import base64
import hashlib
import json
import re

import pytest
from umrs.api.read_cursor import (
    MALFORMED,
    MAX_CURSOR_LEN,
    OTHER_QUERY,
    CursorError,
    decode_cursor,
    encode_cursor,
    scope_key,
)

POS = {"c": "2026-11-20T08:15:30.123456", "j": "job-b"}


def _raw(obj: object) -> str:
    """A token for an arbitrary JSON payload (built the documented way)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def test_cursor_error_is_a_value_error() -> None:
    assert issubclass(CursorError, ValueError)


def test_scope_key_is_16_hex_of_the_canonical_json() -> None:
    filters = {"op": "job_list", "project_id": "p", "user_id": None}
    expected = hashlib.sha256(b'{"op":"job_list","project_id":"p"}').hexdigest()[:16]
    assert scope_key(filters) == expected
    assert re.fullmatch(r"[0-9a-f]{16}", scope_key({}))


def test_scope_key_ignores_order_and_none() -> None:
    a = scope_key({"a": "1", "b": "2", "c": None})
    b = scope_key({"b": "2", "a": "1"})
    assert a == b


def test_scope_key_changes_with_any_value() -> None:
    base = scope_key({"user_id": "alice"})
    assert scope_key({"user_id": "bob"}) != base
    assert scope_key({"user_id": ""}) != scope_key({})
    assert scope_key({"user_id": "alice", "status": "running"}) != base


def test_round_trip() -> None:
    token = encode_cursor(POS, "k1")
    assert decode_cursor(token, "k1") == POS


def test_encoding_is_the_documented_format() -> None:
    token = encode_cursor(POS, "k1")
    assert token == _raw({"k": "k1", "p": POS, "v": 1})
    assert re.fullmatch(r"[A-Za-z0-9_-]+", token)
    assert "=" not in token


def test_decode_returns_a_new_dict() -> None:
    token = encode_cursor(POS, "k1")
    out = decode_cursor(token, "k1")
    assert isinstance(out, dict)
    out["j"] = "changed"
    assert decode_cursor(token, "k1") == POS


def test_another_scope_is_refused() -> None:
    token = encode_cursor(POS, "k1")
    with pytest.raises(CursorError) as exc:
        decode_cursor(token, "k2")
    assert str(exc.value) == OTHER_QUERY == "cursor belongs to another query"


@pytest.mark.parametrize(
    "position",
    [{}, {"c": 1}, {1: "x"}, {"c": None}],
)
def test_encode_rejects_bad_positions(position: dict) -> None:
    with pytest.raises(CursorError, match=f"^{MALFORMED}$"):
        encode_cursor(position, "k1")


def test_encode_rejects_empty_scope() -> None:
    with pytest.raises(CursorError, match=f"^{MALFORMED}$"):
        encode_cursor(POS, "")


_BAD_TOKENS = [
    "",
    "a" * (MAX_CURSOR_LEN + 1),
    "abc+def",
    "abc/def",
    "abc=",
    "ab c",
    "Zm9v\n",
    "a",  # not decodable base64 (1 char)
    base64.urlsafe_b64encode(b"\xff\xfe\xfd").decode().rstrip("="),  # not UTF-8
    base64.urlsafe_b64encode(b"not json").decode().rstrip("="),
    _raw([1, 2]),
    _raw("text"),
    _raw({"k": "k1", "p": POS}),
    _raw({"k": "k1", "p": POS, "v": 1, "x": 0}),
    _raw({"k": "k1", "p": POS, "v": 2}),
    _raw({"k": "k1", "p": POS, "v": True}),
    _raw({"k": "k1", "p": POS, "v": "1"}),
    _raw({"k": 7, "p": POS, "v": 1}),
    _raw({"k": "k1", "p": {}, "v": 1}),
    _raw({"k": "k1", "p": [1], "v": 1}),
    _raw({"k": "k1", "p": {"c": 1}, "v": 1}),
    _raw({"k": "k1", "p": {"c": None}, "v": 1}),
]


@pytest.mark.parametrize("token", _BAD_TOKENS)
def test_decode_rejects_malformed(token: str) -> None:
    with pytest.raises(CursorError) as exc:
        decode_cursor(token, "k1")
    assert str(exc.value) == MALFORMED == "cursor is malformed"


@pytest.mark.parametrize("token", [None, 5, b"abc"])
def test_decode_rejects_non_strings(token: object) -> None:
    with pytest.raises(CursorError, match=f"^{MALFORMED}$"):
        decode_cursor(token, "k1")  # type: ignore[arg-type]


def test_max_length_token_is_not_refused_for_length() -> None:
    """Exactly MAX_CURSOR_LEN characters is allowed (then judged on content)."""
    long_pos = {"c": "x" * 300}
    token = encode_cursor(long_pos, "k1")
    assert len(token) <= MAX_CURSOR_LEN
    assert decode_cursor(token, "k1") == long_pos
