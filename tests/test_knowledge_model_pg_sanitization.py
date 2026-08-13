"""Regression: postgres jsonb must never reject content for NUL bytes.

Live crash (2026-08-13): knowledge→understand→persist crashed with
``psycopg2.errors.UntranslatableCharacter`` when user payload carried raw NUL
bytes inside extracted knowledge content. `_sanitize_pg_json` is the
contract-level guard at every jsonb insert path of pg_store.
"""

from __future__ import annotations

from umrs.knowledge_model.pg_store import _sanitize_pg_json


def test_nul_replaced_recursively():
    dirty = {
        "payload\x00key": "val\x00ue",
        "nested": [{"a\x00": ["b\x00c", 1, None]}, {"ok": "fine"}],
        "num": 5,
        "none": None,
        "bool": True,
    }
    out = _sanitize_pg_json(dirty)
    assert out["payload\ufffdkey"] == "val\ufffdue"
    assert out["nested"][0]["a\ufffd"][0] == "b\ufffdc"
    assert out["num"] == 5
    assert out["none"] is None
    assert out["bool"] is True


def test_clean_passthrough_unchanged():
    clean = {"a": "b", "c": [1, 2]}

    assert _sanitize_pg_json(clean) == clean


def test_scalar_identity():
    assert _sanitize_pg_json(3) == 3
    assert _sanitize_pg_json(None) is None
    assert _sanitize_pg_json("x\x00y") == "x\ufffdy"
