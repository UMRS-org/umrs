"""A gated run's durable job key fits ``jobs.idempotency_key`` (2026-11-14).

The first live signed run (bdrt, maintainer) failed with ``value too long for
type character varying(64)``: the key was ``intent:<64-hex intent id>``.
"""

from __future__ import annotations

from umrs.server import workflow_registry_handler as handler

_INTENT_ID = "af7f5d859ec8442923846c659963801b61e3ef4a340c06e86f68e085d37ebab9"


def test_an_intent_key_fits_the_column_and_is_stable() -> None:
    spec: dict = {"execution_id": "wf-ddef205bdfd1"}
    intent = {"b64": "x", "signature": "y", "id": _INTENT_ID}
    first = handler._stamp_intent(spec, intent, None)
    again = handler._stamp_intent({"execution_id": "wf-other"}, dict(intent), None)
    assert first is not None and len(first) <= 64
    assert first == again  # a second trigger of one intent finds its job
    assert spec["intent"]["id"] == _INTENT_ID


def test_distinct_intents_get_distinct_keys() -> None:
    other = "0" * 64
    a = handler._stamp_intent({}, {"id": _INTENT_ID}, None)
    b = handler._stamp_intent({}, {"id": other}, None)
    assert a != b


def test_a_grant_run_key_fits_the_column() -> None:
    spec = {"execution_id": "wf-" + "e" * 40}
    key = handler._stamp_intent(spec, {"id": "", "grant_id": "g" * 32}, None)
    assert key is not None and len(key) <= 64


def test_a_given_key_and_an_unsigned_run_are_unchanged() -> None:
    assert handler._stamp_intent({}, {"id": _INTENT_ID}, "plan-123") == "plan-123"
    assert handler._stamp_intent({}, None, "plan-123") == "plan-123"
    assert handler._stamp_intent({}, None, None) is None
