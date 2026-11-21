"""U3 S6a: the actor and scope headers fail closed under both names.

Before, core read only ``x-r4-actor`` and ``x-r4-scope``: a request that
sent ``x-umrs-actor: agent`` alone got the human tier on knowledge writes,
and a scope token sent as ``x-umrs-scope`` left the request unscoped. Both
restrictions now hold whichever name carries them (u3a map §8, "a
header-alias fail-closed test covering both names").
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from umrs.api._knowledge_mutation_guard import request_actor, request_session_id
from umrs.api.scope_tokens import _scope_token
from umrs.knowledge.actors import AGENT_ACTOR, HUMAN_ACTOR


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw})


@pytest.mark.parametrize(
    ("headers", "actor"),
    [
        ({}, HUMAN_ACTOR),
        ({"x-r4-actor": "user"}, HUMAN_ACTOR),
        ({"x-umrs-actor": "user"}, HUMAN_ACTOR),
        ({"x-r4-actor": "agent"}, AGENT_ACTOR),
        ({"x-umrs-actor": "agent"}, AGENT_ACTOR),  # human tier on the old code
        ({"x-umrs-actor": "user", "x-r4-actor": "companion"}, AGENT_ACTOR),
        ({"x-umrs-actor": "loop", "x-r4-actor": "user"}, AGENT_ACTOR),
    ],
)
def test_the_actor_fails_closed_over_both_names(headers, actor):
    assert request_actor(_request(headers)) == actor


def test_the_session_id_is_read_under_either_name():
    assert request_session_id(_request({"x-umrs-session-id": " s-1 "})) == "s-1"
    assert request_session_id(_request({"x-r4-session-id": "s-2"})) == "s-2"
    assert request_session_id(_request({})) == ""


def test_a_scope_token_counts_under_either_name():
    assert _scope_token(_request({"x-umrs-scope": "tok"})) == "tok"  # "" on the old code
    assert _scope_token(_request({"x-r4-scope": "tok"})) == "tok"
    assert _scope_token(_request({"x-umrs-scope": "tok", "x-r4-scope": "tok"})) == "tok"
    assert _scope_token(_request({})) == ""


def test_two_different_scope_tokens_are_refused():
    with pytest.raises(HTTPException) as caught:
        _scope_token(_request({"x-umrs-scope": "a", "x-r4-scope": "b"}))
    assert caught.value.status_code == 400
