"""U3 S1 item 11: neutral UMRS names with R4 fallbacks, failing closed.

Env and header values read the ``UMRS_*`` / ``x-umrs-*`` name first, then
the R4 name. The actor and scope headers restrict a request, so either name
being set restricts it; sending the other name cannot shed a restriction.
"""

from __future__ import annotations

import pytest
from umrs.shared import compat_names as cn


def test_env_prefers_the_umrs_name():
    environ = {"UMRS_GATEWAY_URL": "https://u", "R4_GATEWAY_URL": "https://r"}
    assert cn.env("GATEWAY_URL", environ=environ) == "https://u"
    assert cn.env("GATEWAY_URL", environ={"R4_GATEWAY_URL": "https://r"}) == "https://r"
    assert cn.env("GATEWAY_URL", "d", environ={"UMRS_GATEWAY_URL": ""}) == "d"


def test_header_prefers_the_umrs_name_case_insensitively():
    assert cn.header({"X-UMRS-Session-Id": "a", "x-r4-session-id": "b"}, "session-id") == "a"
    assert cn.header({"x-r4-session-id": "b"}, "session-id") == "b"
    assert cn.header({}, "session-id", "none") == "none"


@pytest.mark.parametrize(
    ("headers", "agent"),
    [
        ({}, False),
        ({"x-r4-actor": "user"}, False),
        ({"x-umrs-actor": ""}, False),
        ({"x-r4-actor": "agent"}, True),
        ({"x-umrs-actor": "agent"}, True),
        # fail closed: either name restricts, the other cannot clear it
        ({"x-umrs-actor": "user", "x-r4-actor": "agent"}, True),
        ({"x-umrs-actor": "agent", "x-r4-actor": "user"}, True),
        # only "" or "user" is the person (umrs.knowledge.actors)
        ({"x-r4-actor": "human"}, True),
    ],
)
def test_agent_request_fails_closed(headers, agent):
    assert cn.is_agent_request(headers) is agent


@pytest.mark.parametrize(
    ("headers", "scoped"),
    [
        ({}, False),
        ({"x-r4-scope": "  "}, False),
        ({"x-r4-scope": "tok"}, True),
        ({"x-umrs-scope": "tok", "x-r4-scope": ""}, True),
        ({"x-umrs-scope": "", "x-r4-scope": "tok"}, True),
    ],
)
def test_scoped_request_fails_closed(headers, scoped):
    assert cn.is_scoped_request(headers) is scoped


@pytest.mark.parametrize(
    ("headers", "refused"),
    [
        ({}, False),
        ({"x-r4-actor": "user"}, False),
        ({"x-r4-actor": "agent"}, True),
        ({"x-umrs-actor": "agent"}, True),
        ({"x-umrs-actor": "user", "x-r4-actor": "agent"}, True),
        ({"x-umrs-scope": "tok"}, True),
    ],
)
def test_hpc_account_person_check_uses_the_fail_closed_rule(headers, refused):
    """The first caller of compat_names: the HPC account trust-anchor guard."""
    from types import SimpleNamespace

    from umrs.api import hpc_accounts

    request = SimpleNamespace(headers=headers)
    if not refused:
        assert hpc_accounts._require_person(request) is None
        return
    with pytest.raises(Exception, match="person_required|own act"):
        hpc_accounts._require_person(request)
