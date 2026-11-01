"""connect-info advertises the service-tier default models (ADR-106 D-9, plan S10).

R4 reads ``default_models`` at login and uses the role aliases when the user
named no model. Only aliases this front's roster actually carries are
advertised, so a client is never pointed at a role nobody serves.
"""

from __future__ import annotations

import pytest
from umrs.api import ops
from umrs.orchestrator import provision_roster

ROSTER = """\
pool: test
endpoints:
  - name: glm
    model: glm-5.3-flash
    aliases: [svc-chat]
  - name: nimble
    model: Bespoke-Nimble-9B
    aliases: [svc-decision, nimble]
  - name: plain
    model: some-model
"""


@pytest.fixture()
def roster(tmp_path, monkeypatch):
    path = tmp_path / "pool.yaml"
    path.write_text(ROSTER)
    monkeypatch.setenv(provision_roster.ROSTER_ENV_VAR, str(path))
    monkeypatch.delenv("UMRS_DEFAULT_MODEL_FALLBACK", raising=False)
    provision_roster._cache.clear()
    yield
    provision_roster._cache.clear()


def test_only_served_roles_are_advertised(roster) -> None:
    assert ops.connect_info_default_models() == {"chat": "svc-chat", "decision": "svc-decision"}


def test_fallback_order_comes_from_the_environment(roster, monkeypatch) -> None:
    monkeypatch.setenv("UMRS_DEFAULT_MODEL_FALLBACK", "svc-fast, kimi-k3@rikyu ,")
    assert ops.connect_info_default_models()["fallback"] == ["svc-fast", "kimi-k3@rikyu"]


def test_connect_info_carries_the_map_through_the_response_model(roster) -> None:
    result = ops.ops_connect_info(x_user_id=None)
    dumped = ops.ConnectInfoResponse(**result).model_dump()
    assert dumped["default_models"] == {"chat": "svc-chat", "decision": "svc-decision"}


def test_no_role_alias_means_no_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(provision_roster.ROSTER_ENV_VAR, raising=False)
    monkeypatch.delenv("UMRS_DEFAULT_MODEL_FALLBACK", raising=False)
    provision_roster._cache.clear()
    assert "default_models" not in ops.ops_connect_info(x_user_id=None)
