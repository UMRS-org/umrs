"""connect-info advertises the service-tier default models (ADR-106 D-9, plan S10).

R4 reads ``default_models`` at login and uses the role aliases when the user
named no model. A role is advertised only while one of its models has a
live (READY/DEGRADED) replica on this front. A roster entry that is not yet
staged, or a model that is down, never becomes anyone's default.
"""

from __future__ import annotations

import pytest
from umrs.api import ops
from umrs.orchestrator import endpoint_registry as er
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
  - name: embed
    model: Qwen3-Embedding-0.6B
    aliases: [svc-embed]
  - name: plain
    model: some-model
"""


def _registry(*live: str, down: tuple[str, ...] = ()) -> er.EndpointRegistry:
    registry = er.EndpointRegistry()
    for i, model in enumerate(live + down):
        job = f"j{i}"
        registry.register(project_id="default", model=model, job_id=job, host="h", port=1, backend="slurm")
        status = er.EndpointStatus.READY if model in live else er.EndpointStatus.PROVISIONING
        registry.update_status(job, status)
    return registry


@pytest.fixture(autouse=True)
def roster(tmp_path, monkeypatch):
    path = tmp_path / "pool.yaml"
    path.write_text(ROSTER)
    monkeypatch.setenv(provision_roster.ROSTER_ENV_VAR, str(path))
    monkeypatch.delenv("UMRS_DEFAULT_MODEL_FALLBACK", raising=False)
    provision_roster._cache.clear()
    yield
    provision_roster._cache.clear()


def test_only_roles_with_a_live_replica_are_advertised() -> None:
    registry = _registry("glm-5.3-flash", "Bespoke-Nimble-9B", down=("Qwen3-Embedding-0.6B",))
    assert ops.connect_info_default_models(registry) == {"chat": "svc-chat", "decision": "svc-decision"}


def test_fallback_order_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("UMRS_DEFAULT_MODEL_FALLBACK", "svc-fast, kimi-k3@rikyu ,")
    assert ops.connect_info_default_models(_registry())["fallback"] == ["svc-fast", "kimi-k3@rikyu"]


def test_connect_info_carries_the_map_through_the_response_model(monkeypatch) -> None:
    registry = _registry("glm-5.3-flash")
    monkeypatch.setattr(er, "get_registry", lambda: registry)
    result = ops.ops_connect_info(x_user_id=None)
    assert ops.ConnectInfoResponse(**result).model_dump()["default_models"] == {"chat": "svc-chat"}


def test_nothing_running_means_no_key(monkeypatch) -> None:
    registry = _registry(down=("glm-5.3-flash",))
    monkeypatch.setattr(er, "get_registry", lambda: registry)
    assert "default_models" not in ops.ops_connect_info(x_user_id=None)


def test_role_models_name_the_live_backing_models() -> None:
    # ADR-106 D-10 (2026-11-03): R4 tells the user which model did the work.
    registry = _registry("glm-5.3-flash", "Bespoke-Nimble-9B", down=("Qwen3-Embedding-0.6B",))
    assert ops.connect_info_role_models(registry) == {
        "chat": ["glm-5.3-flash"],
        "decision": ["Bespoke-Nimble-9B"],
    }
