"""Ended-row retention in the Janitor (2026-11-03).

Nothing removed FAILED / TERMINATED registry rows, so a front's
``/umrs/endpoints/health`` stayed "degraded" for good after any failed
provision (rccs and bdrt on the README status). The janitor now removes
ended rows after ``failed_retention_sec``, except a FAILED canary that holds
an ADR-106 upgrade.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from umrs.api.endpoints import _health_counts
from umrs.orchestrator import janitor as janitor_module
from umrs.orchestrator.endpoint_registry import (
    ENDED_AT_KEY,
    EndpointRegistry,
    EndpointStatus,
)
from umrs.orchestrator.janitor import Janitor, JanitorConfig
from umrs.orchestrator.service_controller import DIGEST_KEY

RETENTION_SEC = 3600


def _config(**overrides) -> JanitorConfig:
    base = {
        "enable_ttl_cleanup": False,
        "enable_orphan_cleanup": False,
        "enable_stuck_cleanup": False,
        "enable_liveness_cleanup": False,
        "enable_engine_probe": False,
        "enable_ttl_auto_extend": False,
        "failed_retention_sec": RETENTION_SEC,
    }
    base.update(overrides)
    return JanitorConfig(**base)


def _ago_iso(seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _register(registry: EndpointRegistry, job_id: str) -> None:
    registry.register(
        project_id="proj",
        model="m1",
        job_id=job_id,
        host="node",
        port=8001,
        backend="compose",
    )


def _failed(registry: EndpointRegistry, job_id: str, ended_ago: int, **metadata) -> None:
    _register(registry, job_id)
    registry.mark_failed(job_id, "boom")
    endpoint = registry.get_by_job_id(job_id)
    endpoint.metadata[ENDED_AT_KEY] = _ago_iso(ended_ago)
    endpoint.metadata.update(metadata)


@pytest.fixture
def holds(monkeypatch):
    """The current service recipe digests the janitor sees (default: none)."""
    state = {"digests": set()}
    monkeypatch.setattr(janitor_module, "_service_hold_digests", lambda: state["digests"])
    return state


async def _run(registry: EndpointRegistry, **overrides):
    janitor = Janitor(config=_config(**overrides), registry=registry, backend=object())
    return await janitor.run_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("holds")
class TestEndedRetention:
    async def test_old_failed_row_is_removed_recent_kept(self):
        registry = EndpointRegistry()
        _failed(registry, "old", ended_ago=RETENTION_SEC + 60)
        _failed(registry, "recent", ended_ago=60)

        result = await _run(registry)

        assert result.ended_pruned == 1
        assert result.total_cleaned == 1
        assert registry.get_by_job_id("old") is None
        assert registry.get_by_job_id("recent") is not None

    async def test_terminated_row_is_removed_ready_untouched(self):
        registry = EndpointRegistry()
        _register(registry, "stopped")
        registry.update_status("stopped", EndpointStatus.TERMINATED)
        registry.get_by_job_id("stopped").metadata[ENDED_AT_KEY] = _ago_iso(
            RETENTION_SEC + 60
        )
        _register(registry, "serving")
        registry.mark_ready("serving")
        registry.get_by_job_id("serving").created_at -= timedelta(days=3)

        await _run(registry)

        assert registry.get_by_job_id("stopped") is None
        assert registry.get_by_job_id("serving") is not None

    async def test_row_without_stamp_ages_from_its_latest_time(self):
        registry = EndpointRegistry()
        for job_id, age in (("legacy-old", RETENTION_SEC + 60), ("legacy-new", 60)):
            _register(registry, job_id)
            registry.mark_failed(job_id, "boom")
            endpoint = registry.get_by_job_id(job_id)
            endpoint.metadata.pop(ENDED_AT_KEY)
            endpoint.created_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                seconds=age
            )

        await _run(registry)

        assert registry.get_by_job_id("legacy-old") is None
        assert registry.get_by_job_id("legacy-new") is not None

    async def test_failed_canary_of_current_recipe_holds(self, holds):
        holds["digests"] = {"cur"}
        registry = EndpointRegistry()
        _failed(registry, "canary", ended_ago=RETENTION_SEC * 10, **{DIGEST_KEY: "cur"})
        _failed(registry, "old-canary", ended_ago=RETENTION_SEC * 10, **{DIGEST_KEY: "old"})

        await _run(registry)

        assert registry.get_by_job_id("canary") is not None
        assert registry.get_by_job_id("old-canary") is None

    async def test_unreadable_roster_keeps_every_digest_row(self, holds):
        holds["digests"] = None
        registry = EndpointRegistry()
        _failed(registry, "canary", ended_ago=RETENTION_SEC * 10, **{DIGEST_KEY: "any"})
        _failed(registry, "plain", ended_ago=RETENTION_SEC * 10)

        await _run(registry)

        assert registry.get_by_job_id("canary") is not None
        assert registry.get_by_job_id("plain") is None

    async def test_disabled_keeps_rows(self):
        registry = EndpointRegistry()
        _failed(registry, "old", ended_ago=RETENTION_SEC * 10)

        result = await _run(registry, enable_failed_retention=False)

        assert result.ended_pruned == 0
        assert registry.get_by_job_id("old") is not None


class TestEndedStamp:
    def test_failed_transition_stamps_once(self):
        registry = EndpointRegistry()
        _register(registry, "j")
        registry.mark_failed("j", "first")
        first = registry.get_by_job_id("j").metadata[ENDED_AT_KEY]

        registry.mark_failed("j", "again")

        assert registry.get_by_job_id("j").metadata[ENDED_AT_KEY] == first

    def test_ready_row_has_no_stamp(self):
        registry = EndpointRegistry()
        _register(registry, "j")
        registry.mark_ready("j")
        assert ENDED_AT_KEY not in registry.get_by_job_id("j").metadata


def test_service_hold_digests_reads_service_entries_only(monkeypatch):
    entries = [
        {"model": "svc", "service": {"project_id": "p"}},
        {"model": "keep", "keep_alive": {"project_id": "p"}},
    ]
    monkeypatch.setattr(
        "umrs.orchestrator.provision_roster.roster_entries", lambda: entries
    )
    digests = janitor_module._service_hold_digests()
    assert digests is not None and len(digests) == 1


def test_health_counts_bucket_every_live_status():
    statuses = [
        EndpointStatus.READY,
        EndpointStatus.PENDING,
        EndpointStatus.PROVISIONING,
        EndpointStatus.STARTING,
        EndpointStatus.DEGRADED,
        EndpointStatus.FAILED,
        EndpointStatus.TERMINATED,
        EndpointStatus.DRAINING,
        EndpointStatus.EXPIRED,
    ]
    counts = _health_counts([SimpleNamespace(status=s) for s in statuses])
    assert counts == {"healthy": 1, "provisioning": 3, "unhealthy": 3}
