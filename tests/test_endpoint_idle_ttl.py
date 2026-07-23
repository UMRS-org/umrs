from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta

import pytest
from client_umrs import _resolve_endpoint_idle_ttl_sec
from umrs.client.control_api import UMRSControlAPI
from umrs.client.control_commands import parse_duration_seconds
from umrs.orchestrator.base import JobInfo, JobStatus
from umrs.orchestrator.endpoint_registry import EndpointRegistry
from umrs.orchestrator.janitor import Janitor, JanitorConfig


def test_parse_duration_seconds_accepts_endpoint_ttl_units():
    assert parse_duration_seconds("30m", field_name="idle ttl") == 1800
    assert parse_duration_seconds("2h", field_name="idle ttl") == 7200
    assert parse_duration_seconds("45", field_name="idle ttl") == 45


def test_parse_duration_seconds_rejects_non_positive_values():
    with pytest.raises(ValueError, match="idle ttl must be positive"):
        parse_duration_seconds("0", field_name="idle ttl")


def test_endpoint_idle_ttl_defaults_to_30_minutes_without_experience_config():
    args = Namespace(idle_ttl=None, experience=None)

    assert _resolve_endpoint_idle_ttl_sec(args, None) == 1800


def test_endpoint_idle_ttl_reads_experience_endpoint_config():
    args = Namespace(idle_ttl=None, experience="demo11")
    user_cfg = {
        "experiences": [
            {
                "xp_name": "demo11",
                "endpoint": {"idle_ttl": "45m"},
            }
        ]
    }

    assert _resolve_endpoint_idle_ttl_sec(args, user_cfg) == 2700


def test_endpoint_idle_ttl_cli_flag_overrides_experience_config():
    args = Namespace(idle_ttl="10m", experience="demo11")
    user_cfg = {
        "experiences": [
            {
                "xp_name": "demo11",
                "endpoint": {"idle_ttl": "45m"},
            }
        ]
    }

    assert _resolve_endpoint_idle_ttl_sec(args, user_cfg) == 600


def test_endpoint_mark_used_extends_idle_ttl_expiry():
    registry = EndpointRegistry(default_ttl_sec=60)
    endpoint = registry.register(
        project_id="demo",
        model="sam3",
        job_id="backend-1",
        host="node-1",
        port=11434,
        ttl_sec=30,
        metadata={"idle_ttl_sec": "1800"},
    )
    assert endpoint.metadata["idle_ttl_sec"] == 1800
    endpoint.expires_at = datetime.utcnow() + timedelta(seconds=5)

    updated = registry.mark_used("backend-1")

    assert updated is not None
    assert updated.last_used_at is not None
    assert updated.expires_at is not None
    assert 1700 <= (updated.expires_at - datetime.utcnow()).total_seconds() <= 1800


def test_endpoint_register_ttl_sec_zero_means_no_expiry():
    registry = EndpointRegistry(default_ttl_sec=60)
    endpoint = registry.register(
        project_id="ai4s-platform",
        model="kimi-k2.6",
        job_id="backend-pool-1",
        host="node-1",
        port=11434,
        ttl_sec=0,
    )

    assert endpoint.expires_at is None
    assert endpoint.ttl_remaining_sec is None
    assert registry.list_expired() == []
    # A non-expiring endpoint must never be rejected on TTL grounds.
    assert (
        registry.get("ai4s-platform", "kimi-k2.6", prefer_warm=False, min_ttl_sec=99999) is endpoint
    )


def test_endpoint_register_negative_ttl_sec_means_no_expiry():
    registry = EndpointRegistry(default_ttl_sec=60)
    endpoint = registry.register(
        project_id="ai4s-platform",
        model="gemma-5-31b-it",
        job_id="backend-pool-2",
        host="node-2",
        port=11434,
        ttl_sec=-1,
    )

    assert endpoint.expires_at is None


class _RecordingControlAPI(UMRSControlAPI):
    def __init__(self):
        super().__init__(control_mq=None, client_id="test-client")
        self.submissions = []

    def submit_job(self, **kwargs):
        self.submissions.append(kwargs)
        return {"status": "ok", "job": {"job_id": "job-1"}}


def test_endpoint_up_submits_keepalive_provision_job_with_idle_ttl():
    api = _RecordingControlAPI()

    parsed = api.endpoint_up(
        model="sam3",
        project="demo",
        idle_ttl_sec=2700,
        max_attempts=2,
        partition="gpu",
        node="node-a",
    )

    assert parsed["endpoint"] == {
        "job_id": "job-1",
        "model": "sam3",
        "project_id": "demo",
        "idle_ttl_sec": 2700,
    }
    assert api.submissions == [
        {
            "job_type": "endpoint_provision",
            "spec": {
                "model": "sam3",
                "ttl_sec": 2700,
                "idle_ttl_sec": 2700,
                "tier": "user",
                "gpu_count": 1,
                "node_count": 1,
                "max_model_len": 4096,
            },
            "project": "demo",
            "max_attempts": 2,
            "keep_alive": True,
            "partition": "gpu",
            "node": "node-a",
            "reservation": None,
        }
    ]


class _AliveBackend:
    async def get_job_status(self, job_id: str) -> JobInfo:
        return JobInfo(job_id=job_id, status=JobStatus.RUNNING)


@pytest.mark.asyncio
async def test_janitor_does_not_auto_extend_idle_ttl_endpoints():
    registry = EndpointRegistry(default_ttl_sec=60)
    registry.register(
        project_id="demo",
        model="sam3",
        job_id="backend-1",
        host="node-1",
        port=11434,
        ttl_sec=1,
        metadata={"idle_ttl_sec": 1800},
    )
    endpoint = registry.mark_ready("backend-1")
    assert endpoint is not None
    endpoint.expires_at = datetime.utcnow() + timedelta(seconds=1)
    before = endpoint.expires_at

    janitor = Janitor(
        config=JanitorConfig(
            ttl_auto_extend_sec=600,
            ttl_auto_extend_threshold_sec=120,
        ),
        registry=registry,
        backend=_AliveBackend(),
    )

    assert await janitor._auto_extend_ttl() == 0
    assert registry.get_by_job_id("backend-1").expires_at == before
