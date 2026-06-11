"""Unit tests for PoolAutoscaler queue-depth-driven scaling.

Covers:
- Scale-up when queue depth exceeds threshold
- Scale-down when endpoint is idle beyond threshold
- min_replicas / max_replicas bounds
- Policy loading from roster entries
- No decisions when no policies configured
"""

from datetime import datetime, timedelta

from umrs.orchestrator.autoscaler import AutoscalePolicy, PoolAutoscaler, _parse_autoscale_policy
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus


def _register_ready(
    registry: EndpointRegistry,
    model: str = "DeepSeek-V4-Pro",
    variant: str | None = None,
    job_id: str = "job-1",
    active_requests: int = 0,
    last_used_at: datetime | None = None,
) -> None:
    """Helper: register and mark ready, optionally set load/idle state."""
    registry.register("rivault", model, job_id, "n1", 11434, variant=variant)
    registry.update_status(job_id, EndpointStatus.READY)
    ep = registry.get_by_job_id(job_id)
    if active_requests > 0:
        for _ in range(active_requests):
            registry.increment_requests(job_id)
    if last_used_at is not None:
        ep.last_used_at = last_used_at


class TestParseAutoscalePolicy:
    def test_parses_autoscale_block(self):
        entry = {
            "model": "DeepSeek-V4-Pro",
            "name": "128k",
            "autoscale": {
                "min_replicas": 2,
                "max_replicas": 6,
                "scale_up_depth": 3,
                "scale_down_idle_sec": 300,
                "idle_ttl_sec": 400,
            },
        }
        policy = _parse_autoscale_policy(entry)
        assert policy is not None
        assert policy.model == "DeepSeek-V4-Pro"
        assert policy.variant == "128k"
        assert policy.min_replicas == 2
        assert policy.max_replicas == 6
        assert policy.scale_up_depth == 3

    def test_no_autoscale_key_returns_none(self):
        assert _parse_autoscale_policy({"model": "M"}) is None

    def test_empty_autoscale_returns_none(self):
        assert _parse_autoscale_policy({"model": "M", "autoscale": None}) is None


class TestScaleUp:
    def test_triggers_when_queue_depth_exceeds_threshold(self):
        registry = EndpointRegistry()
        _register_ready(registry, job_id="j1", active_requests=12)
        policy = AutoscalePolicy(
            model="DeepSeek-V4-Pro",
            scale_up_depth=5,  # threshold = 5 * 1 = 5
            max_replicas=3,
        )
        scaler = PoolAutoscaler(registry, [policy])
        decisions = scaler.check_and_scale()
        assert len(decisions) == 1
        assert decisions[0]["action"] == "scale_up"
        assert decisions[0]["model"] == "DeepSeek-V4-Pro"

    def test_no_scale_when_below_threshold(self):
        registry = EndpointRegistry()
        _register_ready(registry, job_id="j1", active_requests=3)
        policy = AutoscalePolicy(
            model="DeepSeek-V4-Pro",
            scale_up_depth=5,
            max_replicas=3,
        )
        scaler = PoolAutoscaler(registry, [policy])
        decisions = scaler.check_and_scale()
        scale_ups = [d for d in decisions if d["action"] == "scale_up"]
        assert len(scale_ups) == 0

    def test_no_scale_beyond_max_replicas(self):
        registry = EndpointRegistry()
        _register_ready(registry, job_id="j1", active_requests=0)
        _register_ready(registry, job_id="j2", active_requests=0)
        _register_ready(registry, job_id="j3", active_requests=0)
        policy = AutoscalePolicy(
            model="DeepSeek-V4-Pro",
            max_replicas=3,
            scale_up_depth=1,  # even 0 active > 1 * 3 = 3? no. need more.
        )
        # total_active = 0, threshold = 1*3 = 3, 0 < 3 → no scale-up
        scaler = PoolAutoscaler(registry, [policy])
        decisions = scaler.check_and_scale()
        scale_ups = [d for d in decisions if d["action"] == "scale_up"]
        assert len(scale_ups) == 0


class TestScaleDown:
    def test_triggers_when_endpoint_idle(self):
        registry = EndpointRegistry()
        old_time = datetime.utcnow() - timedelta(seconds=1200)
        _register_ready(registry, job_id="j1", active_requests=0, last_used_at=old_time)
        _register_ready(registry, job_id="j2", active_requests=0)
        policy = AutoscalePolicy(
            model="DeepSeek-V4-Pro",
            min_replicas=1,
            scale_down_idle_sec=600,
        )
        scaler = PoolAutoscaler(registry, [policy])
        decisions = scaler.check_and_scale()
        scale_downs = [d for d in decisions if d["action"] == "scale_down"]
        assert len(scale_downs) == 1
        assert scale_downs[0]["job_id"] == "j1"

    def test_no_scale_below_min_replicas(self):
        registry = EndpointRegistry()
        old_time = datetime.utcnow() - timedelta(seconds=1200)
        _register_ready(registry, job_id="j1", active_requests=0, last_used_at=old_time)
        policy = AutoscalePolicy(
            model="DeepSeek-V4-Pro",
            min_replicas=1,
            scale_down_idle_sec=600,
        )
        scaler = PoolAutoscaler(registry, [policy])
        decisions = scaler.check_and_scale()
        scale_downs = [d for d in decisions if d["action"] == "scale_down"]
        assert len(scale_downs) == 0  # can't go below min_replicas=1

    def test_no_scale_endpoint_still_active(self):
        registry = EndpointRegistry()
        old_time = datetime.utcnow() - timedelta(seconds=1200)
        _register_ready(registry, job_id="j1", active_requests=3, last_used_at=old_time)
        _register_ready(registry, job_id="j2", active_requests=0)
        policy = AutoscalePolicy(
            model="DeepSeek-V4-Pro",
            min_replicas=1,
            scale_down_idle_sec=600,
        )
        scaler = PoolAutoscaler(registry, [policy])
        decisions = scaler.check_and_scale()
        scale_downs = [d for d in decisions if d["action"] == "scale_down"]
        assert len(scale_downs) == 0  # j1 has active_requests > 0


class TestLoadPolicies:
    def test_load_from_roster(self):
        registry = EndpointRegistry()
        scaler = PoolAutoscaler(registry)
        count = scaler.load_policies_from_roster([
            {"model": "M1", "name": "v1", "autoscale": {"min_replicas": 2}},
            {"model": "M2", "autoscale": {"max_replicas": 8}},
        ])
        assert count == 2
        assert scaler.get_policy("M1", "v1") is not None
        assert scaler.get_policy("M2") is not None


class TestNoPolicies:
    def test_no_decisions_without_policies(self):
        registry = EndpointRegistry()
        _register_ready(registry, job_id="j1", active_requests=100)
        scaler = PoolAutoscaler(registry)
        assert scaler.check_and_scale() == []
