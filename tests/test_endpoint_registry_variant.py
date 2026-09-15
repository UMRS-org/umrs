"""Unit tests for multi-variant endpoint registry features.

Covers:
- _find_candidates() returns all endpoints for a model
- select_endpoint() with least-connections strategy
- Variant filtering in get() and select_endpoint()
- increment_requests / decrement_requests thread safety
- register() with variant parameter
- to_dict() includes variant and active_requests
"""

import threading
from datetime import datetime, timedelta

from umrs.orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
)


def _make_endpoint(
    model: str = "DeepSeek-V4-Pro",
    job_id: str = "job-1",
    variant: str | None = None,
    status: EndpointStatus = EndpointStatus.READY,
    active_requests: int = 0,
    project_id: str = "rivault",
) -> RegisteredEndpoint:
    return RegisteredEndpoint(
        project_id=project_id,
        model=model,
        job_id=job_id,
        host="node1",
        port=11434,
        status=status,
        variant=variant,
        active_requests=active_requests,
    )


class TestFindCandidates:
    def test_returns_all_endpoints_for_same_model(self):
        reg = EndpointRegistry()
        reg.register("p1", "DeepSeek-V4-Pro", "job-1", "n1", 1, variant="1m")
        reg.register("p1", "DeepSeek-V4-Pro", "job-2", "n2", 2, variant="128k")

        candidates = reg._find_candidates("p1", "DeepSeek-V4-Pro")
        assert len(candidates) == 2

    def test_filters_by_variant(self):
        reg = EndpointRegistry()
        reg.register("p1", "DeepSeek-V4-Pro", "job-1", "n1", 1, variant="1m")
        reg.register("p1", "DeepSeek-V4-Pro", "job-2", "n2", 2, variant="128k")

        candidates = reg._find_candidates("p1", "DeepSeek-V4-Pro", variant="128k")
        assert len(candidates) == 1
        assert candidates[0].variant == "128k"

    def test_filters_by_status(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1)
        reg.register("p1", "M", "j2", "n2", 2)
        reg.update_status("j1", EndpointStatus.READY)
        reg.update_status("j2", EndpointStatus.FAILED, "error")

        candidates = reg._find_candidates(
            "p1", "M",
            status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
        )
        assert len(candidates) == 1
        assert candidates[0].job_id == "j1"

    def test_empty_when_no_match(self):
        reg = EndpointRegistry()
        candidates = reg._find_candidates("p1", "NoModel")
        assert candidates == []

    def test_different_projects_isolated(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1)
        reg.register("p2", "M", "j2", "n2", 2)

        assert len(reg._find_candidates("p1", "M")) == 1
        assert len(reg._find_candidates("p2", "M")) == 1

    def test_served_model_name_fallback(self):
        reg = EndpointRegistry()
        ep = reg.register("p1", "qwen3-8b", "j1", "n1", 1)
        ep.metadata["served_model_name"] = "Qwen/Qwen3-8B"

        candidates = reg._find_candidates("p1", "Qwen/Qwen3-8B")
        assert len(candidates) == 1


class TestSelectEndpoint:
    def test_least_connections_picks_lowest(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1, variant="v1")
        reg.register("p1", "M", "j2", "n2", 2, variant="v2")
        reg.update_status("j1", EndpointStatus.READY)
        reg.update_status("j2", EndpointStatus.READY)
        # j1 has 5 requests, j2 has 2
        reg.increment_requests("j1")
        reg.increment_requests("j1")
        reg.increment_requests("j1")
        reg.increment_requests("j1")
        reg.increment_requests("j1")
        reg.increment_requests("j2")
        reg.increment_requests("j2")

        selected = reg.select_endpoint("p1", "M", strategy="least_connections")
        assert selected is not None
        assert selected.job_id == "j2"

    def test_variant_filter(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1, variant="128k")
        reg.register("p1", "M", "j2", "n2", 2, variant="1m")
        reg.update_status("j1", EndpointStatus.READY)
        reg.update_status("j2", EndpointStatus.READY)

        selected = reg.select_endpoint("p1", "M", variant="128k")
        assert selected is not None
        assert selected.variant == "128k"

    def test_returns_none_when_no_usable(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1)
        reg.update_status("j1", EndpointStatus.FAILED, "error")

        selected = reg.select_endpoint("p1", "M")
        assert selected is None

    def test_first_strategy(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1)
        reg.register("p1", "M", "j2", "n2", 2)
        reg.update_status("j1", EndpointStatus.READY)
        reg.update_status("j2", EndpointStatus.READY)

        selected = reg.select_endpoint("p1", "M", strategy="first")
        assert selected is not None
        # "first" returns the first in iteration order (not guaranteed, but
        # should return some endpoint)
        assert selected.job_id in ("j1", "j2")


class TestGetVariantAware:
    def test_variant_lookup(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1, variant="128k")
        reg.register("p1", "M", "j2", "n2", 2, variant="1m")
        reg.update_status("j1", EndpointStatus.READY)
        reg.update_status("j2", EndpointStatus.READY)

        ep = reg.get("p1", "M", variant="128k")
        assert ep is not None
        assert ep.variant == "128k"

    def test_legacy_no_variant_returns_first(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1, variant="128k")
        reg.register("p1", "M", "j2", "n2", 2, variant="1m")
        reg.update_status("j1", EndpointStatus.READY)
        reg.update_status("j2", EndpointStatus.READY)

        ep = reg.get("p1", "M")
        assert ep is not None
        # Legacy path returns first match (any variant)


class TestRequestCounting:
    def test_increment_decrement(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1)
        reg.increment_requests("j1")
        reg.increment_requests("j1")
        assert reg.get_by_job_id("j1").active_requests == 2
        reg.decrement_requests("j1")
        assert reg.get_by_job_id("j1").active_requests == 1

    def test_decrement_never_goes_negative(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1)
        reg.decrement_requests("j1")
        assert reg.get_by_job_id("j1").active_requests == 0

    def test_noop_for_unknown_job(self):
        reg = EndpointRegistry()
        reg.increment_requests("nonexistent")
        reg.decrement_requests("nonexistent")
        # Should not raise

    def test_thread_safety(self):
        reg = EndpointRegistry()
        reg.register("p1", "M", "j1", "n1", 1)
        reg.update_status("j1", EndpointStatus.READY)
        n = 1000
        barrier = threading.Barrier(4)

        def inc_dec():
            barrier.wait()
            for _ in range(n):
                reg.increment_requests("j1")
                reg.decrement_requests("j1")

        threads = [threading.Thread(target=inc_dec) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert reg.get_by_job_id("j1").active_requests == 0


class TestRegisterVariant:
    def test_variant_stored(self):
        reg = EndpointRegistry()
        ep = reg.register("p1", "M", "j1", "n1", 1, variant="128k")
        assert ep.variant == "128k"

    def test_variant_in_to_dict(self):
        reg = EndpointRegistry()
        ep = reg.register("p1", "M", "j1", "n1", 1, variant="128k")
        d = ep.to_dict()
        assert d["variant"] == "128k"
        assert d["active_requests"] == 0

    def test_variant_none_not_in_to_dict(self):
        reg = EndpointRegistry()
        ep = reg.register("p1", "M", "j1", "n1", 1)
        d = ep.to_dict()
        assert "variant" not in d
