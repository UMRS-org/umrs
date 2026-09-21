"""Frontend-venue config resolution + subnet allocation (wave 2026-09-21)."""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.venues.frontend_config import (
    FrontendVenueConfig,
    SubnetPoolExhausted,
    allocate_subnet,
)

_READY_ENV = {
    "UMRS_VENUE_ENABLED": "1",
    "UMRS_VENUE_INFER_HOST": "umrs-infer",
    "UMRS_VENUE_API_UPSTREAM": "169.254.1.2:8000",
    "UMRS_VENUE_WORKER_IMAGE": "localhost/r4-companion@sha256:abc",
    "UMRS_VENUE_NGINX_IMAGE": "docker.io/library/nginx@sha256:def",
    "UMRS_VENUE_SQUID_IMAGE": "docker.io/ubuntu/squid@sha256:123",
}


class TestFromEnv:
    def test_disabled_by_default_and_names_every_blocker(self):
        cfg = FrontendVenueConfig.from_env({})
        assert cfg.enabled is False
        missing = cfg.missing()
        assert "UMRS_VENUE_ENABLED" in missing
        assert "UMRS_VENUE_INFER_HOST" in missing
        assert "UMRS_VENUE_API_UPSTREAM" in missing
        assert "UMRS_VENUE_WORKER_IMAGE" in missing

    def test_ready_env_has_no_blockers(self):
        cfg = FrontendVenueConfig.from_env(_READY_ENV)
        assert cfg.missing() == []
        assert cfg.session_root == Path("/var/lib/umrs/sessions")
        assert cfg.rail_port == 8081

    def test_placeholder_image_is_a_blocker(self):
        env = {**_READY_ENV, "UMRS_VENUE_NGINX_IMAGE": "docker.io/nginx@sha256:PINNED"}
        assert "UMRS_VENUE_NGINX_IMAGE" in FrontendVenueConfig.from_env(env).missing()

    def test_bad_pool_or_policy_are_blockers(self):
        env = {
            **_READY_ENV,
            "UMRS_VENUE_SUBNET_POOL": "not-a-cidr",
            "UMRS_VENUE_WEB_POLICY_DEFAULT": "wide",
        }
        missing = FrontendVenueConfig.from_env(env).missing()
        assert "UMRS_VENUE_SUBNET_POOL" in missing
        assert "UMRS_VENUE_WEB_POLICY_DEFAULT" in missing

    def test_provision_config_carries_rail_and_session_fields(self):
        cfg = FrontendVenueConfig.from_env({**_READY_ENV, "UMRS_VENUE_SESSION_ROOT": "/tmp/x"})
        pc = cfg.provision_config(
            session_id="s1",
            owner="alice",
            source_ids=("11111111-1111-4111-8111-111111111111",),
            default_source_id="11111111-1111-4111-8111-111111111111",
            subnet="172.30.0.0/24",
            web_policy="public",
        )
        assert pc.rail_upstream == "169.254.1.2:8000"
        assert pc.infer_uplink == "http"
        assert pc.worker_image.startswith("localhost/r4-companion")
        assert cfg.session_root == Path("/tmp/x")


class TestAllocateSubnet:
    def test_first_free_slice(self):
        assert allocate_subnet("172.30.0.0/16", []) == "172.30.0.0/24"
        assert allocate_subnet("172.30.0.0/16", ["172.30.0.0/24"]) == "172.30.1.0/24"

    def test_taken_normalized(self):
        assert allocate_subnet("172.30.0.0/16", ["172.30.0.7/24"]) == "172.30.1.0/24"

    def test_exhausted(self):
        with pytest.raises(SubnetPoolExhausted):
            allocate_subnet("10.9.0.0/24", ["10.9.0.0/24"])
