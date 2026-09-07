"""Egress proxy-image declarations per site (WP-01 round 3, ADR-040 D4).

The HPC leg's venue (rccs) now carries the declared proxy image its
provisioning guard requires. These pins keep the declaration from drifting:
rccs declares; rccs-front (the unused front profile) still refuses — the
declared-vs-refused distinction is exactly the signal the D4 check keys on.
"""

from __future__ import annotations

import pytest

from umrs.orchestrator.site_config import load_site_config


class TestRccsDeclares:
    def test_rccs_site_declares_proxy_image(self):
        cfg = load_site_config("deploy/rccs/site.yaml")
        assert cfg.network.egress_proxy_image == "localhost/r4-mcp:latest"

    def test_declared_image_uses_the_in_repo_proxy_entrypoint(self):
        # The declared image must be the r4-mcp image whose entrypoint is
        # `python -m r4_mcp.egress_guard` (rounds 1-2's in-repo proxy) — never
        # an external ghcr reference (that image was unreachable, WP-01 step 0).
        cfg = load_site_config("deploy/rccs/site.yaml")
        assert "r4-mcp" in cfg.network.egress_proxy_image
        assert "ghcr.io" not in cfg.network.egress_proxy_image


class TestFrontStillRefusesByDefault:
    def test_rccs_front_declares_no_image(self):
        # The front profile stays default-deny unless the owner declares one.
        cfg = load_site_config("deploy/rccs-front/site.yaml")
        assert (cfg.network.egress_proxy_image or "") == ""


class TestGateEnvironmentShape:
    def test_site_declared_resolves_through_loader(self):
        cfg = load_site_config("deploy/rccs/site.yaml")
        assert cfg.network.egress_proxy_image is not None
        assert isinstance(cfg.network.egress_proxy_image, str)
        assert cfg.network.egress_proxy_image.startswith("localhost/")
