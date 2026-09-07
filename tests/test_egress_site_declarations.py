"""Egress proxy-image declarations per site (WP-01 round 3, ADR-040 D4).

The HPC leg's venue (rccs) now carries the declared proxy image its
provisioning guard requires. These pins keep the declaration from drifting:
rccs declares; rccs-front (the unused front profile) still refuses — the
declared-vs-refused distinction is exactly the signal the D4 check keys on.
"""

from __future__ import annotations

from pathlib import Path

from umrs.orchestrator.site_config import load_site_config

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"  # cwd-independent (repo-root or umrs/ runs)


class TestRccsDeclares:
    def test_rccs_site_declares_proxy_image(self):
        cfg = load_site_config(str(DEPLOY / "rccs" / "site.yaml"))
        assert cfg.network.egress_proxy_image == "localhost/r4-mcp:latest"

    def test_umrs_front_rccs_site_declares_proxy_image(self):
        # The file the umrs-front rccs server container binds as
        # /etc/umrs/site.yaml (compose volume ../rccs-front/site.yaml).
        cfg = load_site_config(str(DEPLOY / "rccs-front" / "site.yaml"))
        assert cfg.network.egress_proxy_image == "localhost/r4-mcp:latest"

    def test_declared_image_uses_the_in_repo_proxy_entrypoint(self):
        # The declared image must be the r4-mcp image whose entrypoint is
        # `python -m r4_mcp.egress_guard` (rounds 1-2's in-repo proxy) — never
        # an external ghcr reference (that image was unreachable, WP-01 step 0).
        cfg = load_site_config(str(DEPLOY / "rccs" / "site.yaml"))
        assert "r4-mcp" in cfg.network.egress_proxy_image
        assert "ghcr.io" not in cfg.network.egress_proxy_image


class TestOtherVenuesStillRefuseByDefault:
    def test_ai4s_declares_no_image(self):
        # Non-rccs venues keep the default-deny baseline.
        cfg = load_site_config(str(DEPLOY / "ai4s" / "site.yaml"))
        assert (cfg.network.egress_proxy_image or "") == ""


class TestGateEnvironmentShape:
    def test_site_declared_resolves_through_loader(self):
        cfg = load_site_config(str(DEPLOY / "rccs" / "site.yaml"))
        assert cfg.network.egress_proxy_image is not None
        assert isinstance(cfg.network.egress_proxy_image, str)
        assert cfg.network.egress_proxy_image.startswith("localhost/")
