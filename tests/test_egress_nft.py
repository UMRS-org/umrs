"""R2-1: nftables render asserts (egress bridge confined to WAN + proxy)."""

from __future__ import annotations

from umrs.venues.egress_nft import render_apply_script, render_egress_nft


class TestEgressNft:
    def test_confines_bridge_to_wan_and_proxy_only(self):
        rules = render_egress_nft(infer_host="10.99.1.5", infer_port=8443)
        text = rules.text
        # Only infra target allowed: the inference proxy.
        assert "ip daddr 10.99.1.5 tcp dport 8443 accept" in text
        # Infra deny ranges present for both bridge directions.
        for rng in (
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "169.254.0.0/16",
            "127.0.0.0/8",
            "100.64.0.0/10",
            "224.0.0.0/4",
            "240.0.0.0/4",
            "::1/128",
            "fc00::/7",
            "fe80::/10",
        ):
            assert f"daddr {rng} drop" in text
        # Self-bridge traffic and host loopback are out.
        assert "oifname lo drop" in text
        assert "policy drop;" in text

    def test_ssh_latch_defaults_on_and_removable(self):
        with_ssh = render_egress_nft(infer_host="10.99.1.5")
        assert "keep-ssh" in with_ssh.text
        no_ssh = render_egress_nft(infer_host="10.99.1.5", allow_ssh=False)
        assert "keep-ssh" not in no_ssh.text

    def test_apply_script_is_self_verifying(self):
        rules = render_egress_nft(infer_host="10.99.1.5")
        script = render_apply_script(rules)
        assert "nft -f" in script
        assert "nft list table inet umrs-frontend-egress" in script
        assert rules.text in script.split("NFT", 1)[1]
