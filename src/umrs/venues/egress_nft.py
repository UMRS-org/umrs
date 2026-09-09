"""Host nftables ruleset render for the frontend venue (R2-1, design §3.5).

The egress bridge may dial exactly two classes: the internet at large
(WAN via NAT) and the inference proxy on infra-net:8443. Everything else
toward the host and infra-net drops. Rendered declaratively — the Ansible
playbook applies the text; the text itself is the testable artifact.

Operator profile: rootful is the only viable class for a host-enforced
FORWARD chain on umrs-front (rootless netavark filtering is not
userland-addressable); the rootless fallback is named in the ops runbook.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EgressNetRules:
    """One rendered nftables ruleset (deterministic, parameter-ordered)."""

    table_name: str
    text: str


def render_egress_nft(
    *,
    egress_bridge: str = "podman2",
    infer_host: str,
    infer_port: int = 8443,
    allow_ssh: bool = True,
    table: str = "umrs-frontend-egress",
) -> EgressNetRules:
    """Render the host nftables set for the frontend venue.

    Args:
        egress_bridge: bridge interface for the sidecar pods (`egress-net`).
        infer_host: inference proxy hostname/IP (the only infra-net target).
        infer_port: inference TLS port.
        allow_ssh: keep host SSH reachable out of the flush (safety latch; off
            in a hardened headless appliance profile).
    """
    ssh_rule = '    tcp dport 22 accept comment "keep-ssh"' if allow_ssh else ""
    text = f"""table inet {table} {{
  # Frontend-venue egress guard rails (R2-1, design §3.5).
  # Applies: sidecar pods on {egress_bridge} may reach the WAN and the
  # inference proxy ({infer_host}:{infer_port}) ONLY — every other infra-net
  # and the host itself drops.
  chain forward_{table} {{
    type filter hook forward priority filter; policy drop;

    ct state established,related accept
    ct state invalid drop

    iifname "{egress_bridge}" oifname "{egress_bridge}" drop
    iifname "{egress_bridge}" ip daddr {infer_host} tcp dport {infer_port} accept comment "only inference proxy"
    iifname "{egress_bridge}" ip daddr 0.0.0.0/8 drop
    iifname "{egress_bridge}" ip daddr 10.0.0.0/8 drop
    iifname "{egress_bridge}" ip daddr 100.64.0.0/10 drop
    iifname "{egress_bridge}" ip daddr 127.0.0.0/8 drop
    iifname "{egress_bridge}" ip daddr 169.254.0.0/16 drop
    iifname "{egress_bridge}" ip daddr 172.16.0.0/12 drop
    iifname "{egress_bridge}" ip daddr 192.0.0.0/24 drop
    iifname "{egress_bridge}" ip daddr 192.168.0.0/16 drop
    iifname "{egress_bridge}" ip daddr 198.18.0.0/15 drop
    iifname "{egress_bridge}" ip daddr 224.0.0.0/4 drop
    iifname "{egress_bridge}" ip daddr 240.0.0.0/4 drop
    iifname "{egress_bridge}" ip6 daddr ::1/128 drop
    iifname "{egress_bridge}" ip6 daddr fc00::/7 drop
    iifname "{egress_bridge}" ip6 daddr fe80::/10 drop
    iifname "{egress_bridge}" oifname lo drop
    iifname "{egress_bridge}" accept comment "wan nat"
{ssh_rule}
  }}
}}
"""
    return EgressNetRules(table_name=table, text=text)


def render_apply_script(rules: EgressNetRules) -> str:
    """the idempotent apply wrapper (operate-only, never ships a secret)."""
    return f"""#!/bin/sh
# Idempotent apply of {rules.table_name} (R2-1). Run as root.
set -eu
rules_file="$(mktemp)"
trap 'rm -f "$rules_file"' EXIT
cat > "$rules_file" <<'NFT'
{rules.text}NFT
nft -f "$rules_file"
nft list table inet {rules.table_name} >/dev/null
echo "applied: {rules.table_name}"
"""
