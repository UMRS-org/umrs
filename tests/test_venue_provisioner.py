"""R1-3: provisioner state machine + templates + hardening profile.

Fake podman runner drives every step ordering/failure path; templates are
asserted for no-token logging, deny ACLs, CPU-only worker (Amendment 1).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from umrs.jobs.secret_broker import SecretBroker
from umrs.venues.frontend_podman import (
    SELFTEST_SCRIPT,
    FrontendPodmanProvisioner,
    ProvisionConfig,
    render_allowlist_file,
    render_nginx_conf,
    render_squid_conf,
    worker_run_args,
)


def _cfg(**over) -> ProvisionConfig:
    base = {
        "session_id": "sess-t1",
        "owner": "alice",
        "source_ids": ("11111111-1111-4111-8111-111111111111",),
        "default_source_id": "11111111-1111-4111-8111-111111111111",
        "subnet": "172.18.52.0/24",
        "infer_host": "umrs-infer",
        "worker_image": "r4-companion@sha256:abc",
        "nginx_image": "docker.io/nginx@sha256:def",
        "squid_image": "docker.io/squid@sha256:123",
        "gateway_host": "gw.example.org",
        "umrs_host": "umrs.example.org",
    }
    base.update(over)
    return ProvisionConfig(**base)


class FakePodman:
    """Records argv sequences; fails on demand per named step."""

    def __init__(self, fail_at: str | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.fail_at = fail_at
        self.secret_stdin: str | None = None

    def __call__(self, *args, input_text: str | None = None, **_kw) -> subprocess.CompletedProcess:
        self.calls.append(tuple(args))
        if args[:2] == ("secret", "create"):
            self.secret_stdin = input_text
        if self.fail_at and any(self.fail_at in a for a in args):
            return subprocess.CompletedProcess(args, 1, "", f"boom in {' '.join(args)}")
        if args[0] == "inspect":
            payload = [
                {"NetworkSettings": {"Networks": {"egress-net": {"IPAddress": "172.18.52.9"}}}}
            ]
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(args, 0, "", "")


def _provision(fail_at: str | None = None, **cfg_over):
    broker = SecretBroker()
    podman = FakePodman(fail_at)
    provisioner = FrontendPodmanProvisioner(
        broker, podman=podman, session_root=Path("/tmp/umrs-test-sessions")
    )
    import tempfile

    provisioner._root = Path(tempfile.mkdtemp(prefix="umrs-venue-test-"))
    result = provisioner.provision(_cfg(**cfg_over))
    return provisioner, broker, podman, result


class TestHappyPath:
    def test_ordered_steps(self):
        _, _, _, result = _provision()
        assert result.state == "running"
        assert result.steps == [
            "network",
            "lease",
            "secret",
            "configs",
            "egress-pod",
            "bind-egressip",
            "worker",
        ][:5] + ["bind-egress-ip", "worker"] or result.steps == [
            "network",
            "lease",
            "secret",
            "configs",
            "egress-pod",
            "bind-egress-ip",
            "worker",
        ]

    def test_lease_issued_and_ip_bound(self):
        _, broker, _, result = _provision()
        assert result.lease_id
        assert result.egress_ip == "172.18.52.9"
        lease = broker.get_lease(result.lease_id)
        assert lease.session_id == "sess-t1"
        assert lease.egress_ip == "172.18.52.9"

    def test_plaintext_token_only_in_podman_secret(self):
        _, broker, podman, result = _provision()
        assert podman.secret_stdin is not None
        # Token goes via stdin to the secret driver — never into worker env.
        worker_run = next(c for c in podman.calls if c[0] == "run" and "worker-sess-t1" in c)
        env_pairs = [worker_run[i + 1] for i, v in enumerate(worker_run) if v == "--env"]
        assert podman.secret_stdin not in env_pairs
        assert any("R4_VENUE_SESSION_TOKEN_FILE=/run/secrets/session_token" in e for e in env_pairs)


class TestFailurePaths:
    @pytest.mark.parametrize("fail_at", ["sess-sess-t1", "egress-sess-t1", "squid"])
    def test_steps_fail_compensating_teardown(self, fail_at):
        _, broker, podman, result = _provision(fail_at=fail_at)
        assert result.state == "failed"
        assert result.reason
        # Teardown ran: network rm + secret rm + pod rm attempted.
        flattened = [" ".join(c) for c in podman.calls]
        assert any("rm -f worker-sess-t1" in c for c in flattened)
        assert any("pod rm -f egress-sess-t1" in c for c in flattened)
        assert any("network rm sess-sess-t1" in c for c in flattened)
        assert any("secret rm sess-sess-t1-token" in c for c in flattened)

    def test_worker_failure_revokes_lease(self):
        _, broker, _, result = _provision(fail_at="r4-companion@sha256:abc")
        assert result.state == "failed"
        lease = broker.get_lease(result.lease_id)
        assert lease is None or not lease.is_valid


class TestTemplates:
    def test_nginx_conf_bootstrap_http_uplink_has_no_ca(self):
        conf = render_nginx_conf(_cfg(infer_uplink="http"))
        assert "proxy_pass http://" in conf
        assert "internal-ca.pem" not in conf
        assert "proxy_ssl_verify" not in conf

    def test_nginx_conf_https_uplink_needs_ca(self):
        conf = render_nginx_conf(_cfg(infer_uplink="https"))
        assert "proxy_pass https://" in conf
        assert "internal-ca.pem" in conf
        assert "proxy_ssl_verify on" in conf

    def test_squid_deny_set_dedups_shared_hosts(self):
        conf = render_squid_conf(_cfg(gateway_host="data0.x", umrs_host="data0.x"))
        assert conf.count("data0.x") == 1  # duplicate dstdomain entries are a FATAL acl bug

    def test_nginx_conf_never_logs_authorization(self):
        conf = render_nginx_conf(_cfg())
        assert "$http_authorization" not in conf
        assert "listen 8080" in conf
        assert "proxy_buffering off" in conf
        assert "proxy_request_buffering off" in conf
        assert "client_max_body_size 512m" in conf
        assert "s/sess-t1/" in conf

    def test_squid_conf_deny_set(self):
        conf = render_squid_conf(_cfg())
        assert "http_access deny infra_ips" in conf
        assert "10.0.0.0/8" in conf
        assert "169.254.0.0/16" in conf
        # host.containers.internal must NOT sit next to .internal (squid fatal overlap).
        assert "host.containers.internal" not in conf
        assert ".internal" in conf
        assert "gw.example.org" in conf
        assert "http_access deny CONNECT !SSL_ports" in conf
        assert "http_access allow all" in conf

    def test_squid_allowlist_mode(self):
        conf = render_squid_conf(
            _cfg(web_policy="allowlist", web_allow_domains=("pypi.org", "github.com"))
        )
        assert "acl allowed dstdomain" in conf
        assert "http_access deny !allowed" in conf

    def test_squid_allowlist_file_egressguard_schema_parity(self):
        content = render_allowlist_file(("PYPI.ORG", "github.com.", " pypi.org ", ""))
        assert content == "github.com\npypi.org\n"  # normalized, deduped, sorted

    def test_allowlist_file_empty_means_deny_all_fail_closed(self):
        assert render_allowlist_file(()) == ""
        assert render_allowlist_file(("", " ", "..")) == ""

    def test_allowlist_mounts_allow_file(self):
        _, _, podman, _ = _provision(web_policy="allowlist", web_allow_domains=("pypi.org",))
        flattened = [" ".join(c) for c in podman.calls]
        assert any("squid-allow.txt:/etc/squid/allow.txt" in c for c in flattened)
        assert any("squid.conf:/etc/squid/squid.conf" in c for c in flattened)

    def test_selftest_script_self_consistency(self):
        assert "exit 78" in SELFTEST_SCRIPT
        assert "egress" in SELFTEST_SCRIPT
        assert "example.com" in SELFTEST_SCRIPT


class TestWorkerHardening:
    def test_cpu_only_no_gpu(self):
        argv = " ".join(worker_run_args(_cfg()).argv)
        for forbidden in (
            "--gpus",
            "/dev/nvidia",
            "NVIDIA_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "vllm",
            "model_cache",
        ):
            assert forbidden not in argv
        assert "--cap-drop ALL" in argv
        assert "--read-only" in argv
        assert "--security-opt no-new-privileges" in argv
        assert "--network sess-sess-t1" in argv
        assert "egress-net" not in argv  # worker attaches the session net only

    def test_web_policy_none_skips_squid(self):
        _, _, podman, _ = _provision(web_policy="none")
        flattened = [" ".join(c) for c in podman.calls]
        assert not any("squid" in c for c in flattened)


class TestBrokerLeaseShape:
    def test_lease_bindings(self):
        _, broker, _, result = _provision()
        lease = broker.get_lease(result.lease_id)
        assert lease.scope == "inference"
        assert lease.owner_id == "alice"
        assert lease.source_ids == ("11111111-1111-4111-8111-111111111111",)
        assert lease.max_lifetime_at is not None
