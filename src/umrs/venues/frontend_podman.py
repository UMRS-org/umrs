"""Frontend-venue provisioner (R1-3, design §3.5, D-5/D-7, Amendment 1).

Per-session: internal network → broker lease → podman secret → sidecar
configs (nginx inference listener + squid) → dual-homed egress pod →
egress IP pinned on the lease → CPU-only worker container → entrypoint
isolation self-test. Compensating teardown on any step failure.

The podman CLI is injected as a callable so unit tests drive the full
state machine without a live host. Worker is a CPU-only agent sandbox —
no vLLM, no GPU devices, no model mounts (Amendment 1).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSION_DIR_ROOT = Path("/var/lib/umrs/sessions")
SECRET_NAME_PREFIX = "sess-"

PodmanRunner = Callable[..., subprocess.CompletedProcess]


def default_podman_runner(
    *args: str, input_text: str | None = None
) -> subprocess.CompletedProcess:
    """Production runner: podman CLI on PATH, 30 s step timeout."""
    return subprocess.run(  # argv list, no shell
        ["podman", *args],
        capture_output=True,
        text=True,
        input=input_text,
        timeout=30,
        check=False,
    )


@dataclass(frozen=True)
class ProvisionConfig:
    """One session's provisioning inputs (references only — no secrets)."""

    session_id: str
    owner: str
    source_ids: tuple[str, ...]
    default_source_id: str
    subnet: str  # pool-allocated, e.g. "172.18.52.0/24"
    infer_host: str  # inference proxy DNS/IP on the site network
    infer_port: int = 8443
    worker_image: str = "r4-companion@sha256:PINNED"
    nginx_image: str = "docker.io/nginx@sha256:PINNED"
    squid_image: str = "docker.io/ubuntu/squid@sha256:PINNED"
    web_policy: str = "public"  # public | allowlist | none
    web_allow_domains: tuple[str, ...] = ()  # EgressGuard format (R2-7); allowlist mode
    # Inference uplink: "https+internal-ca" (production target, cert bundled)
    # or "http" (bootstrap: umrs-infer plain loopback in the trust zone).
    infer_uplink: str = "https"
    internal_ca_path: str = "/etc/umrs/internal-ca.pem"
    # Override nginx's resolver with an explicit bridge-DNS address (live-leg
    # motivation: infer host comes in as a name on a bridge netns; without it
    # stock nginx can't re-resolve per request). "" = no resolver directive.
    infer_resolve_via: str = ""
    lease_seconds: int = 3600
    max_lifetime_seconds: int = 7 * 86_400
    worker_memory: str = "16g"
    worker_cpus: str = "8"
    project: str = ""
    gateway_host: str = ""
    umrs_host: str = ""


@dataclass
class ProvisionResult:
    session_id: str
    state: str  # "running" | "failed"
    reason: str = ""
    lease_id: str = ""
    egress_ip: str = ""
    steps: list[str] = field(default_factory=list)


# ── Templates ────────────────────────────────────────────────────────────


def _infer_tls_block(cfg: ProvisionConfig) -> str:
    """HTTPS-uplink block vs plain bootstrap HTTP (internal-CA lands later).

    Name-based upstreams (umrs-infer on a bridge) must not bake at startup:
    ``resolver`` + variable re-resolves per request — a proxy recreate whose
    IP changed would otherwise strand every existing session's sidecar.
    """
    needs_resolver = (
        cfg.infer_resolve_via
        and not cfg.infer_host.replace("[", "").replace("]", "").replace(".", "").isdigit()
    )
    resolver = f"    resolver {cfg.infer_resolve_via} valid=10s;\n" if needs_resolver else ""
    if cfg.infer_uplink == "https":
        return f"""{resolver}
      set $infer https://{cfg.infer_host}:{cfg.infer_port}/s/{cfg.session_id}/;
      proxy_pass $infer;
      proxy_ssl_verify on;
      proxy_ssl_trusted_certificate {cfg.internal_ca_path};
      proxy_ssl_server_name on;"""
    return f"""{resolver}
      set $infer http://{cfg.infer_host}:{cfg.infer_port}/s/{cfg.session_id}/;
      proxy_pass $infer;"""


def render_nginx_conf(cfg: ProvisionConfig) -> str:
    """Inference-listener nginx.conf (design §3.4). No $http_authorization logged."""
    return f"""worker_processes 1;
error_log stderr warn;
pid /tmp/nginx.pid;
events {{ worker_connections 256; }}
http {{
  access_log off;
  log_format sess '$time_iso8601 $remote_addr "$request" $status $body_bytes_sent $request_time';
  # D-1: the session token flows through here; auth headers stay out of logs.

  server {{
    listen 8080;
    access_log /dev/stdout sess;
    client_max_body_size 512m;
    location / {{
{_infer_tls_block(cfg)}
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_set_header Host {cfg.infer_host};
      proxy_buffering off;
      proxy_request_buffering off;
      proxy_cache off;
      proxy_read_timeout 3600s;
      proxy_send_timeout 3600s;
      proxy_connect_timeout 10s;
    }}
  }}
  server {{
    listen 8443;
    return 404;  # ingress listener placeholder — gateway-side routing lands with venue wiring
  }}
}}
"""


def _squid_deny_set(cfg: ProvisionConfig) -> str:
    extras = " ".join(sorted({h for h in (cfg.gateway_host, cfg.umrs_host) if h}))
    # host.containers.internal is covered by .internal; squid rejects a
    # host entry that is a subdomain of another acl entry (fatal at parse).
    return (
        "acl infra_ips dst 0.0.0.0/8 10.0.0.0/8 100.64.0.0/10 127.0.0.0/8 "
        "169.254.0.0/16 172.16.0.0/12 192.0.0.0/24 192.168.0.0/16 198.18.0.0/15 "
        "224.0.0.0/4 240.0.0.0/4 ::1/128 fc00::/7 fe80::/10 ::ffff:0:0/96\n"
        f"acl infra_names dstdomain .infra .internal .local localhost {extras}"
    )


def normalize_allow_domain(host: str) -> str:
    """EgressGuard normalization (parse_allowlist parity, R2-7):
    lower-case, whitespace trimmed, one trailing dot stripped.
    """
    return host.strip().lower().rstrip(".")


def render_allowlist_file(domains: tuple[str, ...]) -> str:
    """One domain per line for squid dstdomain (EgressGuard schema parity).

    An empty allowlist renders an empty file — squid denies every destination
    (fail-closed parity with EgressGuard's empty-list rule).
    """
    seen = {normalize_allow_domain(d) for d in domains if normalize_allow_domain(d)}
    return "".join(f"{d}\n" for d in sorted(seen))


def render_squid_conf(cfg: ProvisionConfig) -> str:
    """Forward-proxy config per design §3.4 (web class; no credential injection).

    Allowlist mode reads a mounted `/etc/squid/allow.txt` the provisioner
    writes from ``web_allow_domains`` (EgressGuard schema parity, R2-7).
    """
    allowlist = ""
    if cfg.web_policy == "allowlist":
        allowlist = (
            'acl allowed dstdomain "/etc/squid/allow.txt"\nhttp_access deny !allowed\n'
        )
    return f"""http_port 3128
cache deny all
reply_body_max_size 512 MB
request_body_max_size 64 MB
{_squid_deny_set(cfg)}
acl Safe_ports port 80 443
acl SSL_ports port 443
acl CONNECT method CONNECT
http_access deny infra_ips
http_access deny infra_names
http_access deny !Safe_ports
http_access deny CONNECT !SSL_ports
{allowlist}http_access allow all
logformat sess %ts.%03tu %>a %rm %ru %>Hs %<st %Ss
access_log stdio:/dev/stdout sess
"""


SELFTEST_SCRIPT = """#!/bin/sh
# D-7 isolation self-test: direct egress MUST fail before the agent loop.
set -u
exit_iso() {
  echo "venue_isolation_failed: $1" >&2
  exit 78
}
# 1. Direct TCP to internet must not connect.
( timeout 3 sh -c 'echo > /dev/tcp/1.1.1.1/443' ) 2>/dev/null && exit_iso "direct tcp to 1.1.1.1:443 succeeded"
# 2. External DNS must not resolve.
getent hosts example.com >/dev/null 2>&1 && exit_iso "external DNS resolved"
# 3. Session-internal names MUST resolve.
getent hosts egress >/dev/null 2>&1 || exit_iso "egress does not resolve"
exit 0
"""


@dataclass(frozen=True)
class WorkerRunArgs:
    """Argv for the worker container (§2.5 hardening; Amendment 1 CPU-only)."""

    argv: tuple[str, ...]


def worker_run_args(cfg: ProvisionConfig) -> WorkerRunArgs:
    """CPU-only sandbox. Asserted in tests: no GPU envs, devices, or mounts."""
    argv = (
        "run",
        "-d",
        "--name",
        f"worker-{cfg.session_id}",
        "--network",
        f"sess-{cfg.session_id}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:size=4g",
        "--tmpfs",
        "/run",
        "--pids-limit",
        "1024",
        "--memory",
        cfg.worker_memory,
        "--cpus",
        cfg.worker_cpus,
        "--secret",
        f"sess-{cfg.session_id}-token,type=mount,target=/run/secrets/session_token,mode=0400",
        "--env",
        "OPENAI_BASE_URL=http://egress:8080/v1",
        "--env",
        "ANTHROPIC_BASE_URL=http://egress:8080",
        "--env",
        "HTTPS_PROXY=http://egress:3128",
        "--env",
        "HTTP_PROXY=http://egress:3128",
        "--env",
        "NO_PROXY=egress,localhost,127.0.0.1",
        "--env",
        "R4_VENUE_SESSION_TOKEN_FILE=/run/secrets/session_token",
        cfg.worker_image,
    )
    return WorkerRunArgs(argv=argv)


# ── Provisioner ──────────────────────────────────────────────────────────


class FrontendPodmanProvisioner:
    """Ordered provisioning with compensating teardown (idempotent sweeps)."""

    def __init__(
        self,
        broker: Any,
        podman: PodmanRunner = default_podman_runner,
        session_root: Path = SESSION_DIR_ROOT,
    ) -> None:
        self._broker = broker
        self._podman = podman
        self._root = Path(session_root)

    def provision(self, cfg: ProvisionConfig) -> ProvisionResult:
        result = ProvisionResult(session_id=cfg.session_id, state="running")
        try:
            self._step_network(cfg, result)
            lease, token = self._step_lease(cfg, result)
            self._step_secret(cfg, result, token)
            self._step_configs(cfg, result)
            self._step_egress_pod(cfg, result)
            self._step_bind_egress_ip(cfg, result, lease)
            self._step_worker(cfg, result)
        except ProvisionStepError as exc:
            result.state = "failed"
            result.reason = f"{exc.step}: {exc.detail}"
            self.teardown(cfg.session_id)
        return result

    # ── Steps ─────────────────────────────────────────────────────────

    def _step_network(self, cfg: ProvisionConfig, result: ProvisionResult) -> None:
        self._run_ok(
            "network",
            "create",
            "--internal",
            "--subnet",
            cfg.subnet,
            f"sess-{cfg.session_id}",
        )
        result.steps.append("network")

    def _step_lease(
        self, cfg: ProvisionConfig, result: ProvisionResult
    ) -> tuple[Any, str]:
        """Issue the session lease; returns (lease, plaintext token once)."""
        lease, token = self._broker.issue_lease(
            f"session-{cfg.session_id}",
            "inference-token",
            "inference",
            lease_seconds=cfg.lease_seconds,
            owner_id=cfg.owner,
            session_id=cfg.session_id,
            source_ids=cfg.source_ids,
            default_source_id=cfg.default_source_id,
            project=cfg.project,
            max_lifetime_seconds=cfg.max_lifetime_seconds,
        )
        result.lease_id = lease.lease_id
        result.steps.append("lease")
        return lease, token

    def _step_secret(
        self, cfg: ProvisionConfig, result: ProvisionResult, token: str
    ) -> None:
        """Plaintext token enters ONLY the podman secret driver — never env."""
        name = f"{SECRET_NAME_PREFIX}{cfg.session_id}-token"
        proc = self._podman("secret", "create", name, "-", input_text=token)
        if proc.returncode != 0:
            raise ProvisionStepError("secret", proc.stderr[:120])
        result.steps.append("secret")

    def _step_configs(self, cfg: ProvisionConfig, result: ProvisionResult) -> None:
        session_dir = self._root / cfg.session_id
        session_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        (session_dir / "nginx.conf").write_text(render_nginx_conf(cfg))
        if cfg.web_policy != "none":
            (session_dir / "squid.conf").write_text(render_squid_conf(cfg))
            if cfg.web_policy == "allowlist":
                (session_dir / "squid-allow.txt").write_text(
                    render_allowlist_file(cfg.web_allow_domains)
                )
        (session_dir / "entrypoint-selftest.sh").write_text(SELFTEST_SCRIPT)
        (session_dir / "entrypoint-selftest.sh").chmod(0o700)
        result.steps.append("configs")

    def _step_egress_pod(self, cfg: ProvisionConfig, result: ProvisionResult) -> None:
        session_dir = self._root / cfg.session_id
        self._run_ok(
            "egress-pod",
            "pod",
            "create",
            "--name",
            f"egress-{cfg.session_id}",
            "--network",
            f"sess-{cfg.session_id}:alias=egress",
            "--network",
            "egress-net",
        )
        self._run_ok(
            "nginx",
            "run",
            "-d",
            "--pod",
            f"egress-{cfg.session_id}",
            "--read-only",
            # Stock alpine-nginx chowns its temp dirs (live leg blockers).
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CHOWN",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--tmpfs",
            "/var/cache/nginx:rw,size=32m",
            "--tmpfs",
            "/var/run:rw,size=4m",
            "-v",
            f"{session_dir}/nginx.conf:/etc/nginx/nginx.conf:ro,Z",
            cfg.nginx_image,
        )
        if cfg.web_policy != "none":
            squid_volumes = [
                "-v",
                f"{session_dir}/squid.conf:/etc/squid/squid.conf:ro,Z",
            ]
            if cfg.web_policy == "allowlist":
                squid_volumes += [
                    "-v",
                    f"{session_dir}/squid-allow.txt:/etc/squid/allow.txt:ro,Z",
                ]
            self._run_ok(
                "squid",
                "run",
                "-d",
                "--pod",
                f"egress-{cfg.session_id}",
                "--read-only",
                # Squid drops to the proxy user (uid 13) at boot.
                "--cap-drop",
                "ALL",
                "--cap-add",
                "SETUID",
                "--cap-add",
                "SETGID",
                "--tmpfs",
                "/var/spool/squid:rw,size=32m",
                "--tmpfs",
                "/var/log/squid:rw,size=16m",
                "--tmpfs",
                "/var/run/squid:rw,size=4m",
                *squid_volumes,
                cfg.squid_image,
            )
        result.steps.append("egress-pod")

    def _step_bind_egress_ip(
        self, cfg: ProvisionConfig, result: ProvisionResult, _lease: Any
    ) -> None:
        """lease unused here — bind is keyed on result.lease_id (kept in signature for step shape)."""
        proc = self._podman("inspect", f"egress-{cfg.session_id}")
        if proc.returncode != 0:
            raise ProvisionStepError("bind-egress-ip", proc.stderr[:120])
        import json

        try:
            payload = json.loads(proc.stdout)
            networks = payload[0].get("NetworkSettings", {}).get("Networks", {})
            egress_ip = (networks.get("egress-net") or {}).get("IPAddress", "")
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            raise ProvisionStepError("bind-egress-ip", f"inspect parse: {exc}") from exc
        if not egress_ip:
            raise ProvisionStepError("bind-egress-ip", "egress-net address missing")
        if not self._broker.bind_egress_ip(
            result.lease_id, egress_ip, actor="provisioner"
        ):
            raise ProvisionStepError("bind-egress-ip", "broker refused the bind")
        result.egress_ip = egress_ip
        result.steps.append("bind-egress-ip")

    def _step_worker(self, cfg: ProvisionConfig, result: ProvisionResult) -> None:
        args = worker_run_args(cfg)
        proc = self._podman(*args.argv)
        if proc.returncode != 0:
            raise ProvisionStepError("worker", proc.stderr[:120])
        result.steps.append("worker")

    # ── Teardown ──────────────────────────────────────────────────────

    def teardown(self, session_id: str) -> None:
        """Idempotent, order-fixed teardown; the orphan sweeper calls this."""
        sid = session_id
        for action in (
            ("rm", "-f", f"worker-{sid}"),
            ("pod", "rm", "-f", f"egress-{sid}"),
            ("network", "rm", f"sess-{sid}"),
            ("secret", "rm", f"{SECRET_NAME_PREFIX}{sid}-token"),
        ):
            try:
                self._podman(*action)
            except Exception:  # noqa: BLE001 — teardown must never raise-chain
                logger.warning("teardown step failed: %s", action)
        self._broker.revoke_by_session(sid, actor="sweeper")
        session_dir = self._root / sid
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        logger.info("teardown complete for session %s", sid)

    # ── Runner helper ─────────────────────────────────────────────────

    def _run_ok(self, step: str, *args: str) -> subprocess.CompletedProcess:
        proc = self._podman(*args)
        if proc.returncode != 0:
            raise ProvisionStepError(step, proc.stderr[:160])
        return proc


class ProvisionStepError(RuntimeError):
    """One step failed; the provisioner tears down and reports step+detail."""

    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail
