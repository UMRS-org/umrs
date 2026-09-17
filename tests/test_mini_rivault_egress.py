"""Static guards for mini-RiVault EgressGuard wiring."""

import os
import socket
import subprocess
from pathlib import Path

import yaml
from umrs.orchestrator.base import OrchestratorConfig
from umrs.orchestrator.kubernetes_backend import KubernetesBackend

ROOT = Path(__file__).resolve().parents[2]
UMRS_ROOT = ROOT / "umrs"
MINI_RIVAULT = UMRS_ROOT / "src" / "umrs" / "mini-rivault-llm"
SERVE_MODEL = MINI_RIVAULT / "serve-model.yaml"
SERVE_MODEL_SLURM = MINI_RIVAULT / "serve-model.slurm"
OUT_NGINX = MINI_RIVAULT / "nginx_fs" / "out-nginx.conf"
ASSERT_OFFLINE = MINI_RIVAULT / "assert-offline.sh"
E2E_COMPOSE = ROOT / "docker-compose.e2e.yaml"


MINI_RIVAULT_SERVICES = (
    "litellm",
    "vllm",
    "mcp-vision",
    "mcp-custom",
    "rag",
    "bentoml-vision",
)


def test_mini_rivault_compose_services_use_egress_proxy_env():
    compose = yaml.safe_load(SERVE_MODEL.read_text())

    for service_name in MINI_RIVAULT_SERVICES:
        env = compose["services"][service_name]["environment"]
        if service_name == "vllm":
            # ADR-040 D5 / WP-01 r3b live finding: vLLM stays fully offline.
            # Its proxy env is pinned EMPTY (the egress anchor would inject
            # the guard proxy under the 'egress' profile and trip the
            # container's offline assert) — the pin is the empty-string
            # invariant, not the guard-interpolation one.
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                assert env[key] == ""
            continue
        assert env["HTTP_PROXY"].startswith("${RIVAULT_HTTP_PROXY")
        assert env["HTTPS_PROXY"].startswith("${RIVAULT_HTTPS_PROXY")
        assert env["http_proxy"].startswith("${RIVAULT_HTTP_PROXY")
        assert env["https_proxy"].startswith("${RIVAULT_HTTPS_PROXY")
        assert "egress-guard" in env["NO_PROXY"]
        assert env["NO_PROXY"] == env["no_proxy"]


def test_slurm_wrapper_exports_egress_proxy_defaults():
    text = SERVE_MODEL_SLURM.read_text()

    for key in (
        "RIVAULT_HTTP_PROXY",
        "RIVAULT_HTTPS_PROXY",
        "RIVAULT_NO_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "RIVAULT_EXTRA_PROFILES",
        "MCP_SERVERS_PATH",
        "PROJECT_DATA_PATH",
        "RAG_INDEX_PATH",
        "EGRESS_DOMAIN_ALLOWLIST",
    ):
        assert f"{key}=" in text
    assert "http://egress-guard:8888" in text
    assert "prepare_mcp_mount_overlay" in text


def test_slurm_wrapper_initializes_litellm_db_with_sql_migrations():
    text = SERVE_MODEL_SLURM.read_text()

    assert "LITELLM_DB_SQL_MIGRATIONS=" in text
    assert "run_litellm_db_sql_migrations" in text
    assert "up --remove-orphans --detach ll-postgres" in text
    assert "${GPUoverride[@]}" in text
    assert "/app/litellm-proxy-extras/litellm_proxy_extras/migrations" in text
    assert "psql -v ON_ERROR_STOP=1" in text
    assert "LITELLM_DISABLE_SCHEMA_UPDATE=" in text


def test_custom_mcp_runs_network_transport_when_profile_enabled():
    compose = yaml.safe_load(SERVE_MODEL.read_text())
    command = compose["services"]["mcp-custom"]["command"]

    assert "--transport sse" in command
    assert "--port ${MCP_CUSTOM_PORT:-8080}" in command


def test_forward_proxy_allows_only_rendered_bucket_host():
    nginx_conf = OUT_NGINX.read_text()
    slurm = SERVE_MODEL_SLURM.read_text()

    assert "map $host $egress_proxy_allowed_host" in nginx_conf
    assert "UMRS_S3_BUCKET_HOST 1;" in nginx_conf
    assert "return 403;" in nginx_conf
    assert "s@UMRS_S3_BUCKET_HOST@" in slurm
    assert '_substitute_required "out-nginx.conf"' in slurm


def test_e2e_compute_mock_exports_proxy_env():
    compose = yaml.safe_load(E2E_COMPOSE.read_text())
    env = compose["services"]["compute-mock"]["environment"]

    assert "HTTP_PROXY=http://gateway:8080" in env
    assert "HTTPS_PROXY=http://gateway:8080" in env
    assert any(item.startswith("NO_PROXY=") and "gateway" in item for item in env)


def test_kubernetes_job_manifest_exports_egress_proxy_env():
    backend = KubernetesBackend(
        OrchestratorConfig(
            egress_proxy_url="http://egress-guard.rivault:18888",
            egress_no_proxy="localhost,litellm,vllm,egress-guard",
        )
    )
    manifest = backend._build_job_manifest("mini-rivault-test", "/models/test")
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}

    for key in (
        "RIVAULT_HTTP_PROXY",
        "RIVAULT_HTTPS_PROXY",
        "EGRESS_GUARD_PROXY_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ):
        assert env[key] == "http://egress-guard.rivault:18888"
    assert env["NO_PROXY"] == "localhost,litellm,vllm,egress-guard"
    assert env["no_proxy"] == env["NO_PROXY"]


def _default_network_policy(monkeypatch):
    monkeypatch.delenv("K8S_EGRESS_ALLOWED_CIDRS", raising=False)
    backend = KubernetesBackend(
        OrchestratorConfig(
            egress_proxy_url="http://egress-guard.rivault:18888",
        )
    )
    policy = backend._build_network_policy("mini-rivault-test", "rivault-test")
    egress_rules = policy["spec"]["egress"]
    peers = [peer for rule in egress_rules for peer in rule.get("to", [])]
    return policy, egress_rules, peers


def test_kubernetes_network_policy_defaults_to_dns_and_egress_guard_ports(monkeypatch):
    policy, egress_rules, _peers = _default_network_policy(monkeypatch)
    assert policy["spec"]["policyTypes"] == ["Egress"]
    assert {"protocol": "UDP", "port": 53} in egress_rules[0]["ports"]
    assert {"protocol": "TCP", "port": 53} in egress_rules[0]["ports"]
    assert egress_rules[1]["ports"] == [{"protocol": "TCP", "port": 18888}]


def test_kubernetes_network_policy_default_peers_are_the_guard_only(monkeypatch):
    _policy, _rules, peers = _default_network_policy(monkeypatch)
    assert {"podSelector": {"matchLabels": {"app.kubernetes.io/name": "egress-guard"}}} in peers
    assert {"podSelector": {"matchLabels": {"app": "egress-guard"}}} in peers
    assert not any("ipBlock" in peer for peer in peers)


def test_kubernetes_network_policy_allows_site_cidrs_only_when_configured(monkeypatch):
    monkeypatch.setenv("K8S_EGRESS_ALLOWED_CIDRS", "10.100.0.0/16,10.101.0.0/16")
    policy = KubernetesBackend()._build_network_policy("mini-rivault-test", "rivault-test")
    peers = [peer for rule in policy["spec"]["egress"] for peer in rule.get("to", [])]

    assert {"ipBlock": {"cidr": "10.100.0.0/16"}} in peers
    assert {"ipBlock": {"cidr": "10.101.0.0/16"}} in peers


# ---------------------------------------------------------------------------
# assert-offline.sh is executed, not just read (G-P5-4, ADR-040)
# ---------------------------------------------------------------------------


def _run_assert_offline(**env_overrides: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(ASSERT_OFFLINE)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _closed_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


_OFFLINE_ENV = {"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}


def test_assert_offline_fails_when_offline_env_missing():
    proc = _run_assert_offline(HF_HUB_OFFLINE="1")  # TRANSFORMERS_OFFLINE unset
    assert proc.returncode == 1
    assert "FAIL: TRANSFORMERS_OFFLINE" in proc.stdout


def test_assert_offline_fails_when_proxy_var_set():
    proc = _run_assert_offline(**_OFFLINE_ENV, HTTPS_PROXY="http://egress-guard:8888")
    assert proc.returncode == 1
    assert "FAIL: HTTPS_PROXY is set" in proc.stdout


def test_assert_offline_passes_with_offline_env_and_no_probe():
    proc = _run_assert_offline(**_OFFLINE_ENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK: All offline invariants satisfied." in proc.stdout
    assert "direct egress" not in proc.stdout  # the probe is opt-in


def test_assert_offline_probe_ok_when_direct_connect_is_refused():
    port = _closed_local_port()  # nothing listens here
    proc = _run_assert_offline(
        **_OFFLINE_ENV,
        OFFLINE_PROBE="1",
        OFFLINE_PROBE_HOST="127.0.0.1",
        OFFLINE_PROBE_PORT=str(port),
        OFFLINE_PROBE_TIMEOUT_SEC="2",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"OK: direct egress to 127.0.0.1:{port} is blocked" in proc.stdout


def test_assert_offline_probe_fails_when_direct_connect_succeeds():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        proc = _run_assert_offline(
            **_OFFLINE_ENV,
            OFFLINE_PROBE="1",
            OFFLINE_PROBE_HOST="127.0.0.1",
            OFFLINE_PROBE_PORT=str(port),
            OFFLINE_PROBE_TIMEOUT_SEC="2",
        )
    assert proc.returncode == 1
    assert f"FAIL: direct egress to 127.0.0.1:{port} SUCCEEDED" in proc.stdout
    assert "Refusing to start inference" in proc.stdout


# ---------------------------------------------------------------------------
# ADR-040 D4/D5 wiring: in-repo proxy image + launch refusal
# ---------------------------------------------------------------------------


def test_egress_guard_service_ships_in_the_r4_mcp_image():
    services = yaml.safe_load(SERVE_MODEL.read_text())["services"]
    guard = services["egress-guard"]
    assert guard["profiles"] == ["egress"]
    assert guard["image"].startswith("${EGRESS_GUARD_IMAGE:-localhost/r4-mcp:")
    assert "ghcr.io" not in guard["image"]
    assert "python -m r4_mcp.egress_guard" in guard["command"]


def test_slurm_wrapper_refuses_egress_profile_without_present_image():
    text = SERVE_MODEL_SLURM.read_text()
    assert "EGRESS_GUARD_IMAGE=${EGRESS_GUARD_IMAGE:-localhost/r4-mcp:" in text
    assert "ADR-040 D4" in text
    assert "refusing to launch" in text
    # ADR-045 D3: presence is established through the staging contract
    # (node store -> shared-filesystem archive -> pull) in venue.sh.
    assert 'rivault_ensure_image "${_egress_guard_image}" "${RIVAULT_IMAGE_ARCHIVE_DIR:-}"' in text
    venue = (SERVE_MODEL_SLURM.parent / "venue.sh").read_text()
    assert "podman image exists" in venue and "podman load -i" in venue and "podman pull" in venue


# ── 2026-09-17 localhost-image preflight guard (bdrt 5191/5195/5196 session) ─

def _guard_block() -> str:
    text = SERVE_MODEL_SLURM.read_text()
    start = text.index("Preflight: vLLM image")
    return text[start : start + 2600]


def test_localhost_images_never_silently_substitute_the_default_render():
    block = _guard_block()
    assert "localhost/*|lcr.io/*" in block
    assert "podman image exists" in block
    # The previous shape allowed fallback pull for every image variant —
    # localhost would be swapped to cu130-nightly and boot the WRONG engine.
    head_and_fallback = block.split(";;")[0]
    assert "cu130-nightly" not in head_and_fallback


def test_multi_node_preflight_drops_per_node_when_slurm_present():
    block = _guard_block()
    assert "scontrol show hostnames" in block
    assert "--nodelist" in block
    # worker-spawn missing-image must be fatal BEFORE the run, named to site.
    assert "staged on EVERY node" in block or "must be staged on every node" in block


def test_remote_variants_keep_the_only_substitution_path():
    block = _guard_block()
    # fallthrough case (any non-localhost scheme) retains exactly one legal
    # fallback — pull-then-cu130-nightly, marked WARN, never silent.
    assert 'attempting fallback tag...' in block
    assert block.count("cu130-nightly-aarch64") >= 2


# ── 2026-09-17 --profile capability test correctness (rccs m0 discovery) ────

def test_profile_capability_probes_root_help_not_subcommand_help():
    """The wrapper's `--profile` check must read the GLOBAL `podman-compose -h`
    (`--profile` is a top-level parser flag since 1.6.x where `up --help`
    hides it). The previous `up --help | grep --profile` shape false-failed
    every vendored 1.6.0 install and drove jobs 384645/384652 into the FATAL
    abort lane."""
    text = SERVE_MODEL_SLURM.read_text()
    assert "podman-compose -h" in text
    assert "up --help 2>/dev/null | grep -q -- '--profile'" not in text
