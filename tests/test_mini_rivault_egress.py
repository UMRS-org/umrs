"""Static guards for mini-RiVault EgressGuard wiring."""

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
    assert 's@UMRS_S3_BUCKET_HOST@' in slurm
    assert '_substitute_required "out-nginx.conf"' in slurm


def test_e2e_compute_mock_exports_proxy_env():
    compose = yaml.safe_load(E2E_COMPOSE.read_text())
    env = compose["services"]["compute-mock"]["environment"]

    assert "HTTP_PROXY=http://gateway:8080" in env
    assert "HTTPS_PROXY=http://gateway:8080" in env
    assert any(item.startswith("NO_PROXY=") and "gateway" in item for item in env)


def test_kubernetes_job_manifest_exports_egress_proxy_env():
    backend = KubernetesBackend(OrchestratorConfig(
        egress_proxy_url="http://egress-guard.rivault:18888",
        egress_no_proxy="localhost,litellm,vllm,egress-guard",
    ))
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


def test_kubernetes_network_policy_defaults_to_dns_and_egress_guard(monkeypatch):
    monkeypatch.delenv("K8S_EGRESS_ALLOWED_CIDRS", raising=False)
    backend = KubernetesBackend(OrchestratorConfig(
        egress_proxy_url="http://egress-guard.rivault:18888",
    ))
    policy = backend._build_network_policy("mini-rivault-test", "rivault-test")
    egress_rules = policy["spec"]["egress"]
    peers = [peer for rule in egress_rules for peer in rule.get("to", [])]

    assert policy["spec"]["policyTypes"] == ["Egress"]
    assert {"protocol": "UDP", "port": 53} in egress_rules[0]["ports"]
    assert {"protocol": "TCP", "port": 53} in egress_rules[0]["ports"]
    assert egress_rules[1]["ports"] == [{"protocol": "TCP", "port": 18888}]
    assert {"podSelector": {"matchLabels": {"app.kubernetes.io/name": "egress-guard"}}} in peers
    assert {"podSelector": {"matchLabels": {"app": "egress-guard"}}} in peers
    assert not any("ipBlock" in peer for peer in peers)


def test_kubernetes_network_policy_allows_site_cidrs_only_when_configured(monkeypatch):
    monkeypatch.setenv("K8S_EGRESS_ALLOWED_CIDRS", "10.100.0.0/16,10.101.0.0/16")
    policy = KubernetesBackend()._build_network_policy("mini-rivault-test", "rivault-test")
    peers = [peer for rule in policy["spec"]["egress"] for peer in rule.get("to", [])]

    assert {"ipBlock": {"cidr": "10.100.0.0/16"}} in peers
    assert {"ipBlock": {"cidr": "10.101.0.0/16"}} in peers