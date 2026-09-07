"""
Unit tests for SiteConfig (YAML-based site configuration).
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from umrs.orchestrator.site_config import (
    DatabaseConfig,
    NetworkConfig,
    RuntimeConfig,
    SchedulerConfig,
    SiteConfig,
    VllmConfig,
    _parse_raw,
    default_site_config,
    get_site_config,
    load_site_config,
    reset_site_config,
)

UMRS_ROOT = os.path.dirname(os.path.dirname(__file__))
RCCS_CLOUD_SITE_CONFIG = os.path.join(UMRS_ROOT, "deploy", "rccs-cloud", "site.yaml")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts with a clean singleton."""
    reset_site_config()
    yield
    reset_site_config()


MINIMAL_YAML = textwrap.dedent("""\
    name: test-site
    description: Unit test site
    scheduler:
      kind: slurm
      partition: test-gpu
    runtime:
      kind: podman-compose
      rivault_dir: /opt/rivault
    storage:
      model_cache_dir: /models/hf
    network:
      litellm_port: 9001
      litellm_master_key: sk-test
    vllm:
      model: intfloat/e5-base-v2
      gpu_mem_util: 0.15
    database:
      host: db-host
      port: 5433
""")


FULL_YAML = """\
name: full-site
description: Full test with every field
scheduler:
    kind: slurm
    partition: a100
    account: myaccount
    qos: high
    time_limit: "08:00:00"
    gpus: 4
    cpus: 32
    memory: "256G"
    extra_sbatch:
        - --constraint=a100
        - --exclusive
runtime:
    kind: podman-compose
    rivault_dir: /opt/rivault
    compose_file: custom-compose.yaml
    profiles:
        - harness
        - vllm
    gpu_passthrough: cdi
    cdi_spec: /run/cdi/nvidia.yaml
    cgroup_manager: systemd
    storage_root: /var/tmp/podman_test
storage:
    model_cache_dir: /scratch/models
    sam3_model_path: /scratch/sam3
    hmae_model_path: /scratch/hmae
    hmae_config_path: /scratch/hmae/config.yaml
    hmae_checkpoint: /scratch/hmae/ckpt.pth
    vision_tarball: /scratch/vision.tar
    user_model_path: /scratch/user
    s3_tunnel_host: login01
    s3_tunnel_port: 9999
network:
    litellm_port: 8001
    litellm_master_key: sk-full
    port_range_min: 20000
    port_range_max: 20100
    port_strategy: registry_lock
    egress_proxy_url: "http://egress-guard:8888"
    egress_no_proxy: localhost,vllm,litellm
vllm:
    model: meta/llama-3
    docker_tag: latest-gpu
    gpu_mem_util: 0.90
    max_model_len: 8192
    max_num_seqs: 16
    tensor_parallel_size: 4
database:
    host: pg-host
    port: 5432
    superuser: admin
    superuser_password: s3cret
    user_password: u_pass
    registry_dsn: postgres://admin:s3cret@pg-host:5432/umrs_db
extra:
    custom_key: custom_value
"""


# =============================================================================
# Tests — dataclass defaults
# =============================================================================


class TestSiteConfigDefaults:
    """Verify sane defaults when no YAML / env vars are set."""

    def test_default_name(self):
        cfg = SiteConfig()
        assert cfg.name == "default"

    def test_scheduler_defaults(self):
        s = SchedulerConfig()
        assert s.kind == "slurm"
        assert s.partition == ""
        assert s.time_limit == "04:00:00"
        assert s.gpus == 1

    def test_runtime_defaults(self):
        r = RuntimeConfig()
        assert r.kind == "podman-compose"
        assert r.gpu_passthrough == "auto"
        assert r.profiles == ["harness", "vllm", "vision-bentoml"]

    def test_network_defaults(self):
        n = NetworkConfig()
        assert n.litellm_port == 8001
        assert n.port_range_min == 11434
        assert n.port_range_max == 11534
        assert n.port_strategy == "static_range"


# =============================================================================
# Tests — YAML loading
# =============================================================================


class TestLoadSiteConfig:
    """Test load_site_config from YAML files."""

    def test_load_minimal(self, tmp_path):
        p = tmp_path / "site.yaml"
        p.write_text(MINIMAL_YAML)
        cfg = load_site_config(str(p))

        assert cfg.name == "test-site"
        assert cfg.scheduler.partition == "test-gpu"
        assert cfg.runtime.kind == "podman-compose"
        assert cfg.runtime.rivault_dir == "/opt/rivault"
        assert cfg.storage.model_cache_dir == "/models/hf"
        assert cfg.network.litellm_port == 9001
        assert cfg.network.litellm_master_key == "sk-test"
        assert cfg.vllm.model == "intfloat/e5-base-v2"
        assert cfg.vllm.gpu_mem_util == 0.15
        assert cfg.database.host == "db-host"
        assert cfg.database.port == 5433

    def test_load_full(self, tmp_path):
        p = tmp_path / "site.yaml"
        p.write_text(FULL_YAML)
        cfg = load_site_config(str(p))

        assert cfg.name == "full-site"
        assert cfg.scheduler.account == "myaccount"
        assert cfg.scheduler.qos == "high"
        assert cfg.scheduler.extra_sbatch == ["--constraint=a100", "--exclusive"]
        assert cfg.runtime.gpu_passthrough == "cdi"
        assert cfg.storage.s3_tunnel_host == "login01"
        assert cfg.storage.s3_tunnel_port == 9999
        assert cfg.network.port_strategy == "registry_lock"
        assert cfg.network.egress_proxy_url == "http://egress-guard:8888"
        assert cfg.network.egress_no_proxy == "localhost,vllm,litellm"
        assert cfg.vllm.tensor_parallel_size == 4
        assert cfg.database.registry_dsn == "postgres://admin:s3cret@pg-host:5432/umrs_db"
        assert cfg.extra == {"custom_key": "custom_value"}

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_site_config("/nonexistent/site.yaml")

    def test_unknown_keys_ignored(self, tmp_path):
        p = tmp_path / "site.yaml"
        p.write_text("name: x\nscheduler:\n  kind: slurm\n  unknown_field: 42\n")
        cfg = load_site_config(str(p))
        assert cfg.scheduler.kind == "slurm"

    def test_empty_yaml(self, tmp_path):
        p = tmp_path / "site.yaml"
        p.write_text("")
        cfg = load_site_config(str(p))
        assert cfg.name == "default"

    def test_load_expands_environment_references(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UMRS_TEST_ROOT", "/shared/alice")
        monkeypatch.setenv("USER", "alice")
        p = tmp_path / "site.yaml"
        p.write_text(
            textwrap.dedent("""\
            name: env-site
            runtime:
              rivault_dir: $UMRS_TEST_ROOT/rivault
              repo_dir: /lvs0/dne1/rccs-dmp/${USER}/ai4s-platform/umrs
            storage:
              model_cache_dir: ~/umrs_models/hf
        """)
        )

        cfg = load_site_config(str(p))

        assert cfg.runtime.rivault_dir == "/shared/alice/rivault"
        assert cfg.runtime.repo_dir == "/lvs0/dne1/rccs-dmp/alice/ai4s-platform/umrs"
        assert cfg.storage.model_cache_dir == os.path.expanduser("~/umrs_models/hf")

    def test_blank_litellm_key_uses_environment_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-from-env")
        p = tmp_path / "site.yaml"
        p.write_text(
            textwrap.dedent("""\
            name: env-key-site
            network:
              litellm_port: 8001
              litellm_master_key: ""
        """)
        )

        cfg = load_site_config(str(p))

        assert cfg.network.litellm_master_key == "sk-from-env"

    def test_rccs_cloud_site_config_uses_deployed_runtime_paths(self):
        """Requires rccs-cloud site.yaml — skip when not present (local dev)."""
        import os

        env_val = os.environ.get("UMRS_SITE_CONFIG", "")
        p = Path(env_val) if env_val else Path(RCCS_CLOUD_SITE_CONFIG)

        pytest.skip(f"site.yaml not found at {p}; not available in local dev")
        cfg = load_site_config(str(p))

        assert cfg.runtime.repo_dir == "/lvs0/rccs-dmp/masaru.nagaso/ai4s-platform/umrs"
        assert (
            cfg.runtime.rivault_dir
            == "/lvs0/rccs-dmp/masaru.nagaso/ai4s-platform/umrs/src/umrs/mini-rivault-llm"
        )

    def test_search_path_umrs_site_config_env(self, tmp_path):
        p = tmp_path / "custom_site.yaml"
        p.write_text(MINIMAL_YAML)
        with patch.dict(os.environ, {"UMRS_SITE_CONFIG": str(p)}):
            cfg = load_site_config()
            assert cfg.name == "test-site"

    def test_search_path_uses_resolved_umrs_config_dir(self, tmp_path, monkeypatch):
        xdg_root = tmp_path / "xdg"
        p = xdg_root / "umrs" / "site.yaml"
        p.parent.mkdir(parents=True)
        p.write_text(MINIMAL_YAML)

        monkeypatch.delenv("UMRS_SITE_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_root))
        monkeypatch.chdir(tmp_path)

        cfg = load_site_config()

        assert cfg.name == "test-site"

    def test_search_path_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "site.yaml").write_text(MINIMAL_YAML)
        # Clear UMRS_SITE_CONFIG so CWD is tried
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_site_config()
            assert cfg.name == "test-site"


# =============================================================================
# Tests — to_env / inject_env
# =============================================================================


class TestToEnv:
    """Test SiteConfig ↔ environment variable bridging."""

    def test_to_env_contains_scheduler(self):
        cfg = SiteConfig(
            scheduler=SchedulerConfig(partition="big-gpu", gpus=4, memory="128G"),
        )
        env = cfg.to_env()
        assert env["SLURM_PARTITION"] == "big-gpu"
        assert env["SLURM_GPUS"] == "4"
        assert env["SLURM_MEM"] == "128G"

    def test_to_env_contains_network(self):
        cfg = SiteConfig(
            network=NetworkConfig(
                litellm_port=9001,
                litellm_master_key="sk-x",
                egress_proxy_url="http://egress-guard:8888",
                egress_no_proxy="localhost,vllm",
            ),
        )
        env = cfg.to_env()
        assert env["NGINX_LITELLM_DOCKER_PORT"] == "9001"
        assert env["LITELLM_MASTER_KEY"] == "sk-x"
        assert env["RIVAULT_REMOTE_PORT"] == "9001"
        assert env["RIVAULT_HTTP_PROXY"] == "http://egress-guard:8888"
        assert env["RIVAULT_HTTPS_PROXY"] == "http://egress-guard:8888"
        assert env["EGRESS_GUARD_PROXY_URL"] == "http://egress-guard:8888"
        assert env["RIVAULT_NO_PROXY"] == "localhost,vllm"

    def test_to_env_contains_vllm(self):
        cfg = SiteConfig(
            vllm=VllmConfig(
                model="llama-3",
                docker_tag="nightly",
                gpu_mem_util=0.85,
                enable_prefix_caching=True,
            ),
        )
        env = cfg.to_env()
        assert env["VLLM_MODEL"] == "llama-3"
        assert env["VLLM_DOCKER_TAG"] == "nightly"
        assert env["VLLM_GPU_MEM_UTIL"] == "0.85"
        assert env["VLLM_ENABLE_PREFIX_CACHING"] == "true"

    def test_to_env_enables_postgres_registry_when_dsn_present(self):
        cfg = SiteConfig(
            database=DatabaseConfig(registry_dsn="postgres://umrs:umrs_rccs@localhost:5432/umrs_db")
        )
        env = cfg.to_env()
        assert env["UMRS_REGISTRY_BACKEND"] == "postgres"
        assert (
            env["UMRS_REGISTRY_POSTGRES_DSN"] == "postgres://umrs:umrs_rccs@localhost:5432/umrs_db"
        )

    def test_to_env_omits_empty_values(self):
        cfg = SiteConfig()  # all defaults
        env = cfg.to_env()
        # partition is "" by default → should not appear
        assert "SLURM_PARTITION" not in env

    def test_inject_env_no_overwrite(self, monkeypatch):
        monkeypatch.setenv("SLURM_PARTITION", "already-set")
        cfg = SiteConfig(scheduler=SchedulerConfig(partition="from-yaml"))
        cfg.inject_env(overwrite=False)
        assert os.environ["SLURM_PARTITION"] == "already-set"

    def test_inject_env_overwrite(self, monkeypatch):
        monkeypatch.setenv("SLURM_PARTITION", "already-set")
        cfg = SiteConfig(scheduler=SchedulerConfig(partition="from-yaml"))
        cfg.inject_env(overwrite=True)
        assert os.environ["SLURM_PARTITION"] == "from-yaml"


# =============================================================================
# Tests — default_site_config (env-var fallback)
# =============================================================================


class TestDefaultSiteConfig:
    """Test env-var-only fallback path."""

    def test_reads_env_vars(self, monkeypatch):
        with patch.dict(os.environ, {}, clear=True):
            monkeypatch.setenv("SLURM_PARTITION", "env-part")
            monkeypatch.setenv("RIVAULT_REMOTE_PORT", "9999")
            monkeypatch.setenv("VLLM_MODEL", "custom-model")
            monkeypatch.setenv("POSTGRES_USER", "pg_user_env")

            cfg = default_site_config()
        assert cfg.name == "env-fallback"
        assert cfg.scheduler.partition == "env-part"
        assert cfg.vllm.model == "custom-model"
        assert cfg.database.superuser == "pg_user_env"

    def test_defaults_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = default_site_config()
            assert cfg.scheduler.kind == "slurm"
            assert cfg.network.litellm_port == 8001


# =============================================================================
# Tests — singleton
# =============================================================================


class TestGetSiteConfig:
    """Test singleton behaviour."""

    def test_returns_same_instance(self, tmp_path):
        p = tmp_path / "site.yaml"
        p.write_text(MINIMAL_YAML)
        cfg1 = get_site_config(str(p))
        cfg2 = get_site_config()  # cached
        assert cfg1 is cfg2

    def test_fallback_when_no_yaml(self):
        with patch.dict(os.environ, {"UMRS_SITE_CONFIG": ""}, clear=False):
            # Ensure no site.yaml exists in CWD or home
            cfg = get_site_config()
            assert cfg is not None
            assert cfg.name in ("default", "env-fallback")

    def test_reset_clears_cache(self, tmp_path):
        p = tmp_path / "site.yaml"
        p.write_text(MINIMAL_YAML)
        cfg1 = get_site_config(str(p))
        reset_site_config()
        p.write_text(FULL_YAML)
        cfg2 = get_site_config(str(p))
        assert cfg1.name == "test-site"
        assert cfg2.name == "full-site"


# =============================================================================
# Tests — _parse_raw
# =============================================================================


class TestParseRaw:
    """Test raw dict → SiteConfig conversion."""

    def test_empty_dict(self):
        cfg = _parse_raw({})
        assert cfg.name == "default"
        assert cfg.scheduler.kind == "slurm"

    def test_partial_scheduler(self):
        cfg = _parse_raw({"scheduler": {"partition": "xgpu"}})
        assert cfg.scheduler.partition == "xgpu"
        assert cfg.scheduler.gpus == 1  # default preserved


class TestEgressProxyImage:
    """ADR-040: the site declares the in-stack egress proxy image (or nothing)."""

    def test_network_env_exports_declared_image_only(self):
        from umrs.orchestrator import site_config as sc

        assert "EGRESS_GUARD_IMAGE" not in sc._network_env(NetworkConfig())
        env = sc._network_env(NetworkConfig(egress_proxy_image="localhost/r4-mcp:2026.09"))
        assert env["EGRESS_GUARD_IMAGE"] == "localhost/r4-mcp:2026.09"

    def test_yaml_key_is_loaded(self, tmp_path):
        path = tmp_path / "site.yaml"
        path.write_text(
            textwrap.dedent(
                """
                network:
                  egress_proxy_image: registry.example.org/r4-mcp@sha256:abc
                """
            )
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EGRESS_GUARD_IMAGE", None)
            cfg = load_site_config(str(path))
        assert cfg.network.egress_proxy_image == "registry.example.org/r4-mcp@sha256:abc"

    def test_default_is_empty_meaning_no_proxy_declared(self):
        assert NetworkConfig().egress_proxy_image == ""
