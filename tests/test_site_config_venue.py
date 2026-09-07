"""SiteConfig venue knobs + ``extends`` inheritance (ADR-045)."""

from __future__ import annotations

import dataclasses
import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from umrs.orchestrator import site_config as sc
from umrs.orchestrator.site_config import (
    NetworkConfig,
    RuntimeConfig,
    default_site_config,
    load_site_config,
    reset_site_config,
)

UMRS_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = UMRS_ROOT / "src" / "umrs" / "orchestrator" / "site-config.schema.json"


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_site_config()
    yield
    reset_site_config()


class TestVenueEnvExport:
    def test_runtime_env_exports_only_declared_knobs(self):
        venue_keys = {
            "RIVAULT_STORAGE_ROOT",
            "RIVAULT_NETWORK_CONFIG_DIR",
            "RIVAULT_IMAGE_ARCHIVE_DIR",
            "RIVAULT_CGROUP_MANAGER",
        }
        assert not venue_keys & set(sc._runtime_env(RuntimeConfig(cgroup_manager="")))
        env = sc._runtime_env(
            RuntimeConfig(
                storage_root="/scratch/p/storage",
                network_config_dir="/scratch/p/networks",
                image_archive_dir="/shared/images",
                cgroup_manager="systemd",
            )
        )
        assert env["RIVAULT_STORAGE_ROOT"] == "/scratch/p/storage"
        assert env["RIVAULT_NETWORK_CONFIG_DIR"] == "/scratch/p/networks"
        assert env["RIVAULT_IMAGE_ARCHIVE_DIR"] == "/shared/images"
        assert env["RIVAULT_CGROUP_MANAGER"] == "systemd"

    def test_network_env_exports_pool_only_when_set(self):
        assert "PRIV_SUBNET_POOL" not in sc._network_env(NetworkConfig())
        assert (
            sc._network_env(NetworkConfig(private_subnet_pool="10.200.0.0/16"))["PRIV_SUBNET_POOL"]
            == "10.200.0.0/16"
        )

    def test_defaults_are_empty_so_the_wrapper_decides(self):
        assert RuntimeConfig().image_archive_dir == ""
        assert RuntimeConfig().network_config_dir == ""
        assert NetworkConfig().private_subnet_pool == ""

    def test_env_only_fallback_reads_the_venue_variables(self):
        with patch.dict(
            os.environ,
            {
                "RIVAULT_STORAGE_ROOT": "/a",
                "RIVAULT_NETWORK_CONFIG_DIR": "/b",
                "RIVAULT_IMAGE_ARCHIVE_DIR": "/c",
                "PRIV_SUBNET_POOL": "172.30.0.0/16",
            },
        ):
            cfg = default_site_config()
        assert (
            cfg.runtime.storage_root,
            cfg.runtime.network_config_dir,
            cfg.runtime.image_archive_dir,
        ) == ("/a", "/b", "/c")
        assert cfg.network.private_subnet_pool == "172.30.0.0/16"


class TestExtends:
    def _write(self, path: Path, body: str) -> Path:
        path.write_text(textwrap.dedent(body))
        return path

    def _compute_and_front(self, tmp_path):
        (tmp_path / "compute").mkdir()
        (tmp_path / "front").mkdir()
        self._write(
            tmp_path / "compute" / "site.yaml",
            """
            name: cluster-a
            scheduler:
              kind: slurm-podman
              partition: gpu
            runtime:
              profiles: [harness, vllm]
              image_archive_dir: /shared/images
            network:
              private_subnet_pool: 172.16.0.0/16
            partition_profiles:
              gpu:
                arch: x86_64
                gpu_vendor: nvidia
                needs_netavark_switch: true
            """,
        )
        return self._write(
            tmp_path / "front" / "site.yaml",
            """
            extends: ../compute/site.yaml
            name: cluster-a-front
            scheduler:
              kind: slurm-remote
            runtime:
              profiles: [services]
            partition_profiles:
              gpu:
                needs_netavark_switch: false
            """,
        )

    def test_overlay_scalars_and_lists_replace_and_the_rest_is_inherited(self, tmp_path):
        cfg = load_site_config(str(self._compute_and_front(tmp_path)))
        assert cfg.name == "cluster-a-front"
        assert cfg.scheduler.kind == "slurm-remote"
        assert cfg.scheduler.partition == "gpu"  # inherited
        assert cfg.runtime.profiles == ["services"]  # lists replace
        assert cfg.runtime.image_archive_dir == "/shared/images"  # inherited
        assert cfg.network.private_subnet_pool == "172.16.0.0/16"

    def test_overlay_partition_profiles_merge_field_by_field(self, tmp_path):
        cfg = load_site_config(str(self._compute_and_front(tmp_path)))
        profile = cfg.partition_profiles["gpu"]
        assert profile.arch == "x86_64"
        assert profile.gpu_vendor == "nvidia"
        assert profile.needs_netavark_switch is False  # one field overridden

    def test_extends_path_expands_env_and_accepts_absolute(self, tmp_path):
        base = self._write(tmp_path / "base.yaml", "name: base\nnetwork:\n  litellm_port: 9001\n")
        overlay = self._write(tmp_path / "o.yaml", f"extends: {base}\nname: o\n")
        assert load_site_config(str(overlay)).network.litellm_port == 9001
        with patch.dict(os.environ, {"SITE_BASE_DIR": str(tmp_path)}):
            overlay2 = self._write(
                tmp_path / "o2.yaml", "extends: ${SITE_BASE_DIR}/base.yaml\nname: o2\n"
            )
            assert load_site_config(str(overlay2)).network.litellm_port == 9001

    def test_extends_cycle_raises(self, tmp_path):
        self._write(tmp_path / "a.yaml", "extends: b.yaml\nname: a\n")
        self._write(tmp_path / "b.yaml", "extends: a.yaml\nname: b\n")
        with pytest.raises(ValueError, match="extends cycle"):
            load_site_config(str(tmp_path / "a.yaml"))

    def test_missing_base_is_a_hard_error(self, tmp_path):
        self._write(tmp_path / "a.yaml", "extends: nope.yaml\nname: a\n")
        with pytest.raises(FileNotFoundError):
            load_site_config(str(tmp_path / "a.yaml"))

    def test_extends_key_never_leaks_into_extra(self, tmp_path):
        self._write(tmp_path / "base.yaml", "name: base\n")
        cfg = load_site_config(
            str(self._write(tmp_path / "o.yaml", "extends: base.yaml\nname: o\n"))
        )
        assert "extends" not in cfg.extra


class TestSchemaDrift:
    """The yaml-language-server schema must know every dataclass field."""

    PAIRS = {
        "scheduler": sc.SchedulerConfig,
        "runtime": sc.RuntimeConfig,
        "storage": sc.StorageConfig,
        "network": sc.NetworkConfig,
        "vllm": sc.VllmConfig,
        "step_executor": sc.StepExecutorConfig,
        "database": sc.DatabaseConfig,
        "partition_profile": sc.PartitionProfile,
    }

    def test_every_dataclass_field_is_in_the_schema(self):
        schema = json.loads(SCHEMA.read_text())
        for key, cls in self.PAIRS.items():
            fields = {f.name for f in dataclasses.fields(cls)}
            props = set(schema["$defs"][key]["properties"])
            assert fields <= props, f"{key}: schema lacks {sorted(fields - props)}"
            assert props <= fields, f"{key}: schema has stale {sorted(props - fields)}"

    def test_top_level_keys_including_extends(self):
        schema = json.loads(SCHEMA.read_text())
        top = set(schema["properties"])
        assert {"extends", "strict_secrets", "partition_profiles", "extra"} <= top
        for f in dataclasses.fields(sc.SiteConfig):
            assert f.name in top, f.name

    def test_venue_knobs_documented(self):
        text = SCHEMA.read_text()
        for key in (
            "image_archive_dir",
            "network_config_dir",
            "private_subnet_pool",
            "egress_proxy_image",
        ):
            assert f'"{key}"' in text
