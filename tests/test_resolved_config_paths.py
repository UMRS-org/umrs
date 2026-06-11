import os
from pathlib import Path

from umrs.orchestrator.site_config import SiteConfig
from umrs.orchestrator.slurm_podman_backend import SlurmPodmanBackend
from umrs.pipeline_sdk.connectors.umrs_client import UMRSClientConnector
from umrs.server.config import (
    get_default_db_subdir,
    get_jens_rivault_slurm_script,
    get_rivault_slurm_script,
)
from umrs.server.model_registry_handler import _catalog_path as get_model_catalog_path
from umrs.server.workflow_registry_handler import _catalog_path as get_workflow_catalog_path
from umrs.shared.config import (
    get_resolved_json_server_path,
    get_resolved_model_staging_root,
)
from umrs.shared.storage.s3_presign import _get_presign_endpoint


def test_runtime_helpers_follow_umrs_config(monkeypatch, tmp_path):
    config_root = tmp_path / "config-root"
    expected_root = os.path.realpath(config_root)

    monkeypatch.setenv("UMRS_CONFIG", str(config_root))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("UMRS_MODEL_STAGING_ROOT", raising=False)
    monkeypatch.delenv("RIVAULT_SLURM_SCRIPT", raising=False)
    monkeypatch.delenv("JENS_RIVAULT_SLURM_SCRIPT", raising=False)

    assert get_resolved_json_server_path() == os.path.join(expected_root, "conf_umrs.json")
    assert get_resolved_model_staging_root() == os.path.join(expected_root, "models")
    assert get_default_db_subdir() == os.path.join(expected_root, "tasks_database")
    assert get_model_catalog_path() == os.path.join(expected_root, "model_catalog.json")
    assert get_workflow_catalog_path() == os.path.join(expected_root, "workflow_catalog.json")
    assert get_rivault_slurm_script("example.org") == os.path.join(
        expected_root,
        "rivault_data",
        "serve-model.slurm",
    )
    assert get_rivault_slurm_script("plafrim.example.org") == os.path.join(
        expected_root,
        "rivault_data",
        "serve-model-plafrim.slurm",
    )
    assert get_jens_rivault_slurm_script() == os.path.join(
        expected_root,
        "rivault_data",
        "serve-model-jens.slurm",
    )
    assert UMRSClientConnector().user_config == os.path.join(
        expected_root,
        "users",
        "user_rivault.json",
    )


def test_model_staging_root_prefers_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_CONFIG", str(tmp_path / "config-root"))
    monkeypatch.setenv("UMRS_MODEL_STAGING_ROOT", str(tmp_path / "custom-models"))

    assert get_resolved_model_staging_root() == os.path.realpath(tmp_path / "custom-models")


def test_slurm_podman_backend_uses_resolved_config_defaults(monkeypatch, tmp_path):
    config_root = tmp_path / "config-root"
    expected_root = Path(os.path.realpath(config_root))

    monkeypatch.setenv("UMRS_CONFIG", str(config_root))
    monkeypatch.delenv("RIVAULT_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("SLURM_JOB_INFO_DIR", raising=False)
    monkeypatch.delenv("RIVAULT_SLURM_SCRIPT", raising=False)

    backend = SlurmPodmanBackend(site=SiteConfig(name=""))

    assert backend._output_dir == expected_root / "tmp"
    assert backend.slurm_script == str(expected_root / "serve-model.slurm")


def test_presign_endpoint_reads_from_resolved_umrs_config_dir(monkeypatch, tmp_path):
    config_root = tmp_path / "config-root"
    config_root.mkdir(parents=True)
    (config_root / "s3_presign_endpoint").write_text(
        "http://presign.example:19000\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("UMRS_CONFIG", str(config_root))
    monkeypatch.delenv("S3_PRESIGN_ENDPOINT", raising=False)

    assert _get_presign_endpoint() == "http://presign.example:19000"
