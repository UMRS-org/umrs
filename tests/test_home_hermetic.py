"""The umrs suite never sees the developer's real HOME (plan H-3.6, 2026-10-18).

``tests/conftest.py`` swaps HOME and the XDG base dirs for a session scratch
dir before any umrs module is imported. These pins fail if that swap is
dropped or moved after the umrs imports (import-time defaults such as
``DEFAULT_UMRS_CFG_DIR`` would then capture the real home again).
"""

from __future__ import annotations

import os
import pwd
from pathlib import Path

from umrs.orchestrator.site_config import RuntimeConfig, SiteConfig
from umrs.orchestrator.slurm_podman_backend import SlurmPodmanBackend
from umrs.shared import config as umrs_config
from umrs.shared import model_catalog


def _scratch_home() -> Path:
    home = Path(os.environ["HOME"]).resolve()
    assert home.name.startswith("umrs-tests-home-"), home
    return home


def _under_scratch_home(path: str | Path) -> bool:
    return Path(path).resolve().is_relative_to(_scratch_home())


def test_home_is_the_session_scratch_dir():
    home = _scratch_home()
    assert home != Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    assert os.access(home, os.W_OK)
    assert Path.home().resolve() == home


def test_xdg_base_dirs_live_under_the_scratch_home():
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        assert _under_scratch_home(os.environ[var]), var


def test_import_time_config_default_is_the_scratch_home():
    assert _under_scratch_home(umrs_config.DEFAULT_UMRS_CFG_DIR)
    assert _under_scratch_home(umrs_config.get_resolved_umrs_cfg_dir())


def test_model_catalog_path_is_under_the_scratch_home():
    assert _under_scratch_home(model_catalog._catalog_path())


def test_inherited_path_overrides_are_dropped():
    for var in ("UMRS_CONFIG", "RIVAULT_OUTPUT_DIR", "SLURM_JOB_INFO_DIR"):
        assert var not in os.environ, var


def test_slurm_backend_default_output_dir_is_under_the_scratch_home(tmp_path):
    script = tmp_path / "serve-model.slurm"
    script.write_text("#!/bin/bash\n")
    site = SiteConfig(name="hermetic", runtime=RuntimeConfig(rivault_dir="/opt/rivault"))
    backend = SlurmPodmanBackend(site=site, slurm_script=str(script))
    assert _under_scratch_home(backend._output_dir)
    assert backend._output_dir.is_dir()
