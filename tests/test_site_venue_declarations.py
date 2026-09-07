"""Venue declarations per site + compute-shim convergence pins (ADR-045).

rccs (the WP-01 HPC venue) declares the pool and the image archive dir the
wrapper needs; the front overlay carries no compute-side knobs but must not
drift from the compute file's partition profiles; every podman-runtime
cluster now submits the same compute shim.
"""

from __future__ import annotations

import filecmp
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from umrs.orchestrator.site_config import load_site_config

UMRS_ROOT = Path(__file__).resolve().parents[1]
DEPLOY = UMRS_ROOT / "deploy"
PODMAN_CLUSTERS = ("rccs", "bdrt", "octopus")


class TestRccsDeclaresTheVenue:
    def test_pool_and_archive_dir(self):
        cfg = load_site_config(str(DEPLOY / "rccs" / "site.yaml"))
        assert cfg.network.private_subnet_pool == "172.16.0.0/16"
        assert cfg.runtime.image_archive_dir.endswith("/umrs_models/images")
        env = cfg.to_env()
        assert env["PRIV_SUBNET_POOL"] == "172.16.0.0/16"
        assert env["RIVAULT_IMAGE_ARCHIVE_DIR"] == cfg.runtime.image_archive_dir
        assert env["RIVAULT_CGROUP_MANAGER"] == "cgroupfs"

    def test_archive_dir_is_per_user_not_a_hardcoded_person(self):
        line = next(
            line
            for line in (DEPLOY / "rccs" / "site.yaml").read_text().splitlines()
            if "image_archive_dir:" in line
        )
        assert "${USER}" in line and "masaru" not in line

    def test_archive_dir_shares_the_vision_tarball_tree(self):
        cfg = load_site_config(str(DEPLOY / "rccs" / "site.yaml"))
        assert Path(cfg.runtime.image_archive_dir).parent == Path(cfg.storage.vision_tarball).parent


class TestFrontOverlayDoesNotDrift:
    def test_front_partition_profiles_are_a_subset_of_compute(self):
        compute = load_site_config(str(DEPLOY / "rccs" / "site.yaml")).partition_profiles
        front = load_site_config(str(DEPLOY / "rccs-front" / "site.yaml")).partition_profiles
        assert set(front) <= set(compute)
        for name, profile in front.items():
            assert profile == compute[name], name

    def test_front_points_at_compute_file_for_venue_knobs(self):
        text = (DEPLOY / "rccs-front" / "site.yaml").read_text()
        assert "../rccs/site.yaml" in text
        cfg = load_site_config(str(DEPLOY / "rccs-front" / "site.yaml"))
        assert cfg.runtime.image_archive_dir == "" and cfg.network.private_subnet_pool == ""


class TestAllSitesStillLoad:
    @pytest.mark.parametrize("site", sorted(p.parent.name for p in DEPLOY.glob("*/site.yaml")))
    def test_site_loads_and_pool_is_well_formed(self, site):
        cfg = load_site_config(str(DEPLOY / site / "site.yaml"))
        pool = cfg.network.private_subnet_pool
        assert pool == "" or re.fullmatch(r"\d+\.\d+\.0\.0/16|\d+\.\d+\.\d+\.0/24", pool), (
            site,
            pool,
        )


class TestComputeShimConvergence:
    def test_shims_are_identical_across_podman_clusters(self):
        ref = DEPLOY / "rccs" / "serve-model.slurm"
        for cluster in PODMAN_CLUSTERS[1:]:
            assert filecmp.cmp(ref, DEPLOY / cluster / "serve-model.slurm", shallow=False), cluster

    def test_shim_contract_header_paths_and_exec(self):
        text = (DEPLOY / "rccs" / "serve-model.slurm").read_text()
        assert 'echo "Fully qualified name: $(hostname -f)"' in text  # launcher parses this
        assert "#SBATCH --signal=B:TERM@120" in text  # graceful teardown window
        assert "RIVAULT_WRAPPER_DIR" in text and "UMRS_SITE_CONFIG" in text
        assert text.rstrip().endswith('exec bash --norc --noprofile ./serve-model.slurm "$@"')

    def test_shim_code_has_nothing_site_specific_or_concurrency_hostile(self):
        # Comments may name the legacy parts as history; code may not carry them.
        text = (DEPLOY / "rccs" / "serve-model.slurm").read_text()
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        forbidden = (
            "/lvs0/",
            "masaru",
            "storage.conf",
            "conflist",
            "pkill",
            "podman rm -af",
            "system migrate",
        )
        present = [token for token in forbidden if token in code]
        assert present == []

    def test_shim_syntax(self):
        r = subprocess.run(
            ["bash", "-n", str(DEPLOY / "rccs" / "serve-model.slurm")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
    def test_shim_and_staging_scripts_shellcheck_clean(self):
        files = [
            DEPLOY / "rccs" / "serve-model.slurm",
            DEPLOY / "_common" / "stage_images.sh",
            DEPLOY / "_common" / "cluster-context.sh",
        ]
        r = subprocess.run(
            ["shellcheck", "-s", "bash", *map(str, files)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0, r.stdout

    def test_stage_images_shims_exist_for_podman_clusters(self):
        for cluster in PODMAN_CLUSTERS:
            shim = DEPLOY / cluster / "stage_images.sh"
            assert shim.exists(), cluster
            assert "_common/stage_images.sh" in shim.read_text()


class TestWrapperUsesTheVenueLibrary:
    WRAPPER = UMRS_ROOT / "src" / "umrs" / "mini-rivault-llm" / "serve-model.slurm"

    def test_wrapper_sources_and_calls_the_library(self):
        text = self.WRAPPER.read_text()
        assert 'source "${SCRIPT_DIR}/venue.sh"' in text
        for call in (
            "rivault_venue_prepare",
            "rivault_net_gc",
            "rivault_net_create",
            "rivault_ensure_image",
            "rivault_write_gpu_override",
        ):
            assert call in text, call

    def test_wrapper_no_longer_writes_shared_home_podman_config(self):
        text = self.WRAPPER.read_text()
        assert '> "${HOME}/.config/containers/storage.conf"' not in text
        assert "with any candidate subnet" not in text  # the three-candidate loop is gone

    def test_wrapper_applies_partition_profiles_itself(self):
        text = self.WRAPPER.read_text()
        assert "cfg.partition_env(partition)" in text
        assert 'os.environ.get("SLURM_JOB_PARTITION")' in text
