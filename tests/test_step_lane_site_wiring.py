"""The four umrs-front sites keep uploaded step code off until their HPC lane
is live (ADR-074; superproject docs/development/rccs-step-lane-spec-2026-09-26.md
§1 and §10).

rccs carries every lane knob but ``remote_enabled: false`` and no image — the
converge composes the image only after it verified the staged archive — so the
lane opens only when the maintainer flips ``remote_enabled`` AND an image is
present. bdrt and octopus carry their lane sizing and gpu_form, closed and
without lane paths until onboarded (2026-09-27: bdrt awaits its R2 probe,
octopus a runtime decision; a declared helper + workdir root starts the
reaper's hourly SSH); ai4s has none. All four refuse uploaded code by name, and
each umrs-front server reads only its own cluster's UMRS_<C>_STEP_* from the
converge's .env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jsonschema
import pytest
import yaml
from umrs.orchestrator.site_config import load_site_config, reset_site_config
from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.pipeline_sdk.step import StepContext
from umrs.server import step_reaper
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor, _StepJob
from umrs.server.step_trust import hpc_lane_available, uploaded_code_lane

UMRS_ROOT = Path(__file__).resolve().parents[1]
DEPLOY = UMRS_ROOT / "deploy"
FRONTS = ("rccs-front", "ai4s-front", "bdrt-front", "octopus-front")
RCCS_CHECKOUT = "/lvs0/rccs-dmp/masaru.nagaso/ai4s-platform"
HELPER_REL = "umrs/deploy/_common/umrs_step_helper.py"
VENUE_REL = "umrs/src/umrs/mini-rivault-llm/venue.sh"
#: mng02's home; /home/mnagaso does not exist there (survey 2026-09-26).
OCTOPUS_HOME = "/lustre/home/mnagaso"
SCHEMA = UMRS_ROOT / "src" / "umrs" / "orchestrator" / "site-config.schema.json"
COMPOSE = DEPLOY / "umrs-front" / "docker-compose.yaml"
#: The umrs-front server of each front, keyed by cluster name (the .env's <C>).
CLUSTERS = ("rccs", "ai4s", "bdrt", "octopus")
STEP_ENV = ("UMRS_STEP_IMAGE", "UMRS_STEP_IMAGE_ARCHIVE_DIR", "UMRS_STEP_HELPER",
            "UMRS_STEP_VENUE_SCRIPT")  # fmt: skip

_LANE_ENV = (
    "UMRS_STEP_EXECUTOR",
    "UMRS_STEP_IMAGE",
    "UMRS_STEP_GPU",
    "UMRS_STEP_REMOTE_ENABLED",
    "UMRS_STEP_UPLOADED_CODE",
    "UMRS_STEP_IMAGE_ARCHIVE_DIR",
    "UMRS_STEP_ALLOWED_IMAGES",
    "UMRS_STEP_HELPER",
    "UMRS_STEP_VENUE_SCRIPT",
    "UMRS_SLURM_WORKDIR_ROOT",
    "UMRS_SLURM_STEP_SUBMIT",
    "UMRS_SLURM_QUEUE_TIMEOUT",
    "UMRS_STEP_GPU_FORM",
)


@pytest.fixture
def site_env(monkeypatch):
    """Point the site-config singleton at one front's site.yaml, clean env."""

    def use(front: str, **overrides: str) -> None:
        for key in _LANE_ENV:
            monkeypatch.delenv(key, raising=False)
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("UMRS_SITE_CONFIG", str(DEPLOY / front / "site.yaml"))
        reset_site_config()

    yield use
    reset_site_config()


def _step(front: str):
    return load_site_config(str(DEPLOY / front / "site.yaml")).step_executor


@pytest.mark.parametrize("front", FRONTS)
def test_every_front_declares_the_closed_lane_explicitly(front):
    # Declared, not inherited: a later change of the dataclass defaults must
    # not open a site's lane silently.
    raw = yaml.safe_load((DEPLOY / front / "site.yaml").read_text())["step_executor"]
    assert raw["remote_enabled"] is False
    assert raw["uploaded_code"] == "refuse"


@pytest.mark.parametrize("front", FRONTS)
def test_every_front_keeps_the_lane_closed_and_refuses_uploaded_code(front):
    step = _step(front)
    assert step.remote_enabled is False
    assert step.uploaded_code == "refuse"
    env = load_site_config(str(DEPLOY / front / "site.yaml")).to_env()
    assert "UMRS_STEP_REMOTE_ENABLED" not in env
    assert env["UMRS_STEP_UPLOADED_CODE"] == "refuse"
    assert "UMRS_STEP_IMAGE" not in env


@pytest.mark.parametrize("front", FRONTS)
def test_uploaded_code_is_refused_on_every_front(front, site_env):
    site_env(front)
    assert hpc_lane_available() is False
    assert uploaded_code_lane("in_process") is None


def test_rccs_has_no_default_image_or_gpu_and_workdirs_outside_the_checkout():
    step = _step("rccs-front")
    assert step.kind == "inproc"
    assert step.image == ""
    assert step.gpu is False
    assert step.workdir_root == "/lvs0/rccs-dmp/masaru.nagaso/umrs_steps"
    assert not step.workdir_root.startswith(RCCS_CHECKOUT + "/")


def test_rccs_lane_knobs_point_at_the_pinned_cluster_checkout():
    step = _step("rccs-front")
    assert step.image_archive_dir == "/lvs0/rccs-dmp/masaru.nagaso/umrs_models/images"
    assert step.helper_path == f"{RCCS_CHECKOUT}/{HELPER_REL}"
    assert step.venue_script == f"{RCCS_CHECKOUT}/{VENUE_REL}"
    assert (UMRS_ROOT.parent / VENUE_REL).is_file()


def test_rccs_exports_the_lane_env_without_gpu():
    env = load_site_config(str(DEPLOY / "rccs-front" / "site.yaml")).to_env()
    assert env["UMRS_SLURM_STEP_SUBMIT"] == "remote"
    assert env["UMRS_SLURM_WORKDIR_ROOT"] == "/lvs0/rccs-dmp/masaru.nagaso/umrs_steps"
    assert env["UMRS_STEP_HELPER"] == f"{RCCS_CHECKOUT}/{HELPER_REL}"
    assert env["UMRS_STEP_VENUE_SCRIPT"] == f"{RCCS_CHECKOUT}/{VENUE_REL}"
    assert "UMRS_STEP_GPU" not in env


def test_rccs_verified_image_alone_does_not_open_the_lane(site_env):
    site_env("rccs-front", UMRS_STEP_IMAGE="localhost/umrs-step:0123456789ab")
    assert hpc_lane_available() is False
    assert uploaded_code_lane("in_process") is None


def test_rccs_lane_opens_with_remote_enabled_and_a_verified_image(site_env):
    site_env(
        "rccs-front",
        UMRS_STEP_IMAGE="localhost/umrs-step:0123456789ab",
        UMRS_STEP_REMOTE_ENABLED="1",
    )
    assert hpc_lane_available() is True
    assert uploaded_code_lane("in_process") == "slurm"


def test_rccs_empty_converge_env_outranks_the_site_helper(site_env):
    # The umrs-front compose passes UMRS_STEP_HELPER="" until the converge has
    # verified the staged image; the site.yaml helper must not fill it in.
    site_env(
        "rccs-front",
        UMRS_STEP_IMAGE="localhost/umrs-step:0123456789ab",
        UMRS_STEP_REMOTE_ENABLED="1",
        UMRS_STEP_HELPER="",
    )
    assert hpc_lane_available() is False


def _site_executor(monkeypatch, front: str, **site_overrides) -> RemoteSlurmStepExecutor:
    """The executor as the front's server builds it: site.yaml -> env -> fields."""
    cfg = load_site_config(str(DEPLOY / front / "site.yaml"))
    for key, value in site_overrides.items():
        setattr(cfg.step_executor, key, value)
    for key in [k for k in os.environ if k.startswith(("UMRS_SLURM_", "UMRS_STEP_"))]:
        monkeypatch.delenv(key)
    for key, value in cfg.to_env().items():
        monkeypatch.setenv(key, value)
    return RemoteSlurmStepExecutor(
        ssh_config=SSHControlConfig(ssh_target="login"),
        image="localhost/umrs-step:0123456789ab",
    )


def _rendered(ex: RemoteSlurmStepExecutor) -> str:
    job = _StepJob("wf-0123456789ab", "s1", 1, "umrs-step-wf-0123456789ab-s1",
                   "wf-0123456789ab-s1-1-", workdir=f"{ex.workdir_root}/wf-x-abcdefgh")  # fmt: skip
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output",
                      run_id="wf-0123456789ab")  # fmt: skip
    return ex._job_script(job, ctx, ex.image)


def test_rccs_step_jobs_ask_for_no_memory_and_no_gres(monkeypatch):
    # memory: "" never reaches the environment (to_env drops empty values);
    # the base lane's 32G default would make every ng-dgx-m* submit fail.
    script = _rendered(_site_executor(monkeypatch, "rccs-front"))
    assert "#SBATCH --mem" not in script and "#SBATCH --gres" not in script
    assert "#SBATCH --partition=ng-dgx-m0,ng-dgx-m1,ng-dgx-m2,ng-dgx-m3" in script
    assert "#SBATCH --account=cloud-users" in script


def test_rccs_steps_run_on_plain_podman_bounded_by_nproc(monkeypatch):
    # 2026-10-01: the 2026-09-26 no-cgroup posture needed crun (runc refuses
    # NoCgroups) and RCCS cannot install it (STATE TODO-20260926-CRUN), so
    # the lane runs plain `podman run` on the nodes' stock runtime — no
    # --cgroups flag, no --pids-limit (rootless runc holds no pids cgroup
    # here); RLIMIT_NPROC bounds the step's processes instead.
    script = _rendered(_site_executor(monkeypatch, "rccs-front"))
    for part in ("podman run", "--ulimit nproc=4096"):
        assert part in script
    for part in ("--cgroups", "--cgroup-manager", "--pids-limit", "pids.max"):
        assert part not in script


def test_a_site_asking_for_cgroups_gets_the_pids_limit_and_its_probe(monkeypatch):
    ex = _site_executor(monkeypatch, "rccs-front", cgroups="enabled", pids_limit=512,
                        nproc_limit=0)  # fmt: skip
    script = _rendered(ex)
    assert "--pids-limit=512" in script and "pids.max" in script
    assert "--cgroups=disabled" not in script and "--ulimit nproc" not in script


def test_a_site_memory_value_still_reaches_the_job(monkeypatch):
    script = _rendered(_site_executor(monkeypatch, "rccs-front", memory="16G"))
    assert "#SBATCH --mem=16G" in script


def test_a_site_can_drop_the_pids_limit_its_runtime_cannot_apply(monkeypatch):
    ex = _site_executor(monkeypatch, "rccs-front", pids_limit=0)
    assert ex.pids_limit == 0
    script = _rendered(ex)
    assert "--pids-limit" not in script and "pids.max" not in script


@pytest.mark.parametrize("front", FRONTS)
def test_every_front_site_yaml_validates_against_the_schema(front):
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(yaml.safe_load((DEPLOY / front / "site.yaml").read_text()), schema)


@pytest.mark.parametrize("cluster", CLUSTERS)
def test_each_front_server_reads_only_its_own_clusters_step_env(cluster):
    # An image staged for one cluster's arch must not reach another's server.
    env = yaml.safe_load(COMPOSE.read_text())["services"][f"umrs-server-{cluster}"]["environment"]
    for key in STEP_ENV:
        assert env[key] == "${UMRS_" + cluster.upper() + "_" + key[len("UMRS_") :] + ":-}"
    others = [f"UMRS_{c.upper()}_STEP_" for c in CLUSTERS if c != cluster]
    assert not [v for v in map(str, env.values()) if any(o in v for o in others)]


@pytest.mark.parametrize(
    ("front", "form"), [("rccs-front", "cdi"), ("bdrt-front", "cdi"), ("octopus-front", "rocm")]
)
def test_each_lane_site_hands_its_executor_its_gpu_form(monkeypatch, front, form):
    env = load_site_config(str(DEPLOY / front / "site.yaml")).to_env()
    assert env["UMRS_STEP_GPU_FORM"] == form
    assert _site_executor(monkeypatch, front).gpu_form == form


def test_bdrt_lane_knobs_size_spark_double_jobs_with_no_image():
    step = _step("bdrt-front")
    assert (step.kind, step.partition, step.gpus, step.cpus, step.memory) == (
        "inproc", "spark-double", 0, 8, "",
    )  # fmt: skip
    assert (step.time_limit, step.account, step.image, step.gpu) == ("01:00:00", "", "", False)


#: The lane paths; helper_path + workdir_root start the step reaper.
LANE_PATHS = ("workdir_root", "image_archive_dir", "helper_path", "venue_script")


@pytest.mark.parametrize("front", ["bdrt-front", "octopus-front"])
def test_a_site_not_yet_onboarded_declares_no_lane_paths_and_gets_no_reaper(monkeypatch, front):
    # Per-site wiring review 2026-09-27: a declared helper + workdir root
    # starts the reaper, whose hourly helper sweep SSHes to the cluster with
    # the lane off and which reads the registry every server shares.
    raw = yaml.safe_load((DEPLOY / front / "site.yaml").read_text())["step_executor"]
    assert not set(LANE_PATHS) & set(raw)
    for key in [k for k in os.environ if k.startswith(("UMRS_SLURM_", "UMRS_STEP_"))]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("UMRS_STEP_HELPER", "")  # compose maps the absent .env line to ""
    monkeypatch.setenv("UMRS_SITE_CONFIG", str(DEPLOY / front / "site.yaml"))
    reset_site_config()
    step_reaper.set_step_lane_reaper(None)
    try:
        assert step_reaper.step_lane_reaper() is None
    finally:
        step_reaper.set_step_lane_reaper(None)
        reset_site_config()


def test_no_umrs_code_reads_a_clusters_raw_step_env():
    # env_file: .env hands every server every cluster's UMRS_<C>_STEP_* lines;
    # the isolation is the compose mapping to UMRS_STEP_*, so no code may read
    # a raw per-cluster key (per-site wiring review 2026-09-27).
    raw = [f"UMRS_{c.upper()}_STEP_" for c in CLUSTERS]
    readers = [
        str(path.relative_to(UMRS_ROOT))
        for path in (UMRS_ROOT / "src").rglob("*.py")
        if any(key in path.read_text(encoding="utf-8") for key in raw)
    ]
    assert readers == []


def test_bdrt_posture_waits_for_the_maintainers_r2_probe():
    # Set from the probe's receipt, not guessed: the defaults until then.
    raw = yaml.safe_load((DEPLOY / "bdrt-front" / "site.yaml").read_text())["step_executor"]
    assert not {"cgroups", "pids_limit", "nproc_limit"} & set(raw)


def test_bdrt_every_scheduler_partition_has_an_aarch64_profile():
    cfg = load_site_config(str(DEPLOY / "bdrt-front" / "site.yaml"))
    for name in cfg.scheduler.partition.split(","):
        assert cfg.partition_profiles[name.strip()].arch == "aarch64", name
    assert cfg.partition_profiles["edge-double"] == cfg.partition_profiles["spark-double"]


def test_bdrt_step_jobs_go_to_spark_double_without_mem_gres_or_account(monkeypatch):
    script = _rendered(_site_executor(monkeypatch, "bdrt-front"))
    assert "#SBATCH --partition=spark-double" in script
    for part in ("#SBATCH --mem", "#SBATCH --gres", "#SBATCH --account"):
        assert part not in script


def _strings(node):
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, str):
        yield node


def test_octopus_paths_are_on_the_octopus_home():
    raw = yaml.safe_load((DEPLOY / "octopus-front" / "site.yaml").read_text())
    assert not [s for s in _strings(raw) if s.startswith("/home/")]
    cfg = load_site_config(str(DEPLOY / "octopus-front" / "site.yaml"))
    for path in (
        cfg.scheduler.service_pubkey_path,
        cfg.scheduler.slurm_proxy_path,
        cfg.storage.sam3_model_path,
        cfg.extra["slurm_job_info_dir"],
        cfg.extra["slurm_wrapper_script"],
    ):
        assert path.startswith(OCTOPUS_HOME + "/"), path


def test_octopus_lane_is_closed_on_x86_64_with_rocm_and_no_partition_yet():
    step = _step("octopus-front")
    assert (step.kind, step.image, step.gpu, step.gpu_form) == ("inproc", "", False, "rocm")
    assert step.partition == ""  # set with the runtime decision (mng02 has no podman)
    profiles = load_site_config(str(DEPLOY / "octopus-front" / "site.yaml")).partition_profiles
    assert {p.arch for p in profiles.values()} == {"x86_64"}
