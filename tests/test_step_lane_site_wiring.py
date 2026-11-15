"""The four umrs-front sites keep uploaded step code off until their HPC lane
is live (ADR-074; superproject docs/development/rccs-step-lane-spec-2026-09-26.md
§1 and §10).

rccs carries every lane knob, ``remote_enabled: true`` (maintainer flip
2026-10-02) and the GPU site gate, but no image — the converge composes the
image only after it verified the staged archive — so the lane opens only when
``remote_enabled`` AND an image are present. bdrt and octopus carry their lane sizing and gpu_form, closed and
without lane paths until onboarded (2026-09-27: bdrt awaits its R2 probe,
octopus a runtime decision; a declared helper + workdir root starts the
reaper's hourly SSH). ai4s (RIKYU) opens like rccs since 2026-11-08 (gpus flag);
since 2026-11-09 only spring-8 charges rkp00101, all other projects rkp00043.
All four refuse uploaded code in-process by name, and
each umrs-front server reads only its own cluster's UMRS_<C>_STEP_* from the
converge's .env.

2026-11-12 (ADR-112 B2): bdrt is the gated-run test lane. It declares
remote_enabled, the archive dir, helper and venue paths and image_digests, but
no workdir_root: with the converge's verified image only the gated path opens
(a person's own account, through their gate); the service-account helper lane
stays unavailable, the reaper is the gated-only one (no helper sweep), and
service-account user work stays admins only. rccs, ai4s and bdrt read the
converge-written digests file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jsonschema
import pytest
import yaml
from umrs.orchestrator import partition_choice as pc
from umrs.orchestrator.site_config import load_site_config, reset_site_config
from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.pipeline_sdk.step import StepContext
from umrs.server import step_reaper
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor, _StepJob
from umrs.server.step_executor import (
    EXECUTOR_GPU_COUNT_METADATA_KEY,
    EXECUTOR_GPU_METADATA_KEY,
    StepExecutionError,
)
from umrs.server.step_staging import STAGE_PROJECT_ID_METADATA_KEY
from umrs.server.step_trust import hpc_lane_available, uploaded_code_lane

UMRS_ROOT = Path(__file__).resolve().parents[1]
DEPLOY = UMRS_ROOT / "deploy"
FRONTS = ("rccs-front", "ai4s-front", "bdrt-front", "octopus-front")
#: rccs opened its lane 2026-10-02 (maintainer, after R2); the rest stay closed.
#: bdrt left this list 2026-11-12 (ADR-112 B2 gated test lane).
CLOSED_FRONTS = ("octopus-front",)
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


@pytest.mark.parametrize("front", CLOSED_FRONTS)
def test_every_front_declares_the_closed_lane_explicitly(front):
    # Declared, not inherited: a later change of the dataclass defaults must
    # not open a site's lane silently.
    raw = yaml.safe_load((DEPLOY / front / "site.yaml").read_text())["step_executor"]
    assert raw["remote_enabled"] is False
    assert raw["uploaded_code"] == "refuse"


@pytest.mark.parametrize("front", CLOSED_FRONTS)
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


def test_rccs_has_no_default_image_and_workdirs_outside_the_checkout():
    step = _step("rccs-front")
    assert step.kind == "inproc"
    assert step.image == ""
    # 2026-10-02 R3: the site GPU gate is open; a step still needs routing_hints.gpu.
    assert step.gpu is True
    assert step.workdir_root == "/lvs0/rccs-dmp/masaru.nagaso/umrs_steps"
    assert not step.workdir_root.startswith(RCCS_CHECKOUT + "/")


def test_rccs_lane_knobs_point_at_the_pinned_cluster_checkout():
    step = _step("rccs-front")
    assert step.image_archive_dir == "/lvs0/rccs-dmp/masaru.nagaso/umrs_models/images"
    assert step.helper_path == f"{RCCS_CHECKOUT}/{HELPER_REL}"
    assert step.venue_script == f"{RCCS_CHECKOUT}/{VENUE_REL}"
    assert (UMRS_ROOT.parent / VENUE_REL).is_file()


def test_rccs_exports_the_lane_env_with_the_gpu_gate():
    env = load_site_config(str(DEPLOY / "rccs-front" / "site.yaml")).to_env()
    assert env["UMRS_SLURM_STEP_SUBMIT"] == "remote"
    assert env["UMRS_SLURM_WORKDIR_ROOT"] == "/lvs0/rccs-dmp/masaru.nagaso/umrs_steps"
    assert env["UMRS_STEP_HELPER"] == f"{RCCS_CHECKOUT}/{HELPER_REL}"
    assert env["UMRS_STEP_VENUE_SCRIPT"] == f"{RCCS_CHECKOUT}/{VENUE_REL}"
    assert env["UMRS_STEP_GPU"] == "1"


def test_rccs_lane_opens_once_the_converge_verified_an_image(site_env):
    # 2026-10-02: site.yaml remote_enabled is true, so the converge's verified
    # image is the last switch; without it uploaded code stays refused.
    site_env("rccs-front")
    assert hpc_lane_available() is False
    assert uploaded_code_lane("in_process") is None
    site_env("rccs-front", UMRS_STEP_IMAGE="localhost/umrs-step:0123456789ab")
    assert hpc_lane_available() is True
    assert uploaded_code_lane("in_process") == "slurm"


def test_rccs_declares_its_lane_open_with_refuse_as_the_fallback():
    # Maintainer flip 2026-10-02 after the R2 receipt; without the converge's
    # verified image the lane is still unavailable and uploaded code refused.
    raw = yaml.safe_load((DEPLOY / "rccs-front" / "site.yaml").read_text())["step_executor"]
    assert raw["remote_enabled"] is True
    assert raw["uploaded_code"] == "refuse"
    assert _step("rccs-front").remote_enabled is True


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
        # the front server reads its partition profiles from this same file
        partition_profiles=cfg.partition_profiles,
    )


def _rendered(
    ex: RemoteSlurmStepExecutor, partition: str | None = None, metadata: dict | None = None
) -> str:
    job = _StepJob("wf-0123456789ab", "s1", 1, "umrs-step-wf-0123456789ab-s1",
                   "wf-0123456789ab-s1-1-", workdir=f"{ex.workdir_root}/wf-x-abcdefgh")  # fmt: skip
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output",
                      metadata=dict(metadata or {}), run_id="wf-0123456789ab")  # fmt: skip
    return ex._job_script(job, ctx, ex.image, partition)


def test_rccs_step_jobs_ask_for_no_memory_and_no_gres(monkeypatch):
    # memory: "" never reaches the environment (to_env drops empty values);
    # the base lane's 32G default would make every ng-dgx-m* submit fail.
    script = _rendered(_site_executor(monkeypatch, "rccs-front"))
    assert "#SBATCH --mem" not in script and "#SBATCH --gres" not in script
    # one partition: the rccs job_submit policy refuses a list (R3 2026-10-02)
    assert "#SBATCH --partition=ng-dgx-m3\n" in script
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
    ("front", "form"),
    [("rccs-front", "nvidia-devices"), ("bdrt-front", "cdi"), ("octopus-front", "rocm")],
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


@pytest.mark.parametrize("front", ["octopus-front"])
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


# ── bdrt: the ADR-112 B2 gated test lane (2026-11-12) ────────────────────────

BDRT_CHECKOUT = "/home/mnagaso/ai4s-platform"
BDRT_ARCHIVES = "/home/mnagaso/umrs_models/images"
DIGESTS_IN_SERVER = "/etc/umrs/step-digests/image-digests.json"
STEP_IMAGE = "localhost/umrs-step:0123456789ab"
#: What the converge writes into .env for bdrt once its image verified
#: (umrs-front.yaml "Step lane — write or remove ..."; the compose maps them).
BDRT_CONVERGE_ENV = {
    "UMRS_STEP_IMAGE": STEP_IMAGE,
    "UMRS_STEP_IMAGE_ARCHIVE_DIR": BDRT_ARCHIVES,
    "UMRS_STEP_HELPER": f"{BDRT_CHECKOUT}/{HELPER_REL}",
    "UMRS_STEP_VENUE_SCRIPT": f"{BDRT_CHECKOUT}/{VENUE_REL}",
}


def test_bdrt_declares_the_gated_lane_without_a_workdir_root():
    raw = yaml.safe_load((DEPLOY / "bdrt-front" / "site.yaml").read_text())["step_executor"]
    assert (raw["remote_enabled"], raw["uploaded_code"]) == (True, "refuse")
    assert "workdir_root" not in raw  # no service helper lane, no helper sweep
    step = _step("bdrt-front")
    assert step.image_archive_dir == BDRT_ARCHIVES
    assert step.image_digests == DIGESTS_IN_SERVER
    assert step.helper_path == BDRT_CONVERGE_ENV["UMRS_STEP_HELPER"]
    assert step.venue_script == BDRT_CONVERGE_ENV["UMRS_STEP_VENUE_SCRIPT"]


def test_bdrt_opens_only_the_gated_path_with_the_converge_image(site_env):
    from umrs.server.step_trust import gated_lane_available

    site_env("bdrt-front")  # before the converge verified an image: all closed
    assert (gated_lane_available(), hpc_lane_available()) == (False, False)
    assert uploaded_code_lane("in_process") is None
    site_env("bdrt-front", **BDRT_CONVERGE_ENV)
    assert gated_lane_available() is True
    assert hpc_lane_available() is False  # no workdir root: no service helper lane
    assert uploaded_code_lane("in_process") == "slurm"


def test_bdrt_gets_only_the_gated_reaper_with_the_converge_env(monkeypatch):
    # No workdir root: the reaper is the gated-only one (gate verbs; plan §5
    # B2.0 §10.5), whose helper sweep never SSHes under the platform login.
    for key in [k for k in os.environ if k.startswith(("UMRS_SLURM_", "UMRS_STEP_"))]:
        monkeypatch.delenv(key)
    for key, value in BDRT_CONVERGE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("UMRS_SITE_CONFIG", str(DEPLOY / "bdrt-front" / "site.yaml"))
    reset_site_config()
    step_reaper.set_step_lane_reaper(None)
    try:
        reaper = step_reaper.step_lane_reaper()
        assert (reaper.helper_path, reaper.workdir_root) == ("", "")
        assert reaper.sweep_workdirs() == {"removed": []}
    finally:
        step_reaper.set_step_lane_reaper(None)
        reset_site_config()


def _verified_alice(tables: dict) -> None:
    from umrs.hpc_accounts.store import InMemorySiteAccountStore

    store = InMemorySiteAccountStore("bdrt-front", tables)
    store.issue_key("alice", "tk-000000000001", "SHA256:x")
    store.record_verified(
        "alice",
        "tk-000000000001",
        {
            "site_login": "alice_hpc",
            "verified_login": "alice_hpc",
            "verified_assocs": [{"account": "grp-a"}],
            "gate_version": "2",
            "gate_sha256": "0" * 64,
            "signer_fingerprints": [],
            "uv_required": False,
        },
    )
    store.upsert_link("alice", "bdr-opossum", "grp-a")


@pytest.fixture
def bdrt_open(site_env, monkeypatch):
    """bdrt as its server sees it after the converge: alice enrolled, root admin."""
    from umrs.hpc_accounts.store import InMemorySiteAccountStore
    from umrs.shared import hpc_user_work_access as access

    site_env("bdrt-front", **BDRT_CONVERGE_ENV)
    site = load_site_config(str(DEPLOY / "bdrt-front" / "site.yaml"))
    tables: dict = {"accounts": {}, "links": {}}
    _verified_alice(tables)
    monkeypatch.setattr(access, "current_site", lambda: site)
    monkeypatch.setattr(access, "user_is_admin", lambda user: user == "root")
    monkeypatch.setattr(access, "_store_factory", lambda v: InMemorySiteAccountStore(v, tables))
    return access


def _uploaded(tmp_path: Path) -> tuple[dict, list, str]:
    (tmp_path / "pkg").mkdir()
    steps = [{"name": "s1", "entrypoint": "pkg.step:Step"}]
    return {"steps": steps}, steps, str(tmp_path)


def test_bdrt_service_account_user_work_stays_admins_only_with_the_lane_open(bdrt_open):
    from umrs.orchestrator.hpc_identity import HpcAccessRefused

    with pytest.raises(HpcAccessRefused) as refused:
        bdrt_open.resolve_user_work(user_id="bob", project_id="bdr-opossum", gated=True)
    assert refused.value.reason == "project_not_allowed"
    admin = bdrt_open.resolve_user_work(user_id="root", project_id="bdr-opossum", gated=True)
    assert (admin.mode, admin.login) == ("service_account", "mnagaso")
    person = bdrt_open.resolve_user_work(user_id="alice", project_id="bdr-opossum", gated=True)
    assert (person.mode, person.login, person.slurm_account) == (
        "user_account",
        "alice_hpc",
        "grp-a",
    )


def test_bdrt_runs_uploaded_code_only_under_a_persons_own_account(bdrt_open, tmp_path):
    from umrs.server import workflow_runner as wr

    wf, steps, src = _uploaded(tmp_path)

    def lane(user: str, intent: bool) -> tuple:
        return wr._workflow_executor_name(
            wf, steps, src, injected=False, run_identity=(user, "bdr-opossum", intent)
        )

    assert lane("alice", True) == ("slurm", "")
    name, refusal = lane("bob", False)  # a non-admin on the service account
    assert name is None and "project_not_allowed" in refusal
    assert lane("root", False) == (None, wr._NO_HPC_LANE)  # admin: no helper lane here


def test_bdrt_publishes_a_gate_profile_naming_its_archive_dir():
    from umrs.hpc_accounts.site_profile import build_site_profile

    profile = build_site_profile(_site("bdrt-front"))
    assert profile["image_archive_dir"] == BDRT_ARCHIVES
    assert {p["arch"] for p in profile["partitions"].values()} == {"aarch64"}


@pytest.mark.parametrize("front", FRONTS)
def test_staging_fronts_read_the_converge_written_digests_file(front):
    # The umrs-front converge writes it for rccs, ai4s and bdrt (they stage);
    # octopus stages nothing, so it names none (no_image_digests).
    want = "" if front == "octopus-front" else DIGESTS_IN_SERVER
    assert _step(front).image_digests == want
    if front != "bdrt-front":  # user_account needs the operator's consent there
        assert "user_account" not in _site(front).scheduler.hpc_access.modes_allowed


#: Exactly what the converge writes (deploy/ansible/scripts/
#: test-role-expressions.yaml "Gated-run digests file content" pins it).
CONVERGE_DIGESTS = (
    '{\n  "localhost/umrs-step:0123456789ab": {\n    "aarch64": {\n'
    f'      "archive_sha256": "{"b" * 64}",\n      "image_id": "{"a" * 64}"\n'
    "    }\n  }\n}\n"
)


def test_the_converge_digests_file_is_what_a_bdrt_proposal_pins(monkeypatch, tmp_path):
    from umrs.hpc_accounts.grant_service import site_images
    from umrs.hpc_accounts.site_profile import build_site_profile
    from umrs.server.intent_proposal import _image, load_image_digests, platform_image

    path = tmp_path / "image-digests.json"
    path.write_text(CONVERGE_DIGESTS)
    site = _site("bdrt-front")
    site.step_executor.image_digests = str(path)
    monkeypatch.setenv("UMRS_STEP_IMAGE", STEP_IMAGE)
    arch = build_site_profile(site)["partitions"][site.step_executor.partition]["arch"]
    image = _image(load_image_digests(site), platform_image(site), arch)
    assert image == {"ref": STEP_IMAGE, "image_id": "a" * 64, "archive_sha256": "b" * 64}
    assert site_images(site) == frozenset({(STEP_IMAGE, "a" * 64, "b" * 64)})


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
        cfg.storage.sam3_model_path,
        cfg.extra["slurm_job_info_dir"],
        cfg.extra["slurm_wrapper_script"],
    ):
        assert path.startswith(OCTOPUS_HOME + "/"), path


#: ADR-112 G-8: the maintainer's interim per-site lists ([] = admins only).
DEMO_PROJECTS = ["bdr-opossum", "spring-8", "brc-mouse", "default"]  # default: 2026-11-09
ALLOWED_PROJECTS = {
    "rccs-front": DEMO_PROJECTS,
    "ai4s-front": DEMO_PROJECTS,
    "bdrt-front": [],
    "octopus-front": [],
}
RETIRED_SCHEDULER_KEYS = (
    "slurm_delegation",
    "service_pubkey_path",
    "slurm_proxy_path",
    "require_user_slurm_account",
)


USER_ACCOUNT_MODES = {"bdrt-front": ["service_account", "user_account"]}


@pytest.mark.parametrize("front", FRONTS)
def test_every_front_declares_its_personal_service_account(front):
    # ADR-112 D-1: per_user_ssh is retired; each front declares the login every
    # lane uses, says it is personal, and lists who may run user work on it.
    raw = yaml.safe_load((DEPLOY / front / "site.yaml").read_text())
    assert not set(RETIRED_SCHEDULER_KEYS) & set(raw["scheduler"]), front
    hpc = load_site_config(str(DEPLOY / front / "site.yaml")).scheduler.hpc_access
    assert hpc is not None and hpc.operator, front
    # ADR-112 B1 (2026-11-08): bdrt also allows registration of user accounts;
    # the default (who runs user work) stays the service account everywhere.
    want = (USER_ACCOUNT_MODES.get(front, ["service_account"]), "service_account")
    assert (hpc.modes_allowed, hpc.default_mode) == want, front
    service = hpc.service_account
    assert service is not None
    assert service.login == raw["extra"]["ssh_user"]
    assert (service.personal, service.consent_ref) == (True, "")
    assert service.allowed_projects == ALLOWED_PROJECTS[front]


def test_octopus_lane_is_closed_on_x86_64_with_rocm_and_no_partition_yet():
    step = _step("octopus-front")
    assert (step.kind, step.image, step.gpu, step.gpu_form) == ("inproc", "", False, "rocm")
    assert step.partition == ""  # set with the runtime decision (mng02 has no podman)
    profiles = load_site_config(str(DEPLOY / "octopus-front" / "site.yaml")).partition_profiles
    assert {p.arch for p in profiles.values()} == {"x86_64"}


@pytest.mark.parametrize("front", ["rccs-front", "bdrt-front", "octopus-front", "ai4s-front"])
def test_allowed_images_are_pinned_and_archive_named(front):
    # D-7: an allow-listed image is loaded from <name>-<tag>-<arch>.tar, never
    # pulled; a moving tag (:latest, none) would let a rebuilt archive change
    # what an already-uploaded workflow runs.
    from umrs.server.slurm_step_executor import image_archive_stem

    for image in _step(front).allowed_images or []:
        leaf = image.rsplit("/", 1)[-1]
        assert ":" in leaf and not leaf.endswith(":latest"), image
        assert image_archive_stem(image)


# ── ADR-109 D-3 / V1: per-partition GPU request, ai-pro6000 not yet listed ───

#: rccs's compute-side file and the front's own (each server reads its own).
RCCS_SITES = ("rccs", "rccs-front")
GB10_PARTITIONS = ("ng-dgx-s", "ng-dgx-m0", "ng-dgx-m1", "ng-dgx-m2", "ng-dgx-m3")
PRO6000_GRES = "#SBATCH --gres=gpu:PRO6000-96GB:2"


def _site(site: str):
    return load_site_config(str(DEPLOY / site / "site.yaml"))


@pytest.mark.parametrize("site", RCCS_SITES)
def test_rccs_gb10_profiles_never_ask_for_gres_and_ai_pro6000_does(site):
    profiles = _site(site).partition_profiles
    for name in GB10_PARTITIONS:
        assert profiles[name].gpu_request == "none", name
    pro = profiles["ai-pro6000"]
    assert (pro.arch, pro.gpu_vendor, pro.gpu_memory_gb, pro.gpus) == ("aarch64", "nvidia", 96, 2)
    assert (pro.gpu_request, pro.gres, pro.exclusive) == ("gres", "gpu:PRO6000-96GB:2", True)


@pytest.mark.parametrize("site", RCCS_SITES)
def test_rccs_site_yaml_validates_against_the_schema(site):
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(yaml.safe_load((DEPLOY / site / "site.yaml").read_text()), schema)


@pytest.mark.parametrize("site", RCCS_SITES)
def test_ai_pro6000_is_in_no_candidate_list_before_its_v2_receipts(site):
    cfg = _site(site)
    assert "ai-pro6000" not in pc.split_partitions(cfg.scheduler.partition)
    assert "ai-pro6000" not in pc.split_partitions(cfg.step_executor.partition)
    for roster in (DEPLOY / site).glob("pool-*.yaml"):
        assert "ai-pro6000" not in roster.read_text(), roster.name
    if site == "rccs-front":
        assert cfg.step_executor.partition == "ng-dgx-m3"  # the live step lane


@pytest.mark.parametrize("partition", ["ng-dgx-m3", "ng-dgx-s"])
def test_rccs_gb10_step_scripts_are_byte_identical_with_their_profiles(monkeypatch, partition):
    ex = _site_executor(monkeypatch, "rccs-front")
    gpu_step = {EXECUTOR_GPU_METADATA_KEY: True}  # site gpu gate on + the hint
    script = _rendered(ex, partition, gpu_step)
    ex.partition_profiles = {}  # the rule before ADR-109: no profile consulted
    assert _rendered(ex, partition, gpu_step) == script
    assert f"#SBATCH --partition={partition}\n" in script
    assert "#SBATCH --gres" not in script and "#SBATCH --exclusive" not in script
    assert '"${_step_gpu[@]}"' in script  # the GPU still reaches the pod (nvidia-devices)


def test_rccs_front_ai_pro6000_step_script_asks_for_its_whole_node(monkeypatch):
    ex = _site_executor(monkeypatch, "rccs-front", partition="ng-dgx-m3,ai-pro6000")
    script = _rendered(ex, "ai-pro6000", {EXECUTOR_GPU_METADATA_KEY: True})
    lines = script.splitlines()
    assert "#SBATCH --partition=ai-pro6000" in lines
    assert PRO6000_GRES in lines and "#SBATCH --exclusive" in lines
    # a step without the GPU hint holds the exclusive node but asks no GPU
    plain = _rendered(ex, "ai-pro6000").splitlines()
    assert PRO6000_GRES not in plain and "#SBATCH --exclusive" in plain


def test_rccs_front_two_gpu_steps_are_eligible_only_on_ai_pro6000(monkeypatch):
    ex = _site_executor(monkeypatch, "rccs-front", partition="ng-dgx-m3,ai-pro6000")

    def ctx(count: int) -> StepContext:
        return StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output",
                           metadata={EXECUTOR_GPU_COUNT_METADATA_KEY: count}, run_id="wf-1")  # fmt: skip

    assert ex._lane_partitions(ctx(1)) == ["ng-dgx-m3", "ai-pro6000"]
    assert ex._lane_partitions(ctx(2)) == ["ai-pro6000"]
    live = _site_executor(monkeypatch, "rccs-front")  # today: ng-dgx-m3 only
    with pytest.raises(StepExecutionError, match="gpu_count 2"):
        live._lane_partitions(ctx(2))


@pytest.mark.parametrize("site", RCCS_SITES)
def test_rccs_serving_args_of_every_gb10_candidate_are_unchanged(site):
    from umrs.orchestrator.remote_slurm_backend import _partition_shaper

    cfg = _site(site)
    args = ["sbatch", "--parsable", "--mem=0", "--partition=x", "--time=23:00:00",
            "--cpus-per-task=8", "--account=cloud-users", "/deploy/serve-model.slurm"]  # fmt: skip
    shape = _partition_shaper(cfg, None)
    for name in pc.split_partitions(cfg.scheduler.partition):
        assert shape(args, name) == pc.with_partition(args, name), name
    assert shape(args, "ai-pro6000")[3:6] == [
        "--partition=ai-pro6000", "--gres=gpu:PRO6000-96GB:2", "--exclusive",
    ]  # fmt: skip


# ── RIKYU (ai4s-front): the lane opens 2026-11-08 ────────────────────────────

AI4S_CHECKOUT = "/data1/rkp00043/rku00126/ai4s-platform"


def test_ai4s_lane_is_declared_open_like_rccs_with_rikyu_limits():
    step = _step("ai4s-front")
    assert (step.kind, step.partition, step.account, step.image) == (
        "inproc",
        "gpu",
        "rkp00043",
        "",
    )
    assert (step.remote_enabled, step.uploaded_code, step.gpu, step.gpu_form) == (
        True,
        "refuse",
        True,
        "nvidia-devices",
    )
    # RIKYU per-GPU job limit: 32 cores / 400 GB; refuses --gres=gpu:N, takes --gpus=N.
    assert (step.gpus, step.gpu_flag, step.cpus, step.memory) == (1, "gpus", 32, "240G")
    assert step.workdir_root == "/data1/rkp00101/rku00126/umrs_steps"
    assert step.image_archive_dir == "/data1/rkp00101/rku00126/umrs_models/images"
    assert step.helper_path == f"{AI4S_CHECKOUT}/{HELPER_REL}"
    assert step.venue_script == f"{AI4S_CHECKOUT}/{VENUE_REL}"
    assert step.allowed_images == _step("rccs-front").allowed_images


def test_ai4s_account_override_is_scoped_to_spring8():
    assert _step("ai4s-front").project_accounts == {"spring-8": "rkp00101"}


def test_ai4s_serving_account_does_not_change():
    cfg = load_site_config(str(DEPLOY / "ai4s-front" / "site.yaml"))
    assert cfg.scheduler.account == "rkp00043"


def test_ai4s_exports_the_gpus_flag_and_a_job_private_venue_root():
    cfg = load_site_config(str(DEPLOY / "ai4s-front" / "site.yaml"))
    env = cfg.to_env()
    assert env["UMRS_STEP_GPU_FLAG"] == "gpus"
    assert env["UMRS_SLURM_ACCOUNT"] == "rkp00043"
    assert json.loads(env["UMRS_SLURM_PROJECT_ACCOUNTS"]) == {"spring-8": "rkp00101"}
    assert env["UMRS_STEP_REMOTE_ENABLED"] == "1"
    remote_env = cfg.extra["ssh_remote_env"]
    assert remote_env["RIVAULT_VENUE_ROOT"].startswith("/tmp/")  # shared nodes: no common graphroot
    assert remote_env["RIVAULT_INIT_BINARY"].startswith(
        "/data1/rkp00101/"
    )  # RIKYU podman ships no catatonit
    assert remote_env["SLURM_CONF_SERVER"]


def test_ai4s_executor_renders_gpus_not_gres(monkeypatch):
    executor = _site_executor(monkeypatch, "ai4s-front")
    assert executor.gpu_flag == "gpus"
    assert executor.account == "rkp00043"
    assert executor.project_accounts == {"spring-8": "rkp00101"}


def test_ai4s_steps_charge_only_spring8_to_aimon(monkeypatch):
    executor = _site_executor(monkeypatch, "ai4s-front")
    cases = {
        "spring-8": "rkp00101",
        "bdr-opossum": "rkp00043",
        "brc-mouse": "rkp00043",
        "default": "rkp00043",
        None: "rkp00043",
    }
    for project, account in cases.items():
        script = _rendered(executor, metadata={STAGE_PROJECT_ID_METADATA_KEY: project})
        assert f"#SBATCH --account={account}\n" in script, project
