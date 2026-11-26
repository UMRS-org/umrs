"""ADR-091 local staged podman lane for uploaded step code (slice S-D64-2).

Before this slice a site with no HPC lane refused uploaded code by name (or,
with ``uploaded_code: legacy_in_process``, ran it in the UMRS process); the
only podman executor, ``PodmanExecutor``, exchanges pickles, defaults to host
networking and hands the step the object-store endpoint, so it is never
admitted for uploaded code (ADR-091 D-2). Now a site that sets
``step_executor.uploaded_code_lane: podman`` plans and runs uploaded code on
``PodmanStagedStepExecutor``: the server stages files in, the container has
no network, no capability and no credential, the reply is JSON only and the
server collects ``out/`` under the staging caps.

The container is a fake ``podman`` (a script) that runs the REAL step worker
against the staged directories it was handed with ``-v``, as the container
would see them, and records its argv and its own environment.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml
from step_lane_harness import IMAGE, FakeS3, write_step_tree
from umrs.orchestrator.site_config import StepExecutorConfig, load_site_config
from umrs.pipeline_sdk.step import StepContext
from umrs.server import step_trust
from umrs.server import workflow_runner as wr
from umrs.server.podman_staged_step_executor import (
    LOCAL_CAPS,
    PodmanStagedStepExecutor,
    out_tar,
    read_reply,
)
from umrs.server.remote_slurm_step_executor import STEP_TIMEOUT_METADATA_KEY
from umrs.server.step_executor import (
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_IMAGE_METADATA_KEY,
    EXECUTOR_SOURCE_DIR_METADATA_KEY,
    PodmanExecutor,
    StepExecutionError,
    build_step_executor,
)
from umrs.server.step_staging import (
    STAGE_PROJECT_ID_METADATA_KEY,
    STAGE_REQUEST_METADATA_KEY,
    register_bucket_registry,
)
from umrs.server.step_trust import (
    LEGACY_IN_PROCESS,
    PODMAN_STAGED_LANE,
    uploaded_code_lane,
)
from umrs.workflow_plans import quote_workflow

UMRS_ROOT = Path(__file__).resolve().parents[1]
SRC = UMRS_ROOT / "src"
SCHEMA = UMRS_ROOT / "src" / "umrs" / "orchestrator" / "site-config.schema.json"

OWNER = "alice"
IN_BUCKET = "alice.proj.data"
OUT_BUCKET = "alice.proj.data-output"
RUN = "wf-0123456789ab"
SECRET = "do-not-leak-0123456789"

_STEP_ENV = (
    "UMRS_SLURM_STEP_SUBMIT",
    "UMRS_STEP_REMOTE_ENABLED",
    "UMRS_SLURM_WORKDIR_ROOT",
    "UMRS_STEP_HELPER",
    "UMRS_STEP_IMAGE",
    "UMRS_STEP_UPLOADED_CODE",
    "UMRS_STEP_UPLOADED_CODE_LANE",
    "UMRS_STEP_ALLOWED_IMAGES",
    "UMRS_STEP_EXECUTOR",
    "UMRS_STEP_EXECUTOR_TIMEOUT",
    "UMRS_STEP_GPU",
    "UMRS_STEP_WORKDIR",
)

FAKE_PODMAN = r"""#!__PYTHON__
import json, os, pickle, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
with open(os.path.join(HERE, "calls.jsonl"), "a") as fh:
    fh.write(json.dumps({"argv": args, "env": dict(os.environ)}) + "\n")
if args[0] != "run":
    sys.exit(0)
beh_path = os.path.join(HERE, "behaviour.json")
beh = json.load(open(beh_path)) if os.path.exists(beh_path) else {}
mounts = {}
for i, a in enumerate(args):
    if a == "-v":
        src, target, _mode = args[i + 1].rsplit(":", 2)
        mounts[target] = src
time.sleep(beh.get("sleep", 0))
if beh.get("pickle_only"):
    with open(os.path.join(mounts["/work/reply"], "output.pkl"), "wb") as fh:
        fh.write(pickle.dumps({"status": "ok"}))
    sys.exit(0)
if beh.get("rc"):
    print("the container failed on purpose")
    sys.exit(beh["rc"])
env = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": "/tmp",
    "PYTHONPATH": "__SRC__",
    "UMRS_STEP_INPUT": mounts["/work/input.pkl"],
    "UMRS_STEP_OUTPUT_JSON": mounts["/work/reply"] + "/output.json",
    "UMRS_STEP_STAGED": "1",
    "UMRS_STEP_SOURCE_DIR": mounts["/src"],
}
code = (
    "import sys\n"
    "from umrs.pipeline_sdk import staged_s3\n"
    "staged_s3.StagedS3Client.__init__.__defaults__ = (sys.argv[1], sys.argv[2], None)\n"
    "from umrs.server import step_worker\n"
    "sys.exit(step_worker.main())\n"
)
rc = subprocess.call(
    [sys.executable, "-c", code, mounts["/work/in"], mounts["/work/out"]], env=env
)
if beh.get("deep"):  # a tree deeper than Python recurses
    fd = os.open(mounts["/work/reply"], os.O_RDONLY)
    for _ in range(beh["deep"]):
        os.mkdir("d", dir_fd=fd)
        fd, old = os.open("d", os.O_RDONLY, dir_fd=fd), fd
        os.close(old)
if beh.get("lock"):  # a directory the step made read-only for its owner
    os.chmod(mounts["/work/reply"], 0o500)
sys.exit(rc)
"""


@dataclasses.dataclass
class Podman:
    bin: Path

    def behave(self, **behaviour) -> None:
        (self.bin.parent / "behaviour.json").write_text(json.dumps(behaviour))

    def calls(self) -> list[dict]:
        path = self.bin.parent / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def runs(self) -> list[dict]:
        return [c for c in self.calls() if c["argv"][0] == "run"]


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    """No site.yaml; the lane's env only as each test sets it."""
    register_bucket_registry(None)
    monkeypatch.setattr(step_trust, "_load_site_env", lambda: None)
    monkeypatch.setattr("umrs.shared.uploaded_code.load_site_env", lambda: None)
    for name in _STEP_ENV:
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch
    register_bucket_registry(None)


@pytest.fixture
def opted_in(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_UPLOADED_CODE_LANE", "podman")


@pytest.fixture
def podman(tmp_path) -> Podman:
    bin_dir = tmp_path / "fakepodman"
    bin_dir.mkdir()
    script = FAKE_PODMAN.replace("__PYTHON__", sys.executable).replace("__SRC__", str(SRC))
    path = bin_dir / "podman"
    path.write_text(script)
    path.chmod(0o755)
    return Podman(path)


@pytest.fixture
def tree(tmp_path) -> Path:
    return write_step_tree(tmp_path / "bundle")


@pytest.fixture
def root(tmp_path) -> Path:
    path = tmp_path / "steps"
    path.mkdir()
    return path


def _executor(podman: Podman, root: Path, **kw) -> PodmanStagedStepExecutor:
    values = {
        "image": IMAGE,
        "podman_bin": str(podman.bin),
        "workdir_root": str(root),
        "poll_interval": 0.05,
        "caps": dataclasses.replace(LOCAL_CAPS, min_free_bytes=1),
    }
    values.update(kw)
    return PodmanStagedStepExecutor(**values)


def _ctx(s3, tree, *, stage_inputs=None, params=None, metadata=None) -> StepContext:
    meta = {
        EXECUTOR_SOURCE_DIR_METADATA_KEY: str(tree),
        STAGE_PROJECT_ID_METADATA_KEY: "proj",
        STAGE_REQUEST_METADATA_KEY: {
            "owner": OWNER,
            "step": "s1",
            "stage_inputs": stage_inputs or {},
            "params": params or {},
        },
    }
    meta.update(metadata or {})
    return StepContext(
        s3=s3,
        input_bucket=IN_BUCKET,
        input_keys=[],
        output_bucket=OUT_BUCKET,
        config={"step_name": "s1", "api_key": "sk-platform"},
        metadata=meta,
        run_id=RUN,
    )


def _video(ex, tree, entrypoint="steps.lane:Upper", metadata=None, bucket=IN_BUCKET):
    s3 = FakeS3({(bucket, "in/video.txt"): b"frames"})
    ctx = _ctx(
        s3,
        tree,
        stage_inputs={"params": ["video"]},
        params={"video": f"s3://{bucket}/in/video.txt"},
        metadata=metadata,
    )
    config = {
        "step_name": "s1",
        "input_key": "in/video.txt",
        "api_key": "sk-platform",
        "headers": {"Authorization": "Bearer platform"},
    }
    return ex.execute(entrypoint=entrypoint, config=config, ctx=ctx), s3


# ── lane admission (step_trust) and the plan ────────────────────────────────


_WORKFLOW = {
    "name": "relax",
    "steps": [{"name": "s1", "entrypoint": "m:S1", "routing_hints": {"gpu": True}}],
}


def test_opted_in_site_plans_uploaded_code_onto_the_podman_lane(tmp_path, opted_in):
    (tmp_path / "m.py").write_text("")  # m:S1 is uploaded code
    quote = quote_workflow(_WORKFLOW, str(tmp_path))
    assert quote.refusal == ""
    assert quote.lane == PODMAN_STAGED_LANE
    assert [(s.lane, s.attempt_seconds, s.gpu) for s in quote.steps] == [
        (PODMAN_STAGED_LANE, 3600, False)  # no GPU without the site's switch
    ]


def test_the_plan_bounds_the_attempt_by_the_executor_timeout(tmp_path, opted_in, monkeypatch):
    (tmp_path / "m.py").write_text("")
    monkeypatch.setenv("UMRS_STEP_EXECUTOR_TIMEOUT", "900")
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    [step] = quote_workflow(_WORKFLOW, str(tmp_path)).steps
    assert (step.attempt_seconds, step.gpu) == (900, True)


def test_without_the_opt_in_uploaded_code_is_still_refused(tmp_path):
    (tmp_path / "m.py").write_text("")
    quote = quote_workflow(_WORKFLOW, str(tmp_path))
    assert quote.lane is None and "HPC lane" in quote.refusal
    assert uploaded_code_lane(None) is None


@pytest.mark.parametrize("value", ["docker", "podman-staged", "PODMAN ", "yes"])
def test_only_podman_opts_in(monkeypatch, value):
    monkeypatch.setenv("UMRS_STEP_UPLOADED_CODE_LANE", value)
    expected = PODMAN_STAGED_LANE if value.strip().lower() == "podman" else None
    assert uploaded_code_lane(None) == expected


def test_the_opt_in_wins_over_the_legacy_switch(monkeypatch, opted_in):
    monkeypatch.setenv("UMRS_STEP_UPLOADED_CODE", LEGACY_IN_PROCESS)
    assert uploaded_code_lane("in_process") == PODMAN_STAGED_LANE


def test_an_available_hpc_lane_still_wins(monkeypatch, opted_in):
    for name, value in {
        "UMRS_SLURM_STEP_SUBMIT": "remote",
        "UMRS_SLURM_WORKDIR_ROOT": "/cluster/steps",
        "UMRS_STEP_REMOTE_ENABLED": "1",
        "UMRS_STEP_HELPER": "/cluster/umrs_step_helper.py",
        "UMRS_STEP_IMAGE": IMAGE,
    }.items():
        monkeypatch.setenv(name, value)
    assert uploaded_code_lane(None) == "slurm"


def test_the_runner_takes_the_podman_lane_without_an_hpc_identity(tmp_path, opted_in):
    (tmp_path / "m.py").write_text("")
    steps = _WORKFLOW["steps"]
    lane = wr._workflow_executor_name(
        _WORKFLOW, steps, str(tmp_path), injected=False, run_identity=("alice", "p")
    )
    assert lane == (PODMAN_STAGED_LANE, "")


def test_the_runner_still_refuses_without_the_opt_in(tmp_path):
    (tmp_path / "m.py").write_text("")
    steps = _WORKFLOW["steps"]
    name, refusal = wr._workflow_executor_name(
        _WORKFLOW, steps, str(tmp_path), injected=False, run_identity=("alice", "p")
    )
    assert name is None and "HPC lane" in refusal


def test_platform_template_steps_keep_their_own_lane(tmp_path, opted_in):
    (tmp_path / "m.py").write_text("")
    workflow = {
        "steps": [
            {"name": "s1", "entrypoint": "m:S1"},
            {"name": "t", "entrypoint": "umrs.pipeline_sdk.templates:LLMStep"},
        ]
    }
    lanes = wr._split_step_lanes(
        object(), workflow, workflow["steps"], str(tmp_path), PODMAN_STAGED_LANE
    )
    assert isinstance(lanes, wr._StepLanes)
    assert type(lanes.template).__name__ == "InProcessExecutor"


def test_the_lane_name_builds_the_staged_executor_never_podman_executor():
    assert isinstance(build_step_executor("podman-staged"), PodmanStagedStepExecutor)
    assert PodmanStagedStepExecutor.name == PODMAN_STAGED_LANE != "podman"


def test_podman_still_builds_the_platform_step_executor(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_IMAGE", IMAGE)
    assert isinstance(build_step_executor("podman"), PodmanExecutor)


# ── site.yaml ───────────────────────────────────────────────────────────────


def _site_env(tmp_path: Path, step_executor: dict) -> dict[str, str]:
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump({"name": "t", "step_executor": step_executor}))
    return load_site_config(str(path)).to_env()


def test_the_site_key_defaults_off_and_exports_nothing():
    assert StepExecutorConfig().uploaded_code_lane == ""


def test_the_site_key_exports_the_opt_in_and_the_local_workdir(tmp_path):
    env = _site_env(
        tmp_path,
        {
            "kind": "inproc",
            "uploaded_code_lane": "podman",
            "workdir_root": "/srv/umrs-steps",
            "memory": "16g",
            "cpus": 4,
        },
    )
    assert env["UMRS_STEP_UPLOADED_CODE_LANE"] == "podman"
    assert env["UMRS_STEP_WORKDIR"] == "/srv/umrs-steps"
    assert (env["UMRS_STEP_MEMORY"], env["UMRS_STEP_CPUS"]) == ("16g", "4")


def test_a_slurm_kind_keeps_its_cluster_root_out_of_the_local_lane(tmp_path):
    env = _site_env(
        tmp_path,
        {"kind": "slurm-remote", "uploaded_code_lane": "podman", "workdir_root": "/c"},
    )
    assert "UMRS_STEP_WORKDIR" not in env
    off = _site_env(tmp_path, {"kind": "inproc", "workdir_root": "/srv/x"})
    assert "UMRS_STEP_UPLOADED_CODE_LANE" not in off and "UMRS_STEP_WORKDIR" not in off


def test_the_schema_accepts_only_podman():
    schema = json.loads(SCHEMA.read_text())
    lane = schema["$defs"]["step_executor"]["properties"]["uploaded_code_lane"]
    jsonschema.validate("podman", lane)
    jsonschema.validate("", lane)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate("docker", lane)


# ── the executor ────────────────────────────────────────────────────────────


def test_an_uploaded_step_runs_end_to_end(podman, root, tree, opted_in):
    result, s3 = _video(_executor(podman, root), tree)
    physical = f"runs/{RUN}/s1/result/out.txt"
    assert result.output_keys == [physical]
    assert s3.objects[(OUT_BUCKET, physical)] == b"FRAMES!"
    assert result.extra["config_keys"] == ["input_key", "step_name"]  # no platform key
    assert "slurm" not in result.extra  # the step's forged executor fact is gone
    assert list(root.iterdir()) == []  # the workdir is removed


@pytest.fixture
def run_call(podman, root, tree, opted_in, monkeypatch) -> dict:
    """One staged run with credentials and an endpoint in the server's env."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", SECRET)
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://store.internal:9000")
    _video(_executor(podman, root, memory="2g", pids_limit=256), tree)
    [call] = podman.runs()
    return call


@pytest.mark.parametrize(
    "flag",
    [
        "--network=none",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--unsetenv-all",
        "--pull=never",
        "--http-proxy=false",
        "--userns=keep-id",
        "--pids-limit=256",
        "--memory=2g",
    ],
)
def test_the_argv_carries_the_hardening_flags(run_call, flag):
    assert flag in run_call["argv"]


def test_the_argv_names_the_image_and_our_entrypoint(run_call):
    argv = run_call["argv"]
    assert argv[argv.index("--entrypoint") :] == [
        "--entrypoint",
        "python",
        IMAGE,
        "-m",
        "umrs.server.step_worker",
    ]


@pytest.mark.parametrize(
    "absent", ["AWS_", SECRET, "store.internal", "--network=host", "--privileged"]
)
def test_the_argv_carries_no_credential(run_call, absent):
    assert absent not in " ".join(run_call["argv"])


def test_the_podman_cli_gets_no_server_variable(run_call):
    names = set(run_call["env"])
    assert not {n for n in names if n.startswith(("AWS_", "S3_"))}
    assert SECRET not in json.dumps(run_call["env"])


def test_the_container_env_is_the_staged_set(run_call):
    argv = run_call["argv"]
    names = {argv[i + 1].split("=", 1)[0] for i, a in enumerate(argv) if a == "-e"}
    assert names == {
        "PATH",
        "HOME",
        "LANG",
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        "UMRS_STEP_INPUT",
        "UMRS_STEP_OUTPUT_JSON",
        "UMRS_STEP_STAGED",
        "UMRS_STEP_SOURCE_DIR",
    }


def test_the_payload_carries_no_platform_key(podman, root, tree, opted_in):
    ex = _executor(podman, root, keep_workdir=True)
    _video(ex, tree)
    [workdir] = list(root.iterdir())
    payload = pickle.loads((workdir / "input.pkl").read_bytes())
    assert "api_key" not in payload["config"] and "headers" not in payload["config"]
    assert payload["s3_config"] == {}
    assert STAGE_REQUEST_METADATA_KEY not in payload["ctx"]["metadata"]
    assert (workdir / "in" / "manifest.json").is_file()
    assert (workdir / "src" / "steps" / "lane.py").is_file()  # a copy, not a mount


def test_a_pickle_reply_is_refused(podman, root, tree, opted_in):
    podman.behave(pickle_only=True)
    with pytest.raises(StepExecutionError, match="no JSON reply"):
        _video(_executor(podman, root), tree)


def test_an_out_of_scope_bucket_is_refused_before_anything_runs(podman, root, tree, opted_in):
    with pytest.raises(StepExecutionError, match="bob.proj.data"):
        _video(_executor(podman, root), tree, bucket="bob.proj.data")
    assert podman.runs() == [] and list(root.iterdir()) == []


def test_without_the_opt_in_the_executor_runs_nothing(podman, root, tree):
    with pytest.raises(StepExecutionError, match="uploaded_code_lane"):
        _video(_executor(podman, root), tree)
    assert podman.calls() == []


def test_a_gated_run_is_refused(podman, root, tree, opted_in):
    ex = _executor(podman, root)
    ex.bind_run(RUN, gated=object())
    with pytest.raises(StepExecutionError, match="lane_not_gated"):
        _video(ex, tree)
    assert podman.calls() == []


def test_an_image_off_the_allow_list_is_refused(podman, root, tree, opted_in):
    meta = {EXECUTOR_IMAGE_METADATA_KEY: "docker.io/evil/image:1"}
    with pytest.raises(StepExecutionError, match="not an allowed step image"):
        _video(_executor(podman, root), tree, metadata=meta)
    assert podman.calls() == []


def test_a_link_left_in_out_is_refused(podman, root, tree, opted_in):
    with pytest.raises(StepExecutionError, match="link"):
        _video(_executor(podman, root), tree, entrypoint="steps.lane:Sneaky")


def test_a_failing_step_fails_by_name(podman, root, tree, opted_in):
    with pytest.raises(StepExecutionError, match="exploded on purpose"):
        _video(_executor(podman, root), tree, entrypoint="steps.lane:Boom")


def test_a_failing_container_reports_its_log(podman, root, tree, opted_in):
    podman.behave(rc=3)
    with pytest.raises(StepExecutionError, match="failed on purpose"):
        _video(_executor(podman, root), tree)


def test_the_step_deadline_stops_the_container(podman, root, tree, opted_in):
    podman.behave(sleep=30)
    meta = {STEP_TIMEOUT_METADATA_KEY: 0.3}
    with pytest.raises(StepExecutionError, match="deadline"):
        _video(_executor(podman, root), tree, metadata=meta)
    removed = [c["argv"] for c in podman.calls() if c["argv"][0] == "rm"]
    assert removed and removed[0][:2] == ["rm", "-f"]


def test_a_cancelled_run_stops_the_container(podman, root, tree, opted_in):
    podman.behave(sleep=30)
    ex = _executor(podman, root)
    ex.bind_run(RUN, cancelled=lambda: True)
    with pytest.raises(StepExecutionError, match="cancelled"):
        _video(ex, tree)


def test_a_gpu_step_gets_cdi_devices_on_a_gpu_site(podman, root, tree, opted_in, monkeypatch):
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    _video(_executor(podman, root), tree, metadata={EXECUTOR_GPU_METADATA_KEY: True})
    argv = podman.runs()[0]["argv"]
    assert argv[argv.index("--device") + 1] == "nvidia.com/gpu=all"


def test_an_unsupported_gpu_form_is_refused(podman, root, tree, opted_in, monkeypatch):
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    ex = _executor(podman, root, gpu_form="nvidia-devices")
    with pytest.raises(StepExecutionError, match="gpu_form"):
        _video(ex, tree, metadata={EXECUTOR_GPU_METADATA_KEY: True})


def _out_dir(tmp_path: Path) -> Path:
    out = tmp_path / "wd" / "out" / OUT_BUCKET
    out.mkdir(parents=True)
    return out


def _shapes(out: Path, tmp_path: Path) -> dict:
    return {
        "fifo": lambda: os.mkfifo(out / "f"),
        "hard link": lambda: os.link(tmp_path / "host.txt", out / "h"),
        "deep tree": lambda: (out / ("d/" * 70)).mkdir(parents=True),
        "linked dir": lambda: (out / "l").symlink_to(tmp_path),
    }


@pytest.mark.parametrize("shape", ["fifo", "hard link", "deep tree", "linked dir"])
def test_out_shapes_a_step_can_leave_are_refused(tmp_path, shape):
    out = _out_dir(tmp_path)
    (tmp_path / "host.txt").write_text("host file")
    _shapes(out, tmp_path)[shape]()
    with open(os.devnull, "wb") as sink, pytest.raises(StepExecutionError):
        out_tar(str(tmp_path / "wd"), sink, 1000)


def test_a_linked_reply_is_refused(tmp_path):
    (tmp_path / "wd" / "reply").mkdir(parents=True)
    (tmp_path / "host.json").write_text("{}")
    (tmp_path / "wd" / "reply" / "output.json").symlink_to(tmp_path / "host.json")
    with pytest.raises(StepExecutionError, match="link"):
        read_reply(str(tmp_path / "wd"))


def test_an_oversized_reply_is_refused(tmp_path):
    (tmp_path / "wd" / "reply").mkdir(parents=True)
    (tmp_path / "wd" / "reply" / "output.json").write_bytes(b" " * 65)
    with pytest.raises(StepExecutionError, match="exceeds 64 bytes"):
        read_reply(str(tmp_path / "wd"), cap=64)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the mode bits")
def test_a_directory_the_step_locked_is_still_removed(podman, root, tree, opted_in):
    podman.behave(lock=True)
    result, _s3 = _video(_executor(podman, root), tree)
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert list(root.iterdir()) == []  # not left behind on the host's disk


def test_a_tree_deeper_than_python_recurses_is_still_removed(podman, root, tree, opted_in):
    podman.behave(deep=1100)
    result, _s3 = _video(_executor(podman, root), tree)
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert list(root.iterdir()) == []


def test_a_relative_workdir_root_refuses_by_name(podman, tree, opted_in):
    with pytest.raises(StepExecutionError, match="absolute"):
        _video(_executor(podman, Path("rel")), tree)
    assert os.path.exists("rel") is False
