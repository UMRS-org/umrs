"""A staged step's routed-endpoint socket (ADR-109 D-5): job script, lane,
runner and router.

The job script of a step whose routing decision names an endpoint starts
the forwarder (platform image, host network, the credential file mounted
into it only) before the step, waits for its socket, mounts that socket
into the step pod at ``/run/umrs/endpoint.sock`` and names it in
``UMRS_STEP_ENDPOINT_SOCKET``; the step pod keeps ``--network=none``; the
EXIT trap stops the forwarder after the step. A step without one gets the
unchanged script. The lane writes the credential as ``endpoint.json``
(0600, helper ``put``) and never into ``input.pkl`` or the job script.
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from step_lane_harness import FakeS3, make_cluster, write_step_tree
from test_staged_job_script import (
    BASE_ENV,
    DEFAULT_PROFILE,
    FAKE_PODMAN,
    STEP_COMMAND,
    STEP_ENV,
    WD,
    _flag_values,
    _job,
    _tokens,
)
from test_umrs_step_helper import helper, run
from umrs.orchestrator.site_config import StepExecutorConfig, _step_executor_env
from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.pipeline_sdk.step import StepContext
from umrs.pipeline_sdk.workflow import StepSpec
from umrs.server import workflow_runner as wr
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor
from umrs.server.slurm_step_executor import EndpointForward, render_staged_job_script
from umrs.server.step_executor import (
    EXECUTOR_ENDPOINT_METADATA_KEY,
    EXECUTOR_SOURCE_DIR_METADATA_KEY,
    StepExecutionError,
    _worker_metadata,
)
from umrs.server.step_router import RoutingDecision, StepRouter, endpoint_serving_backend
from umrs.server.step_staging import STAGE_REQUEST_METADATA_KEY, register_bucket_registry

SECRET = "sk-endpoint-0123456789abcdef"
FORWARD = EndpointForward("gb10-07", 8401, "bentoml", 1000, 2000, 3000)
SOCKET_ENV = "UMRS_STEP_ENDPOINT_SOCKET=/run/umrs/endpoint.sock"


def _forwarder_line(script: str) -> str:
    [line] = [x for x in script.splitlines() if "umrs.server.step_endpoint_forwarder" in x]
    return line


def _forwarder_tokens(script: str) -> list[str]:
    line = _forwarder_line(script).removesuffix(" >&2 &")
    line = line.replace('"${_step_ctr}-endpoint"', "FWD").replace('"${_step_trusted}"', "TRUSTED")
    return shlex.split(line.replace('"${_step_seccomp}"', "SECCOMP"))


# ── the rendered script ─────────────────────────────────────────────────────


def test_a_step_without_a_routed_endpoint_gets_no_forwarder_and_no_socket():
    script = render_staged_job_script(_job())
    for needle in ("step_endpoint_forwarder", "UMRS_STEP_ENDPOINT_SOCKET", "endpoint.json",
                   "endpoint.sock", "-endpoint", "--network=host"):  # fmt: skip
        assert needle not in script


def test_the_step_pod_gets_the_socket_and_its_env_name_and_keeps_no_network():
    tokens = _tokens(render_staged_job_script(_job(endpoint=FORWARD)))
    assert "--network=none" in tokens and "--network=host" not in tokens
    assert _flag_values(tokens, "-v") == [
        f"{WD}/in:/work/in:ro",
        f"{WD}/out:/work/out:rw",
        f"{WD}/reply:/work/reply:rw",
        f"{WD}/src:/src:ro",
        f"{WD}/input.pkl:/work/input.pkl:ro",
        f"{WD}/reply/endpoint.sock:/run/umrs/endpoint.sock:rw",
    ]
    assert _flag_values(tokens, "-e") == [*BASE_ENV, *STEP_ENV, SOCKET_ENV]
    assert tokens[-5:] == STEP_COMMAND


def test_the_credential_file_is_mounted_into_the_forwarder_only():
    script = render_staged_job_script(_job(endpoint=FORWARD))
    step = " ".join(_tokens(script))
    assert "endpoint.json" not in step and "umrs-endpoint" not in step
    assert _flag_values(_forwarder_tokens(script), "-v") == [
        f"{WD}/reply:/run/umrs:rw",
        f"{WD}/endpoint.json:/umrs-endpoint.json:ro",
    ]


def test_the_forwarder_is_trusted_code_in_the_steps_posture_on_the_host_network():
    tokens = _forwarder_tokens(render_staged_job_script(_job(endpoint=FORWARD)))
    assert tokens[:4] == ["podman", "run", "--rm", "--init"]
    assert "--network=host" in tokens and "--network=none" not in tokens
    for flag in ("--read-only", "--cap-drop=all", "--unsetenv-all", "--pull=never",
                 "--userns=keep-id", "--http-proxy=false", "--image-volume=ignore"):  # fmt: skip
        assert flag in tokens, flag
    assert "seccomp=SECCOMP" in tokens and "--pids-limit=512" in tokens
    at = tokens.index("--entrypoint")
    assert tokens[at:] == [
        "--entrypoint", "python", "TRUSTED", "-I", "-m", "umrs.server.step_endpoint_forwarder",
        "--socket", "/run/umrs/endpoint.sock", "--credential", "/umrs-endpoint.json",
        "--upstream", "gb10-07:8401", "--kind", "bentoml",
        "--max-request-bytes", "1000", "--max-response-bytes", "2000",
        "--max-total-bytes", "3000",
    ]  # fmt: skip


def test_the_forwarder_starts_before_the_step_and_stops_with_it():
    lines = render_staged_job_script(_job(endpoint=FORWARD)).splitlines()
    start = lines.index(_forwarder_line("\n".join(lines)))
    step = next(i for i, x in enumerate(lines) if x.endswith(" 2>&1 | _step_log >&2"))
    assert start < step
    trap = lines.index("_step_cleanup() {")
    stops = [x.strip() for x in lines[trap : lines.index("trap _step_cleanup EXIT")]]
    assert 'podman stop -t 5 "${_step_ctr}-endpoint" >/dev/null 2>&1 || true' in stops


def test_the_script_is_valid_bash():
    script = render_staged_job_script(_job(endpoint=FORWARD))
    proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


# ── the script run against a fake podman ────────────────────────────────────

#: The forwarder container: binds the step's socket and serves until stopped
#: (or, with ``fwd_fails``, exits at once without one).
FORWARDER_CASE = r"""    if [[ "$*" == *step_endpoint_forwarder* ]]; then
      [[ -f "$state/fwd_fails" ]] && exit 1
      echo $$ > "$state/fwd_pid"
      exec "$FAKE_PYTHON" -c 'import os, socket, time
s = socket.socket(socket.AF_UNIX)
s.bind(os.environ["WD"] + "/reply/endpoint.sock")
s.listen()
time.sleep(30)'
    fi
"""
FAKE_PODMAN_FWD = FAKE_PODMAN.replace(
    '    if [[ "$*" == *pids.max* ]]; then', FORWARDER_CASE + '    if [[ "$*" == *pids.max* ]]; then'
).replace(
    "  pull)",
    '  stop) [[ -f "$state/fwd_pid" ]] && kill "$(cat "$state/fwd_pid")"; exit 0 ;;\n  pull)',
)


@pytest.fixture
def node(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "podman").write_text(FAKE_PODMAN_FWD)
    (bin_dir / "podman").chmod(0o755)
    (bin_dir / "squeue").write_text("#!/bin/bash\nexit 1\n")
    (bin_dir / "squeue").chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    archive_dir = tmp_path / "images"
    archive_dir.mkdir()
    archive = archive_dir / f"umrs-step-0123456789ab-{platform.machine()}.tar"
    archive.write_bytes(b"tar")
    (archive_dir / (archive.name + ".id")).write_text("f" * 64 + "\n")
    (state / "have_id").write_text("f" * 64)
    (state / "label").write_text("1")
    (state / "seccomp_base.json").write_text(json.dumps(DEFAULT_PROFILE))
    # the step records whether its socket was there when it ran
    (state / "run_script").write_text(
        '[[ -S "$WD/reply/endpoint.sock" ]] && echo "STEP SAW SOCKET" >> "$state/podman.log"\n'
    )
    venue = tmp_path / "venue.sh"
    venue.write_text('rivault_venue_prepare() { echo "prepared $1"; }\n')
    wd = tmp_path / "wd"
    for sub in ("out", "reply", "log"):
        (wd / sub).mkdir(parents=True)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_STATE": str(state),
        "FAKE_PYTHON": sys.executable,
        "SLURM_JOB_ID": "4242",
        "WD": str(wd),
    }

    def run_script(**overrides) -> subprocess.CompletedProcess:
        script = render_staged_job_script(
            _job(workdir=str(wd), image_archive_dir=str(archive_dir), venue_script=str(venue),
                 **overrides)
        )  # fmt: skip
        return subprocess.run(
            ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60
        )

    def log() -> list[str]:
        path = state / "podman.log"
        return path.read_text().splitlines() if path.exists() else []

    return SimpleNamespace(run=run_script, log=log, state=state, wd=wd)


def test_the_step_runs_with_its_socket_up_and_the_forwarder_goes_after_it(node):
    proc = node.run(endpoint=FORWARD)
    assert proc.returncode == 0, proc.stderr
    log = node.log()
    fwd = next(i for i, x in enumerate(log) if "step_endpoint_forwarder" in x)
    saw = log.index("STEP SAW SOCKET")
    stop = log.index("stop -t 5 umrs-step-4242-endpoint")
    assert fwd < saw < stop
    assert log[stop + 1] == "rm -f umrs-step-4242-endpoint"
    assert log[fwd].startswith("run --rm --init --name umrs-step-4242-endpoint ")


def test_a_forwarder_that_does_not_come_up_fails_the_job_before_the_step(node):
    (node.state / "fwd_fails").write_text("")
    proc = node.run(endpoint=FORWARD)
    assert proc.returncode == 78
    assert "the endpoint forwarder did not start" in proc.stderr
    assert "STEP SAW SOCKET" not in node.log()
    assert not any("/work/input.pkl" in x for x in node.log())  # no step pod


def test_a_step_without_an_endpoint_runs_without_a_socket(node):
    assert node.run().returncode == 0
    log = node.log()
    assert "STEP SAW SOCKET" not in log
    assert not any("endpoint" in x for x in log)


# ── the lane: credential file, payload, script ──────────────────────────────

RUN = "wf-0123456789ab"
IN_BUCKET = "alice.proj.data"


@pytest.fixture(autouse=True)
def _no_bucket_registry():
    register_bucket_registry(None)
    yield
    register_bucket_registry(None)


def _handover(**over) -> dict:
    return {
        "api_base": "http://gb10-07:8401",
        "kind": "vllm",
        "api_key": SECRET,
        "headers": {"X-Tenant": "t-1"},
        **over,
    }


def _step_ctx(tmp_path, metadata: dict) -> StepContext:
    tree = write_step_tree(tmp_path / "bundle")
    stage = {
        "owner": "alice",
        "step": "s1",
        "params": {"video": f"s3://{IN_BUCKET}/in/video.txt"},
        "stage_inputs": {"params": ["video"]},
    }
    return StepContext(
        s3=FakeS3({(IN_BUCKET, "in/video.txt"): b"frames"}),
        input_bucket=IN_BUCKET,
        input_keys=[],
        output_bucket=IN_BUCKET + "-output",
        config={"step_name": "s1"},
        metadata={
            EXECUTOR_SOURCE_DIR_METADATA_KEY: str(tree),
            STAGE_REQUEST_METADATA_KEY: stage,
            **metadata,
        },
        run_id=RUN,
    )


def _lane_run(tmp_path, monkeypatch, metadata: dict):
    cluster = make_cluster(tmp_path, monkeypatch)
    ex = cluster.executor(keep_workdir=True)
    recorded: list[dict] = []
    ex.bind_run(RUN, on_job=lambda _step, job: recorded.append(dict(job)))
    config = {"step_name": "s1", "input_key": "in/video.txt"}
    ex.execute(entrypoint="steps.lane:Upper", config=config, ctx=_step_ctx(tmp_path, metadata))
    return cluster, Path(recorded[-1]["workdir"])


def test_the_lane_writes_the_credential_beside_the_job_and_nowhere_else(tmp_path, monkeypatch):
    cluster, wd = _lane_run(tmp_path, monkeypatch, {EXECUTOR_ENDPOINT_METADATA_KEY: _handover()})
    cred = wd / "endpoint.json"
    assert stat.S_IMODE(cred.stat().st_mode) == 0o600
    assert json.loads(cred.read_text()) == {
        "headers": {"X-Tenant": "t-1", "Authorization": f"Bearer {SECRET}"}
    }
    script = (wd / "job.sbatch").read_text()
    assert "--upstream gb10-07:8401 --kind vllm" in script
    assert SECRET not in script and "X-Tenant" not in script
    payload = (wd / "input.pkl").read_bytes()
    assert SECRET.encode() not in payload
    assert EXECUTOR_ENDPOINT_METADATA_KEY not in pickle.loads(payload)["ctx"]["metadata"]
    assert cluster.helper_verbs()[3:6] == ["put", "put", "put"]  # input.pkl, endpoint.json, job


def test_a_step_without_a_routed_endpoint_gets_no_credential_file(tmp_path, monkeypatch):
    _cluster, wd = _lane_run(tmp_path, monkeypatch, {})
    assert not (wd / "endpoint.json").exists()
    assert "step_endpoint_forwarder" not in (wd / "job.sbatch").read_text()


@pytest.mark.parametrize(
    "over",
    [{"api_base": "https://gb10-07:8401"}, {"api_base": "http://gb10-07"}, {"kind": "smtp"},
     {"headers": {"X-A": "b\r\nX-Evil: c"}}],
)  # fmt: skip
def test_an_endpoint_the_job_cannot_forward_fails_before_any_ssh(tmp_path, monkeypatch, over):
    cluster = make_cluster(tmp_path, monkeypatch)
    ex = cluster.executor()
    ctx = _step_ctx(tmp_path, {EXECUTOR_ENDPOINT_METADATA_KEY: _handover(**over)})
    with pytest.raises(StepExecutionError, match="cannot be forwarded"):
        ex.execute(entrypoint="steps.lane:Upper", config={}, ctx=ctx)
    assert cluster.commands == []


def test_the_executor_reads_the_cap_env(monkeypatch):
    env = _step_executor_env(
        StepExecutorConfig(kind="slurm-remote", endpoint_max_total_bytes=777,
                           endpoint_max_request_bytes=55)
    )  # fmt: skip
    assert env["UMRS_STEP_ENDPOINT_MAX_TOTAL_BYTES"] == "777"
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("UMRS_SLURM_WORKDIR_ROOT", "/w")
    ex = RemoteSlurmStepExecutor(ssh_config=SSHControlConfig(ssh_target="login"))
    assert (ex.endpoint_max_request_bytes, ex.endpoint_max_total_bytes) == (55, 777)
    assert ex.endpoint_max_response_bytes == 256 * 1024 * 1024  # 0 = the default


def test_the_endpoint_never_reaches_any_workers_metadata():
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="o",
                      metadata={EXECUTOR_ENDPOINT_METADATA_KEY: _handover(), "k": 1})  # fmt: skip
    assert _worker_metadata(ctx) == {"k": 1}


def test_the_helper_writes_the_credential_file_private(tmp_path):
    root = tmp_path / "steps"
    made = run("mkwd", root, "wf-abc-s1-1-")
    wd = Path(made.stdout.decode().strip())
    assert run("put", root, wd, "endpoint.json", stdin=b'{"headers": {}}').returncode == 0
    assert stat.S_IMODE((wd / "endpoint.json").stat().st_mode) == 0o600
    big = b"x" * (helper.PUT_CAPS["endpoint.json"] + 1)
    assert run("put", root, wd, "endpoint.json", stdin=big).returncode == helper.EXIT_REFUSED


# ── the runner and the router ───────────────────────────────────────────────


class _SocketLane:
    endpoint_socket = True


def _decision(**kw) -> RoutingDecision:
    values = {"api_base": "http://gb10-07:8401", "api_key": SECRET, "endpoint_id": "job-1",
              "headers": {"X-Tenant": "t-1"}, "resolved_via": "registry_capability"}  # fmt: skip
    values.update(kw)
    return RoutingDecision(**values)


def test_the_runner_hands_a_socket_lane_the_routed_endpoint():
    metadata: dict = {}
    wr._bind_step_endpoint(_SocketLane(), metadata, _decision(serving_backend="bentoml"))
    assert metadata[EXECUTOR_ENDPOINT_METADATA_KEY] == {
        "api_base": "http://gb10-07:8401",
        "kind": "bentoml",
        "api_key": SECRET,
        "headers": {"X-Tenant": "t-1"},
    }
    wr._bind_step_endpoint(_SocketLane(), metadata, _decision(serving_backend="vllm"))
    assert metadata[EXECUTOR_ENDPOINT_METADATA_KEY]["kind"] == "vllm"


@pytest.mark.parametrize(
    ("executor", "decision"),
    [
        (_SocketLane(), _decision(endpoint_id=None, resolved_via="explicit_config")),
        (_SocketLane(), _decision(api_base=None)),
        (SimpleNamespace(name="podman"), _decision()),
        (MagicMock(), _decision()),
    ],
)
def test_no_named_endpoint_or_no_socket_lane_means_no_hand_over(executor, decision):
    metadata: dict = {}
    wr._bind_step_endpoint(executor, metadata, decision)
    assert metadata == {}


def test_the_remote_lane_is_a_socket_lane():
    assert RemoteSlurmStepExecutor.endpoint_socket is True


@pytest.mark.parametrize(
    ("row", "backend"),
    [
        ({"serving_backend": "bentoml"}, "bentoml"),
        ({"service_kind": "bentoml_vision"}, "bentoml"),
        ({"model_type": "vision"}, "bentoml"),
        ({"serving_backend": "vllm"}, "vllm"),
        ({}, None),
    ],
)
def test_the_endpoint_rows_backend_picks_the_kind(row, backend):
    assert endpoint_serving_backend(row) == backend


def test_a_registry_decision_carries_its_endpoints_backend(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    router = StepRouter(
        find_models=lambda *_: [{"name": "sam3.1", "scope": "system", "model_type": "vision"}],
        find_endpoint=lambda *_: {
            "litellm_url": "http://gb10-07:8401", "job_id": "j::sam3.1",
            "serving_backend": "bentoml", "service_kind": "bentoml_vision",
        },
    )  # fmt: skip
    spec = StepSpec(name="seg", entrypoint="steps:Seg", requires=["vision"], model="sam3.1")
    decision = router.resolve(spec)
    assert (decision.endpoint_id, decision.serving_backend) == ("j::sam3.1", "bentoml")
    assert decision.api_base == "http://gb10-07:8401"
