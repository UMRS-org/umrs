"""The staged lane with the generated job script under REAL podman (spec §11).

Opt-in: ``UMRS_STEP_PODMAN_TESTS=1`` with rootless podman on the host, and
either ``UMRS_STEP_TEST_IMAGE`` naming a step image the stager built or the
``docker.io/library/python:3.11-slim`` base already present locally (a test
image is then built from it with ``--pull=never``: the umrs source copied into
site-packages plus the step-contract label, as ``docker/Containerfile.step``
does). Nothing is pulled.

The whole lane runs through the fake cluster of ``step_lane_harness``: the
real helper stages the workdir, and the fake ``sbatch`` runs the rendered
``job.sbatch`` with bash (venue stub, image-ensure lines, the pids probe, the
watchdog, the capped log and the hardened ``podman run``). Adversarial steps
then report what they could reach from inside the container.
"""

from __future__ import annotations

import dataclasses
import errno
import json
import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from step_lane_harness import SRC, FakeS3, make_cluster
from umrs.pipeline_sdk.step import StepContext
from umrs.server.remote_slurm_step_executor import STEP_TIMEOUT_METADATA_KEY
from umrs.server.step_executor import EXECUTOR_SOURCE_DIR_METADATA_KEY, StepExecutionError
from umrs.server.step_staging import DEFAULT_CAPS, STAGE_REQUEST_METADATA_KEY, MiB

BASE = "docker.io/library/python:3.11-slim"
OUT_BUCKET = "alice.proj.data-output"
RUN = "wf-00000000feed"
CANARY = "UMRS_TEST_CANARY_SECRET"

pytestmark = pytest.mark.skipif(
    os.environ.get("UMRS_STEP_PODMAN_TESTS") != "1" or shutil.which("podman") is None,
    reason="opt-in: UMRS_STEP_PODMAN_TESTS=1 and rootless podman",
)

CONTAINERFILE = f"""FROM {BASE}
LABEL org.umrs.step-contract="1"
COPY site-packages/ /usr/local/lib/python3.11/site-packages/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
USER 65534:65534
WORKDIR /tmp
"""

STEPS = r"""
import ctypes, errno, json, os, pickle, socket, time
from umrs.pipeline_sdk import Step, StepResult


def _err(exc):
    return errno.errorcode.get(exc.errno, type(exc).__name__)


def _connect(host, port):
    try:
        socket.create_connection((host, port), timeout=3).close()
        return "connected"
    except OSError as exc:
        return _err(exc)


def _write(path):
    try:
        with open(path, "w") as fh:
            fh.write("x")
        return "written"
    except OSError as exc:
        return _err(exc)


def _fallocate(path):
    # FALLOC_FL_KEEP_SIZE (1) reserves blocks the apparent size never shows;
    # posix_fallocate falls back to writing only when fallocate says EOPNOTSUPP.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        rc = libc.fallocate(fd, 1, ctypes.c_long(0), ctypes.c_long(1 << 20))
        keep = "reserved" if rc == 0 else ctypes.get_errno()
        try:
            os.posix_fallocate(fd, 0, 64 * 1024)
            posix = os.fstat(fd).st_size
        except OSError as exc:
            posix = _err(exc)
        return [keep, posix]
    finally:
        os.close(fd)
        os.unlink(path)


def _fork(n):
    pids, error = [], None
    try:
        for _ in range(n):
            pid = os.fork()
            if pid == 0:
                time.sleep(30)
                os._exit(0)
            pids.append(pid)
    except OSError as exc:
        error = _err(exc)
    for pid in pids:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
    return len(pids), error


class Probe(Step):
    name = "probe"
    def setup(self, config): pass
    def teardown(self): pass
    def run(self, ctx):
        forked, fork_error = _fork(600)
        report = {
            "env": sorted(os.environ),
            "uid": os.getuid(),
            "net_local": _connect("127.0.0.1", 9000),
            "net_external": _connect("1.1.1.1", 443),
            "writes": {p: _write(p) for p in ("/work/in/x", "/src/x", "/work/x", "/x", "/tmp/x")},
            "forked": forked,
            "fork_error": fork_error,
            "big": _write_big("/work/out/big", 2 * 1024 * 1024),
            "fallocate": _fallocate("/work/out/held"),
        }
        ctx.upload_bytes("probe.json", json.dumps(report).encode())
        return StepResult(output_keys=["probe.json"])


def _write_big(path, size):
    try:
        with open(path, "wb") as fh:
            fh.write(b"\0" * size)
        return "written"
    except OSError as exc:
        return _err(exc)
    finally:
        if os.path.exists(path):
            os.unlink(path)


class Sneaky(Step):
    name = "sneaky"
    def setup(self, config): pass
    def teardown(self): pass
    def run(self, ctx):
        os.makedirs(f"/work/out/{ctx.output_bucket}", exist_ok=True)
        os.symlink(os.path.expanduser("~/.ssh/id_rsa"), f"/work/out/{ctx.output_bucket}/k")
        return StepResult(output_keys=["k"])


class PickleReply(Step):
    name = "pickle-reply"
    def setup(self, config): pass
    def teardown(self): pass
    def run(self, ctx):
        with open("/work/reply/output.json", "wb") as fh:
            fh.write(pickle.dumps({"status": "ok"}))
        os._exit(0)


class Hide(Step):
    name = "hide"
    def setup(self, config): pass
    def teardown(self): pass
    def run(self, ctx):
        # write + search, no read: the watchdog (the job's uid, no DAC
        # override) cannot list it while the step fills it by name.
        hidden = "/work/out/h"
        os.mkdir(hidden)
        os.chmod(hidden, 0o300)
        for i in range(12):
            with open(f"{hidden}/part{i}", "wb") as fh:
                fh.write(b"\0" * (900 * 1024))
        time.sleep(120)
        return StepResult()


class Flood(Step):
    name = "flood"
    def setup(self, config): pass
    def teardown(self): pass
    def run(self, ctx):
        out = f"/work/out/{ctx.output_bucket}"
        os.makedirs(out, exist_ok=True)
        for i in range(12):
            with open(f"{out}/part{i}", "wb") as fh:
                fh.write(b"\0" * (900 * 1024))
        time.sleep(120)
        return StepResult()
"""


def _image_id(image: str) -> str:
    return subprocess.run(
        ["podman", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


@pytest.fixture(scope="module")
def step_image(tmp_path_factory):
    named = os.environ.get("UMRS_STEP_TEST_IMAGE")
    if named:
        yield named
        return
    have = subprocess.run(["podman", "image", "exists", BASE], check=False)
    if have.returncode != 0:
        pytest.skip(f"{BASE} is not present locally (nothing is pulled)")
    ctx = tmp_path_factory.mktemp("step-image")
    shutil.copytree(
        SRC / "umrs",
        ctx / "site-packages" / "umrs",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (ctx / "Containerfile").write_text(CONTAINERFILE)
    tag = f"localhost/umrs-step-test:{uuid.uuid4().hex[:12]}"
    subprocess.run(
        ["podman", "build", "--pull=never", "-q", "-t", tag, str(ctx)],
        check=True, capture_output=True, timeout=600,
    )  # fmt: skip
    yield tag
    subprocess.run(["podman", "rmi", "-f", tag], capture_output=True, check=False)


@pytest.fixture
def lane(tmp_path, monkeypatch, step_image):
    cluster = make_cluster(tmp_path, monkeypatch)
    # SELinux hosts: the step's mounts must carry a container label.
    subprocess.run(["chcon", "-t", "container_file_t", str(tmp_path / "lustre")],
                   capture_output=True, check=False)  # fmt: skip
    archive_dir = tmp_path / "images"
    archive_dir.mkdir()
    name, _, tag = step_image.rsplit("/", 1)[-1].partition(":")
    archive = archive_dir / f"{name}-{tag or 'latest'}-{platform.machine()}.tar"
    archive.write_bytes(b"")  # the node already holds the image: the .id matches
    (archive_dir / f"{archive.name}.id").write_text(_image_id(step_image) + "\n")
    venue = tmp_path / "venue.sh"
    venue.write_text("rivault_venue_prepare() { :; }\n")
    tree = tmp_path / "bundle" / "adv"
    tree.mkdir(parents=True)
    (tree / "__init__.py").write_text("")
    (tree / "steps.py").write_text(STEPS)
    monkeypatch.setenv(CANARY, "leak-me")
    cluster.behave(real_job=True)
    caps = dataclasses.replace(
        DEFAULT_CAPS, min_free_bytes=1, max_out_file_bytes=1 * MiB, max_out_bytes=4 * MiB
    )
    ex = cluster.executor(
        image=step_image, image_archive_dir=str(archive_dir), venue_script=str(venue),
        caps=caps, reply_cap=1 * MiB,
    )  # fmt: skip
    return cluster, ex, tree.parent


def _run(lane, step: str, s3=None):
    cluster, ex, tree = lane
    s3 = s3 if s3 is not None else FakeS3()
    ctx = StepContext(
        s3=s3,
        input_bucket="alice.proj.data",
        input_keys=[],
        output_bucket=OUT_BUCKET,
        metadata={
            EXECUTOR_SOURCE_DIR_METADATA_KEY: str(tree),
            STAGE_REQUEST_METADATA_KEY: {"owner": "alice", "step": step},
            STEP_TIMEOUT_METADATA_KEY: 300,
        },
        run_id=RUN,
    )
    return ex.execute(entrypoint=f"adv.steps:{step}", config={}, ctx=ctx), s3


def _check_env(env: list[str]) -> None:
    assert sorted(k for k in env if k.startswith("UMRS_")) == [
        "UMRS_STEP_INPUT", "UMRS_STEP_OUTPUT_JSON", "UMRS_STEP_SOURCE_DIR", "UMRS_STEP_STAGED",
    ]  # fmt: skip
    assert not [k for k in env if k.startswith(("AWS_", "S3_", "MINIO", "LITELLM"))]
    assert CANARY not in env


def _check_reach(report: dict) -> None:
    assert report["uid"] == os.getuid() != 0
    assert "connected" not in (report["net_local"], report["net_external"])
    writes = dict(report["writes"])
    assert writes.pop("/tmp/x") == "written"
    assert set(writes.values()) == {"EROFS"}, writes  # in, src, the rootfs: read-only


def _check_limits(report: dict) -> None:
    assert report["forked"] < 512
    assert report["fork_error"] == "EAGAIN"
    assert report["big"] == "EFBIG"  # --ulimit fsize = max_out_file_bytes
    # the step's seccomp profile: no reservation, a written (bounded) fallback
    assert report["fallocate"] == [errno.EOPNOTSUPP, 64 * 1024]


def test_the_step_reaches_nothing_but_its_files(lane):
    result, s3 = _run(lane, "Probe")
    report = json.loads(s3.objects[(OUT_BUCKET, result.output_keys[0])])
    _check_env(report["env"])
    _check_reach(report)
    _check_limits(report)
    assert lane[0].workdirs() == []


def test_a_symlink_to_the_accounts_ssh_key_never_leaves_the_cluster(lane):
    with pytest.raises(StepExecutionError, match="not a regular, singly-linked file"):
        _run(lane, "Sneaky")


def test_a_pickle_reply_is_refused(lane):
    with pytest.raises(StepExecutionError, match="step reply is not valid JSON"):
        _run(lane, "PickleReply")


def test_the_watchdog_stops_a_step_flooding_the_shared_filesystem(lane):
    with pytest.raises(StepExecutionError, match="passed their caps"):
        _run(lane, "Flood")
    cluster = lane[0]
    assert cluster.workdirs() == []
    assert not list(Path(cluster.root).glob("*/out"))


def test_the_watchdog_stops_a_step_hiding_its_outputs_from_it(lane):
    with pytest.raises(StepExecutionError, match="hidden from the watchdog"):
        _run(lane, "Hide")
    assert lane[0].workdirs() == []  # rmwd gets through the 0300 directory too
