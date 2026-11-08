"""End to end: a two-step run through the person's REAL gate (ADR-112 B2).

Plan §5 B2.0 §12 S binds the gate (``r4_contracts.hpc_gate``), the worker
(JSON mode), the server (resolver, proposal, admission, runner, gated lane)
and r4 (``r4.hpc_intents``, ``r4.hpc_signing``) together. This file runs them
as one system, extending the pattern of ``step_lane_harness.py``:

- a temp directory is the person's HPC home; the gate is installed into it
  exactly as ``r4 hpc add`` installs it (the server's ``transport-key`` and
  ``site-profile`` answers, r4's ``build_install`` and fixed install script,
  then the server's ``verify`` and ``links`` routes);
- ``ssh`` on the server's PATH is a fake sshd: it answers ``ssh -G`` with a
  fixed target, and otherwise finds the ``authorized_keys`` line carrying
  the offered key and runs ITS forced command (``/usr/bin/python3 -I
  <home>/.umrs/gate/umrs_gate.py --site <site>``) with
  ``SSH_ORIGINAL_COMMAND`` set; it never reads a real ``~/.ssh``;
- the gate's ``slurm.path`` is a directory of fake ``sbatch`` / ``squeue`` /
  ``sacct`` / ``scancel`` / ``sacctmgr``. The fake ``sbatch`` runs the
  gate-written ``job.sbatch`` under ``bash -p`` as the job, with a fake
  ``podman`` that loads the operator's docker archive and runs the step pod
  as the REAL ``umrs.server.step_worker`` in JSON mode (``pickle.loads`` is
  patched to raise inside it) over the gate-written ``input.json``,
  ``in/manifest.json``, ``src/`` and ``out/``;
- r4 asks for the proposal, recomputes it from the export, shows it, signs
  it with a real ``ssh-keygen`` key generated in ``tmp_path`` (the test hook
  for a passphrase-less key) and triggers; the server admits the run and
  its runner drives both steps through the gate.

The tampering cases each refuse by name before the step runs.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import os
import platform
import pwd
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from step_lane_harness import FakeS3

hpc_gate = pytest.importorskip("r4_contracts.hpc_gate")
hs = pytest.importorskip("r4.hpc_signing")
hpc_remote = pytest.importorskip("r4.hpc_remote")
hpc_intents = pytest.importorskip("r4.hpc_intents")
hpc_site_profile = pytest.importorskip("r4.hpc_site_profile")

from umrs.api import hpc_accounts as api  # noqa: E402
from umrs.api.workflow_export import build_export_zip  # noqa: E402
from umrs.hpc_accounts.store import InMemorySiteAccountStore  # noqa: E402
from umrs.orchestrator.site_config import (  # noqa: E402
    HpcAccessConfig,
    ServiceAccountAccess,
    SiteConfig,
    UserAccountAccess,
)
from umrs.orchestrator.ssh_control import SSHControlConfig  # noqa: E402
from umrs.server import gated_step_lane as lane  # noqa: E402
from umrs.server import intent_proposal  # noqa: E402
from umrs.server import workflow_runner as wr  # noqa: E402
from umrs.server.hpc_user_work import workflow_user_work_identity  # noqa: E402
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor  # noqa: E402
from umrs.server.run_intent_admission import admit_run  # noqa: E402
from umrs.server.step_staging import DEFAULT_CAPS  # noqa: E402
from umrs.server.workflow_execution_registry import ExecutionRecord  # noqa: E402
from umrs.shared import hpc_user_work_access as access  # noqa: E402

GATE_PYTHON = "/usr/bin/python3"
KEYGEN = "/usr/bin/ssh-keygen"
SITE = "e2e-gated"
DOMAIN = "platform.example.org"
USER = "alice"
PROJECT = "p1"
ACCOUNT = "grp-a"
WORKFLOW = "chain"
VERSION = "1.0.0"
IMAGE = "localhost/umrs-step:e2e0123456"
HEADERS = {"X-User-ID": USER, "X-Verified-Project-ID": PROJECT}
LOGIN = pwd.getpwuid(os.geteuid()).pw_name
ARCH = platform.machine() if platform.machine() in ("x86_64", "aarch64") else "x86_64"
UMRS_SRC = Path(__file__).resolve().parents[1] / "src"
INPUT = b"hello gated world"
LABELS = {"org.umrs.step-contract": "1", "org.umrs.step-payload": "pickle,json"}
_LANE_ENV = {
    "UMRS_SLURM_STEP_SUBMIT": "remote",
    "UMRS_SLURM_WORKDIR_ROOT": "/lvs0/x/umrs_steps",
    "UMRS_STEP_REMOTE_ENABLED": "1",
    "UMRS_STEP_HELPER": "/lvs0/x/umrs_step_helper.py",
    "UMRS_STEP_IMAGE": IMAGE,
}


def _tools_ok() -> bool:
    tools = (GATE_PYTHON, KEYGEN, "/bin/bash", "/usr/bin/sha256sum")
    if not all(os.access(tool, os.X_OK) for tool in tools):
        return False
    probe = [GATE_PYTHON, "-I", "-c", "import sys; sys.exit(sys.version_info < (3, 9))"]
    return subprocess.run(probe, check=False, timeout=30).returncode == 0


pytestmark = pytest.mark.skipif(
    not _tools_ok(), reason="needs /usr/bin/python3 >= 3.9, ssh-keygen, bash and sha256sum"
)

# ── the fake transport: ssh -G, and sshd's forced command ────────────────────

FAKE_SSH = r"""#!__PYTHON__
import json, os, re, shlex, sys

HOME, PATH, LOG, LOGIN = __HOME__, __PATH__, __LOG__, __LOGIN__
args = sys.argv[1:]
if args[:1] == ["-G"]:
    print("hostname login.example.org")
    print("port 22")
    sys.exit(0)
key, dest, command = args[args.index("-i") + 1], args[-2], args[-1]
with open(LOG, "a") as fh:
    fh.write(json.dumps(command) + "\n")
pub = open(key + ".pub").read().split()[1]
forced = None
path = os.path.join(HOME, ".ssh", "authorized_keys")
lines = open(path).read().splitlines() if os.path.exists(path) else []
for line in lines:
    options, sep, rest = line.partition(" ssh-ed25519 ")
    if sep and rest.split(" ")[0] == pub:
        found = re.search(r'(?:^|,)command="([^"]*)"', options)
        forced = found.group(1) if found else None
if forced is None or dest.split("@")[0] != LOGIN:
    sys.stderr.write("Permission denied (publickey).\r\n")
    sys.exit(255)
argv = shlex.split(forced)
env = {"HOME": HOME, "PATH": PATH, "LC_ALL": "C", "SSH_ORIGINAL_COMMAND": command}
os.execve(argv[0], argv, env)
"""

# ── the fake cluster: Slurm runs the gate's job script as the job ────────────

FAKE_SLURM = r"""#!__PYTHON__
import fcntl, json, os, subprocess, sys

STATE, NODE = __STATE__, __NODE__
LIVE = {"PENDING", "RUNNING"}
USAGE = "{}|{}|00:00:07|cpu=1,mem=1G|1|grp-a|cpu|2026-11-08T10:00:00|2026-11-08T10:00:07"


def load(name, default):
    path = os.path.join(STATE, name)
    return json.load(open(path)) if os.path.exists(path) else default


def opt(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def update(change):
    with open(os.path.join(STATE, "lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load("jobs.json", {"next": 1000, "jobs": {}})
        out = change(data)
        json.dump(data, open(os.path.join(STATE, "jobs.json"), "w"))
    return out


def queue(data, wd):
    job = str(data["next"])
    data["next"] += 1
    data["jobs"][job] = {"workdir": wd, "state": "RUNNING"}
    return job


def finish(job, state):
    def change(data):
        data["jobs"][job]["state"] = state
    update(change)


def sbatch(args):
    if load("behaviour.json", {}).get("sbatch_rc"):
        sys.stderr.write("sbatch: error: Batch job submission failed\n")
        sys.exit(1)
    wd = [a[len("--chdir="):] for a in args if a.startswith("--chdir=")][0]
    job = update(lambda data: queue(data, wd))
    env = dict(NODE, SLURM_JOB_ID=job)
    with open(os.path.join(wd, "log", "slurm.out"), "ab") as out, \
         open(os.path.join(wd, "log", "slurm.err"), "ab") as err:
        rc = subprocess.call(["/bin/bash", "-p", args[-1]], cwd=wd, env=env,
                             stdin=subprocess.DEVNULL, stdout=out, stderr=err)
    finish(job, "COMPLETED" if rc == 0 else "FAILED")
    print(f"{job};fakecluster")


def squeue(args):
    live = {k: v for k, v in load("jobs.json", {"jobs": {}})["jobs"].items() if v["state"] in LIVE}
    if "-j" in args:
        if opt(args, "-j") not in live:
            sys.stderr.write("slurm_load_jobs error: Invalid job id specified\n")
            sys.exit(1)
        print(live[opt(args, "-j")]["state"])
        return
    for k, v in live.items():
        print(f"{k} {v['workdir']}" if opt(args, "-o") == "%i %Z" else k)


def sacct(args):
    known = load("jobs.json", {"jobs": {}})["jobs"]
    ids = opt(args, "-j").split(",") if "-j" in args else list(known)
    fmt = opt(args, "-o")
    for k in [k for k in ids if k in known]:
        state, wd = known[k]["state"], known[k]["workdir"]
        rows = {"State": state, "JobID,WorkDir": f"{k}|{wd}", "JobID,State": f"{k}|{state}"}
        print(rows.get(fmt, USAGE.format(k, state)))


def scancel(args):
    finish(args[-1], "CANCELLED")


def sacctmgr(args):
    sys.stdout.write("c1|grp-a|cpu|normal\nc1|grp-b||\n")


tool = os.path.basename(sys.argv[0])
with open(os.path.join(STATE, "calls.jsonl"), "a") as fh:
    fh.write(json.dumps([tool, *sys.argv[1:]]) + "\n")
{"sbatch": sbatch, "squeue": squeue, "sacct": sacct, "scancel": scancel,
 "sacctmgr": sacctmgr}[tool](sys.argv[1:])
"""

FAKE_PODMAN = r"""#!__PYTHON__
import hashlib, json, os, subprocess, sys, tarfile

STATE, SHIM, SRC = __STATE__, __SHIM__, __SRC__
args = sys.argv[1:]
if args and args[0].startswith("--cgroup-manager"):
    args = args[1:]
store = os.path.join(STATE, "store")
os.makedirs(store, exist_ok=True)


def inspect(fmt, ref):
    path = os.path.join(store, ref + ".json")
    if not os.path.exists(path):
        sys.exit(125)
    labels = json.load(open(path)).get("config", {}).get("Labels") or {}
    if fmt == "{{.Id}}":
        print(ref)
    else:
        print(labels.get(fmt.split('"')[1], "<no value>"))


def load(path):
    with tarfile.open(path) as tar:
        manifest = json.load(tar.extractfile("manifest.json"))
        config = tar.extractfile(manifest[0]["Config"]).read()
    image_id = hashlib.sha256(config).hexdigest()
    open(os.path.join(store, image_id + ".json"), "wb").write(config)
    with open(os.path.join(STATE, "loads.jsonl"), "a") as fh:
        fh.write(json.dumps(image_id) + "\n")


def values(rest, flag):
    return [rest[i + 1] for i, word in enumerate(rest[:-1]) if word == flag]


def host_path(mounts, path):
    for host, ctr in mounts:
        if path == ctr or path.startswith(ctr + "/"):
            return host + path[len(ctr):]
    return path


def step(rest):
    mounts = [tuple(v.split(":")[:2]) for v in values(rest, "-v")]
    env = {"PATH": "/usr/bin:/bin", "HOME": STATE, "PYTHONPATH": SRC,
           "PYTHONDONTWRITEBYTECODE": "1",
           "FAKE_IN": host_path(mounts, "/work/in"), "FAKE_OUT": host_path(mounts, "/work/out")}
    for item in values(rest, "-e"):
        name, _, value = item.partition("=")
        if name.startswith("UMRS_STEP_"):
            env[name] = host_path(mounts, value)
    with open(os.path.join(STATE, "pods.jsonl"), "a") as fh:
        fh.write(json.dumps({"argv": rest, "env": env}) + "\n")
    sys.exit(subprocess.call([sys.executable, SHIM], env=env))


def run(rest):
    if "-c" in rest:
        print("{}")
    elif "/sys/fs/cgroup/pids.max" in rest:
        print([a for a in rest if a.startswith("--pids-limit=")][0].split("=")[1])
    else:
        step(rest)


if args[:2] == ["image", "inspect"]:
    inspect(args[3], args[4])
elif args[:2] == ["load", "-i"]:
    load(args[2])
elif args[:1] == ["info"]:
    print(os.path.join(STATE, "seccomp.json") if "SECCOMP" in args[2] else "")
elif args[:1] == ["run"]:
    run(args[1:])
"""

#: The step pod: the real worker, JSON mode, with no pickle to fall back on.
WORKER_SHIM = """import os, pickle, sys


def _no_pickle(*_a, **_k):
    raise AssertionError("pickle.loads was called in JSON mode")


pickle.loads = _no_pickle
from umrs.pipeline_sdk import staged_s3

staged_s3.StagedS3Client.__init__.__defaults__ = (
    os.environ.pop("FAKE_IN"), os.environ.pop("FAKE_OUT"), None
)
from umrs.server import step_worker

sys.exit(step_worker.entry([]))
"""

# ── the workflow: two steps, the second chained on the first ─────────────────

STEPS_PY = """from umrs.pipeline_sdk import Step, StepResult


class First(Step):
    name = "first"

    def setup(self, config):
        self.config = config

    def teardown(self):
        pass

    def run(self, ctx):
        data = ctx.download_bytes(self.config["input_key"])
        ctx.upload_bytes("first/out.txt", data.upper())
        seen = {"threshold": self.config["threshold"], "label": self.config["label"]}
        return StepResult(output_keys=["first/out.txt"], extra={"seen": seen})


class Second(Step):
    name = "second"

    def setup(self, config):
        self.config = config

    def teardown(self):
        pass

    def run(self, ctx):
        data = b"".join(ctx.download_parent_bytes(key) for key in ctx.input_keys)
        ctx.upload_bytes("second/out.txt", data + b"!")
        parent = ctx.parent_results["first"]
        extra = {"parent_keys": list(parent.output_keys), "input_keys": list(ctx.input_keys)}
        return StepResult(output_keys=["second/out.txt"], extra=extra)
"""

WORKFLOW_JSON = {
    "name": WORKFLOW,
    "version": VERSION,
    "execution_mode": "slurm",
    "steps": [
        {
            "name": "first",
            "entrypoint": "steps.chain:First",
            "config": {"threshold": 0.25, "label": None},
            "stage_inputs": {"params": ["input_key"]},
            "depends_on": [],
        },
        {"name": "second", "entrypoint": "steps.chain:Second", "depends_on": ["first"]},
    ],
}


def _script(path: Path, text: str, **values: Any) -> None:
    text = text.replace("__PYTHON__", sys.executable)
    for name, value in values.items():
        text = text.replace(f"__{name}__", repr(value))
    path.write_text(text)
    path.chmod(0o755)


def _docker_archive(salt: str = "") -> tuple[bytes, str]:
    """A docker-archive tar (manifest.json + config); (bytes, image_id)."""
    config = json.dumps({"architecture": "amd64", "config": {"Labels": LABELS}, "salt": salt})
    raw = config.encode()
    image_id = hashlib.sha256(raw).hexdigest()
    manifest = json.dumps([{"Config": f"{image_id}.json", "RepoTags": [IMAGE], "Layers": []}])
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w") as tar:
        for name, data in (("manifest.json", manifest.encode()), (f"{image_id}.json", raw)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return sink.getvalue(), image_id


# ── the world ────────────────────────────────────────────────────────────────


@dataclass
class Cluster:
    """The person's HPC home, the fake Slurm and podman, the archives."""

    tmp: Path
    home: Path
    slurm_bin: Path
    slurm: Path
    node: Path
    archives: Path
    ssh_log: Path

    @property
    def gate(self) -> Path:
        return self.home / ".umrs" / "gate"

    def calls(self, tool: str) -> list[list[str]]:
        path = self.slurm / "calls.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        return [row for row in rows if row[0] == tool]

    def pods(self) -> list[dict[str, Any]]:
        path = self.node / "pods.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def verbs(self) -> list[str]:
        if not self.ssh_log.exists():
            return []
        return [json.loads(x).split()[0] for x in self.ssh_log.read_text().splitlines()]

    def log_entries(self) -> list[dict[str, Any]]:
        path = self.gate / "log" / "gate.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _mkdirs(tmp: Path) -> Cluster:
    names = ("hpc-home", "slurm-bin", "slurm", "node", "archives", "xdg", "fake-ssh")
    for name in names:
        (tmp / name).mkdir(mode=0o700)
    for name in ("slurm-bin", "archives"):
        (tmp / name).chmod(0o755)
    (tmp / "node" / "seccomp.json").write_text("{}")
    return Cluster(
        tmp, tmp / "hpc-home", tmp / "slurm-bin", tmp / "slurm", tmp / "node",
        tmp / "archives", tmp / "ssh.jsonl",
    )  # fmt: skip


def _install_fakes(c: Cluster) -> None:
    node_bin = c.node / "bin"
    node_bin.mkdir(mode=0o700)
    shim = c.node / "worker_shim.py"
    shim.write_text(WORKER_SHIM)
    _script(node_bin / "podman", FAKE_PODMAN, STATE=str(c.node), SHIM=str(shim), SRC=str(UMRS_SRC))
    node_env = {
        "UMRS_GATE_CLEAN": "1", "PATH": f"{node_bin}:{c.slurm_bin}:/usr/bin:/bin",
        "HOME": str(c.home), "LANG": "C.UTF-8", "XDG_RUNTIME_DIR": str(c.tmp / "xdg"),
    }  # fmt: skip
    for tool in ("sbatch", "squeue", "sacct", "scancel", "sacctmgr"):
        _script(c.slurm_bin / tool, FAKE_SLURM, STATE=str(c.slurm), NODE=node_env)
    session_path = f"{c.slurm_bin}:/usr/bin:/bin"
    _script(
        c.tmp / "fake-ssh" / "ssh", FAKE_SSH,
        HOME=str(c.home), PATH=session_path, LOG=str(c.ssh_log), LOGIN=LOGIN,
    )  # fmt: skip


def _image(c: Cluster) -> dict[str, str]:
    data, image_id = _docker_archive()
    sha = hashlib.sha256(data).hexdigest()
    (c.archives / f"{sha}.tar").write_bytes(data)
    digests = c.tmp / "image-digests.json"
    digests.write_text(json.dumps({IMAGE: {ARCH: {"image_id": image_id, "archive_sha256": sha}}}))
    return {"ref": IMAGE, "image_id": image_id, "archive_sha256": sha, "digests": str(digests)}


def _profile_block(c: Cluster) -> dict[str, Any]:
    """The site's ``hpc_access.user_account.profile`` overrides (§2)."""
    partition = {
        "arch": ARCH, "max_cpus": 4, "max_gpus": 0, "max_time": 600, "mem": False,
        "gpu_request": "none", "gres": "", "gpu_flag": "gres", "exclusive": False,
    }  # fmt: skip
    return {
        "image_archive_dir": str(c.archives),
        "slurm": {"path": str(c.slurm_bin), "owner_uids": [os.geteuid()], "queue_seconds": 60},
        "partitions": {"cpu": partition},
        "venue": {"root": str(c.tmp / "venue")},
        "caps": {"min_free_bytes": 0, "max_workdirs": 16},
        "watch_seconds": 1,
    }


def _site(c: Cluster, digests: str) -> SiteConfig:
    site = SiteConfig(name=SITE)
    site.extra = {"ssh_target": "login.example.org", "ssh_port": 22, "ssh_user": "op"}
    site.scheduler.hpc_access = HpcAccessConfig(
        operator="Centre X",
        modes_allowed=["service_account", "user_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(login="op", personal=True, allowed_projects=[]),
        user_account=UserAccountAccess(
            methods=["gated_key"], signing_domain=DOMAIN, profile=_profile_block(c)
        ),
    )
    se = site.step_executor
    se.partition, se.cpus, se.memory, se.gpus = "cpu", 1, "", 0
    se.time_limit, se.queue_timeout, se.image, se.image_digests = (
        "00:10:00", "00:01:00", IMAGE, digests,
    )  # fmt: skip
    return site


def _source_tree(tmp: Path) -> Path:
    src = tmp / "registered" / WORKFLOW
    (src / "steps").mkdir(parents=True)
    (src / "steps" / "__init__.py").write_text("")
    (src / "steps" / "chain.py").write_text(STEPS_PY)
    (src / "workflow.json").write_text(json.dumps(WORKFLOW_JSON, indent=2))
    return src


@dataclass
class World:
    cluster: Cluster
    site: SiteConfig
    store: InMemorySiteAccountStore
    api: TestClient
    s3: FakeS3
    source: Path
    image: dict[str, str]
    records: list[ExecutionRecord] = dataclasses.field(default_factory=list)

    @property
    def entry(self) -> dict[str, Any]:
        digest = hashlib.sha256((self.source / "steps" / "chain.py").read_bytes()).hexdigest()
        return {"source_path": str(self.source), "version": VERSION, "source_sha256": digest}

    def workflow_json(self) -> dict[str, Any]:
        return json.loads((self.source / "workflow.json").read_text())

    def identity(self) -> Any:
        return workflow_user_work_identity(
            self.workflow_json(), str(self.source), user_id=USER, project_id=PROJECT
        )


def _api_app(store: InMemorySiteAccountStore, site: SiteConfig, monkeypatch) -> TestClient:
    monkeypatch.setattr(api, "_load_site", lambda: site)
    monkeypatch.setattr(api, "_make_store", lambda _req, _venue: store)
    monkeypatch.setattr(api, "_live_memberships", lambda _req, user: {USER: [PROJECT]}.get(user))
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def _signing_key() -> Path:
    key = hs.key_path("ed25519")
    key.parent.mkdir(parents=True, mode=0o700)
    argv = [KEYGEN, "-q", "-t", "ed25519", "-N", "", "-C", "umrs-signing", "-f", str(key)]
    subprocess.run(argv, check=True, timeout=30)
    return key


def _install_as_r4(w: World) -> None:
    """``r4 hpc add`` and ``verify``: offer, profile, install, records."""
    offer = hs.parse_offer(_ok(w.api.post("/umrs/sites/me/hpc/transport-key", json={"rotate": False}, headers=HEADERS)))
    profile = hpc_site_profile.parse_site_profile(
        _ok(w.api.get("/umrs/sites/me/hpc/site-profile", headers=HEADERS)),
        venue=offer.site, domain=offer.domain,
    )  # fmt: skip
    key = _signing_key()
    plan = hs.build_install(
        offer, principal=USER, home=str(w.cluster.home), signer=key, kind="ed25519",
        site_profile=profile.data,
    )  # fmt: skip
    proc = subprocess.run(
        ["sh", "-c", hpc_remote.INSTALL_SCRIPT],
        input=hpc_remote.install_payload(plan),
        capture_output=True,
        env={"HOME": str(w.cluster.home), "PATH": "/usr/bin:/bin"},
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0 and proc.stdout.strip() == b"installed", proc.stderr
    hs.record_site_signer(SITE, "ed25519", key)
    hpc_site_profile.remember(SITE, profile)
    hs.update_site_record(SITE, venue=offer.site, domain=offer.domain, site_profile_sha256=profile.sha256)


def _verify_and_link(w: World) -> None:
    key = hs.key_path("ed25519")
    body = {"login": LOGIN, "signers": [hs.public_key_info(key)[2]], "uv_required": False, "allow_non_sk": True}
    account = _ok(w.api.post("/umrs/sites/me/hpc/verify", json=body, headers=HEADERS))["account"]
    assert account["status"] == "active", account
    hs.update_site_record(SITE, login=account["verified_login"])
    _ok(w.api.post("/umrs/sites/me/hpc/links", json={"project_id": PROJECT, "slurm_account": ACCOUNT}, headers=HEADERS))


def _ok(response: Any) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def _environment(tmp: Path, c: Cluster, monkeypatch) -> None:
    for name, value in _LANE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.setenv("UMRS_TRANSPORT_KEY_DIR", str(tmp / "transport-keys"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp / "r4-config"))
    monkeypatch.setenv("HOME", str(tmp / "server-home"))
    monkeypatch.setenv("PATH", f"{c.tmp / 'fake-ssh'}:{os.environ['PATH']}")
    monkeypatch.delenv("R4_COMPANION_SESSION", raising=False)
    monkeypatch.setattr(hs, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(hs, "_ALLOW_UNENCRYPTED_KEYS", True)


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    """The person's gate installed and verified, the run's store holding the
    first step's input, the server wired to this site and this store."""
    cluster = _mkdirs(tmp_path)
    _install_fakes(cluster)
    _environment(tmp_path, cluster, monkeypatch)
    image = _image(cluster)
    site, store = _site(cluster, image["digests"]), InMemorySiteAccountStore(SITE)
    monkeypatch.setattr(access, "current_site", lambda: site)
    monkeypatch.setattr(access, "user_is_admin", lambda _u: False)
    access.set_site_account_store_factory(lambda _venue: store)
    s3 = FakeS3({("wf-runner", "data/in.txt"): INPUT})
    monkeypatch.setattr(intent_proposal, "proposal_s3", lambda _params, _project: s3)
    monkeypatch.setattr(wr, "_resolve_s3_client", lambda *_a, **_k: (s3, "wf-runner"))
    w = World(cluster, site, store, _api_app(store, site, monkeypatch), s3, _source_tree(tmp_path), image)
    _install_as_r4(w)
    _verify_and_link(w)
    yield w
    access.set_site_account_store_factory(None)


# ── r4 talking to this server ────────────────────────────────────────────────


class Workflows:
    def __init__(self, world: World) -> None:
        self.world = world

    def export_bytes(self, name: str, *, version: str, max_bytes: int) -> bytes:
        assert (name, version) == (WORKFLOW, VERSION)
        result = build_export_zip(name, self.world.entry)
        try:
            data = Path(result.path).read_bytes()
        finally:
            os.unlink(result.path)
        assert len(data) <= max_bytes
        return data


class R4Client:
    """r4's ``Client`` surface the signing flow uses, answered by the server's
    own code: the account routes over HTTP, the proposal by
    ``intent_proposal.propose`` and the trigger by ``admit_run``."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.workflows = Workflows(world)
        self.admitted: list[Any] = []

    def hpc_account(self) -> dict[str, Any]:
        return _ok(self.world.api.get("/umrs/sites/me/hpc", headers=HEADERS))

    def site_me(self) -> dict[str, Any]:
        return {"user_id": USER}

    def _params(self, body: dict[str, Any]) -> dict[str, Any]:
        return {**dict(body.get("params") or {}), "project_id": PROJECT}

    def workflow_intent_proposal(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        w = self.world
        resolved = SimpleNamespace(
            name=name, version=VERSION, entry=w.entry, verified_project=PROJECT,
            user_id=USER, params=self._params(body),
        )  # fmt: skip
        return intent_proposal.propose(resolved, w.workflow_json(), str(w.source), w.identity(), body.get("inputs"))

    def workflow_trigger_signed(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        run = {"workflow": name, "version": VERSION, "project": PROJECT, "principal": USER}
        admission, error = admit_run(self.world.identity(), body, self._params(body), run, _no_run)
        assert error is None, error
        self.admitted.append(admission)
        return {"execution_id": admission.execution_id, "status": "accepted"}


def _no_run(_execution_id: str) -> bool:
    return False


def _sign_and_trigger(w: World) -> Any:
    """r4's whole §11 flow; the server's admission of the signed run."""
    client = R4Client(w)
    request = hpc_intents.TriggerRequest(
        workflow=WORKFLOW, version=None, params={"input_key": "data/in.txt"},
        input_files=[], project=PROJECT, site=SITE, yes=True,
    )  # fmt: skip
    shown: list[str] = []
    answer = hpc_intents.run_gated_trigger(client, request, confirm=lambda _q: False, out=shown.append)
    assert hpc_intents.D10_TEXT.split(".")[0] in "\n".join(shown)
    [admission] = client.admitted
    assert answer["intent_id"] == admission.intent.intent_id
    return admission.intent


# ── the run ─────────────────────────────────────────────────────────────────


def _executor(**over: Any) -> RemoteSlurmStepExecutor:
    values: dict[str, Any] = {
        "ssh_config": SSHControlConfig(ssh_target="login", ssh_user="op"),
        "enabled": True,
        "poll_interval": 0,
        "unreachable_budget": 2.0,
        "caps": dataclasses.replace(DEFAULT_CAPS, min_free_bytes=1),
    }
    values.update(over)
    executor = RemoteSlurmStepExecutor(**values)

    def no_helper(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the helper lane (the platform login) was used")

    executor._ssh_run = no_helper  # type: ignore[method-assign]
    executor._stream = no_helper  # type: ignore[method-assign]
    return executor


def _record(intent: Any) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=intent.run_id,
        workflow_name=WORKFLOW,
        user_id=USER,
        params=dict(intent.body["params"]),
        intent_b64=intent.b64,
        intent_signature=intent.signature,
        intent_id=intent.intent_id,
    )


def _run(w: World, intent: Any, **executor: Any) -> ExecutionRecord:
    record = _record(intent)
    wr.run_workflow_record(
        record,
        source_path=str(w.source),
        workflow_json=w.workflow_json(),
        executor=_executor(**executor),
        project_id=PROJECT,
        user_id=USER,
        verified_project=PROJECT,
    )
    w.records.append(record)
    return record


def _resigned(w: World, intent: Any, **change: Any) -> Any:
    """The same run with *change* applied, signed by the person's own key
    (as if r4 had been talked into it): the gate is the last word."""
    from umrs.hpc_accounts.intents import decode_signed_intent

    body = {**intent.body, **change}
    payload = hpc_gate.canonical_bytes(body, json_values=True)
    namespace = hpc_gate.namespace("intent", DOMAIN)
    signature = hs.sign(payload, namespace, hs.key_path("ed25519"))
    return decode_signed_intent(base64.b64encode(payload).decode(), signature)


def _out(w: World, step: str, key: str, run_id: str) -> bytes:
    return w.s3.objects[("wf-runner-output", f"runs/{run_id}/{step}/{key}")]


def _pod_view(pod: dict[str, Any]) -> tuple[Any, ...]:
    """(signed entrypoint, image id literal, pickle input, JSON input name)."""
    env = pod["env"]
    json_input = Path(env["UMRS_STEP_INPUT_JSON"]).name
    return (env["UMRS_STEP_ENTRYPOINT"], pod["argv"][-3], env.get("UMRS_STEP_INPUT"), json_input)


def _extra(record: ExecutionRecord, step: str) -> dict[str, Any]:
    """The step's own ``result.extra`` as the runner recorded it."""
    return record.steps[step].extra["extra"]


def _error(record: ExecutionRecord) -> str:
    return str(record.error or "") + json.dumps(record.steps, default=str)


# ── the happy path ───────────────────────────────────────────────────────────


def test_a_signed_two_step_run_goes_through_the_real_gate(world):
    intent = _sign_and_trigger(world)
    record = _run(world, intent)
    assert record.status == "succeeded", _error(record)
    run_id = intent.run_id
    assert _out(world, "first", "first/out.txt", run_id) == INPUT.upper()
    assert _out(world, "second", "second/out.txt", run_id) == INPUT.upper() + b"!"
    assert len(world.cluster.calls("sbatch")) == 2


def test_the_steps_ran_as_the_worker_in_json_mode_from_gate_written_files(world):
    intent = _sign_and_trigger(world)
    record = _run(world, intent)
    assert record.status == "succeeded", _error(record)
    image_id = world.image["image_id"]
    assert [_pod_view(p) for p in world.cluster.pods()] == [
        ("steps.chain:First", image_id, None, "input.json"),
        ("steps.chain:Second", image_id, None, "input.json"),
    ]
    first = _extra(record, "first")["seen"]
    assert first == {"threshold": 0.25, "label": None}  # a float and a null reached the step


def test_the_chained_step_reads_the_parent_outputs_under_physical_keys(world):
    intent = _sign_and_trigger(world)
    record = _run(world, intent)
    assert record.status == "succeeded", _error(record)
    physical = f"runs/{intent.run_id}/first/first/out.txt"
    extra = _extra(record, "second")
    assert extra["input_keys"] == [physical] and extra["parent_keys"] == [physical]


def test_a_stage_in_dropped_halfway_is_recovered_with_a_fresh_workdir(world, monkeypatch):
    intent = _sign_and_trigger(world)
    original, dropped = lane.feed_stage_in, []

    def drop_once(sink: Any, *args: Any) -> int:
        if dropped:
            return original(sink, *args)
        buffer = io.BytesIO()
        original(buffer, *args)
        sink.write(buffer.getvalue()[:700])  # half a header and some of a file
        dropped.append(True)
        raise OSError("the connection dropped")

    monkeypatch.setattr(lane, "feed_stage_in", drop_once)
    record = _run(world, intent)
    assert record.status == "succeeded", _error(record)
    verbs = world.cluster.verbs()
    first = verbs[verbs.index("lookup") : verbs.index("submit")]
    assert first == ["lookup", "mkwd", "stage-in", "rmwd", "mkwd", "stage-in"]
    assert len(world.cluster.calls("sbatch")) == 2


def test_the_server_drove_only_gate_verbs_and_the_gate_logged_each_submit(world):
    intent = _sign_and_trigger(world)
    record = _run(world, intent)
    assert record.status == "succeeded", _error(record)
    per_step = ["lookup", "mkwd", "stage-in", "submit", "status", "read-reply", "collect-out", "usage", "rmwd"]
    assert world.cluster.verbs()[-2 * len(per_step) :] == per_step * 2
    submits = [e for e in world.cluster.log_entries() if e["verb"] == "submit" and e["result"] == "ok"]
    assert [f"intent={intent.intent_id}" in e["detail"] for e in submits] == [True, True]
    assert hpc_gate.verify_log(str(world.cluster.gate / "log" / "gate.jsonl"))


def test_the_witness_sees_only_intents_r4_signed(world):
    intent = _sign_and_trigger(world)
    assert _run(world, intent).status == "succeeded"
    data = (world.cluster.gate / "log" / "gate.jsonl").read_bytes()
    report = hpc_intents.witness_site(SITE, data)
    assert report.chain_ok and not report.forked and report.unknown == []
    assert {ident for _n, ident in report.submits} == {intent.intent_id}


def test_the_step_records_name_the_gated_transport_and_the_jobs_measured_usage(world):
    intent = _sign_and_trigger(world)
    record = _run(world, intent)
    assert record.status == "succeeded", _error(record)
    [job] = record.steps["first"].extra["slurm_jobs"].values()
    assert (job["transport"], job["intent_id"]) == (f"gated:{SITE}:{USER}", intent.intent_id)
    assert (job["access_mode"], job["site_login"], job["slurm_account"]) == ("user_account", LOGIN, ACCOUNT)
    assert job["gate_log"]["n"] > 0 and job["cleaned"] is True
    measured = _extra(record, "first")["slurm"]
    assert (measured["job_id"], measured["elapsed_seconds"]) == (job["job_id"], 7)


# ── tampering: each refused by name, the step never runs ─────────────────────


def _refused(w: World, record: ExecutionRecord, name: str) -> None:
    assert record.status == "failed"
    assert name in _error(record), _error(record)
    assert w.cluster.pods() == []


def test_an_extra_src_file_is_refused_by_the_gate(world, monkeypatch):
    intent = _sign_and_trigger(world)
    original = lane._add_sources

    def with_extra(tar: Any, files: list[dict[str, Any]], source_dir: str) -> None:
        original(tar, files, source_dir)
        extra = b"print('not signed')\n"
        lane._add_checked(tar, "src/steps/extra.py", io.BytesIO(extra), len(extra), hashlib.sha256(extra).hexdigest())

    monkeypatch.setattr(lane, "_add_sources", with_extra)
    _refused(world, _run(world, intent), "src_mismatch")
    assert world.cluster.calls("sbatch") == []


def test_a_swapped_input_is_refused_before_the_gate_seals_anything(world):
    intent = _sign_and_trigger(world)
    world.s3.objects[("wf-runner", "data/in.txt")] = b"HELLO GATED WORLX"
    _refused(world, _run(world, intent), "input_changed")
    assert world.cluster.calls("sbatch") == []


def test_a_swapped_input_from_a_compromised_server_is_refused_by_the_gate(world, monkeypatch):
    intent = _sign_and_trigger(world)
    world.s3.objects[("wf-runner", "data/in.txt")] = b"HELLO GATED WORLX"
    monkeypatch.setattr(lane._Hashing, "check", lambda _self, _sha: None)
    _refused(world, _run(world, intent), "digest_mismatch")
    assert world.cluster.calls("sbatch") == []


def test_a_swapped_image_archive_is_refused_by_the_gate(world):
    intent = _sign_and_trigger(world)
    path = world.cluster.archives / f"{world.image['archive_sha256']}.tar"
    path.write_bytes(_docker_archive("swapped")[0])
    _refused(world, _run(world, intent), "image_mismatch")
    assert world.cluster.calls("sbatch") == []


def test_a_changed_entrypoint_is_refused_before_anything_is_sent(world):
    intent = _sign_and_trigger(world)
    changed = world.workflow_json()
    changed["steps"][0]["entrypoint"] = "steps.chain:Second"
    (world.source / "workflow.json").write_text(json.dumps(changed))
    before = len(world.cluster.verbs())
    _refused(world, _run(world, intent), "entrypoint_changed")
    assert world.cluster.verbs()[before:] == []


def test_a_retried_run_adopts_its_jobs_and_never_submits_twice(world):
    intent = _sign_and_trigger(world)
    assert _run(world, intent, keep_workdir=True).status == "succeeded"
    again = _run(world, intent, keep_workdir=True)  # lookup finds each step's job
    assert again.status == "succeeded", _error(again)
    assert len(world.cluster.calls("sbatch")) == 2
    assert _out(world, "second", "second/out.txt", intent.run_id) == INPUT.upper() + b"!"


def test_a_replayed_step_is_refused_by_name_and_never_runs_twice(world):
    intent = _sign_and_trigger(world)
    assert _run(world, intent).status == "succeeded"
    again = _run(world, intent)  # the workdirs are gone: nothing to adopt
    assert again.status == "failed" and "replayed" in _error(again), _error(again)
    transport = lane.transport_for(SITE, USER, LOGIN)
    proc = transport.call("mkwd", intent.intent_id, "first")
    assert (proc.returncode, lane._refusal_of(proc)[0]) == (4, "replayed")
    assert len(world.cluster.calls("sbatch")) == 2 and len(world.cluster.pods()) == 2


def test_an_expired_intent_is_refused_by_the_gate(world):
    intent = _sign_and_trigger(world)
    late = int(time.time()) - 4000
    expired = _resigned(world, intent, issued=late, expires=late + 3600)
    _refused(world, _run(world, expired), "expired")
    assert world.cluster.calls("sbatch") == []


def test_an_intent_for_another_site_is_refused_by_the_gate(world):
    intent = _sign_and_trigger(world)
    _refused(world, _run(world, _resigned(world, intent, site="other-site")), "wrong_site")
    assert world.cluster.calls("sbatch") == []


def test_a_stage_in_after_submit_is_refused_sealed(world):
    intent = _sign_and_trigger(world)
    record = _run(world, intent, keep_workdir=True)
    assert record.status == "succeeded", _error(record)
    wd = _extra(record, "first")["slurm_job"]["workdir"]
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w", format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo("src/steps/late.py")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    transport = lane.transport_for(SITE, USER, LOGIN)
    proc = transport.call("stage-in", wd, stdin=sink.getvalue())
    assert (proc.returncode, lane._refusal_of(proc)[0]) == (4, "sealed")
    assert len(world.cluster.calls("sbatch")) == 2
