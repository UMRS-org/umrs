"""Fake cluster for the staged step lane tests (build spec §11).

The SSH hop becomes a local ``bash -c``: the REAL login-node helper runs
against a tmp workdir root, and fake ``sbatch`` / ``squeue`` / ``sacct`` /
``scancel`` keep their jobs in a JSON state file. The fake ``sbatch`` runs
the real step worker against the staged directories (``in/`` through the
file shim, ``out/``, ``reply/output.json``) exactly as the job's container
would see them, then reports the job through the states a test asks for.
Like Slurm, ``sbatch`` records ``--chdir`` as given and ``getcwd()`` (the
physical path) otherwise; ``squeue`` reports ``--comment`` and ``sacct``
reports none (RCCS stores no job comments: ``AccountingStoreFlags=(null)``).
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor
from umrs.server.step_staging import DEFAULT_CAPS

UMRS_ROOT = Path(__file__).resolve().parents[1]
HELPER = UMRS_ROOT / "deploy" / "_common" / "umrs_step_helper.py"
SRC = UMRS_ROOT / "src"
IMAGE = "localhost/umrs-step:0123456789ab"

FAKE_SLURM = r"""#!__PYTHON__
import fcntl, json, os, re, subprocess, sys, time

STATE = os.environ["FAKE_SLURM_DIR"]
LIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED"}


class Locked:
    def __enter__(self):
        self.fh = open(os.path.join(STATE, "lock"), "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        path = os.path.join(STATE, "jobs.json")
        self.data = json.load(open(path)) if os.path.exists(path) else {"next": 1000, "jobs": {}}
        return self.data

    def __exit__(self, *exc):
        json.dump(self.data, open(os.path.join(STATE, "jobs.json"), "w"))
        self.fh.close()


def behaviour():
    path = os.path.join(STATE, "behaviour.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def log(*args):
    with open(os.path.join(STATE, "calls.log"), "a") as fh:
        fh.write(json.dumps([os.path.basename(sys.argv[0]), *args]) + "\n")


def current(job):
    return job["states"][min(job["i"], len(job["states"]) - 1)]


def run_job(wd):
    err = open(os.path.join(wd, "log", "slurm.err"), "ab")
    if behaviour().get("real_job"):  # the generated script, real podman
        out = open(os.path.join(wd, "log", "slurm.out"), "ab")
        env = dict(os.environ, SLURM_JOB_ID=str(os.getpid()))
        script = os.path.join(wd, "job.sbatch")
        return subprocess.call(["bash", script], cwd=wd, stdout=out, stderr=err, env=env)
    env = dict(os.environ, PYTHONPATH=os.environ["FAKE_SLURM_SRC"])
    return subprocess.call([sys.executable, os.environ["FAKE_SLURM_JOB"], wd], stderr=err, env=env)


def sbatch(args):
    beh = behaviour()
    path = args[-1]
    chdir = [a[len("--chdir="):] for a in args if a.startswith("--chdir=")]
    wd = chdir[0] if chdir else os.getcwd()
    script = open(path).read()
    name = re.search(r"^#SBATCH --job-name=(\S+)", script, re.M).group(1)
    comment = re.search(r"^#SBATCH --comment=(\S+)", script, re.M)
    for rel, text in beh.get("files", {}).items():
        with open(os.path.join(wd, rel), "w") as fh:
            fh.write(text)
    for rel, target in beh.get("symlinks", {}).items():
        os.symlink(target, os.path.join(wd, rel))
    states = list(beh.get("states", ["PENDING", "RUNNING", "COMPLETED"]))
    if beh.get("run", True) and run_job(wd) != 0 and states[-1] == "COMPLETED":
        states[-1] = "FAILED"
    for rel, mode in beh.get("chmod_after", {}).items():  # the step's last act
        os.chmod(os.path.join(wd, rel), int(mode, 8))
    queued = beh.get("sbatch_rc", 0) == 0 or beh.get("queue_anyway", False)
    job_id = None
    if queued:
        with Locked() as data:
            job_id = str(data["next"])
            data["next"] += 1
            data["jobs"][job_id] = {"name": name, "workdir": wd, "states": states, "i": 0,
                                    "cancel_to": beh.get("cancel_to", ["CANCELLED"]),
                                    "comment": comment.group(1) if comment else ""}
    if beh.get("sbatch_rc", 0):
        sys.stderr.write(beh.get("sbatch_err", "sbatch: error: Socket timed out on send/recv"))
        sys.exit(beh["sbatch_rc"])
    print(f"{job_id};fakecluster")


def squeue(args):
    if "--me" in args:
        if behaviour().get("find_rc", 0):
            sys.stderr.write("slurm_load_jobs error: Unable to contact slurm controller")
            sys.exit(1)
        names = [a[7:] for a in args if a.startswith("--name=")]  # none: every job
        with Locked() as data:
            for jid, job in sorted(data["jobs"].items()):
                if (not names or job["name"] == names[0]) and current(job) in LIVE:
                    print(f"{jid}|{current(job)}|{job['name']}|{job['workdir']}|"
                          f"{job.get('comment', '')}")
        return
    jid = args[args.index("-j") + 1]
    with Locked() as data:
        job = data["jobs"].get(jid)
        if job is None or current(job) not in LIVE:
            sys.stderr.write("slurm_load_jobs error: Invalid job id specified\n")
            sys.exit(1)
        state = current(job)
        job["i"] += 1
    if state == "PENDING":
        time.sleep(behaviour().get("pending_sleep", 0))
    print(state)


def sacct(args):
    with Locked() as data:
        if "--job" in args:  # accounting (measure_slurm_accounting)
            if behaviour().get("acct_rc", 0):
                sys.exit(behaviour()["acct_rc"])
            print("00:00:07|")
            return
        if "-j" in args:
            job = data["jobs"].get(args[args.index("-j") + 1])
            if job is not None:
                print(current(job))
            return
        if behaviour().get("find_sacct_rc", 0):
            sys.stderr.write("sacct: error: Problem talking to the database")
            sys.exit(behaviour()["find_sacct_rc"])
        name = next(a for a in args if a.startswith("--name="))[7:]
        for jid, job in sorted(data["jobs"].items()):
            if job["name"] == name:
                print(f"{jid}|{current(job)}|{name}|{job['workdir']}|")


def scancel(args):
    if behaviour().get("scancel_rc", 0):
        sys.stderr.write("scancel: error: Kill job error on job id\n")
        sys.exit(behaviour()["scancel_rc"])
    with Locked() as data:
        job = data["jobs"][args[0]]
        if current(job) in LIVE:
            job["states"], job["i"] = list(job["cancel_to"]), 0


verb = os.path.basename(sys.argv[0])
log(*sys.argv[1:])
{"sbatch": sbatch, "squeue": squeue, "sacct": sacct, "scancel": scancel}[verb](sys.argv[1:])
"""

FAKE_JOB = """import os, sys
wd = sys.argv[1]
# The container's environment: the four -e of the job script and nothing else.
keep = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
os.environ.clear()
os.environ.update(
    keep,
    UMRS_STEP_INPUT=f"{wd}/input.pkl",
    UMRS_STEP_OUTPUT_JSON=f"{wd}/reply/output.json",
    UMRS_STEP_STAGED="1",
    UMRS_STEP_SOURCE_DIR=f"{wd}/src",
)
from umrs.pipeline_sdk import staged_s3
staged_s3.StagedS3Client.__init__.__defaults__ = (f"{wd}/in", f"{wd}/out", None)
from umrs.server import step_worker
sys.exit(step_worker.main())
"""


class FakeS3:
    """The server's object store: get/head/put/list/upload_fileobj."""

    def __init__(self, objects: dict[tuple[str, str], bytes] | None = None) -> None:
        self.objects = dict(objects or {})

    def head_object(self, *, Bucket, Key, **_):
        body = self.objects[(Bucket, Key)]
        return {"ContentLength": len(body), "ETag": '"e"', "ContentType": "text/plain"}

    def get_object(self, *, Bucket, Key, **_):
        body = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def put_object(self, *, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.read()

    def upload_fileobj(self, fileobj, bucket, key, **_):
        self.objects[(bucket, key)] = fileobj.read()

    def list_objects_v2(self, *, Bucket, Prefix="", **_):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        return {"Contents": [{"Key": k, "Size": len(self.objects[(Bucket, k)])} for k in keys]}


@dataclass
class Cluster:
    tmp: Path
    root: Path
    state: Path
    commands: list[str] = field(default_factory=list)

    def behave(self, **behaviour: Any) -> None:
        (self.state / "behaviour.json").write_text(json.dumps(behaviour))

    def jobs(self) -> dict[str, dict[str, Any]]:
        path = self.state / "jobs.json"
        return json.loads(path.read_text())["jobs"] if path.exists() else {}

    def add_job(self, job_id: str, name: str, workdir: str, states: list[str], **extra) -> None:
        path = self.state / "jobs.json"
        data = json.loads(path.read_text()) if path.exists() else {"next": 1000, "jobs": {}}
        data["jobs"][job_id] = {
            "name": name, "workdir": workdir, "states": states, "i": 0,
            "cancel_to": extra.get("cancel_to", ["CANCELLED"]),
            "comment": extra.get("comment", ""),
        }  # fmt: skip
        path.write_text(json.dumps(data))

    def calls(self, verb: str | None = None) -> list[list[str]]:
        path = self.state / "calls.log"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        return [r for r in rows if verb is None or r[0] == verb]

    def helper_verbs(self) -> list[str]:
        verbs = []
        for command in self.commands:
            parts = command.split()
            if len(parts) > 3 and parts[2] == str(HELPER):
                verbs.append(parts[3])
        return verbs

    def workdirs(self) -> list[Path]:
        return sorted(self.root.iterdir()) if self.root.exists() else []

    def executor(self, **overrides: Any) -> RemoteSlurmStepExecutor:
        values: dict[str, Any] = {
            "ssh_config": SSHControlConfig(ssh_target="login"),
            "workdir_root": str(self.root),
            "helper_path": str(HELPER),
            "helper_python": sys.executable,
            "venue_script": "/cluster/umrs/src/umrs/mini-rivault-llm/venue.sh",
            "image_archive_dir": "/cluster/images",
            "image": IMAGE,
            "enabled": True,
            "poll_interval": 0,
            "partition": "ng-dgx-m0",
            "account": "cloud-users",
            "memory": "",
            "gpus": 0,
            "time_limit": "01:00:00",
            "queue_timeout": "06:00:00",
            "settle_seconds": 5.0,
            "unreachable_budget": 2.0,
            "cancel_patience": 2.0,
            "caps": dataclasses.replace(DEFAULT_CAPS, min_free_bytes=1),
        }
        values.update(overrides)
        ex = RemoteSlurmStepExecutor(**values)
        commands = self.commands

        def local_argv(command: str) -> list[str]:
            commands.append(command)
            return ["bash", "-c", command]

        ex._remote_argv = local_argv  # type: ignore[method-assign]
        return ex


def make_cluster(tmp_path: Path, monkeypatch, *, symlinked_root: bool = False) -> Cluster:
    """A fake cluster; *symlinked_root* puts the workdir root behind a
    symlinked parent, as on RCCS (``/lvs0/rccs-dmp`` -> ``/lvs0/dne1/rccs-dmp``)."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    state = tmp_path / "slurm"
    state.mkdir()
    script = FAKE_SLURM.replace("__PYTHON__", sys.executable)
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    job = tmp_path / "fake_job.py"
    job.write_text(FAKE_JOB)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(state))
    monkeypatch.setenv("FAKE_SLURM_JOB", str(job))
    monkeypatch.setenv("FAKE_SLURM_SRC", str(SRC))
    if symlinked_root:
        (tmp_path / "dne1" / "lustre").mkdir(parents=True)
        (tmp_path / "lustre").symlink_to(tmp_path / "dne1" / "lustre")
    else:
        (tmp_path / "lustre").mkdir()
    return Cluster(tmp=tmp_path, root=tmp_path / "lustre" / "umrs_steps", state=state)


def write_step_tree(root: Path) -> Path:
    """An uploaded workflow tree with the steps the tests run."""
    pkg = root / "steps"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "lane.py").write_text(
        "import os\n"
        "from umrs.pipeline_sdk import Step, StepResult\n\n"
        "class Upper(Step):\n"
        "    name = 'upper'\n"
        "    def setup(self, config): self.config = config\n"
        "    def teardown(self): pass\n"
        "    def run(self, ctx):\n"
        "        data = b''\n"
        "        if self.config.get('input_key'):\n"
        "            data = ctx.download_bytes(self.config['input_key'])\n"
        "        for key in ctx.input_keys:\n"
        "            data += ctx.download_parent_bytes(key)\n"
        "        out = self.config.get('out', 'result/out.txt')\n"
        "        ctx.upload_bytes(out, data.upper() + b'!')\n"
        "        extra = {'config_keys': sorted(self.config),\n"
        "                 'slurm': {'gpu_seconds': 1e9}}\n"
        "        return StepResult(output_keys=[out], extra=extra)\n\n"
        "class Boom(Step):\n"
        "    name = 'boom'\n"
        "    def setup(self, config): pass\n"
        "    def teardown(self): pass\n"
        "    def run(self, ctx): raise RuntimeError('step exploded on purpose')\n\n"
        "class Sneaky(Step):\n"
        "    name = 'sneaky'\n"
        "    def setup(self, config): pass\n"
        "    def teardown(self): pass\n"
        "    def run(self, ctx):\n"
        "        target = os.path.join('/work/out' if os.path.isdir('/work/out') else\n"
        "                              os.environ['UMRS_STEP_SOURCE_DIR'] + '/../out',\n"
        "                              ctx.output_bucket)\n"
        "        os.makedirs(target, exist_ok=True)\n"
        "        os.symlink(os.path.expanduser('~/.ssh/id_rsa'), os.path.join(target, 'k'))\n"
        "        return StepResult(output_keys=['k'])\n\n"
        "class Stray(Step):\n"
        "    name = 'stray'\n"
        "    def setup(self, config): pass\n"
        "    def teardown(self): pass\n"
        "    def run(self, ctx):\n"
        "        out = '/work/out' if os.path.isdir('/work/out') else (\n"
        "            os.environ['UMRS_STEP_SOURCE_DIR'] + '/../out')\n"
        "        with open(os.path.join(out, 'arr.npy'), 'wb') as fh:\n"
        "            fh.write(b'\\0' * (256 * 1024))  # past the pipe buffer\n"
        "        return StepResult(output_keys=[])\n",
        encoding="utf-8",
    )
    return root
