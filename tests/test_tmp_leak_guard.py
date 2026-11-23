"""Guard (2026-11-23): a test run leaves nothing behind in the temp dir.

/tmp (tmpfs, ~1M inodes) ran out of inodes; this suite had left ~25,000
dirs there (venue-route-*, plan-venue-*, slurm-step-*, ...): tests called
``tempfile.mkdtemp`` without cleanup, and the step executors keep their
``slurm-step-*`` workdir on ``keep_workdir=True``. The conftest now gives
every test a scratch tempdir it removes afterwards.

The child run uses this suite's conftest (as a plugin) under its own TMPDIR
and does what the leaking tests did; afterwards that TMPDIR must be empty.
"""

import os
import subprocess
import sys
from pathlib import Path

import umrs.server.slurm_step_executor as _step

TESTS_DIR = Path(__file__).resolve().parent
# From a real module file: under the r4 parent repo ``umrs`` can resolve
# as a namespace package (the submodule dir on sys.path), whose __file__
# is None (r4 CI, 2026-11-24). src/umrs/server/x.py -> src.
SRC_DIR = Path(_step.__file__).resolve().parents[2]

CHILD = '''
import tempfile

from umrs.server.slurm_step_executor import SlurmStepExecutor


def test_leaky_patterns():
    tempfile.mkdtemp(prefix="venue-route-")
    SlurmStepExecutor(image="img", keep_workdir=True)._new_workdir()
'''


def _child_env(probe: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("UMRS_KB_EXPORT_DIR", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    paths = [str(TESTS_DIR), str(SRC_DIR), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    env["TMPDIR"] = str(probe)
    return env


def test_a_run_leaves_its_temp_dir_empty(tmp_path):
    probe = tmp_path / "tmp"
    probe.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    (work / "test_child_leak.py").write_text(CHILD)
    cmd = [
        sys.executable, "-m", "pytest", "-q", "-p", "conftest", "-p", "no:cacheprovider",
        "-c", os.devnull, "--rootdir", str(work), "--basetemp", str(tmp_path / "bt"),
        str(work / "test_child_leak.py"),
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=work, env=_child_env(probe), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert sorted(os.listdir(probe)) == []
