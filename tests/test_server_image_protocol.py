"""The server image installs umrs-protocol locally and ships one gate (U2.0 U-A1).

ADR-117 D-2: umrs depends on ``umrs-protocol`` (this repo's ``protocol/``),
and ``r4_contracts.<m>`` for the moved modules is a ``sys.modules`` alias of
``umrs_protocol.<m>``. Neither package is published (PyPI holds only 0.0.0
name reservations), so Dockerfile.server installs both from
the build context, then ``pip check`` proves the local edges resolve and the
smoke test proves ``r4_contracts.hpc_gate is umrs_protocol.hpc_gate`` and that
``gate_sha256()`` hashes the file the image runs. On the U-A0 Dockerfile and
smoke script every test here fails.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

UMRS_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = UMRS_ROOT / "docker" / "Dockerfile.server"
SMOKE = UMRS_ROOT / "docker" / "server-image" / "smoke.py"
GATE = UMRS_ROOT / "protocol" / "src" / "umrs_protocol" / "hpc_gate.py"
GATE_PINS = UMRS_ROOT / "protocol" / "tests" / "golden" / "gate.json"
CHECK = "import runpy, sys; print(runpy.run_path(sys.argv[1])['check_protocol_alias']())"


def _runtime_stage() -> list[str]:
    """Dockerfile.server's runtime stage (after the last FROM), continuations joined."""
    text = re.sub(r"\\\n", " ", DOCKERFILE.read_text(encoding="utf-8"))
    lines = text.splitlines()
    last_from = max(i for i, line in enumerate(lines) if line.startswith("FROM "))
    return [line for line in lines[last_from:] if line and not line.startswith("#")]


def _install_run() -> str:
    runs = [line for line in _runtime_stage() if "-e /app/umrs-src" in line]
    assert len(runs) == 1, runs
    return runs[0]


def test_runtime_stage_copies_protocol_from_the_umrs_checkout() -> None:
    assert "COPY ./umrs/protocol /app/umrs-protocol" in _runtime_stage()


def test_protocol_installs_first_without_resolving() -> None:
    run = _install_run()
    names = ("umrs-protocol", "r4-contracts", "umrs-src")
    order = [run.index(f"--no-deps -e /app/{name}") for name in names]
    assert order == sorted(order)


def test_runtime_stage_runs_pip_check_after_the_local_installs() -> None:
    run = _install_run()
    assert run.index("pip check") > run.index("-e /app/umrs-src")


def test_runtime_stage_asserts_one_gate_module() -> None:
    run = _install_run()
    assert "assert g is p" in run
    assert "r4_contracts.hpc_gate as g" in run and "umrs_protocol.hpc_gate as p" in run


def test_smoke_main_runs_the_alias_check() -> None:
    main = SMOKE.read_text(encoding="utf-8").split("def main()", 1)[1]
    assert "check_protocol_alias()" in main


def _run_check(pythonpath: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([*pythonpath, env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-c", CHECK, str(SMOKE)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_smoke_alias_check_passes_and_reports_the_pinned_sha() -> None:
    pytest.importorskip("r4_contracts", reason="the image installs r4-contracts (core-only run)")
    done = _run_check([str(UMRS_ROOT / "protocol" / "src")])
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == json.loads(GATE_PINS.read_text())["sha256"]


def test_smoke_alias_check_fails_on_a_second_gate_copy(tmp_path: Path) -> None:
    """The pre-U2.0b layout: r4_contracts carries its own full gate file."""
    package = tmp_path / "r4_contracts"
    package.mkdir()
    (package / "__init__.py").write_text("")
    shutil.copyfile(GATE, package / "hpc_gate.py")
    done = _run_check([str(tmp_path), str(UMRS_ROOT / "protocol" / "src")])
    assert done.returncode != 0
    assert "not umrs_protocol's" in done.stderr
