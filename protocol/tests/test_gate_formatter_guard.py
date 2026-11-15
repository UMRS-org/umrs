"""The formatter never touches the gate, even when the gate is named (U2.0 spec §3.1).

``hpc_gate.py`` bytes are a release (``test_gate_release.py``). The ruff
config excludes it from formatting, but ruff only honours an exclude for a
path passed on the command line when ``force-exclude`` is on, and the lint
gates (``scripts/ci-lint-local.sh``, ci.yaml, pre-commit) pass explicit file
lists. Without it, a plain ``ruff format <gate path>`` after a ruff upgrade
would rewrite the gate.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE_REL = "src/umrs_protocol/hpc_gate.py"


def _ruff_config() -> dict:
    tomllib = pytest.importorskip("tomllib")  # stdlib from Python 3.11
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["ruff"]


def test_format_excludes_the_gate() -> None:
    assert GATE_REL in _ruff_config()["format"]["exclude"]


def test_excludes_hold_for_explicit_paths() -> None:
    assert _ruff_config().get("force-exclude") is True


def test_ruff_format_skips_the_named_gate(tmp_path: Path) -> None:
    """Behavioural check: a badly formatted file at the gate path is not reported."""
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff not installed")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    gate = tmp_path / GATE_REL
    gate.parent.mkdir(parents=True)
    gate.write_text("x  =  ( 1 )\n", encoding="utf-8")
    result = subprocess.run(
        [ruff, "format", "--check", GATE_REL],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
