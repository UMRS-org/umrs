"""gb10_reclaim.py — the GB10 pre-start CUDA reclaim (rccs job 458139, 2026-11-05).

Run as the wrapper runs it (a script inside the vLLM container), against a fake
``torch`` that models unified memory: allocations succeed up to a limit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/umrs/mini-rivault-llm/gb10_reclaim.py"

FAKE_TORCH = """
import os

G = 1 << 30
_limit = int(os.environ.get("FAKE_LIMIT_GIB", "1000")) * G
_used = [0]
uint8 = "uint8"


class _Tensor:
    def __init__(self, n):
        self.n = n

    def __del__(self):
        _used[0] -= self.n


def empty(n, dtype=None, device=None):
    if _used[0] + n > _limit:
        raise RuntimeError("CUDA out of memory")
    _used[0] += n
    return _Tensor(n)


class cuda:
    @staticmethod
    def mem_get_info():
        if os.environ.get("FAKE_NO_GPU"):
            raise RuntimeError("no CUDA device")
        return (59 * G if not os.environ.get("FAKE_AFTER") else 102 * G, 121 * G)

    @staticmethod
    def empty_cache():
        os.environ["FAKE_AFTER"] = "1"
"""


def _run(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin", **env},
        check=False,
    )


@pytest.mark.parametrize(("limit", "held"), [("1000", 100), ("40", 40)])
def test_reclaim_allocates_up_to_the_target_or_the_first_failure(tmp_path, limit, held):
    proc = _run(tmp_path, RIVAULT_MN_PRESTART_RECLAIM_GIB="100", FAKE_LIMIT_GIB=limit)
    assert proc.returncode == 0
    assert f"(held {held} GiB)" in proc.stdout
    assert "cuda free 59.0 GiB -> 102.0 GiB" in proc.stdout


@pytest.mark.parametrize("gib", ["", "0", "lots"])
def test_reclaim_is_a_no_op_without_a_target(tmp_path, gib):
    proc = _run(tmp_path, RIVAULT_MN_PRESTART_RECLAIM_GIB=gib)
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")


def test_reclaim_never_blocks_the_launch(tmp_path):
    proc = _run(tmp_path, RIVAULT_MN_PRESTART_RECLAIM_GIB="100", FAKE_NO_GPU="1")
    assert proc.returncode == 0
    assert "[gb10-reclaim] skipped: RuntimeError: no CUDA device" in proc.stderr
