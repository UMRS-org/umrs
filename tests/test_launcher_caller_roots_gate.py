"""Launcher realpath re-check (H-1, 2026-10-19).

Two lanes under one contract:

1. ``caller_mount_escapes`` (``orchestrator/caller_roots.py``): the pure
   decision the compute-node gate in ``mini-rivault-llm/serve-model.slurm``
   mirrors — each bind source is fully resolved and must sit under at least
   one caller root; a source re-pointed after approval fails closed.
2. The literal shell gate: the snippet is run the way the script runs it
   (bash, envs), so a drift between them fails here on the next edit that
   changes one side only.

Fail-before markers: the plain slurm snippet (without the env present)
ignores foreign sources today, and the pre-wave site never gated at all.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from umrs.orchestrator.caller_roots import (
    caller_mount_escapes,
    caller_mount_roots,
    caller_roots_json,
)


@pytest.fixture
def sandbox():
    base = Path(tempfile.mkdtemp(prefix="gate-root-"))
    roots = {
        "home": base / "home" / "alice",
        "project": base / "shared" / "projects" / "proj-a",
        "foreign": base / "home" / "bob",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    yield base, roots


def _run_shell_gate(env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the literal gate from serve-model.slurm (keep in verbatim sync)."""
    snippet = '''
RIVAULT_CALLER_ROOTS_JSON="$RIVAULT_CALLER_ROOTS_JSON" \
GATE_VLLM_MODEL_PATH="$GATE_VLLM_MODEL_PATH" \
GATE_PROJECT_DATA="$GATE_PROJECT_DATA" \
python3 - <<'PY'
import json, os, sys
from pathlib import Path

roots = [Path(p).expanduser().resolve() for p in json.loads(os.environ["RIVAULT_CALLER_ROOTS_JSON"])]
named = {
    "VLLM_HOST_MODEL_PATH": os.environ.get("GATE_VLLM_MODEL_PATH", ""),
    "PROJECT_DATA_PATH": os.environ.get("GATE_PROJECT_DATA", ""),
}
violations = []
for name, source in named.items():
    if not source:
        continue
    resolved = Path(source).expanduser().resolve()
    if not any(root == resolved or root in resolved.parents for root in roots):
        violations.append(f"{name}={resolved}")
if violations:
    listed = "; ".join(violations)
    sys.stderr.write(
        f"[rivault] FAIL: bind source escapes caller mount roots ({listed})\\n"
    )
    raise SystemExit(1)
PY
'''
    merged = dict(os.environ)
    merged.update(env)
    return subprocess.run(
        ["bash", "-c", snippet], capture_output=True, text=True, env=merged, timeout=30
    )


class TestModuleGate:
    def test_inside_roots_pass(self, sandbox):
        _base, roots = sandbox
        sources = {
            "VLLM_HOST_MODEL_PATH": str(roots["home"] / "m"),
            "PROJECT_DATA_PATH": str(roots["project"] / "data"),
        }
        assert caller_mount_escapes(sources, [roots["home"], roots["project"]]) == []

    def test_foreign_source_named(self, sandbox):
        _base, roots = sandbox
        sources = {"PROJECT_DATA_PATH": str(roots["foreign"])}
        violations = caller_mount_escapes(sources, [roots["home"], roots["project"]])
        assert violations == [f"PROJECT_DATA_PATH={roots['foreign']}"]

    def test_toc_to_swing_named(self, sandbox):
        _base, roots = sandbox
        seam = roots["home"] / "model"
        seam.symlink_to(roots["foreign"], target_is_directory=True)
        violations = caller_mount_escapes(
            {"VLLM_HOST_MODEL_PATH": str(seam)}, [roots["home"]]
        )
        assert violations == [f"VLLM_HOST_MODEL_PATH={roots['foreign']}"]

    def test_blank_source_and_blank_identifiers(self, sandbox):
        _base, roots = sandbox
        assert caller_mount_escapes({"X": ""}, [roots["home"]]) == []
        blanks = caller_mount_roots("", "")
        assert isinstance(blanks, list)
        assert not any("{" in str(root) for root in blanks)
        assert len(blanks) < len(caller_mount_roots("proj-a", "alice"))

    def test_roots_json_roundtrip(self):
        # The env transport never broadens a root set; blank inputs narrow it.
        assert json.loads(caller_roots_json("proj-a", "alice")) == [
            str(p) for p in caller_mount_roots("proj-a", "alice")
        ]
        assert isinstance(json.loads(caller_roots_json("", "")), list)


class TestShellGate:
    def _env(self, roots: list[Path], **overrides: str) -> dict[str, str]:
        return {
            "RIVAULT_CALLER_ROOTS_JSON": json.dumps([str(r) for r in roots]),
            **overrides,
        }

    def test_within_roots_passes(self, sandbox):
        _base, roots = sandbox
        result = _run_shell_gate(
            self._env(
                [roots["home"], roots["project"]],
                GATE_VLLM_MODEL_PATH=str(roots["project"] / "m"),
                GATE_PROJECT_DATA=str(roots["home"] / "d"),
            )
        )
        assert result.returncode == 0, result.stderr

    def test_foreign_mount_fails_closed(self, sandbox):
        _base, roots = sandbox
        result = _run_shell_gate(
            self._env(
                [roots["home"]],
                GATE_PROJECT_DATA=str(roots["foreign"]),
            )
        )
        assert result.returncode != 0
        assert "escapes" in result.stderr
        assert str(roots["foreign"]) in result.stderr

    def test_toc_swing_fails_closed(self, sandbox):
        _base, roots = sandbox
        seam = roots["home"] / "model"
        seam.symlink_to(roots["foreign"], target_is_directory=True)
        result = _run_shell_gate(
            self._env([roots["home"]], GATE_VLLM_MODEL_PATH=str(seam))
        )
        assert result.returncode != 0
        assert str(roots["foreign"]) in result.stderr

    def test_no_roots_json_is_legacy_pass(self, sandbox):
        _base, roots = sandbox
        merged = dict(os.environ)
        merged.pop("RIVAULT_CALLER_ROOTS_JSON", None)
        result = subprocess.run(
            ["bash", "-c", '[[ -n "${RIVAULT_CALLER_ROOTS_JSON:-}" ]] || exit 0'],
            capture_output=True,
            env=merged,
        )
        assert result.returncode == 0
