"""DEC-064-WD / ADR-091 D-5: the local podman step lane's workdir root.

``step_executor.workdir_root`` (site.yaml) / ``UMRS_STEP_WORKDIR`` (env) is the
parent of the per-step workdir the podman executor bind-mounts into its
sibling step container. Before slice S-D64-1 (2026-11-24) ``PodmanExecutor``
always called a bare ``tempfile.mkdtemp`` and ignored the knob, so a
containerised UMRS (the stand-alone stack) could not put the workdir on an
identical host/container path. Unset keeps the old behaviour; a configured
root that is not a directory refuses by name.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from umrs.orchestrator.site_config import SiteConfig, StepExecutorConfig
from umrs.pipeline_sdk.step import StepContext, StepResult
from umrs.server.step_executor import (
    PodmanExecutor,
    StepExecutionError,
    build_step_executor,
    step_workdir,
)

SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "umrs"
    / "orchestrator"
    / "site-config.schema.json"
)


def _ctx() -> StepContext:
    return StepContext(
        s3=None,
        input_bucket="b",
        input_keys=[],
        output_bucket="b",
        config={},
        run_id="t",
    )


class _FakePodman:
    """Records the ``-v`` source of each podman call and writes an ok reply."""

    def __init__(self) -> None:
        self.mounts: list[str] = []
        self.calls = 0
        self._real_run = subprocess.run

    def __call__(self, cmd, **kwargs):
        if os.path.basename(str(cmd[0])) != "podman":
            return self._real_run(cmd, **kwargs)
        self.calls += 1
        source = cmd[cmd.index("-v") + 1].split(":")[0]
        self.mounts.append(source)
        with open(os.path.join(source, "output.pkl"), "wb") as fh:
            fh.write(pickle.dumps({"status": "ok", "result": StepResult()}))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")


@pytest.fixture
def fake(monkeypatch) -> _FakePodman:
    podman = _FakePodman()
    monkeypatch.setattr(subprocess, "run", podman)
    return podman


@pytest.fixture
def system_tmp(monkeypatch, tmp_path) -> Path:
    """A scratch system temp dir, so 'unchanged' is observable."""
    sys_tmp = tmp_path / "system-tmp"
    sys_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sys_tmp))
    return sys_tmp


def _run(executor: PodmanExecutor) -> None:
    executor.execute(entrypoint="pkg.mod:Cls", config={}, ctx=_ctx())


def test_env_root_puts_the_workdir_under_it(monkeypatch, tmp_path, fake, system_tmp):
    root = tmp_path / "step-root"
    root.mkdir()
    monkeypatch.setenv("UMRS_STEP_WORKDIR", str(root))
    _run(PodmanExecutor(image="img"))
    assert Path(fake.mounts[0]).parent == root
    assert Path(fake.mounts[0]).name.startswith("umrs-step-")
    assert not Path(fake.mounts[0]).exists()  # removed after the step


def test_unset_keeps_the_system_temp_dir(monkeypatch, fake, system_tmp):
    monkeypatch.delenv("UMRS_STEP_WORKDIR", raising=False)
    _run(PodmanExecutor(image="img"))
    assert Path(fake.mounts[0]).parent == system_tmp


def test_explicit_empty_root_overrides_the_env(monkeypatch, tmp_path, fake, system_tmp):
    monkeypatch.setenv("UMRS_STEP_WORKDIR", str(tmp_path))
    _run(PodmanExecutor(image="img", workdir_root=""))
    assert Path(fake.mounts[0]).parent == system_tmp


def test_missing_root_refuses_by_name_before_podman(monkeypatch, tmp_path, fake):
    monkeypatch.setenv("UMRS_STEP_WORKDIR", str(tmp_path / "absent"))
    with pytest.raises(StepExecutionError) as info:
        _run(PodmanExecutor(image="img"))
    assert info.value.phase == "setup"
    assert "UMRS_STEP_WORKDIR" in str(info.value)
    assert fake.calls == 0


def test_built_podman_executor_reads_the_env(monkeypatch, tmp_path, fake):
    root = tmp_path / "built"
    root.mkdir()
    monkeypatch.setenv("UMRS_STEP_IMAGE", "img")
    monkeypatch.setenv("UMRS_STEP_WORKDIR", str(root))
    executor = build_step_executor("podman")
    assert isinstance(executor, PodmanExecutor)
    _run(executor)
    assert Path(fake.mounts[0]).parent == root


def test_step_workdir_helper_is_shared(tmp_path):
    # The staged podman executor (S-D64-2) creates its workdir the same way.
    made = Path(step_workdir("umrs-staged-", str(tmp_path)))
    assert made.parent == tmp_path and made.is_dir()


def _env(kind: str, root: str) -> dict[str, str]:
    site = SiteConfig(step_executor=StepExecutorConfig(kind=kind, workdir_root=root))
    return site.to_env()


def test_site_knob_exports_the_env_for_the_podman_lane():
    env = _env("podman", "/srv/umrs-steps")
    assert env["UMRS_STEP_WORKDIR"] == "/srv/umrs-steps"


@pytest.mark.parametrize("kind", ["slurm-remote", "slurm", "inproc"])
def test_cluster_workdir_root_stays_out_of_the_local_lane(kind):
    env = _env(kind, "/lustre/umrs_steps")
    assert env["UMRS_SLURM_WORKDIR_ROOT"] == "/lustre/umrs_steps"
    assert "UMRS_STEP_WORKDIR" not in env


def test_no_root_exports_nothing():
    env = _env("podman", "")
    assert "UMRS_STEP_WORKDIR" not in env and "UMRS_SLURM_WORKDIR_ROOT" not in env


def test_schema_accepts_and_documents_the_key():
    schema = json.loads(SCHEMA.read_text())
    site = {"name": "solo", "step_executor": {"kind": "podman", "workdir_root": "/srv/s"}}
    jsonschema.validate(site, schema)
    prop = schema["$defs"]["step_executor"]["properties"]["workdir_root"]
    assert "UMRS_STEP_WORKDIR" in prop["description"]
