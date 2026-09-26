"""Server policy for uploaded step code (ADR-074; build spec 2026-09-26 §1-§2).

- ``step_executor`` in site.yaml carries the lane knobs into the env;
- the HPC lane counts as available only with SSH submit, a workdir root, the
  site's ``remote_enabled``, the login-node helper and a platform image;
- without it, ``uploaded_code`` refuses by name or (explicitly) runs the
  pre-ADR-074 lanes with a warning on every run;
- an uploaded step's config never carries the routed key, headers or routing
  record; ``routing_hints.image`` is the platform image or allow-listed;
- no ``AWS_*`` credential value reaches a step container or job script.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import jsonschema
import pytest
import yaml
from umrs.orchestrator.site_config import (
    StepExecutorConfig,
    load_site_config,
    reset_site_config,
)
from umrs.pipeline_sdk.step import StepResult
from umrs.server import step_trust
from umrs.server import workflow_runner as wr
from umrs.server.step_executor import PodmanExecutor, WorkerPoolExecutor
from umrs.server.step_router import RoutingDecision
from umrs.server.step_trust import (
    LEGACY_IN_PROCESS,
    hpc_lane_available,
    step_image_refusal,
    uploaded_code_lane,
    uploaded_code_policy,
)
from umrs.server.workflow_execution_registry import ExecutionRecord

UMRS_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = UMRS_ROOT / "src" / "umrs" / "orchestrator" / "site-config.schema.json"

_LANE_ENV = {
    "UMRS_SLURM_STEP_SUBMIT": "remote",
    "UMRS_SLURM_WORKDIR_ROOT": "/lvs0/x/umrs_steps",
    "UMRS_STEP_REMOTE_ENABLED": "1",
    "UMRS_STEP_HELPER": "/lvs0/x/umrs/deploy/_common/umrs_step_helper.py",
    "UMRS_STEP_IMAGE": "localhost/umrs-step:0123456789ab",
}
_POLICY_ENV = (
    *_LANE_ENV,
    "UMRS_STEP_UPLOADED_CODE",
    "UMRS_STEP_ALLOWED_IMAGES",
    "UMRS_STEP_EXECUTOR",
)


@pytest.fixture
def no_site(monkeypatch):
    """No site.yaml injection and no lane variable (conftest restores env)."""
    monkeypatch.setattr(step_trust, "_load_site_env", lambda: None)
    for key in _POLICY_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# ── §1 site config ─────────────────────────────────────────────────────────


def test_step_executor_defaults_keep_the_lane_closed():
    cfg = StepExecutorConfig()
    assert cfg.remote_enabled is False
    assert cfg.uploaded_code == "refuse"
    assert cfg.queue_timeout == "06:00:00"
    assert cfg.allowed_images == [] and cfg.helper_path == "" and cfg.venue_script == ""
    assert cfg.gpu_form == "cdi"


def _site_yaml(tmp_path: Path, step_executor: dict) -> Path:
    path = tmp_path / "site.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "lane-test",
                "scheduler": {"kind": "slurm-remote"},
                "step_executor": step_executor,
            }
        )
    )
    return path


_FULL_LANE = {
    "kind": "inproc",
    "workdir_root": "/lvs0/x/umrs_steps",
    "image": "localhost/umrs-step:0123456789ab",
    "remote_enabled": True,
    "uploaded_code": "legacy_in_process",
    "image_archive_dir": "/lvs0/x/umrs_models/images",
    "allowed_images": ["localhost/umrs-step-gpu:0123456789ab", "registry.example/a:1"],
    "helper_path": "/lvs0/x/umrs/deploy/_common/umrs_step_helper.py",
    "venue_script": "/lvs0/x/umrs/src/umrs/mini-rivault-llm/venue.sh",
    "queue_timeout": "02:00:00",
    "gpu_form": "rocm",
}


def test_site_yaml_exports_every_lane_knob(tmp_path):
    env = load_site_config(str(_site_yaml(tmp_path, _FULL_LANE))).to_env()
    assert env["UMRS_STEP_REMOTE_ENABLED"] == "1"
    assert env["UMRS_STEP_UPLOADED_CODE"] == "legacy_in_process"
    assert env["UMRS_STEP_IMAGE_ARCHIVE_DIR"] == "/lvs0/x/umrs_models/images"
    assert env["UMRS_STEP_ALLOWED_IMAGES"] == (
        "localhost/umrs-step-gpu:0123456789ab,registry.example/a:1"
    )
    assert env["UMRS_STEP_HELPER"] == _FULL_LANE["helper_path"]
    assert env["UMRS_STEP_VENUE_SCRIPT"] == _FULL_LANE["venue_script"]
    assert env["UMRS_SLURM_QUEUE_TIMEOUT"] == "02:00:00"
    assert env["UMRS_SLURM_STEP_SUBMIT"] == "remote"


def test_site_yaml_exports_the_gpu_form_cdi_by_default(tmp_path):
    env = load_site_config(str(_site_yaml(tmp_path, _FULL_LANE))).to_env()
    assert env["UMRS_STEP_GPU_FORM"] == "rocm"
    default = load_site_config(str(_site_yaml(tmp_path, {}))).to_env()
    assert default["UMRS_STEP_GPU_FORM"] == "cdi"


def test_default_site_exports_refuse_and_no_lane(tmp_path):
    env = load_site_config(str(_site_yaml(tmp_path, {}))).to_env()
    assert "UMRS_STEP_REMOTE_ENABLED" not in env
    assert env["UMRS_STEP_UPLOADED_CODE"] == "refuse"
    assert env["UMRS_SLURM_QUEUE_TIMEOUT"] == "06:00:00"
    for key in ("UMRS_STEP_HELPER", "UMRS_STEP_VENUE_SCRIPT", "UMRS_STEP_ALLOWED_IMAGES"):
        assert key not in env


def test_schema_accepts_the_lane_block_and_refuses_an_unknown_switch():
    schema = json.loads(SCHEMA.read_text())
    doc = {"name": "s", "step_executor": dict(_FULL_LANE)}
    jsonschema.validate(doc, schema)
    doc["step_executor"]["uploaded_code"] = "allow"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)
    doc["step_executor"].update(uploaded_code="refuse", gpu_form="opencl")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


def test_the_gate_reads_site_yaml_when_nothing_was_injected(tmp_path, monkeypatch):
    for key in _POLICY_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("UMRS_SITE_CONFIG", str(_site_yaml(tmp_path, _FULL_LANE)))
    reset_site_config()
    try:
        assert hpc_lane_available() is True
    finally:
        reset_site_config()


# ── §1 the HPC-lane gate ───────────────────────────────────────────────────


def test_lane_available_only_with_every_piece(no_site):
    for key, value in _LANE_ENV.items():
        no_site.setenv(key, value)
    assert hpc_lane_available() is True


@pytest.mark.parametrize("missing", sorted(_LANE_ENV))
def test_lane_closed_when_one_piece_is_missing(no_site, missing):
    for key, value in _LANE_ENV.items():
        if key != missing:
            no_site.setenv(key, value)
    assert hpc_lane_available() is False


def test_remote_enabled_must_be_true(no_site):
    for key, value in _LANE_ENV.items():
        no_site.setenv(key, value)
    no_site.setenv("UMRS_STEP_REMOTE_ENABLED", "0")
    assert hpc_lane_available() is False


def test_a_local_sbatch_is_not_the_hpc_lane(no_site, tmp_path):
    """The unhardened local Slurm lane never counts for uploaded code."""
    sbatch = tmp_path / "sbatch"
    sbatch.write_text("#!/bin/sh\n")
    sbatch.chmod(0o755)
    no_site.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    no_site.setenv("UMRS_SLURM_WORKDIR_ROOT", "/shared/steps")
    assert hpc_lane_available() is False


# ── §2 the uploaded_code transition switch ─────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "refuse"), ("refuse", "refuse"), ("legacy_in_process", LEGACY_IN_PROCESS)],
)
def test_uploaded_code_policy(no_site, raw, expected):
    if raw is not None:
        no_site.setenv("UMRS_STEP_UPLOADED_CODE", raw)
    assert uploaded_code_policy() == expected


def test_an_unknown_switch_refuses(no_site, caplog):
    no_site.setenv("UMRS_STEP_UPLOADED_CODE", "allow")
    with caplog.at_level(logging.WARNING, logger="umrs.server.step_trust"):
        assert uploaded_code_policy() == "refuse"
    assert "UMRS_STEP_UPLOADED_CODE" in caplog.text


def test_a_requested_hpc_lane_does_not_stand_while_the_lane_is_closed(no_site):
    no_site.setattr(step_trust, "hpc_lane_available", lambda: False)
    assert uploaded_code_lane("slurm") is None
    assert uploaded_code_lane("slurm-remote") is None


def test_legacy_switch_applies_only_without_the_lane(no_site):
    no_site.setenv("UMRS_STEP_UPLOADED_CODE", "legacy_in_process")
    no_site.setattr(step_trust, "hpc_lane_available", lambda: False)
    assert uploaded_code_lane("inproc") == LEGACY_IN_PROCESS
    no_site.setattr(step_trust, "hpc_lane_available", lambda: True)
    assert uploaded_code_lane("inproc") == "slurm"  # ignored when the lane is live


# ── §2 the runner ──────────────────────────────────────────────────────────


def _upload(root: Path, marker: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "lanemod.py").write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            from umrs.pipeline_sdk import Step, StepResult

            class UserStep(Step):
                name = "user"
                def setup(self, config): pass
                def teardown(self): pass
                def run(self, ctx):
                    Path({str(marker)!r}).write_text("ran here")
                    return StepResult(output_keys=["ok"])
            """
        ),
        encoding="utf-8",
    )
    return str(root)


@pytest.fixture
def clean_modules(monkeypatch):
    monkeypatch.setattr(wr, "_KNOWN_SOURCE_ROOTS", [])
    saved_path = list(sys.path)
    yield
    sys.path[:] = saved_path
    sys.modules.pop("lanemod", None)


def _wf(mode: str = "in_process", **step) -> dict:
    spec = {"name": "s1", "entrypoint": "lanemod:UserStep", "config": {}, **step}
    return {"execution_mode": mode, "steps": [spec]}


def test_legacy_switch_runs_uploaded_code_in_process_with_a_warning(
    tmp_path, no_site, clean_modules, caplog
):
    no_site.setenv("UMRS_STEP_UPLOADED_CODE", "legacy_in_process")
    no_site.setattr(step_trust, "hpc_lane_available", lambda: False)
    marker = tmp_path / "marker"
    record = ExecutionRecord(execution_id="e-legacy", workflow_name="w", user_id="u")
    with caplog.at_level(logging.WARNING, logger="umrs.server.workflow_runner"):
        wr.run_workflow_record(
            record, source_path=_upload(tmp_path / "wf", marker), workflow_json=_wf()
        )
    assert record.status == "succeeded", record.error
    assert marker.exists()  # the pre-ADR-074 in-process lane, by explicit choice
    assert "legacy_in_process" in caplog.text


def test_refuse_switch_refuses_a_manifest_slurm_lane_by_name(tmp_path, no_site, clean_modules):
    no_site.setattr(step_trust, "hpc_lane_available", lambda: False)
    built: list = []
    no_site.setattr(wr, "build_step_executor", lambda **kw: built.append(kw))
    marker = tmp_path / "marker"
    record = ExecutionRecord(execution_id="e-refuse", workflow_name="w", user_id="u")
    wr.run_workflow_record(
        record, source_path=_upload(tmp_path / "wf", marker), workflow_json=_wf("slurm")
    )
    assert record.status == "failed"
    assert "connected HPC" in (record.error or "")
    assert built == [] and not marker.exists()


class _PlatformRouter:
    def resolve(self, _step_spec, **_kw):  # noqa: ANN001, ANN003
        return RoutingDecision(
            api_base="http://llm.internal/v1",
            api_key="sk-platform",
            model="m",
            headers={"Authorization": "Bearer sk-platform"},
            resolved_via="registry",
        )


class _CapturingHpc:
    name = "slurm"

    def __init__(self) -> None:
        self.configs: list[dict] = []
        self.metadata: list[dict] = []

    def execute(self, *, entrypoint, config, ctx):  # noqa: ANN001, ARG002
        self.configs.append(dict(config))
        self.metadata.append(dict(ctx.metadata))
        return StepResult(output_keys=["ok"])


def _run_on_hpc(tmp_path, monkeypatch, workflow_json, executor=None):
    monkeypatch.setattr(step_trust, "hpc_lane_available", lambda: True)
    hpc = executor or _CapturingHpc()
    monkeypatch.setattr(wr, "build_step_executor", lambda **_kw: hpc)
    source = _upload(tmp_path / "wf", tmp_path / "marker")
    record = ExecutionRecord(execution_id="e-hpc", workflow_name="w", user_id="u")
    wr.run_workflow_record(
        record, source_path=source, workflow_json=workflow_json, router=_PlatformRouter()
    )
    return record, hpc


def test_uploaded_step_config_carries_no_routed_credential(tmp_path, no_site, clean_modules):
    workflow = _wf(config={"api_key": "author-own", "headers": {"X-A": "1"}})
    record, hpc = _run_on_hpc(tmp_path, no_site, workflow)
    assert record.status == "succeeded", record.error
    config = hpc.configs[0]
    assert "api_key" not in config and "headers" not in config
    assert "_routing" not in config
    assert config["api_base"] == "http://llm.internal/v1"  # the route, not the key
    assert "sk-platform" not in json.dumps(config, default=str)


def test_a_platform_step_keeps_its_routed_credential(tmp_path, no_site):
    """Unchanged for installed templates: only uploaded code loses the key."""
    hpc = _CapturingHpc()
    record = ExecutionRecord(execution_id="e-plat", workflow_name="w", user_id="u")
    wr.run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "steps": [
                {
                    "name": "llm",
                    "entrypoint": "umrs.pipeline_sdk.templates.llm_step:LLMStep",
                    "config": {},
                }
            ]
        },
        router=_PlatformRouter(),
        executor=hpc,
    )
    assert record.status == "succeeded", record.error
    assert hpc.configs[0]["api_key"] == "sk-platform"


class _CapturingInproc(_CapturingHpc):
    name = "inproc"

    def __init__(self) -> None:
        super().__init__()
        self.entrypoints: list[str] = []
        self.shut = False

    def execute(self, *, entrypoint, config, ctx):  # noqa: ANN001
        self.entrypoints.append(entrypoint)
        return super().execute(entrypoint=entrypoint, config=config, ctx=ctx)

    def shutdown(self) -> None:
        self.shut = True


_LLM = "umrs.pipeline_sdk.templates.llm_step:LLMStep"


def test_a_mixed_workflow_keeps_its_template_steps_off_the_networkless_lane(
    tmp_path, no_site, clean_modules
):
    no_site.setattr(step_trust, "hpc_lane_available", lambda: True)
    hpc, inproc = _CapturingHpc(), _CapturingInproc()
    built: list = []

    def build(name=None):
        built.append(name)
        return hpc if name == "slurm" else inproc

    no_site.setattr(wr, "build_step_executor", build)
    workflow = _wf()
    workflow["steps"].append(
        {"name": "llm", "entrypoint": _LLM, "config": {}, "depends_on": ["s1"]}
    )
    record = ExecutionRecord(execution_id="e-mixed", workflow_name="w", user_id="u")
    wr.run_workflow_record(
        record,
        source_path=_upload(tmp_path / "wf", tmp_path / "marker"),
        workflow_json=workflow,
        router=_PlatformRouter(),
    )
    assert record.status == "succeeded", record.error
    assert built == ["slurm", "inproc"]
    assert len(hpc.configs) == 1 and "api_key" not in hpc.configs[0]  # the uploaded step
    assert inproc.entrypoints == [_LLM]  # the template, on the manifest's own lane
    assert inproc.shut


def test_a_workflow_of_uploaded_steps_only_gets_one_lane(tmp_path, no_site, clean_modules):
    record, hpc = _run_on_hpc(tmp_path, no_site, _wf())
    assert record.status == "succeeded", record.error
    assert len(hpc.configs) == 1


@pytest.mark.parametrize("staged", ["", "1"])
def test_llm_step_never_mocks_a_failed_call_on_the_staged_lane(monkeypatch, staged):
    from umrs.pipeline_sdk.step import StepContext
    from umrs.pipeline_sdk.templates.llm_step import LLMStep

    monkeypatch.setenv("UMRS_STEP_STAGED", staged)
    monkeypatch.delenv("UMRS_LLM_MOCK", raising=False)
    step = LLMStep()
    step.setup({"prompt_template": "hi", "api_base": "http://127.0.0.1:9/v1", "timeout_sec": 2})
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output")
    if staged:
        with pytest.raises(RuntimeError, match="no mock answer is returned"):
            step.run(ctx)
    else:
        assert "response_text" in step.run(ctx).extra  # the legacy mock fallback


@pytest.mark.parametrize("staged", ["", "1"])
def test_vision_step_never_mocks_a_failed_call_on_the_staged_lane(monkeypatch, staged):
    from umrs.pipeline_sdk.templates import vision_step

    monkeypatch.setenv("UMRS_STEP_STAGED", staged)
    monkeypatch.delenv("UMRS_VISION_MOCK", raising=False)
    step = vision_step.VisionStep()
    step.setup({"mode": "caption", "api_base": "http://127.0.0.1:9/v1", "timeout_sec": 2})
    image = vision_step._ImageRef(bytes_data=b"\x89PNG")
    if staged:
        with pytest.raises(RuntimeError, match="no mock perception is returned"):
            step._invoke(None, image)
    else:
        assert step._invoke(None, image).meta["mock"] is True


def test_an_image_outside_the_allow_list_is_refused_by_name(tmp_path, no_site, clean_modules):
    no_site.setenv("UMRS_STEP_IMAGE", "localhost/umrs-step:0123456789ab")
    workflow = _wf(routing_hints={"image": "ghcr.io/someone/own:latest"})
    record, hpc = _run_on_hpc(tmp_path, no_site, workflow)
    assert record.status == "failed"
    assert "is not an allowed step image" in (record.steps["s1"].error or "")
    assert hpc.configs == []  # never reached the lane


@pytest.mark.parametrize(
    "image", ["localhost/umrs-step:0123456789ab", "registry.example/allowed:1"]
)
def test_the_platform_image_and_allow_listed_images_run(tmp_path, no_site, clean_modules, image):
    no_site.setenv("UMRS_STEP_IMAGE", "localhost/umrs-step:0123456789ab")
    no_site.setenv(
        "UMRS_STEP_ALLOWED_IMAGES", "registry.example/other:2, registry.example/allowed:1"
    )
    record, hpc = _run_on_hpc(tmp_path, no_site, _wf(routing_hints={"image": image}))
    assert record.status == "succeeded", record.error
    assert image in hpc.metadata[0].values()


def test_step_image_refusal_rules(no_site):
    no_site.setenv("UMRS_STEP_IMAGE", "localhost/umrs-step:abc")
    assert step_image_refusal(None, "slurm") is None
    assert step_image_refusal("localhost/umrs-step:abc", "slurm") is None
    assert step_image_refusal("evil:1", "inproc") is None  # no image runs there
    assert step_image_refusal("evil:1", "slurm")
    assert step_image_refusal(42, "podman")
    assert step_image_refusal("evil:1", "")  # an unnamed executor is not exempt


# ── §2 no AWS_* credential value passthrough ───────────────────────────────


_CREDS = {
    "AWS_ACCESS_KEY_ID": "AKIA-test",
    "AWS_SECRET_ACCESS_KEY": "secret-test",
    "AWS_SESSION_TOKEN": "token-test",
}


def _assert_no_credential(argv: list[str]) -> None:
    joined = " ".join(argv)
    for name, value in _CREDS.items():
        assert name not in joined and value not in joined


def test_podman_executor_forwards_no_credential(monkeypatch, tmp_path):
    for name, value in _CREDS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    cmd = PodmanExecutor(image="img:1")._build_command(
        container_name="c", workdir=str(tmp_path), s3_config={}, image="img:1", gpu=False
    )
    _assert_no_credential(cmd)
    assert "AWS_REGION=us-east-1" in cmd  # a region is no credential


def test_pool_worker_forwards_no_credential(monkeypatch):
    for name, value in _CREDS.items():
        monkeypatch.setenv(name, value)
    seen: list[list[str]] = []

    class _Proc:
        def poll(self):
            return None

    def _popen(cmd, **_kw):  # noqa: ANN001, ANN003
        seen.append(list(cmd))
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", _popen)
    WorkerPoolExecutor(image="img:1")._start_worker("img:1", False)
    _assert_no_credential(seen[0])
