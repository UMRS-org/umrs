"""U3 S6e-2 (ADR-117, multi-site X-5): core UMRS hard-codes no site for the
cluster loop and fine-tune.

The front declares the base URL its cluster jobs call back
(``network.gateway_base`` -> ``UMRS_GATEWAY_BASE``), the loop driver image
(``runtime.loop_job_image`` -> ``LOOP_JOB_IMAGE``) and the one site it trains
on (``extra.finetune_site``, optional ``extra.finetune_wrapper``). An
undeclared one refuses by name, before anything is submitted. These pins read
only this umrs checkout, so they run in UMRS-org's standalone CI too.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from umrs.jobs import cluster_venue, fine_tune
from umrs.jobs.models import Job, JobAttempt, JobType
from umrs.orchestrator import site_config as sc
from umrs.orchestrator.hpc_identity import SERVICE_ACCOUNT

UMRS = Path(__file__).resolve().parents[1]
LOOP_SCRIPT = UMRS / "src" / "umrs" / "knowledge" / "_cluster_loop_job.sh"
CORE_FILES = [
    UMRS / "src" / "umrs" / "jobs" / "cluster_venue.py",
    UMRS / "src" / "umrs" / "jobs" / "fine_tune.py",
    LOOP_SCRIPT,
]
_ENV_NAMES = ("UMRS_GATEWAY_BASE", "R4_GATEWAY_BASE", "LOOP_JOB_IMAGE", "R4_REPO_ROOT")
_SPEC = {"target_model": "m", "dataset_ref": "s3://b/k.jsonl", "adapter_name": "a"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _write_site(tmp_path: Path, monkeypatch, body: str) -> Path:
    path = tmp_path / "site.yaml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("UMRS_SITE_CONFIG", str(path))
    return path


_DECLARED_SITE = """\
name: site-a
network:
  gateway_base: https://gw.example.test/base
runtime:
  loop_job_image: registry.example.test/loop:2
extra:
  finetune_site: site-a
"""


# ── (e) no site literal in core ──────────────────────────────────────────────


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_loop_and_finetune_files_name_no_site(path):
    text = path.read_text(encoding="utf-8")
    for literal in ("data0.ai.r-ccs.riken.jp", "r-ccs", '"ai4s"', "deploy/ai4s", "deploy/rccs"):
        assert literal not in text, literal
    assert "r4-mcp" not in text


# ── (a) cluster loop: declared or refused ────────────────────────────────────


def _job() -> Job:
    return Job(job_id="job-1", project_id="p", user_id="alice",
               job_type=JobType.RESEARCH_LOOP, spec={}, idempotency_key="k")  # fmt: skip


def _submit(monkeypatch) -> dict:
    """Run one cluster-loop submit with seams; returns what reached sbatch."""
    seen: dict = {}

    async def submit_wrapped(wrapper: str, env: dict):
        seen.update(wrapper=wrapper, env=env)
        return SimpleNamespace(job_id="9")

    monkeypatch.setattr(cluster_venue, "_gateway_hmac_secret", lambda: "s")
    monkeypatch.setattr(cluster_venue, "_ledger_pending", lambda *_a: SimpleNamespace(tag="t"))
    monkeypatch.setattr(cluster_venue, "_ledger_submitted", lambda *_a: None)
    identity = SimpleNamespace(mode=SERVICE_ACCOUNT, to_metrics=lambda: {})
    submitter = cluster_venue.ClusterSlurmSubmitter(submit_wrapped, identity=lambda _j: identity)
    attempt = JobAttempt(attempt_id="att-1", job_id="job-1", attempt_no=1, backend="slurm")
    asyncio.run(submitter.submit(job=_job(), attempt=attempt, spec={"loop_id": "L1"}))
    return seen


def test_cluster_loop_refuses_without_a_gateway_base(monkeypatch):
    monkeypatch.setattr(cluster_venue, "_loaded_site", lambda: None)
    monkeypatch.setenv("LOOP_JOB_IMAGE", "loop:1")
    with pytest.raises(ValueError, match=r"network\.gateway_base / UMRS_GATEWAY_BASE"):
        _submit(monkeypatch)


def test_cluster_loop_refuses_without_a_loop_image(monkeypatch):
    monkeypatch.setattr(cluster_venue, "_loaded_site", lambda: None)
    monkeypatch.setenv("UMRS_GATEWAY_BASE", "https://gw.example.test")
    with pytest.raises(ValueError, match=r"runtime\.loop_job_image / LOOP_JOB_IMAGE"):
        _submit(monkeypatch)


def test_cluster_loop_uses_the_site_declarations(tmp_path, monkeypatch):
    _write_site(tmp_path, monkeypatch, _DECLARED_SITE)
    monkeypatch.setenv("R4_REPO_ROOT", str(tmp_path))
    seen = _submit(monkeypatch)
    env = seen["env"]
    assert env["UMRS_GATEWAY_BASE"] == env["R4_GATEWAY_BASE"] == "https://gw.example.test/base"
    assert env["LOOP_JOB_IMAGE"] == "registry.example.test/loop:2"
    # No slurm_job_info_dir: the local fallback is this site's own deploy dir.
    assert env["RIVAULT_WRAPPER_DIR"] == str(tmp_path / "deploy" / "site-a")
    assert seen["wrapper"] == str(tmp_path / "deploy" / "site-a" / "loop-job.slurm")


def test_env_overrides_the_site_declarations(tmp_path, monkeypatch):
    _write_site(tmp_path, monkeypatch, _DECLARED_SITE)
    monkeypatch.setenv("R4_GATEWAY_BASE", "https://legacy.example.test")
    monkeypatch.setenv("LOOP_JOB_IMAGE", "loop:env")
    assert cluster_venue._require_loop_declarations() == ("https://legacy.example.test", "loop:env")


def test_deploy_dir_follows_the_job_info_dir(tmp_path, monkeypatch):
    _write_site(
        tmp_path,
        monkeypatch,
        "name: site-b\nextra:\n  slurm_job_info_dir: /remote/umrs/deploy/b/jobs\n",
    )
    assert cluster_venue._cluster_deploy_dir() == "/remote/umrs/deploy/b"


# ── (b) the job script derives its egress host from the gateway base ─────────


def _declarations_block() -> str:
    text = LOOP_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"# BEGIN loop-job declarations.*?# END loop-job declarations", text, re.S)
    assert match, "declarations block missing"
    return match.group(0)


def _run_block(env: dict[str, str]) -> subprocess.CompletedProcess:
    script = f'set -euo pipefail\n{_declarations_block()}\necho "HOST=${{GATEWAY_HOST}} IMAGE=${{LOOP_IMAGE}}"\n'
    return subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin", **env},
                          capture_output=True, text=True, check=False)  # fmt: skip


@pytest.mark.parametrize(
    ("base", "host"),
    [
        ("https://gw.example.test/site", "gw.example.test"),
        ("http://user@gw.example.test:8443/a/b", "gw.example.test"),
        ("gw.example.test", "gw.example.test"),
    ],
)
def test_script_derives_the_allowlist_host(base, host):
    done = _run_block({"UMRS_GATEWAY_BASE": base, "LOOP_JOB_IMAGE": "loop:1"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"HOST={host} IMAGE=loop:1" in done.stdout


def test_script_reads_the_legacy_gateway_name():
    done = _run_block({"R4_GATEWAY_BASE": "https://old.example.test/x", "LOOP_JOB_IMAGE": "i"})
    assert "HOST=old.example.test" in done.stdout


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({"LOOP_JOB_IMAGE": "loop:1"}, "network.gateway_base / UMRS_GATEWAY_BASE"),
        (
            {"UMRS_GATEWAY_BASE": "https://gw.example.test"},
            "runtime.loop_job_image / LOOP_JOB_IMAGE",
        ),
        ({"UMRS_GATEWAY_BASE": "https:///path", "LOOP_JOB_IMAGE": "i"}, "names no host"),
    ],
)
def test_script_refuses_undeclared_values_by_name(env, named):
    done = _run_block(env)
    assert done.returncode == 1
    assert "FATAL" in done.stdout and named in done.stdout


def test_script_allowlist_and_no_proxy_use_the_derived_host():
    text = LOOP_SCRIPT.read_text(encoding="utf-8")
    assert 'ALLOWLIST="${GATEWAY_HOST}"' in text
    assert 'NO_PROXY="localhost,127.0.0.1,${GATEWAY_HOST}"' in text


# ── (c) fine-tune trains only on the declared site ───────────────────────────


def test_finetune_refuses_when_no_site_is_declared(monkeypatch):
    monkeypatch.setattr(fine_tune, "_site_extra", lambda: {})
    with pytest.raises(ValueError, match=r"not enabled on this front.*extra\.finetune_site"):
        fine_tune._validate_spec(dict(_SPEC))


def test_finetune_defaults_to_and_accepts_the_declared_site(monkeypatch):
    monkeypatch.setattr(fine_tune, "_site_extra", lambda: {"finetune_site": "site-a"})
    assert fine_tune._validate_spec(dict(_SPEC))["site"] == "site-a"
    assert fine_tune._validate_spec({**_SPEC, "site": "site-a"})["site"] == "site-a"


def test_finetune_rejects_another_site(monkeypatch):
    monkeypatch.setattr(fine_tune, "_site_extra", lambda: {"finetune_site": "site-a"})
    with pytest.raises(ValueError, match="pin is 'site-a' only.*not 'site-b'"):
        fine_tune._validate_spec({**_SPEC, "site": "site-b"})


def test_finetune_wrapper_defaults_to_the_declared_site(monkeypatch, tmp_path):
    monkeypatch.setattr(fine_tune, "_site_extra", lambda: {"finetune_site": "site-a"})
    monkeypatch.setenv("R4_REPO_ROOT", str(tmp_path))
    expected = tmp_path / "deploy" / "site-a" / "finetune-adapter.slurm"
    assert fine_tune._finetune_wrapper("site-a") == str(expected)


def test_finetune_wrapper_override(monkeypatch):
    extra = {"finetune_site": "site-a", "finetune_wrapper": "/remote/ft.slurm"}
    monkeypatch.setattr(fine_tune, "_site_extra", lambda: extra)
    assert fine_tune._finetune_wrapper("site-a") == "/remote/ft.slurm"


def test_finetune_reads_the_site_yaml_extra(tmp_path, monkeypatch):
    _write_site(tmp_path, monkeypatch, _DECLARED_SITE)
    assert fine_tune._declared_finetune_site() == "site-a"


# ── (d) site_config parses and exports the new keys ──────────────────────────


def test_site_config_parses_and_exports_the_new_keys(tmp_path, monkeypatch):
    cfg = sc.load_site_config(str(_write_site(tmp_path, monkeypatch, _DECLARED_SITE)))
    assert cfg.network.gateway_base == "https://gw.example.test/base"
    assert cfg.runtime.loop_job_image == "registry.example.test/loop:2"
    assert sc._network_env(cfg.network)["UMRS_GATEWAY_BASE"] == "https://gw.example.test/base"
    assert sc._runtime_env(cfg.runtime)["LOOP_JOB_IMAGE"] == "registry.example.test/loop:2"


def test_undeclared_keys_are_not_exported():
    assert sc.NetworkConfig().gateway_base == "" and sc.RuntimeConfig().loop_job_image == ""
    assert "UMRS_GATEWAY_BASE" not in sc._network_env(sc.NetworkConfig())
    assert "LOOP_JOB_IMAGE" not in sc._runtime_env(sc.RuntimeConfig())


@pytest.mark.parametrize("name", ["UMRS_GATEWAY_BASE", "R4_GATEWAY_BASE"])
def test_env_only_fallback(monkeypatch, name):
    monkeypatch.setenv(name, "https://gw.example.test")
    monkeypatch.setenv("LOOP_JOB_IMAGE", "loop:1")
    cfg = sc.default_site_config()
    assert (cfg.network.gateway_base, cfg.runtime.loop_job_image) == (
        "https://gw.example.test",
        "loop:1",
    )


# ── site declarations keep production behaviour ──────────────────────────────


def _deploy_site(name: str) -> dict:
    return yaml.safe_load((UMRS / "deploy" / name / "site.yaml").read_text(encoding="utf-8"))


def test_rccs_front_declares_what_core_used_to_default():
    site = _deploy_site("rccs-front")
    assert site["network"]["gateway_base"] == "https://data0.ai.r-ccs.riken.jp/rccs"
    assert site["runtime"]["loop_job_image"] == "localhost/r4-mcp:latest"


def test_ai4s_front_declares_the_finetune_site():
    assert _deploy_site("ai4s-front")["extra"]["finetune_site"] == "ai4s"


@pytest.mark.parametrize("front", ["ai4s-front", "bdrt-front", "octopus-front"])
def test_other_fronts_declare_no_cluster_loop(front):
    site = _deploy_site(front)
    assert not (site.get("network") or {}).get("gateway_base")
    assert not (site.get("runtime") or {}).get("loop_job_image")
    if front != "ai4s-front":
        assert not (site.get("extra") or {}).get("finetune_site")
