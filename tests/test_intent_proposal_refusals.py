"""What the proposal refuses before anything runs (ADR-112 B2; plan §5 B2.0
§10.3; fixer findings 2026-11-08).

- a step the router would send to a model endpoint (``requires`` /
  ``model_type``): B2 signs no endpoint, so the step would fail
  ``endpoint_changed`` mid-run after earlier steps already ran on the
  person's allocation (``endpoint_not_supported``);
- partitions of another arch than the trusted image's (``mixed_arch``): the
  trusted image runs the seccomp transform, the pids probe and the
  forwarder in every job;
- a step named ``intent`` (``bad_step_name``): the gate reserves it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from umrs.hpc_accounts.intents import IntentInvalid
from umrs.server.intent_proposal import ProposalRequest, build_intent

X86 = {"image_id": "1" * 64, "archive_sha256": "2" * 64}
ARM = {"image_id": "3" * 64, "archive_sha256": "4" * 64}


def _part(arch: str, gpus: int) -> dict:
    return {
        "arch": arch, "max_cpus": 4, "max_gpus": gpus, "max_time": 3600, "mem": False,
        "gpu_request": "", "gres": "", "gpu_flag": "gres", "exclusive": False,
    }  # fmt: skip


def _request(tmp_path, steps: list[dict], gpu_arch: str = "aarch64") -> ProposalRequest:
    digests = tmp_path / "digests.json"
    digests.write_text(json.dumps({"img": {"x86_64": X86, "aarch64": ARM}}))
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "mod.py").write_text("class A: pass\nclass B: pass\n")
    site = SimpleNamespace(
        name="s1",
        step_executor=SimpleNamespace(image_digests=str(digests), image="img", memory="", cpus=2),
    )
    profile = {
        "partitions": {"cpu": _part("x86_64", 0), "gpu": _part(gpu_arch, 4)},
        "gpu": {"enabled": True},
        "slurm": {"queue_seconds": 600},
    }
    return ProposalRequest(
        workflow="w", version="1.0.0", source_sha256="5" * 64, project="p", principal="alice",
        login="alice", account="acct", params={"project_id": "p"},
        workflow_json={"steps": steps}, source_path=str(src), site=site, profile=profile,
        profile_sha256="6" * 64,
    )  # fmt: skip


A = {"name": "a", "entrypoint": "mod:A"}
GPU_B = {"name": "b", "entrypoint": "mod:B", "depends_on": ["a"], "routing_hints": {"gpu": True}}


def _refused(req: ProposalRequest) -> str:
    with pytest.raises(IntentInvalid) as exc:
        build_intent(req, s3=None, now=1_800_000_000)
    return exc.value.reason


def test_a_step_on_a_partition_of_another_arch_is_mixed_arch(tmp_path):
    assert _refused(_request(tmp_path, [A, GPU_B])) == "mixed_arch"


def test_steps_of_one_arch_pass_and_carry_no_endpoint(tmp_path):
    body = build_intent(
        _request(tmp_path, [A, GPU_B], gpu_arch="x86_64"), s3=None, now=1_800_000_000
    )
    assert body["trusted_image"]["image_id"] == X86["image_id"]
    assert not any("endpoint" in step for step in body["steps"])


@pytest.mark.parametrize("routing", [{"requires": ["llm"]}, {"model_type": "vision"}])
def test_a_step_that_reads_a_model_endpoint_is_refused(tmp_path, routing):
    step = {**A, **routing}
    assert _refused(_request(tmp_path, [step], gpu_arch="x86_64")) == "endpoint_not_supported"


def test_the_step_name_the_gate_reserves_is_bad_step_name(tmp_path):
    step = {**A, "name": "intent"}
    assert _refused(_request(tmp_path, [step], gpu_arch="x86_64")) == "bad_step_name"
