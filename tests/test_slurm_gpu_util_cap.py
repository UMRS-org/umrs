"""The partition's gpu_mem_util_max applies to EVERY wrapper launch.

Live rccs 2026-09-24: `r4 endpoint up --model Bonsai-8B` still requested
0.9 (109.5 GiB) against 98.1 of 121.6 GiB free on a GB10 and died at vLLM
startup — the cap lived only in the DeploymentSpec path, and the endpoint
path builds VLLM_GPU_MEM_UTIL from the serving profile. The Slurm submit now
caps the env itself; an unset value counts as the wrapper default (0.95).
"""

from __future__ import annotations

from types import SimpleNamespace

from umrs.orchestrator.remote_slurm_backend import _cap_gpu_mem_util_env


def _site(**caps: float) -> SimpleNamespace:
    return SimpleNamespace(
        partition_profiles={
            name: SimpleNamespace(gpu_mem_util_max=cap) for name, cap in caps.items()
        }
    )


_RCCS = _site(**{"ng-dgx-m0": 0.78, "ng-dgx-m1": 0.78})


def test_profile_value_above_the_cap_is_capped():
    env = _cap_gpu_mem_util_env(_RCCS, "ng-dgx-m0", {"VLLM_GPU_MEM_UTIL": "0.9", "X": "1"})
    assert env == {"VLLM_GPU_MEM_UTIL": "0.78", "X": "1"}


def test_unset_value_is_capped_not_left_to_the_wrapper_default():
    assert _cap_gpu_mem_util_env(_RCCS, "ng-dgx-m0", None) == {"VLLM_GPU_MEM_UTIL": "0.78"}


def test_a_lower_request_is_kept():
    env = {"VLLM_GPU_MEM_UTIL": "0.7"}
    assert _cap_gpu_mem_util_env(_RCCS, "ng-dgx-m0", env) == env


def test_comma_list_takes_the_strictest_cap():
    site = _site(a=0.8, b=0.6)
    assert _cap_gpu_mem_util_env(site, "a,b", {})["VLLM_GPU_MEM_UTIL"] == "0.6"


def test_partition_without_a_cap_is_untouched():
    env = {"VLLM_GPU_MEM_UTIL": "0.95"}
    assert _cap_gpu_mem_util_env(_site(other=0.5), "gpu-big", env) is env
    assert _cap_gpu_mem_util_env(_site(), "x", None) is None
