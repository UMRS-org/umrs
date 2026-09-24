"""Per-partition gpu_mem_util_max: model servers on shared-GPU partitions.

rccs GB10 nodes (121.6 GiB) have ~97 GiB free once the job's harness
services + vision sidecar initialise CUDA; registry serving profiles carry
0.9 by default, so vLLM refused to start ("Free memory on device cuda:0
(98.29/121.63 GiB) ... less than desired GPU memory utilization (0.9,
109.46 GiB)", slurm-384694). The partition ceiling caps the request.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from umrs.orchestrator.deployment_spec import _cap_gpu_mem_util
from umrs.orchestrator.site_config import PartitionProfile

REPO = Path(__file__).resolve().parents[1]


def _site(**profiles):
    return SimpleNamespace(partition_profiles=profiles)


def test_cap_applies_below_the_request():
    site = _site(**{"ng-dgx-m0": PartitionProfile(gpu_mem_util_max=0.78)})
    assert _cap_gpu_mem_util(0.9, site, "ng-dgx-m0") == 0.78


def test_request_under_the_cap_is_kept():
    site = _site(**{"ng-dgx-m0": PartitionProfile(gpu_mem_util_max=0.78)})
    assert _cap_gpu_mem_util(0.6, site, "ng-dgx-m0") == 0.6


def test_no_cap_declared_or_unknown_partition_is_unchanged():
    site = _site(**{"a100": PartitionProfile()})
    assert _cap_gpu_mem_util(0.9, site, "a100") == 0.9
    assert _cap_gpu_mem_util(0.9, site, "not-a-partition") == 0.9
    assert _cap_gpu_mem_util(0.9, SimpleNamespace(), "x") == 0.9


def test_default_profile_declares_no_cap():
    assert PartitionProfile().gpu_mem_util_max == 0.0


def test_rccs_gb10_partitions_declare_the_cap():
    for rel in ("deploy/rccs/site.yaml", "deploy/rccs-front/site.yaml"):
        profiles = yaml.safe_load((REPO / rel).read_text())["partition_profiles"]
        caps = {
            k: v.get("gpu_mem_util_max") for k, v in profiles.items() if k.startswith("ng-dgx-m")
        }
        assert caps and set(caps.values()) == {0.78}, (rel, caps)
