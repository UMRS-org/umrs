#!/usr/bin/env python3
"""Hardware microbenchmarks behind LLM serving numbers (torch + NCCL only).

Per rank: device memory copy bandwidth, pinned host<->device bandwidth, bf16
GEMM throughput. Across ranks: NCCL all_reduce latency and bus bandwidth from
8 KiB to 1 GiB (tensor-parallel decode moves small messages twice a layer;
prefill moves large ones). Launch one process per GPU with RANK, WORLD_SIZE,
LOCAL_RANK, MASTER_ADDR, MASTER_PORT set; rank 0 writes the JSON (--out).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist

GIB = 1 << 30


def timed(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def mem_bw(dev: torch.device) -> dict:
    n = 2 * GIB
    a = torch.empty(n, dtype=torch.uint8, device=dev)
    b = torch.empty_like(a)
    t = timed(lambda: b.copy_(a), 10)
    h = torch.empty(GIB, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(GIB, dtype=torch.uint8, device=dev)
    h2d = timed(lambda: d.copy_(h, non_blocking=True), 10)
    d2h = timed(lambda: h.copy_(d, non_blocking=True), 10)
    return {"d2d_copy_GBps": 2 * n / t / 1e9, "h2d_GBps": GIB / h2d / 1e9, "d2h_GBps": GIB / d2h / 1e9}


def gemm(dev: torch.device) -> dict:
    out = {}
    for m in (4096, 8192):
        a = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
        b = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
        t = timed(lambda a=a, b=b: a @ b, 20)
        out[f"bf16_gemm_{m}_TFLOPS"] = 2 * m**3 / t / 1e12
    # decode-shaped GEMV-like product: a few tokens against a large weight
    w = torch.randn(7168, 7168 * 2, dtype=torch.bfloat16, device=dev)
    x = torch.randn(4, 7168, dtype=torch.bfloat16, device=dev)
    t = timed(lambda: x @ w, 50)
    out["bf16_skinny_4x7168x14336_GBps_weights"] = w.numel() * 2 / t / 1e9
    return out


def allreduce(dev: torch.device, world: int) -> list[dict]:
    rows = []
    size = 8 * 1024
    while size <= GIB:
        t = torch.ones(size // 2, dtype=torch.bfloat16, device=dev)
        iters = 50 if size <= 64 * 1024 * 1024 else 10
        sec = timed(lambda t=t: dist.all_reduce(t), iters, warmup=5)
        algbw = size / sec / 1e9
        rows.append(
            {"bytes": size, "lat_us": sec * 1e6, "algbw_GBps": algbw, "busbw_GBps": algbw * 2 * (world - 1) / world}
        )
        size *= 4
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--skip-local", action="store_true")
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    info = {
        "rank": rank,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(dev),
        "cc": ".".join(map(str, torch.cuda.get_device_capability(dev))),
        "mem_total_GiB": torch.cuda.get_device_properties(dev).total_memory / GIB,
    }
    if not args.skip_local:
        info.update(mem_bw(dev))
        info.update(gemm(dev))
    dist.barrier()
    ar = allreduce(dev, world)
    infos = [None] * world
    dist.all_gather_object(infos, info)
    if rank == 0:
        env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
        res = {
            "label": args.label,
            "world": world,
            "ranks": infos,
            "allreduce": ar,
            "nccl_env": env,
            "torch": torch.__version__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(
            json.dumps(
                {
                    "label": args.label,
                    "allreduce_8KiB_us": ar[0]["lat_us"],
                    "allreduce_1GiB_busbw": ar[-1]["busbw_GBps"],
                    "rank0": info,
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
