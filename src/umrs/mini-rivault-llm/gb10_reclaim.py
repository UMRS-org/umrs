"""Pre-start memory reclaim for GB10 unified-memory nodes (multi-node vLLM).

vLLM refuses to start a rank when CUDA's free memory is below
``--gpu-memory-utilization x total``. On rccs GB10 nodes CUDA reports only
the memory that is free *right now*: an idle ng-dgx-m2 node showed 59.4 GiB
free of 121.6 GiB (MemAvailable 104.5 GB) because ~45 GiB sat in Lustre
page cache and slab. That memory is reclaimable: allocating 100 GiB in CUDA
succeeded, and after freeing it CUDA reported 102.7 GiB free (probe jobs
458165/458166, 2026-11-05). DeepSeek-V4-Flash job 458139 died on that check.

This script performs the reclaim: it allocates up to
``RIVAULT_MN_PRESTART_RECLAIM_GIB`` GiB in 2 GiB chunks, frees them and exits,
so the vLLM rank that starts next sees the memory as free. It never fails
the launch: any error is printed and the script exits 0.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

GIB = 1 << 30
CHUNK_GIB = 2


def reclaim(target_gib: int, alloc: Callable[[int], object]) -> int:
    """Allocate up to *target_gib* GiB in chunks; return the GiB obtained.

    Allocation stops at the first failure; the held chunks are released when
    the list goes out of scope.
    """
    held: list[object] = []
    while len(held) * CHUNK_GIB < target_gib:
        try:
            held.append(alloc(CHUNK_GIB * GIB))
        except Exception:  # noqa: BLE001 - out of memory ends the reclaim
            break
    return len(held) * CHUNK_GIB


def _target_gib() -> int:
    try:
        return max(0, int(os.environ.get("RIVAULT_MN_PRESTART_RECLAIM_GIB", "0")))
    except ValueError:
        return 0


def main() -> int:
    target = _target_gib()
    if target == 0:
        return 0
    try:
        import torch

        free_before = torch.cuda.mem_get_info()[0] / GIB
        got = reclaim(target, lambda n: torch.empty(n, dtype=torch.uint8, device="cuda"))
        torch.cuda.empty_cache()
        free_after = torch.cuda.mem_get_info()[0] / GIB
    except Exception as exc:  # noqa: BLE001 - never block the launch
        print(f"[gb10-reclaim] skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0
    print(f"[gb10-reclaim] cuda free {free_before:.1f} GiB -> {free_after:.1f} GiB (held {got} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
