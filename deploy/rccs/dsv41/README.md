# DeepSeek-V4.1-Flash-1017 on rccs: staging, first launches, benchmarks (2026-11-12)

Maintainer ask (2026-11-12): serve DeepSeek-V4.1-Flash-1017 as the always-on chat service on an `ng-dgx-m` pair
(the MiaAI-Lab TensorFold recipe) and on both `ai-pro6000` nodes, then benchmark how architecture, GPU and
interconnect shape serving performance. The roster entries are `deepseek-v4.1-flash-1017` and
`deepseek-v4.1-flash-1017-nvfp4` in `../pool-rccs.yaml`; these scripts stage what they need and run the
same adapters (`tensorfold_runner.sh`, `sglang_runner.sh`) outside the UMRS stack for first launches and the
benchmark. They expect copies of the adapters, `venue.sh` and `gb10_reclaim.py` in `~/tmp/dsv41/mr/` and the
benchmark tools in `~/tmp/dsv41/bench/` on the rccs login node.

| Script | What it does |
| --- | --- |
| `tf_build.sbatch` | builds the recipe's engine image (recipe `c7f2b851`, upstream `zig/Dockerfile`, sm_121) on a GB10 node in a throwaway podman store, saves it to `umrs_models/images` (`-aarch64.tar` plus the `-arm64.tar` name venue.sh looks for) |
| `dsv41_stage_local.sbatch` | on every node of an m pair: Engram shards 47/48 and a local EXL3 copy from Hugging Face, then both ranks' weight files, all under `/var/tmp/umrs-tf` (~583 GB a node); the served folder on Lustre |
| `tf_direct.sbatch` | launches the pair through `tensorfold_runner.sh`, runs `serve_bench.py` and the ablations (`TF_RDMA=0`, `TF_DRAFTS` flip), then serves; `KEEP=1` queues a singleton successor (stopgap keep-alive until the roster's service block is converged; stop it with `scancel -n dsv41-tf-direct`) |
| `m_hw.sbatch` | GB10 pair microbenchmarks: raw RDMA (`ib_write_bw`/`ib_write_lat`, out-of-band over the mgmt NIC) and `hw_bench.py` over RoCE and over TCP |
| `nvfp4_dl.sbatch` | the NVFP4 checkpoint into `~/umrs_models_hs` (home, `/hs/work0`) from a compute node |
| `pro6000_hold.sbatch` → `pro6000_entry.sh` → `pro6000_stage2.sh` | the historical direct driver on the partition now named `ng-dgx-l`: node probe, Singularity SGLang TP=4 launch, `serve_bench.py`, `hw_bench.py`, then serve; its chain stops on a failure marker. Jobs 468620 and 468760 failed readiness; do not requeue it as a substitute for managed preflight |
| `serve_bench.py` | stdlib OpenAI streaming client: decode concurrency sweep, prefill length sweep, mixed long-prompt streams (TTFT, inter-token latency, tok/s) |
| `hw_bench.py` | torch/NCCL: device copy, pinned host<->device copy, bf16 GEMM, all_reduce latency and bus bandwidth |
| `summarize.py` | medians per phase from `serve_bench.py` JSON |

rccs facts these scripts encode (measured 2026-11-12):

- the Lustre project quota on `/lvs0/rccs-dmp/masaru.nagaso` is 5 TB and full, so large data goes to node NVMe
  (`/var/tmp`, 3.6 TB a node) or home (`/hs/work0`, 1 TB user quota);
- `ng-dgx-m*` nodes have `RealMemory=1`, so a job's default memory request is 1 MB: every job passes `--mem=0`;
- compute nodes download from Hugging Face at ~550 MB/s, the login node at ~60 MB/s (Engram needs `hf_xet`);
- the QSFP port accepts RoCE but firewalld rejects TCP on it: rendezvous and NCCL bootstrap use `enP7s7`;
- a throwaway podman store's database pins the first job's runroot: drop `db.sql`, `libpod/` and
  `overlay-containers/` before reusing it in a new job.

Current rollout facts (2026-11-18):

- direct TensorFold `499019` is RUNNING on `ng-dgx-m0`; managed attempts failed because the batch shell lacked
  `VLLM_API_KEY`. The wrapper exports one key before the env heredoc and TensorFold source, but managed health is not
  proven until the reviewed rccs convergence and a new readiness receipt;
- `ng-dgx-l` was UP with both ai-pro6000 nodes idle and no PENDING job at the check. This is capacity, not runtime
  health. Only the NVFP4 roster entry and its required direct-backend scheduler override candidate use the new name;
  TensorFold remains `ng-dgx-m2,ng-dgx-m0,ng-dgx-m1`, and the step lane remains `ng-dgx-m3`;
- rank logs from both 468620 and 468760 show `CUDA error: invalid device ordinal` in the readiness phase. No NVFP4
  benchmark results exist, the registration and aliases remain, and the `service:` block stays commented pending
  on-demand Singularity preflight, provision and health. `dsv41-serving-bench.html` is updated source only; the live
  Claude artifact has not been republished from this implementation.
