# UMRS benchmark scripts

Load, latency and ingestion benchmarks for a running UMRS server, written by
Emmanuel Jeannot on upstream `main` and ported to `ai4s-platform` (plan U1 A8).
Only the code is here. The logs and CSVs of the published runs, which these
scripts and the R plots read from `log/`, stay on the archived upstream main
under the tag `archive/main-pre-merge-2026-11` (plan M-U5). Fetch that tag's
`src/scripts/benchmarks/log/` to re-plot them.

Run the Python scripts from `src/` with the UMRS environment active. Each one
has `--help`. Nothing names a host, user or key: the server address comes
from the connection profile (`--server-ip` / `--control-port` override it),
S3 and DB credentials come from your s3cmd config, user JSON or DB
credentials file.

## Scripts

| Script | What it does |
|---|---|
| `launch_N_dir.sh N [run]` | Starts `monitor_users.py`, then N clients (`user1_1..N`) that each `upload_dir --blocking` `../data/img` into a per-run prefix of `bucket-3`. Logs the client host, CPU and GPU into `log/xp_dir_<N>_run<run>.log`. Run it from `src/`. |
| `launch_N_dir_simple.sh`, `Launch_N_dir_mixed.sh`, `launch_N_file.sh` | Variants: no monitor; each client uploads to `bucket-3` and `bucket-1`; one file per client. |
| `monitor_users.py N` | Polls `STATUS_ALL` until N users are idle; prints per-user durations. |
| `check_user_status.py USER` | Exit 0 when the user is idle, 1 while it still runs. |
| `bench_latency.py` | Concurrent ZMQ control-plane latency (`status-all`, `status`, `ping`); `--sweep 1,5,10` over N users, `--csv` appends results. |
| `bench_knn.py` | KNN search latency with a pre-uploaded query image; `--sweep` over N or `--sweep-k` over k. Needs the `KNN` command (`request_knn`), which other U1 lanes bring to `ai4s-platform`. |
| `knn_sweep_cold_warm.sh` | Cold then hot KNN call per user, k and repeat through `client_umrs.py knn`. Paths come from `UMRS_BENCH_CLIENT`, `UMRS_BENCH_IMAGE`, `UMRS_BENCH_CONFIG_DIR` and `UMRS_BENCH_LOG_DIR` (defaults inside the repo and `~/.config/umrs/n_users/users`). Upstream named it `knn_sweep_cold_warm.py`. |
| `parse_server_log.py LOG [--csv]` | Per-task timings from a server log: auth, PROCESS_OBJECTS, batch, S3 check, run, completion, plus every `[TIMING]` field in the CSV. |
| `bench_baremetal_ingest.py -c USER.json` | The same ingest steps without the UMRS server: S3 upload, optional already-indexed check, DataLoader decode, ResNet-50, store into its own PostgreSQL database. Writes `log/baremetal_<tag>/baremetal.csv`. `--drop-db` removes the database. Its `plot_breakdown.R` was never published upstream. |
| `populate.py` | Fills `user1-<z>` buckets for the mass-upload sweep with boto3: one upload of `data/images/5000`, then server-side copies to `5000ab`, `5000ac`, and so on. `--user Z:TARGET` repeats. Resumable. |
| `populate_buckets.py` | The older mass upload through `client_umrs.py upload_dir`, 7000 images per round. Resumable. |
| `embed_images.py` | Stand-alone ResNet-50 or CLIP embeddings for image files, without UMRS. |
| `perf_test.py` | `perf_test()` helper (no CLI): in-process S3 read, embed and DB store throughput. |
| `plot_*.R`, `multi_usersR.R` | ggplot2 figures. They read `log/*.csv` or hold the measured numbers inline, and write PDF/PNG files into `log/` or the working directory. Run them from this directory with `Rscript`. |

`populate.py` connects with TLS verification off, as upstream did for
self-signed MinIO. Pass `--verify-tls` against a real certificate.
