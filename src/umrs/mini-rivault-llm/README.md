# Mini-RiVault LLM + Vision Runtime

This directory contains the canonical runtime assets used by UMRS to launch
GPU inference stacks on Slurm-managed compute nodes. The stack is profile-
driven and is designed to support text inference through vLLM/LiteLLM and
vision inference through BentoML.

## What lives here

- `serve-model.yaml` — the main compose definition.
- `serve-model.slurm` — the source-side Slurm launcher used on compute nodes.
- `x_nvidia-override.yaml` / `x_rocm-override.yaml` — runtime-specific GPU overrides.
- `litellm_fs/`, `nginx_fs/`, `coredns_fs/`, `mcpo_fs/` — service configuration fragments.
- `tests/` — focused runtime tests and helpers.

Site-specific deploy wrappers under `deploy/*/serve-model.slurm` normally call
into this source tree after loading `site.yaml`-derived configuration.

## Runtime Profiles

The compose stack is profile-based. The most important profiles are:

- `harness` — shared infrastructure inside the job, including NGINX and support services.
- `vllm` — text/embedding inference through vLLM + LiteLLM.
- `vision-bentoml` — vision inference through the BentoML service.

The deployed wrapper usually exports:

```bash
RIVAULT_PROFILES="harness vllm vision-bentoml"
```

## Launch Flow

For the normal Slurm+Podman deployment path:

1. A site wrapper in `deploy/<site>/serve-model.slurm` loads `site.yaml` and
      site-specific defaults.
2. The wrapper exports the resolved runtime variables such as profile set,
      model cache paths, LiteLLM auth token, and output-log location.
3. The wrapper executes this directory's `serve-model.slurm`.
4. `serve-model.slurm` chooses the appropriate override file and runs the
      profile set from `serve-model.yaml` through Podman Compose.

### Venue prerequisites

- **podman-compose must parse `--profile`.** Distro shims that spoof the
  embedded podman version (dgx-spark prints 4.9.3 but ships a 1.0.x CLI)
  fail at runtime; the script capability-gates with `up --help`, and on
  failure installs a vendored `podman-compose >= 1.2` into
  `$HOME/venvs/pc-all` (durable across jobs; PATH prefix picks it up).
  Fail-closed: no `--profile`-capable compose ⇒ the job exits before
  touching state.
- **Static host-publish ports** (4000 LiteLLM, 3001 vision): at most one
  mini-rivault per node — always submit with `--nodelist <node>` (site pool
  YAMLs already forward `nodelist` → `#SBATCH --nodelist=...`).
- **dgx-spark carries no CNI plugins** → `network_backend: netavark` in the
  partition profile; `venue.sh` honours it via
  `RIVAULT_PARTITION_NETWORK_BACKEND` (§6b-3).

The preferred orchestrator path for this branch is
`src/umrs/orchestrator/slurm_podman_backend.py`.

## Main Services

| Component | Purpose |
|---|---|
| vLLM | text/embedding model serving |
| LiteLLM | OpenAI-compatible API layer in front of vLLM |
| BentoML vision service | generic vision inference endpoint (`sam3`, `hmae`, custom adapters) |
| NGINX | front door for LiteLLM and `/vision/` routes |
| CoreDNS / support services | internal service discovery and runtime support |

## Important Environment Inputs

These are normally exported by the deploy wrapper or `site.yaml` bridge:

### sts-renewal profile (WP-11 round 3)

Add `sts-renewal` to `RIVAULT_PROFILES` to run the credential renewal
sidecar in the same network (`genai-private-net`):

- It POSTs `${UMRS_BASE_URL}/umrs/jobs/${RIVAULT_JOB_ID}/attempts/
  ${RIVAULT_ATTEMPT_ID}/credentials/sts` every
  `min(ttl/2, ttl-180)` (floor 60s, `RIVAULT_STS_TTL_SEC` default 3000).
- It atomically re-renders `./sts_fs/sts.env` (`S3_ENDPOINT_URL`,
  `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_SESSION_TOKEN`) — the
  ADR-043 fix for session-cap vs days-long serving (no provision-time
  env-freeze).
- Auth: `UMRS_AUTH_TOKEN` falls back to `R4_API_KEY`; the STS identity
  itself stays server-side (only the minted contract travels).
- `rivault_venue_write_container_configs` runs as usual; `sts_fs/` exists
  in-tree with `.gitkeep`.
- Repo-contract pins: `umrs/tests/test_sts_renewal.py`. Live flip remains
  an ops-day change (the pencil-marked leg, ADR-043 rollout block).

- `RIVAULT_DIR`
- `RIVAULT_PROFILES`
- `VLLM_PATH_TO_HUGGINGFACE`
- `VLLM_MODEL`
- `VLLM_DOCKER_TAG`
- `LITELLM_MASTER_KEY`
- `SAM3_MODEL_PATH`
- `HMAE_MODEL_PATH`
- `HMAE_CONFIG_PATH`
- `HMAE_CHECKPOINT`

## Direct Source Launch

For debugging on a machine that already has the required container runtime and
site configuration available:

```bash
cd src/umrs/mini-rivault-llm
export UMRS_SITE_CONFIG=/path/to/site.yaml
bash serve-model.slurm
```

The source launcher now honors `UMRS_SITE_CONFIG` via `site_config.py` instead
of relying only on generic fallback values.

## Notes

- This directory is the source of truth for the Mini-RiVault runtime assets.
- The deploy wrappers are expected to remain thin and site-specific.
- The vision service source code itself lives under
     `src/umrs/vision_service/`.
- Public deploy guidance lives under the site deployment docs in `deploy/`
      (for example `deploy/rccs-cloud/README.md`) and in
      `doc/admin_guide.md`.

## See Also

- [../../../deploy/rccs-cloud/README.md](../../../deploy/rccs-cloud/README.md)
- [../../../deploy/hpbd-atom/README.md](../../../deploy/hpbd-atom/README.md)
- [../../../doc/admin_guide.md](../../../doc/admin_guide.md)
- [../vision_service/service.py](../vision_service/service.py)
