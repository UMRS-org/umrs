# Demo 00: User Onboarding & Initialization

Creates the `demo_user` account, pushes it to the server to trigger
provisioning, and verifies the new buckets are visible.

See [../README.md](../README.md) for shared setup, connection profiles, and
SSH tunnel guidance.

## What it does

1. **Login** — authenticates and auto-creates the connection profile and user
   config in one step (idempotent once the profile exists)
2. **Show active profile** — confirms the connection profile is set
3. **Pre-flight check** — verifies `umrs-client` can reach the server
4. **Onboard** — calls `reload` to trigger server-side provisioning:
   - MinIO IAM user + per-bucket access policy
   - (No additional experiences yet — subsequent demos add their own)
5. **Confirm** — shows provisioned status and bucket list

## Usage

```bash
# Navigate from either repo level (auto-detects parent vs subdirectory)
cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/00_onboarding"
# See the numbered steps (Step 1, Step 2, etc.) below in this README
```

## Prerequisites

- `umrs-client` installed and in PATH
- UMRS server running and reachable
- Admin has enrolled `demo_user` server-side (see Step 1 below)

---

## Step 1 (Admin): Enroll the User

Run once on the server node before the user can onboard.

```bash
# Start infrastructure (postgres + minio + umrs-server)
cd deploy/rccs-cloud
./start.sh

# Bootstrap the admin CLI (first time only — token shown once)
umrs-server admin bootstrap --admin-user admin --json

# Enroll demo_user
# Run from the deploy directory — conf_umrs.json is discovered automatically
umrs-server admin user add demo_user \
    --fullname "Demo User" \
    --role user \
    --slurm-account rccs-dmp --site rccs-front \
    --issue-token --token-days 90
```

The printed JWT is given to the user. The admin record is written to
`user_files/demo_user.json` and the user is registered in `conf_umrs.json`.

**Note:** No `--config conf_umrs.json` flag is needed when running from the
deploy directory. The admin CLI auto-discovers `conf_umrs.json` from the
current working directory.

### Manage enrolled users

```bash
umrs-server admin user list
umrs-server admin user show demo_user
umrs-server admin user disable demo_user
umrs-server admin user delete demo_user --purge -y
```

---

## Step 2 (User): Client-Side Setup

`umrs-client login` handles all client-side setup in a single command — it
creates the connection profile **and** the user config automatically.

```bash
# Authenticate: creates connection profile + user config in one step
umrs-client login --host <server-host> --api-key <token-from-admin>

# Trigger provisioning (idempotent)
umrs-client reload

# Confirm buckets are visible
umrs-client bucket list
```

> **Note:** `umrs-client login` automatically downloads and imports your user config
> from the server — no need to run `umrs-client config import` separately.

### Experience recovery and removal

If you remove an experience from your user JSON and reload, UMRS stops routing to
that experience but keeps the MinIO buckets and Postgres/pgvector data. Use this
for reversible cleanup:

```bash
umrs-client experience remove <xp_name>
umrs-client reload
```

If you need to rebuild the JSON stanza from existing storage state:

```bash
umrs-client experience recover <xp_name> \
   --username demo_user \
   --s3cfg ~/.config/umrs/s3cfg_demo_user \
   --output-path recovered-experience.json
```

Review the recovered model fields before reload. MinIO/Postgres can identify the
bucket, table, and vector shape, but may not prove the original model name.

### Available experiences

Each demo ships with an `experience.json` fragment that registers a
modality + model + bucket + DB table. Load them in order as your
workload grows:

| Demo | Experience name | Modality | Model | Bucket alias |
|------|-----------------|----------|-------|--------------|
| [02](../02_text_embedding/) | Text Embedding | text | `intfloat/e5-base-v2` (768-d) | `demo-texts` / `demo-textfiles` |
| [03](../03_sam3_segmentation/) | SAM3 Segmentation | image | `sam3.1` | `demo-sam3` |
| [04](../04_vision_inference/) | Vision Inference | image | `sam3.1` | `demo-images` |
| [09](../09_multistep_vision_pipeline/) | Vision Pipeline | image | `facebook/sam2-hiera-small` | `demo-pipeline` |
| [10](../10_vision_multi_mode/) | Vision Multi Mode | image | `facebook/sam2-hiera-small` | `demo-vision-multi` |
| [14](../14_video_analysis/) | Video Analysis | movie | `opencv-motion-analysis` | `demo-video` |
| [16](../16_custom_workflow/) | Custom Workflow | text | `custom-workflow` | `demo-custom-text` |

Load any of these before running the corresponding demo:

```bash
# Example: load text embedding experience before running Demo 02
umrs-client experience add ../02_text_embedding/experience.json
umrs-client reload

# Example: load SAM3 segmentation experience before running Demo 03
umrs-client experience add ../03_sam3_segmentation/experience.json
umrs-client reload
```

### SSH tunnel (remote HPC)

If the server is on a remote cluster, open an SSH tunnel first:

```bash
ssh -N -L 5555:localhost:5555 -L 5556:localhost:5556 rccs-cloud &

umrs-client login --host 127.0.0.1 --mode ssh_tunnel --api-key <token>
```

### Advanced: manual profile and config setup

If you need more control than `login` provides:

```bash
# Profile only (no login)
umrs-client profile init --name local --mode direct
umrs-client profile init --name rccs --mode ssh_tunnel --ssh-host rccs-cloud

# Config only (no login)
umrs-client config init --username demo_user --fullname "Demo User"
```

---

## What `reload` Provisions

All steps are **idempotent** — re-running with an unchanged config is a no-op.

| Resource | What is created |
|----------|-----------------|
| MinIO IAM user | One per username; random password written to `~/.config/umrs/s3cfg_demo_user` |
| S3 buckets | One per modality (`demo_user/<xp_name>/<alias>`); output bucket where declared |
| IAM policies | Per-bucket `allow-<bucket>` policy + base `list-only` policy attached to user |
| Postgres role | One per `sql_username` in the user config's `experiences` array |
| Postgres database | One per experience with `pgvector` extension enabled |
| Embedding table | One per modality with columns matching the model's vector size |

---

## Expected Result

After onboarding, `list_buckets` shows the aliases from the user config.
Run any numbered demo (02, 03, 04 …) — each adds its own experience.
Run `99_cleanup` to tear everything down when finished.
