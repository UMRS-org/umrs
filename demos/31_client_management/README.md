# Demo 31: Client Management & Utility Commands

This demo covers UMRS client commands that are **not** exercised by the
workload-focused demos (00–30). These are management, lifecycle, and
utility commands that every UMRS user should know.

---

## Commands Covered

| Command | Purpose | Prerequisite Demo |
|---------|---------|-------------------|
| `info` | Show connection setup, transport modes, auth, S3 status | None |
| `completion` | Generate shell completion scripts | None |
| `profile list` | List available connection profiles | None |
| `profile show` | Show effective connection resolved from profile/env/flags | None |
| `config list` | List discovered config candidates | None |
| `config show` | Show resolved config stack for server, user, and S3 | None |
| `experience add` | Merge experience fragment(s) from a JSON file | Demo 00 |
| `experience list` | List experiences in the current user config | None |
| `job list` | List durable jobs with optional filters | Demo 11 |
| `endpoint up` | Provision a reusable endpoint | None |
| `endpoint list` | List endpoint provision jobs | None |
| `endpoint status` | Show endpoint provision job details | None |
| `endpoint stop` | Stop an endpoint provision job | None |
| `workflow list` | List registered workflows | None |
| `workflow run` | Trigger execution of a registered workflow | Demo 16 |
| `workflow upload` | Upload a workflow bundle to the server | Demo 16 |
| `workflow status` | Show execution status for a workflow run | Demo 14 |
| `workflow errors` | Show error logs for a failed workflow execution | Demo 14 |
| `workflow visualize` | Render the workflow DAG as ASCII art | Demo 15 |
| `model delete` | Remove a registered model from the catalog | Demo 07 |
| `purge_remote_buckets` | Delete extra remote S3 buckets not in user config | Demo 06 |

---

## Prerequisites

- UMRS client installed: `which umrs-client`
- At least one connection profile configured (see Demo 00)
- UMRS server reachable for server-dependent commands

---

## Step-by-Step Tutorial

### Step 1: View Connection Info

The `info` command provides a quick overview of your current connection
setup, including transport modes, authentication status, and S3
configuration.

```bash
umrs-client info
```

**Expected output:**
```
  ● Server ready  |  profile=rccs-cloud  |  data0.ai.r-ccs.riken.jp:50090

  Transport:
    Control channel : ZMQ (tcp://data0.ai.r-ccs.riken.jp:50090)
    Data channel    : ZMQ (tcp://data0.ai.r-ccs.riken.jp:50091)
    HTTP gateway    : https://data0.ai.r-ccs.riken.jp/ai4s

  Authentication:
    Mode            : JWT
    Token expires   : 2026-12-31T23:59:59Z

  S3 / MinIO:
    Endpoint        : http://data0.ai.r-ccs.riken.jp:9000
    Access key      : mnagaso
    Buckets         : 3 configured
```

**Common issues:**
- `Server not reachable` → Check your SSH tunnel or network connectivity
- `No profile selected` → Run `umrs-client profile use <name>`

---

### Step 2: Generate Shell Completion

The `completion` command generates shell completion scripts for bash,
zsh, or fish. This enables tab-completion for all `umrs-client` commands.

```bash
umrs-client completion --shell zsh
```

**Expected output:**
```
#compdef umrs-client
# zsh completion for umrs-client
...
```

**Setup (zsh):**
```bash
umrs-client completion --shell zsh > ~/.zsh/completions/_umrs-client
# Or add to .zshrc:
# eval "$(umrs-client completion --shell zsh)"
```

**Setup (bash):**
```bash
umrs-client completion --shell bash > /etc/bash_completion.d/umrs-client
# Or:
# source <(umrs-client completion --shell bash)
```

---

### Step 3: List Connection Profiles

The `profile list` command shows all available connection profiles.

```bash
umrs-client profile list
```

**Expected output:**
```
  Profile         Mode         Server                          Status
  -------         ----         ------                          ------
  rccs-cloud      direct       data0.ai.r-ccs.riken.jp:50090   ✓ current
  ai4s            direct       data0.ai.r-ccs.riken.jp:50100   available
  local           direct       127.0.0.1:5555                  available
```

---

### Step 4: Show Effective Connection

The `profile show` command displays the effective connection that would
be used, resolving from profile, environment variables, and CLI flags.

```bash
umrs-client profile show
```

**Expected output:**
```
  Active profile: rccs-cloud
  Mode:           direct (ZMQ)
  Server host:    data0.ai.r-ccs.riken.jp
  Control port:   50090
  Data port:      50091
  HTTP gateway:   https://data0.ai.r-ccs.riken.jp/ai4s
  Tunnel mode:    false
```

---

### Step 5: List Config Candidates

The `config list` command lists all discovered configuration files for
server, user, and S3.

```bash
umrs-client config list
```

**Expected output:**
```
  Server configs:
    ~/.config/umrs/conf_umrs.json          (current)
    ~/deploy/ai4s-platform/conf_umrs.json  (available)

  User configs:
    ~/.config/umrs/users/user1.json        (current)
    ~/.config/umrs/users/demo_user.json    (available)

  S3 configs:
    ~/.config/umrs/s3cfg                   (current)
    ~/.config/umrs/s3cfg_demo_user         (available)
```

---

### Step 6: Show Resolved Config Stack

The `config show` command displays the resolved configuration stack,
showing which files are active and what settings they provide.

```bash
umrs-client config show
```

**Expected output:**
```
  Server config:  ~/.config/umrs/conf_umrs.json
    server_host:  data0.ai.r-ccs.riken.jp
    control_port: 50090
    data_port:    50091

  User config:    ~/.config/umrs/users/user1.json
    username:     user1
    experiences:  3 (text, sam3, vision)

  S3 config:      ~/.config/umrs/s3cfg
    host_base:    data0.ai.r-ccs.riken.jp:9000
    access_key:   mnagaso
```

---

### Step 7: Add an Experience

The `experience add` command merges experience fragment(s) from a JSON
file into the current user config. This is the standard way to add a
new workload experience.

```bash
# Create a sample experience fragment
cat > my_experience.json << 'EOF'
{
  "xp_name": "My Text Experience",
  "sql_username": "my_text",
  "modalities": [
    {
      "modality_name": "text",
      "task": "embedding",
      "s3": {
        "bucket_alias": "my-text",
        "need_output_bucket": true,
        "data": {"type": "file", "mime_type": "text/plain"}
      },
      "model": {
        "model_name": "intfloat/e5-base-v2",
        "vector_size": "768",
        "model_type": "embedding",
        "return_type": "file"
      },
      "sql": {
        "store_embeddings": "yes",
        "table_name": "my_text_emb",
        "meta_data": []
      }
    }
  ]
}
EOF

# Add the experience to your user config
umrs-client experience add my_experience.json --update
```

**Expected output:**
```
  ✓ Added experience(s): My Text Experience
```

**Common issues:**
- `Fragment not found` → Run from the directory containing the JSON file
- `Experience already exists` → Use `--update` to replace it

---

### Step 8: List Experiences

The `experience list` command shows all experiences in the current user
config.

```bash
umrs-client experience list
```

**Expected output:**
```
  Experiences in ~/.config/umrs/users/user1.json:

    Name                    Modality    Bucket Alias    Task
    ----                    --------    ------------    ----
    Text Embedding          text        user1-text      embedding
    SAM3 Segmentation       image       user1-sam3      segmentation
    My Text Experience      text        my-text         embedding
```

---

### Step 9: List Durable Jobs

The `job list` command lists durable jobs with optional filters for
status, type, and project.

```bash
# List all jobs
umrs-client job list

# List running jobs only
umrs-client job list --status running

# List endpoint_provision jobs
umrs-client job list --type endpoint_provision

# Limit to 10 results
umrs-client job list --limit 10
```

**Expected output:**
```
  Job ID          Type                  Status     Created
  ------          ----                  ------     -------
  job-abc123      endpoint_provision    running    2026-07-02T10:30:00Z
  job-def456      index_build           completed  2026-07-02T09:15:00Z
  job-ghi789      cleanup               planned    2026-07-02T11:00:00Z
```

---

### Step 10: Provision an Endpoint

The `endpoint up` command provisions a reusable endpoint (e.g., a vLLM
instance) that can be used for multiple inference requests.

```bash
# Provision an endpoint with a specific model
umrs-client endpoint up --model Qwen/Qwen2.5-7B-Instruct

# With custom idle timeout (default: 30m)
umrs-client endpoint up --model Qwen/Qwen2.5-7B-Instruct --idle-ttl 60m

# With GPU count specification
umrs-client endpoint up --model Qwen/Qwen2.5-7B-Instruct --gpu-count 1
```

**Expected output:**
```
  ✓ Endpoint provision queued as durable job job-xxx
  → Use 'umrs-client endpoint status job-xxx' to check progress
```

---

### Step 11: List Endpoint Provision Jobs

The `endpoint list` command lists all endpoint provision jobs.

```bash
umrs-client endpoint list
```

**Expected output:**
```
  Job ID          Model                       Status     GPUs
  ------          -----                       ------     ----
  job-xxx         Qwen/Qwen2.5-7B-Instruct    running    1
  job-yyy         e5-base-v2                  completed  0
```

---

### Step 12: Check Endpoint Status

The `endpoint status` command shows detailed status for a specific
endpoint provision job.

```bash
umrs-client endpoint status job-xxx
```

**Expected output:**
```
  Job ID:       job-xxx
  Model:        Qwen/Qwen2.5-7B-Instruct
  Status:       running
  Created:      2026-07-02T10:30:00Z
  Started:      2026-07-02T10:30:05Z
  Endpoint URL: http://compute-node:8000/v1
  GPUs:         1
  Idle TTL:     30m
```

---

### Step 13: Stop an Endpoint

The `endpoint stop` command stops a running endpoint provision job,
releasing the allocated resources.

```bash
umrs-client endpoint stop job-xxx
```

**Expected output:**
```
  ✓ Endpoint job-xxx stopped
  → Allocated resources have been released
```

---

### Step 14: List Registered Workflows

The `workflow list` command lists all registered workflows on the
server.

```bash
umrs-client workflow list
```

**Expected output:**
```
  Name                    Version    Status      Steps
  ----                    -------    ------      -----
  vision-pipeline-demo    1.0        registered  3
  video-motion-analysis   1.0        registered  1
  custom-text-counter     1.0        registered  2
```

---

### Step 15: Run a Registered Workflow

The `workflow run` command (also available as `run-workflow`) triggers
execution of a registered workflow on the server.

```bash
# Run a workflow with default settings
umrs-client workflow run vision-pipeline-demo

# Run with a specific input bucket
umrs-client workflow run vision-pipeline-demo --bucket user1-pipeline

# Run with additional parameters
umrs-client workflow run vision-pipeline-demo --params '{"max_images": 5}'
```

**Expected output:**
```
  ✓ Workflow 'vision-pipeline-demo' execution queued as durable job job-xxx
  → Use 'umrs-client job status job-xxx' to check progress
```

---

### Step 16: Delete a Model

The `model delete` command removes a registered model from the catalog.
This does not delete the model files on disk unless `--delete-files` is
specified.

```bash
# Delete a model (prompt for confirmation)
umrs-client model delete my-old-model

# Delete without confirmation
umrs-client model delete my-old-model --force

# Delete and remove files from disk
umrs-client model delete my-old-model --delete-files --force
```

**Expected output:**
```
  ✓ Model 'my-old-model' deleted from catalog
  → Model files retained on disk (use --delete-files to remove)
```

---

### Step 17: Purge Remote Buckets

The `purge_remote_buckets` command deletes extra remote S3 buckets that
are not declared in the user config. This is useful for cleaning up
orphaned buckets from deleted experiences.

```bash
# Preview which buckets would be purged (dry run)
umrs-client purge_remote_buckets --dry-run

# Purge all extra buckets
umrs-client purge_remote_buckets --yes

# Purge a specific bucket
umrs-client purge_remote_buckets --bucket orphaned-bucket --yes
```

**Expected output:**
```
  Scanning for extra remote S3 buckets...
  Found 2 extra bucket(s):
    - orphaned-bucket-1 (47 objects)
    - orphaned-bucket-2 (0 objects)

  ✓ Purged 2 bucket(s)
```

**Warning:** This operation is irreversible. Always use `--dry-run` first
to preview which buckets will be deleted.

---

## Command Quick Reference

```bash
# Connection management
umrs-client info                          # Show connection info
umrs-client profile list                  # List connection profiles
umrs-client profile show                  # Show effective connection
umrs-client profile use <name>            # Set active profile
umrs-client profile import <file>         # Import a profile JSON
umrs-client profile init --name <name>    # Create a new profile

# Config management
umrs-client config list                   # List config candidates
umrs-client config show                   # Show resolved config stack
umrs-client config use --user <file>      # Set active user config
umrs-client config init                   # Create new user config
umrs-client config add-experience <file>  # Add experience fragment
umrs-client config import <file>          # Import experience(s)

# Experience management
umrs-client experience add <file>         # Add experience fragment
umrs-client experience list               # List experiences
umrs-client experience remove <name>      # Remove experience
umrs-client experience recover <name>     # Recover experience JSON

# Job management
umrs-client job list                      # List durable jobs
umrs-client job submit <type>             # Submit a new job
umrs-client job status <id>               # Show job status
umrs-client job cancel <id>               # Cancel a job

# Endpoint management
umrs-client endpoint up --model <name>    # Provision an endpoint
umrs-client endpoint list                 # List endpoint jobs
umrs-client endpoint status <id>          # Show endpoint status
umrs-client endpoint stop <id>            # Stop an endpoint

# Workflow management
umrs-client workflow list                 # List registered workflows
umrs-client workflow run <name>           # Run a registered workflow
umrs-client workflow upload <dir>         # Upload a workflow bundle
umrs-client workflow status <exec-id>     # Show execution status
umrs-client workflow results <exec-id>    # Fetch results for a completed run
umrs-client workflow cancel <exec-id>     # Cancel a running execution
umrs-client workflow errors <exec-id>     # Show error logs for failed Attempts
umrs-client workflow visualize <name>     # Render the workflow DAG as ASCII art
umrs-client workflow validate-sandbox <dir>  # Validate Dockerfile sandbox

# Model management
umrs-client model list                    # List registered models
umrs-client model register <name> <path>  # Register a local model
umrs-client model download <repo>         # Download from HuggingFace
umrs-client model upload <path>           # Upload model to staging
umrs-client model delete <name>           # Delete a model

# Bucket management
umrs-client list_buckets                  # List bucket aliases
umrs-client purge_remote_buckets          # Delete extra S3 buckets

# Shell completion
umrs-client completion --shell <shell>    # Generate completion script
```

---

## Related Demos

- [Demo 00: User Onboarding](../00_onboarding/) — First-run setup and provisioning
- [Demo 06: Server Maintenance](../06_server_maintenance/) — `check`, `update`, `recompute`
- [Demo 07: Model Registry](../07_model_registry/) — Model registration and download
- [Demo 11: Job Submission](../11_job_submission/) — Job submit/cancel lifecycle
- [Demo 16: Custom Workflow](../16_custom_workflow/) — Upload and execute custom workflows
- [Demo 22: Monitor](../22_monitor/) — Live-refreshing TUI dashboard
- [Demo 25: Endpoint Provisioning](../25_endpoint_provisioning/) — REST API for endpoints
