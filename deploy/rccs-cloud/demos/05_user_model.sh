#!/usr/bin/env bash
# =============================================================================
# User-Defined Model Demo — rccs-cloud deployment
# =============================================================================
# Demonstrates running a custom user Step against data already on the server.
# The step reads text files from S3, computes word/char/line stats, and writes
# JSON result files back to S3 — all without modifying the UMRS server code.
#
# Prerequisites:
#   - SSH tunnels running (ssh_tunnel.sh)
#   - umrs package installed (setup_client.sh)
#   - Text files uploaded (01_text_embedding.sh)
#
# The workflow runs locally using the pipeline_sdk with a mock S3 connector
# pointing at the live MinIO instance.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
USER_CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
INPUT_BUCKET="${UMRS_TEXT_BUCKET:-user1-texts}"

log()  { echo "[user_model] $*"; }
warn() { echo "[user_model] WARNING: $*"; }
die()  { echo "[user_model] ERROR: $*" >&2; exit 1; }

CLIENT="umrs-client -c $USER_CONF"

# ─── Preflight ───────────────────────────────────────────────────────────────
[[ -f "$USER_CONF" ]] || die "User config not found. Run setup_client.sh first."

python3 -c "from umrs.pipeline_sdk.workflow import WorkflowDefinition; print('Pipeline SDK OK')" 2>/dev/null || \
    die "umrs package not importable. Run setup_client.sh first."

echo
log "Step 1: Ping server"
$CLIENT ping || die "Server not reachable. Is ssh_tunnel.sh running?"

# ─── Step 2: Run the user step locally against live S3 ───────────────────────
echo
log "Step 2: Run TextStatsStep against bucket '$INPUT_BUCKET' on MinIO"
log "  (reads files from S3, computes stats, writes JSON results back to S3)"

python3 - <<PYEOF
import json, sys, os
sys.path.insert(0, "$REPO_ROOT/src")

import boto3
from botocore.client import Config
from umrs.pipeline_sdk.workflow import WorkflowDefinition, WorkflowPipeline, load_steps

# ── Connect to MinIO via SSH tunnel ─────────────────────────────────
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

# ── Discover objects in bucket ──────────────────────────────────────
bucket = "$INPUT_BUCKET"
try:
    resp = s3.list_objects_v2(Bucket=bucket)
except Exception as e:
    print(f"ERROR: Cannot list bucket '{bucket}': {e}")
    print("Make sure you have uploaded text files first (01_text_embedding.sh).")
    sys.exit(1)

objects = [o["Key"] for o in resp.get("Contents", [])]
if not objects:
    print(f"No objects found in bucket '{bucket}'. Run 01_text_embedding.sh first.")
    sys.exit(1)

print(f"  Found {len(objects)} object(s) in '{bucket}':")
for k in objects[:5]:
    print(f"    {k}")
if len(objects) > 5:
    print(f"    ... and {len(objects) - 5} more")
print()

# ── Build workflow: text_stats step ─────────────────────────────────
scripts_dir = "$REPO_ROOT/demos/07_user_model/example_scripts"
wf_data = {
    "name": "text-stats-demo",
    "version": "1.0",
    "description": "Compute text statistics via user-defined step",
    "source": {"type": "local_path", "path": scripts_dir},
    "requirements": [],
    "resources": {"gpu_count": 0, "cpu_limit": "1", "memory_limit": "512Mi", "timeout_seconds": 120},
    "steps": [{"name": "text_stats", "entrypoint": "text_stats:TextStatsStep", "config": {"output_prefix": "stats/"}}],
    "output": {"return_type": "file", "embedding_vector_size": 0},
}
wf = WorkflowDefinition.from_dict(wf_data)
wf.validate()
steps = load_steps(wf)
pipeline = WorkflowPipeline.from_workflow(wf, steps)

# ── Run ─────────────────────────────────────────────────────────────
print("Running TextStatsStep ...")
embs, checksums, metadata, saved_keys = pipeline.run_as_embeddings(
    s3, objects[:10], bucket, bucket  # input and output in same bucket
)

print()
print(f"  Processed {len(saved_keys)} file(s).")

# Print a sample result
for key in saved_keys[:3]:
    if key:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            stats = json.loads(obj["Body"].read())
            src = stats.get("source_key", key)
            print(f"  {src}: {stats['words']} words, {stats['characters']} chars, {stats['lines']} lines")
        except Exception:
            pass
PYEOF

echo
log "Done! Stats JSON files written to the 'stats/' prefix in bucket '$INPUT_BUCKET'."
log "Browse them: umrs-client -c $USER_CONF --output text list_buckets"
