#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Demo 06 — Structural Defect Analysis Pipeline
#
# Runs a 4-step UDM workflow:
#   1. SAM3 segmentation     → detect structures in materials images
#   2. Region extraction      → crop detected regions into patches
#   3. HMAE analysis          → feature embeddings + anomaly scoring
#   4. Structure discovery    → clustering + dataset-level report
#
# Prerequisites:
#   - Mini-Rivault stack running with vision-bentoml profile
#   - MinIO accessible at S3_ENDPOINT_URL
#   - UMRS package installed (pip install -e src/)
#
# Usage:
#   bash run_demo.sh                                   # defaults
#   PROMPT="crack" bash run_demo.sh                    # custom prompt
#   IMAGE_DIR=/path/to/images bash run_demo.sh         # custom images
#   BENTOML_URL=http://host:11434/vision bash run_demo.sh
# ──────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Configuration (override via environment) ──────────────────────────────
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://localhost:19000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"
BENTOML_URL="${BENTOML_URL:-http://localhost:11434/vision}"
BUCKET="${BUCKET:-defect-analysis}"
PROMPT="${PROMPT:-defect}"
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-0.3}"
N_CLUSTERS="${N_CLUSTERS:-4}"
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/test_data}"
RESULTS_DIR="${SCRIPT_DIR}/results"
WORKFLOW="${SCRIPT_DIR}/workflow.json"

# ── Colour helpers ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
step()  { echo -e "\n${CYAN}═══ $* ═══${NC}"; }

# ──────────────────────────────────────────────────────────────────────────
step "1/6  Preflight checks"
# ──────────────────────────────────────────────────────────────────────────

command -v python3 >/dev/null || fail "python3 not found"

# Check that UMRS is importable
python3 -c "from umrs.steps import UMRSStep" 2>/dev/null \
  || fail "umrs package not installed — run: pip install -e src/"

# Check BentoML health
info "Checking BentoML at ${BENTOML_URL} ..."
if curl -sf "${BENTOML_URL}/healthz" >/dev/null 2>&1; then
    ok "BentoML vision service healthy"
else
    warn "BentoML health check failed — SAM3/HMAE steps will fail"
    warn "Make sure Mini-Rivault is running with vision-bentoml profile"
fi

# Check S3/MinIO
info "Checking MinIO at ${S3_ENDPOINT_URL} ..."
if curl -sf "${S3_ENDPOINT_URL}/minio/health/live" >/dev/null 2>&1; then
    ok "MinIO healthy"
else
    warn "MinIO health check failed at ${S3_ENDPOINT_URL}"
fi

# ──────────────────────────────────────────────────────────────────────────
step "2/6  Prepare test data"
# ──────────────────────────────────────────────────────────────────────────

# Find images in the test data directory
shopt -s nullglob
images=("${IMAGE_DIR}"/*.{png,jpg,jpeg,tif,tiff,PNG,JPG})
shopt -u nullglob

if [[ ${#images[@]} -eq 0 ]]; then
    warn "No images found in ${IMAGE_DIR}"
    info "Generating synthetic test images for demo ..."
    python3 - <<'PYEOF'
import os, sys
import numpy as np
from PIL import Image, ImageDraw

out_dir = os.environ.get("IMAGE_DIR", "test_data")
os.makedirs(out_dir, exist_ok=True)

rng = np.random.RandomState(42)

for i in range(3):
    # Create a 512x512 grayscale image with synthetic structures
    bg = rng.randint(100, 160, size=(512, 512), dtype=np.uint8)
    img = Image.fromarray(bg, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    # Add "normal" grain boundaries (thin lines)
    for _ in range(8):
        x1, y1 = rng.randint(0, 512, size=2)
        x2, y2 = rng.randint(0, 512, size=2)
        draw.line([(x1, y1), (x2, y2)], fill=(80, 80, 80), width=2)

    # Add "pores" (dark circles)
    for _ in range(3):
        cx, cy = rng.randint(50, 462, size=2)
        r = rng.randint(10, 30)
        draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)], fill=(40, 40, 40))

    # Add "cracks" on some images (bright jagged lines)
    if i >= 1:
        pts = [(rng.randint(0, 512), rng.randint(0, 512)) for _ in range(5)]
        draw.line(pts, fill=(220, 220, 220), width=3)

    path = os.path.join(out_dir, f"sample_{i:03d}.png")
    img.save(path)
    print(f"  Created: {path}")
PYEOF
    images=("${IMAGE_DIR}"/*.png)
fi

info "Found ${#images[@]} images in ${IMAGE_DIR}"

# ──────────────────────────────────────────────────────────────────────────
step "3/6  Upload images to S3"
# ──────────────────────────────────────────────────────────────────────────

python3 - "${BUCKET}" "${S3_ENDPOINT_URL}" "${S3_ACCESS_KEY}" "${S3_SECRET_KEY}" "${images[@]}" <<'PYEOF'
import sys, os
import boto3

bucket = sys.argv[1]
endpoint = sys.argv[2]
access_key = sys.argv[3]
secret_key = sys.argv[4]
files = sys.argv[5:]

s3 = boto3.client("s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)

# Create bucket if needed
try:
    s3.head_bucket(Bucket=bucket)
except:
    s3.create_bucket(Bucket=bucket)
    print(f"  Created bucket: {bucket}")

for f in files:
    key = os.path.basename(f)
    s3.upload_file(f, bucket, key)
    print(f"  Uploaded: {key}")
PYEOF

ok "Images uploaded to s3://${BUCKET}/"

# ──────────────────────────────────────────────────────────────────────────
step "4/6  Run defect analysis workflow"
# ──────────────────────────────────────────────────────────────────────────

# Build the s3 keys list
S3_KEYS=""
for img in "${images[@]}"; do
    S3_KEYS="${S3_KEYS}$(basename "$img"),"
done
S3_KEYS="${S3_KEYS%,}"  # strip trailing comma

info "Running 4-step workflow:"
info "  1. SAM3 segmentation  (prompt=${PROMPT})"
info "  2. Region extraction"
info "  3. HMAE analysis      (features + anomaly)"
info "  4. Structure discovery (${N_CLUSTERS} clusters)"
echo ""

python3 - <<PYEOF
import sys, os, json, logging

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from umrs.steps.loader import ScriptLoader
from umrs.steps.executor import WorkflowExecutor

# Load and patch workflow config with runtime values
workflow_path = "${WORKFLOW}"
with open(workflow_path) as f:
    wf_dict = json.load(f)

# Override config values from environment
for step in wf_dict["steps"]:
    cfg = step["config"]
    if "bentoml_url" in cfg:
        cfg["bentoml_url"] = "${BENTOML_URL}"
    if "prompt" in cfg:
        cfg["prompt"] = "${PROMPT}"
    if "anomaly_threshold" in cfg:
        cfg["anomaly_threshold"] = float("${ANOMALY_THRESHOLD}")
    if "n_clusters" in cfg:
        cfg["n_clusters"] = int("${N_CLUSTERS}")

# Write patched workflow to temp file
import tempfile
patched = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
json.dump(wf_dict, patched, indent=2)
patched.close()

# Load workflow + steps
wf_def = ScriptLoader.load_workflow(patched.name)
steps = ScriptLoader.load_steps(wf_def)
executor = WorkflowExecutor(wf_def, steps)

# Create S3 client
import boto3
s3 = boto3.client("s3",
    endpoint_url="${S3_ENDPOINT_URL}",
    aws_access_key_id="${S3_ACCESS_KEY}",
    aws_secret_access_key="${S3_SECRET_KEY}",
)

# Run
s3_keys = "${S3_KEYS}".split(",")
print(f"\nProcessing {len(s3_keys)} images ...\n")

embeddings, checksums, metadata, saved_keys = executor.run(
    s3, s3_keys, "${BUCKET}", "${BUCKET}",
)

print(f"\n{'='*60}")
print(f"Workflow complete!")
print(f"  Images processed: {len(s3_keys)}")
print(f"  Embeddings: {len([e for e in embeddings if e is not None])}")
print(f"  Output keys: {saved_keys}")
print(f"{'='*60}")

os.unlink(patched.name)
PYEOF

ok "Workflow complete"

# ──────────────────────────────────────────────────────────────────────────
step "5/6  Download results"
# ──────────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_DIR}"

python3 - "${BUCKET}" "${S3_ENDPOINT_URL}" "${S3_ACCESS_KEY}" "${S3_SECRET_KEY}" "${RESULTS_DIR}" <<'PYEOF'
import sys, os
import boto3

bucket = sys.argv[1]
endpoint = sys.argv[2]
access_key = sys.argv[3]
secret_key = sys.argv[4]
results_dir = sys.argv[5]

s3 = boto3.client("s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)

# Download all analysis results
paginator = s3.get_paginator("list_objects_v2")
count = 0
for prefix in ["analysis/", "patches/", "embeddings/", "anomaly/"]:
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            dest = os.path.join(results_dir, key)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)
            count += 1
            print(f"  Downloaded: {key}")

# Also download overlay images
for page in paginator.paginate(Bucket=bucket):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if "_sam3_seg" in key or "_recon" in key or "_mae" in key:
            dest = os.path.join(results_dir, key)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)
            count += 1
            print(f"  Downloaded: {key}")

print(f"\nTotal files downloaded: {count}")
PYEOF

ok "Results saved to ${RESULTS_DIR}/"

# ──────────────────────────────────────────────────────────────────────────
step "6/6  Summary"
# ──────────────────────────────────────────────────────────────────────────

if [[ -f "${RESULTS_DIR}/analysis/report.json" ]]; then
    echo ""
    info "Analysis report:"
    python3 -c "
import json
with open('${RESULTS_DIR}/analysis/report.json') as f:
    r = json.load(f)
print(f'  Total patches analysed: {r.get(\"total_patches\", \"?\")}')
print(f'  Clusters: {r.get(\"n_clusters\", \"?\")}')
cs = r.get('cluster_summary', {})
for name, s in cs.items():
    anom = s.get('anomaly_count', 0)
    label = ' ⚠️' if anom > 0 else ''
    print(f'    {name}: {s[\"count\"]} patches, {anom} anomalies{label}')
"
    echo ""
fi

ok "Demo 06 complete. Results in: ${RESULTS_DIR}/"
info "Key outputs:"
info "  ${RESULTS_DIR}/analysis/report.json       — full analysis report"
info "  ${RESULTS_DIR}/analysis/all_embeddings.npy — N×768 embedding matrix"
info "  ${RESULTS_DIR}/analysis/projection_2d.npy  — 2-D UMAP/PCA projection"
info "  ${RESULTS_DIR}/patches/                    — cropped region patches"
info "  ${RESULTS_DIR}/*_sam3_seg.png              — segmentation overlays"
