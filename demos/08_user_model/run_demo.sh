#!/usr/bin/env bash
# -----------------------------------------------------------------------
# Demo 08 — User-Defined Model Workflow (offline / unit-test mode)
#
# This script exercises the UDM subsystem *without* a running UMRS
# server or MinIO — it uses a mock S3 and runs the executor directly.
# -----------------------------------------------------------------------
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Use the repo venv if available, else system Python
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

echo "=== Demo 08: User-Defined Model Workflow ==="
echo "Python: $PYTHON"
echo ""

# Run the built-in test suite for the steps package
echo "--- Running unit tests ---"
"$PYTHON" -m pytest "${REPO_ROOT}/tests/test_user_steps.py" -v --tb=short
echo ""

# Run the inline integration smoke test
echo "--- Running integration smoke test ---"
"$PYTHON" - <<'PYEOF'
import json, sys, os

# Adjust path so we can import from src/
sys.path.insert(0, os.path.join(os.environ.get("PYTHONPATH", "").split(":")[0]))

from pathlib import Path
from unittest.mock import MagicMock

from umrs.pipeline_sdk.workflow import load_workflow, load_steps, WorkflowDefinition, WorkflowPipeline

# ── Locate demo artefacts ───────────────────────────────────────────
demo_dir = Path(__file__).resolve().parent if "__file__" in dir() else Path(os.environ.get("SCRIPT_DIR", "."))
demo_dir = Path(os.environ.get("SCRIPT_DIR", os.path.dirname(os.path.abspath("."))))

# Use the repo-relative path
repo = Path(os.environ["PYTHONPATH"].split(":")[0]).parent
scripts_dir = repo / "demos" / "08_user_model" / "example_scripts"
wf_path = repo / "demos" / "08_user_model" / "workflow_text.json"

# Patch workflow source path to point at the actual scripts
with open(wf_path) as f:
    wf_data = json.load(f)
wf_data["source"]["path"] = str(scripts_dir)
wf = WorkflowDefinition.from_dict(wf_data)
wf.validate()

# ── Mock S3 ─────────────────────────────────────────────────────────
store = {}

def put_object(Bucket, Key, Body, **kw):
    store[f"{Bucket}/{Key}"] = Body if isinstance(Body, bytes) else Body.encode()

def get_object(Bucket, Key):
    data = store[f"{Bucket}/{Key}"]
    body = MagicMock()
    body.read.return_value = data
    return {"Body": body, "ETag": '"aaa"', "ContentLength": len(data), "Metadata": {}}

def head_object(Bucket, Key):
    data = store[f"{Bucket}/{Key}"]
    return {"ETag": '"aaa"', "ContentLength": len(data), "Metadata": {}}

s3 = MagicMock()
s3.put_object = MagicMock(side_effect=put_object)
s3.get_object = MagicMock(side_effect=get_object)
s3.head_object = MagicMock(side_effect=head_object)

# Seed input data
store["demo-in/hello.txt"] = b"Hello World!\nThis is a test file.\n"
store["demo-in/readme.txt"] = b"UMRS user-defined workflows rock.\n"

# ── Load steps & run ────────────────────────────────────────────────
steps = load_steps(wf)

# Build a WorkflowPipeline from the pre-patched workflow + loaded steps
pipeline = WorkflowPipeline.from_workflow(wf, steps)

embs, checksums, metadata, saved_keys = pipeline.run_as_embeddings(
    s3, ["hello.txt", "readme.txt"], "demo-in", "demo-out"
)

print(f"  Input keys:  ['hello.txt', 'readme.txt']")
print(f"  Saved keys:  {saved_keys}")
print(f"  Checksums:   {checksums}")
print()

# Verify outputs exist
for sk in saved_keys:
    if sk:
        full_key = f"demo-out/{sk}"
        assert full_key in store, f"Missing output: {full_key}"
        result = json.loads(store[full_key])
        print(f"  {sk}: {result}")

print()
print("Integration smoke test PASSED ✓")
PYEOF

echo ""
echo "=== Demo 08 complete ==="
