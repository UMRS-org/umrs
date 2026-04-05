#!/usr/bin/env bash
# =============================================================================
# Demo 21: Remote Plugin Execution (Local Simulation)
# =============================================================================
# Demonstrates the plugin system with different source types and
# execution backends. Runs locally — no server required.
#
# Shows:
#   1. PluginSpec definition (git repo, S3/wheel, local path)
#   2. Plugin validation
#   3. Plugin execution simulation
#   4. Audit log of plugin operations
#
# Prerequisites:
#   - Python 3.10+ with umrs package installed (pip install -e src/)
#   - No server required (local simulation)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="plugins" source "${SCRIPT_DIR}/../lib/common.sh"

RESULTS_DIR="${SCRIPT_DIR}/results"

banner "Demo 21: Remote Plugin Execution"
echo "This demo runs locally — no server required."
echo ""

# ---------- Step 1: Prepare plugin definitions ----------
log "Step 1: Preparing plugin definitions..."
mkdir -p "$RESULTS_DIR"

python3 - "$RESULTS_DIR" << 'PYEOF'
import json, sys, time
from pathlib import Path

results_dir = Path(sys.argv[1])

# ── 1. Define Plugin Specs ──────────────────────────────────
print("=== Plugin Spec Definitions ===\n")

plugins = [
    {
        "name": "image-embedder",
        "version": "1.2.0",
        "source_type": "git_repo",
        "source_uri": "https://github.com/example/image-embedder.git",
        "source_ref": "v1.2.0",
        "entrypoint": "image_embedder:ImageEmbedderPlugin",
        "config": {"model": "clip-vit-base", "batch_size": 32},
        "description": "Compute CLIP embeddings for uploaded images",
    },
    {
        "name": "pdf-parser",
        "version": "0.9.0",
        "source_type": "s3_wheel",
        "source_uri": "s3://plugins/pdf_parser-0.9.0-py3-none-any.whl",
        "entrypoint": "pdf_parser:PDFParserPlugin",
        "config": {"extract_images": True, "ocr_enabled": False},
        "description": "Extract text and tables from PDF documents",
    },
    {
        "name": "local-analyzer",
        "version": "0.1.0",
        "source_type": "local_path",
        "source_uri": "/opt/plugins/local_analyzer",
        "entrypoint": "analyzer:AnalyzerPlugin",
        "config": {"threshold": 0.75},
        "description": "Analyze data patterns (local development plugin)",
    },
]

for p in plugins:
    print(f"  Plugin: {p['name']} v{p['version']}")
    print(f"    Source: {p['source_type']} → {p['source_uri']}")
    print(f"    Entry:  {p['entrypoint']}")
    print(f"    Desc:   {p['description']}")
    print()

# Save specs
(results_dir / "plugin_specs.json").write_text(json.dumps(plugins, indent=2))

# ── 2. Validate Plugin Specs ────────────────────────────────
print("=== Plugin Validation ===\n")

audit_log = []

def audit(action, plugin_name, status, detail=""):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": action,
        "plugin": plugin_name,
        "status": status,
        "detail": detail,
    }
    audit_log.append(entry)
    icon = "✓" if status == "ok" else "✗" if status == "error" else "⚠"
    print(f"  {icon} [{action}] {plugin_name}: {status}" + (f" — {detail}" if detail else ""))

for p in plugins:
    name = p["name"]

    # Check required fields
    required = ["name", "version", "source_type", "source_uri", "entrypoint"]
    missing = [f for f in required if not p.get(f)]
    if missing:
        audit("validate", name, "error", f"missing fields: {missing}")
        continue

    # Check source type
    valid_sources = ["git_repo", "s3_wheel", "local_path"]
    if p["source_type"] not in valid_sources:
        audit("validate", name, "error", f"invalid source_type: {p['source_type']}")
        continue

    # Check entrypoint format
    if ":" not in p["entrypoint"]:
        audit("validate", name, "error", f"entrypoint must be module:ClassName")
        continue

    audit("validate", name, "ok", f"source={p['source_type']}")

# ── 3. Simulate Plugin Execution ────────────────────────────
print("\n=== Plugin Execution (Simulated) ===\n")

for p in plugins:
    name = p["name"]
    source = p["source_type"]

    audit("fetch", name, "ok", f"fetched from {source}")

    # Simulate execution based on source type
    if source == "git_repo":
        audit("install", name, "ok", "cloned and installed via pip")
    elif source == "s3_wheel":
        audit("install", name, "ok", "downloaded wheel and installed")
    elif source == "local_path":
        audit("install", name, "ok", "loaded from local path")

    # Simulate running the plugin
    audit("execute", name, "ok", f"config={json.dumps(p.get('config', {}))}")

    # Simulate result
    result = {
        "plugin": name,
        "status": "completed",
        "outputs": [f"{name}_output.json"],
        "duration_ms": 150 + hash(name) % 500,
    }
    audit("complete", name, "ok", f"outputs={result['outputs']}")

# ── 4. Write Audit Log ──────────────────────────────────────
print("\n=== Audit Log Summary ===\n")

(results_dir / "audit_log.json").write_text(json.dumps(audit_log, indent=2))

total = len(audit_log)
ok_count = sum(1 for e in audit_log if e["status"] == "ok")
err_count = sum(1 for e in audit_log if e["status"] == "error")
print(f"  Total entries : {total}")
print(f"  Successful    : {ok_count}")
print(f"  Errors        : {err_count}")
print(f"  Audit log     : {results_dir / 'audit_log.json'}")

# ── 5. Plugin Lifecycle Summary ──────────────────────────────
print("\n=== Plugin Lifecycle Summary ===")
print("  validate → fetch → install → execute → complete")
print("  Each stage is logged with timestamp and status")
print("  Failed plugins are quarantined and reported")
PYEOF

ok "Plugin simulation complete"

log ""
log "Results saved to: ${RESULTS_DIR}/"

banner "Demo 21 Complete"
