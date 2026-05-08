#!/usr/bin/env bash
# =============================================================================
# Demo 15: Complex DAG Workflow
# =============================================================================
# Demonstrates a multi-step pipeline with parallel branches and fan-in
# using the local Pipeline SDK and DAGScheduler.
#
# Prerequisites:
#   - Python 3.10+ with umrs package installed (pip install -e src/)
#   - No server required (runs locally)
#
# DAG Structure:
#   [load_data]
#       │
#   ┌───┴───┐
#   │       │
#   [analyze_text]  [analyze_metadata]
#   │       │
#   └───┬───┘
#       │
#   [merge_results]
#       │
#   [generate_report]
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_TAG="complex_dag" source "${SCRIPT_DIR}/../lib/common.sh"

RESULTS_DIR="${SCRIPT_DIR}/results"
STEPS_DIR="${SCRIPT_DIR}/steps"

banner "Demo 15: Complex DAG Workflow"
echo "This demo runs locally — no server required."
echo ""

# ---------- Step 1: Prepare step scripts ----------
log "Step 1: Preparing step scripts..."
mkdir -p "$STEPS_DIR" "$RESULTS_DIR"

cat > "${STEPS_DIR}/steps.py" << 'PYEOF'
"""Step implementations for the complex DAG demo."""

class LoadData:
    name = "load_data"
    def setup(self, config): self.config = config
    def teardown(self): pass
    def run(self, ctx):
        data = {"texts": ["hello world", "complex analysis"], "meta": {"source": "demo"}}
        ctx.metadata["loaded_data"] = data
        return type("R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": data})()

class AnalyzeText:
    name = "analyze_text"
    def setup(self, config): self.config = config
    def teardown(self): pass
    def run(self, ctx):
        data = ctx.metadata.get("loaded_data", {})
        texts = data.get("texts", [])
        analysis = {"word_count": sum(len(t.split()) for t in texts), "doc_count": len(texts)}
        ctx.metadata["text_analysis"] = analysis
        return type("R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": analysis})()

class AnalyzeMetadata:
    name = "analyze_metadata"
    def setup(self, config): self.config = config
    def teardown(self): pass
    def run(self, ctx):
        data = ctx.metadata.get("loaded_data", {})
        meta = data.get("meta", {})
        result = {"field_count": len(meta), "source": meta.get("source", "unknown")}
        ctx.metadata["meta_analysis"] = result
        return type("R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": result})()

class MergeResults:
    name = "merge_results"
    def setup(self, config): self.config = config
    def teardown(self): pass
    def run(self, ctx):
        merged = {
            "text": ctx.metadata.get("text_analysis", {}),
            "meta": ctx.metadata.get("meta_analysis", {}),
        }
        ctx.metadata["merged"] = merged
        return type("R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": merged})()

class GenerateReport:
    name = "generate_report"
    def setup(self, config): self.config = config
    def teardown(self): pass
    def run(self, ctx):
        merged = ctx.metadata.get("merged", {})
        report = f"=== Analysis Report ===\nText: {merged.get('text', {})}\nMeta: {merged.get('meta', {})}\n"
        ctx.metadata["report"] = report
        return type("R", (), {"output_keys": [], "embeddings": None, "checksums": {}, "metadata": {"report": report}})()
PYEOF
ok "Step scripts created"

# ---------- Step 2: Create workflow.json ----------
log ""
log "Step 2: Creating workflow.json with DAG dependencies..."

cat > "${STEPS_DIR}/workflow.json" << 'JSON_EOF'
{
  "name": "complex-dag-demo",
  "version": "1.0",
  "source": {
    "type": "local_path",
    "path": "."
  },
  "execution_mode": "in_process",
  "steps": [
    {"name": "load_data",        "entrypoint": "steps:LoadData",        "config": {},  "depends_on": []},
    {"name": "analyze_text",     "entrypoint": "steps:AnalyzeText",     "config": {},  "depends_on": ["load_data"]},
    {"name": "analyze_metadata", "entrypoint": "steps:AnalyzeMetadata", "config": {},  "depends_on": ["load_data"]},
    {"name": "merge_results",    "entrypoint": "steps:MergeResults",    "config": {},  "depends_on": ["analyze_text", "analyze_metadata"]},
    {"name": "generate_report",  "entrypoint": "steps:GenerateReport",  "config": {},  "depends_on": ["merge_results"]}
  ],
  "output": {
    "return_type": "file"
  }
}
JSON_EOF
ok "workflow.json created"

# ---------- Step 3: Run pipeline via Python ----------
log ""
log "Step 3: Executing DAG pipeline..."

python3 - "$STEPS_DIR" "$RESULTS_DIR" << 'PYEOF'
import sys, json, importlib.util
from pathlib import Path

steps_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])

# Load workflow definition
wf = json.loads((steps_dir / "workflow.json").read_text())

# Load step module
spec = importlib.util.spec_from_file_location("steps", steps_dir / "steps.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Build step instances
step_classes = {
    "load_data": mod.LoadData,
    "analyze_text": mod.AnalyzeText,
    "analyze_metadata": mod.AnalyzeMetadata,
    "merge_results": mod.MergeResults,
    "generate_report": mod.GenerateReport,
}

# Simple DAG execution (mirrors DAGScheduler logic)
from collections import defaultdict, deque

specs = wf["steps"]
name_to_idx = {s["name"]: i for i, s in enumerate(specs)}
adj = defaultdict(list)
in_deg = [0] * len(specs)

for i, s in enumerate(specs):
    for dep in s.get("depends_on", []):
        j = name_to_idx[dep]
        adj[j].append(i)
        in_deg[i] += 1

# Topological sort into levels
queue = deque(i for i, d in enumerate(in_deg) if d == 0)
levels = []
while queue:
    level = list(queue)
    levels.append(level)
    next_q = deque()
    for node in level:
        for neighbor in adj[node]:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                next_q.append(neighbor)
    queue = next_q

print(f"DAG has {len(levels)} execution levels:")
for lvl_num, lvl in enumerate(levels):
    names = [specs[i]["name"] for i in lvl]
    marker = " (parallel)" if len(names) > 1 else ""
    print(f"  Level {lvl_num}: {names}{marker}")

# Execute level by level
shared_metadata = {}

class FakeCtx:
    def __init__(self, meta):
        self.metadata = meta
        self.config = {}

ctx = FakeCtx(shared_metadata)

for lvl_num, lvl in enumerate(levels):
    for idx in lvl:
        name = specs[idx]["name"]
        instance = step_classes[name]()
        instance.setup(specs[idx].get("config", {}))
        result = instance.run(ctx)
        instance.teardown()
        print(f"  ✓ {name} completed")

# Save report
report = shared_metadata.get("report", "No report generated")
(results_dir / "report.txt").write_text(report)
print(f"\nReport saved to {results_dir / 'report.txt'}")
PYEOF

ok "DAG pipeline executed successfully"

# ---------- Step 4: Show results ----------
log ""
log "Step 4: Results:"
echo ""
cat "${RESULTS_DIR}/report.txt"

banner "Demo 15 Complete"
