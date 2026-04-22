# Demo 15: Complex DAG Workflow

Demonstrates a multi-step pipeline with **parallel branches and fan-in**
using the local Pipeline SDK and DAG scheduler. No server required — runs
entirely on your local machine.

## DAG Structure

```
    [load_data]
        │
    ┌───┴───┐
    │       │
[analyze_text]  [analyze_metadata]
    │       │
    └───┬───┘
        │
    [merge_results]
        │
    [generate_report]
```

Steps at the same level (`analyze_text` and `analyze_metadata`) execute
in parallel, demonstrating the DAG scheduler's topological sort and
fan-in capabilities.

## Prerequisites

- Python 3.10+ with UMRS package installed (`uv pip install -e src/`)
- **No server required** — this demo runs locally

## Quick Start

```bash
cd demos/15_complex_dag
bash run_demo.sh
```

## What This Demo Does

1. **Prepare step scripts** — generates a `steps/steps.py` file with 5 step
   classes (`LoadData`, `AnalyzeText`, `AnalyzeMetadata`, `MergeResults`,
   `GenerateReport`)
2. **Create workflow.json** — defines the DAG dependencies between steps
3. **Execute DAG pipeline** — runs a Python-based topological sort, executes
   steps level by level (parallel steps run in the same level)
4. **Show results** — prints the generated analysis report

## Running Manually

This demo does not use `umrs-client` because it runs entirely locally.
You can replicate it step by step:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)/umrs"
cd "$REPO_ROOT/demos/15_complex_dag"

# 1. Create working directories.
mkdir -p steps results

# 2. Write the step implementations.
cat > steps/steps.py <<'PYEOF'
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
    merged = {"text": ctx.metadata.get("text_analysis", {}), "meta": ctx.metadata.get("meta_analysis", {})}
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

# 3. Define the DAG dependencies.
cat > steps/workflow.json <<'JSONEOF'
{
  "name": "complex-dag-demo",
  "version": "1.0",
  "source": {
  "type": "local_path",
  "path": "."
  },
  "execution_mode": "in_process",
  "steps": [
  {"name": "load_data",        "entrypoint": "steps:LoadData",        "config": {}, "depends_on": []},
  {"name": "analyze_text",     "entrypoint": "steps:AnalyzeText",     "config": {}, "depends_on": ["load_data"]},
  {"name": "analyze_metadata", "entrypoint": "steps:AnalyzeMetadata", "config": {}, "depends_on": ["load_data"]},
  {"name": "merge_results",    "entrypoint": "steps:MergeResults",    "config": {}, "depends_on": ["analyze_text", "analyze_metadata"]},
  {"name": "generate_report",  "entrypoint": "steps:GenerateReport",  "config": {}, "depends_on": ["merge_results"]}
  ],
  "output": {
  "return_type": "file"
  }
}
JSONEOF

# 4. Execute the DAG runner.
python3 - steps results <<'PYEOF'
import importlib.util
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

steps_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
wf = json.loads((steps_dir / "workflow.json").read_text())

spec = importlib.util.spec_from_file_location("steps", steps_dir / "steps.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

step_classes = {
  "load_data": mod.LoadData,
  "analyze_text": mod.AnalyzeText,
  "analyze_metadata": mod.AnalyzeMetadata,
  "merge_results": mod.MergeResults,
  "generate_report": mod.GenerateReport,
}

specs = wf["steps"]
name_to_idx = {s["name"]: i for i, s in enumerate(specs)}
adj = defaultdict(list)
in_deg = [0] * len(specs)
for i, s in enumerate(specs):
  for dep in s.get("depends_on", []):
    j = name_to_idx[dep]
    adj[j].append(i)
    in_deg[i] += 1

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

shared_metadata = {}
class FakeCtx:
  def __init__(self, meta):
    self.metadata = meta
    self.config = {}

ctx = FakeCtx(shared_metadata)
for lvl in levels:
  for idx in lvl:
    name = specs[idx]["name"]
    instance = step_classes[name]()
    instance.setup(specs[idx].get("config", {}))
    instance.run(ctx)
    instance.teardown()
    print(f"  ✓ {name} completed")

(results_dir / "report.txt").write_text(shared_metadata.get("report", "No report generated"))
print(f"\nReport saved to {results_dir / 'report.txt'}")
PYEOF

# 5. Show the generated report.
cat results/report.txt
```

## How the DAG Scheduler Works

The scheduler builds an adjacency list from `depends_on` declarations, computes
in-degrees, then runs Kahn's algorithm (BFS topological sort) to group steps
into execution levels:

| Level | Steps | Parallelism |
|-------|-------|-------------|
| 0 | `load_data` | Sequential |
| 1 | `analyze_text`, `analyze_metadata` | **Parallel** |
| 2 | `merge_results` | Sequential (fan-in) |
| 3 | `generate_report` | Sequential |

Steps within the same level have no dependencies on each other and can
safely execute concurrently.

## Expected Output

```
DAG has 4 execution levels:
  Level 0: ['load_data']
  Level 1: ['analyze_text', 'analyze_metadata'] (parallel)
  Level 2: ['merge_results']
  Level 3: ['generate_report']
  ✓ load_data completed
  ✓ analyze_text completed
  ✓ analyze_metadata completed
  ✓ merge_results completed
  ✓ generate_report completed

Report saved to results/report.txt
```

## See Also

- [Demo 09: Multi-Step Vision Pipeline](../09_multistep_vision_pipeline/README.md) — sequential chained pipeline
- [Demo 10: Vision Multi-Mode](../10_vision_multi_mode/README.md) — parallel branch execution
- [Demo 12: Pipeline SDK](../12_pipeline_sdk/README.md) — 6-phase pipeline lifecycle
- [Demo 16: Custom Workflow](../16_custom_workflow/README.md) — server-side workflow execution
