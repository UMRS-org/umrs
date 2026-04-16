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
# 1. Create the steps directory and results directory
mkdir -p steps results

# 2. Create the step implementations (steps/steps.py)
#    Each class implements: name, setup(), teardown(), run(ctx)
#    - LoadData:        loads sample texts and metadata
#    - AnalyzeText:     counts words and documents
#    - AnalyzeMetadata: inspects metadata fields
#    - MergeResults:    combines text + metadata analysis
#    - GenerateReport:  formats the final report
cat > steps/steps.py << 'EOF'
# ... (see run_demo.sh for full content)
EOF

# 3. Create the workflow.json with DAG dependencies
#    The "depends_on" array defines which steps must complete
#    before a given step can execute.
cat > steps/workflow.json << 'EOF'
{
  "name": "complex-dag-demo",
  "version": "1.0",
  "steps": [
    {"name": "load_data",        "depends_on": []},
    {"name": "analyze_text",     "depends_on": ["load_data"]},
    {"name": "analyze_metadata", "depends_on": ["load_data"]},
    {"name": "merge_results",    "depends_on": ["analyze_text", "analyze_metadata"]},
    {"name": "generate_report",  "depends_on": ["merge_results"]}
  ]
}
EOF

# 4. Execute the pipeline via Python
#    The script performs a topological sort to determine execution levels:
#      Level 0: load_data
#      Level 1: analyze_text, analyze_metadata  (parallel)
#      Level 2: merge_results
#      Level 3: generate_report
python3 - steps/ results/ << 'PYEOF'
import sys, json, importlib.util
from pathlib import Path
# ... (DAG scheduler logic — see run_demo.sh)
PYEOF

# 5. View the generated report
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
