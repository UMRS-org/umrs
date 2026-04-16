# Demo 16: Custom Server-Side Workflow

Upload a custom workflow (Python step scripts + `workflow.json`) to the
UMRS server, send data through it, and download processed results.

## Prerequisites

- UMRS server running and reachable (SSH tunnels if remote: ports 5555, 5556, 9000)
- `umrs-client` installed and in PATH

## Quick Start

```bash
cd demos/16_custom_workflow
bash run_demo.sh
```

## What This Demo Does

1. **Pre-flight** — verifies server connectivity
2. **Prepare workflow files** — creates `workflow.json` (pipeline definition)
   and `counter.py` (custom word-counting step)
3. **Upload workflow** — sends the workflow package to the server via
   `umrs-client upload-workflow`
4. **Upload sample data** — sends text files to the workflow's bucket
5. **Download results** — retrieves the processed output
   (`word_count_results.json`)

## Running Manually

```bash
# 1. Verify server connectivity
#    Sends a ZMQ ping to confirm the server is alive and your config is valid.
umrs-client ping

# 2. Prepare your workflow files
#    Create a directory with:
#    - workflow.json: defines the pipeline name, steps, and output type
#    - counter.py:    Python step class that processes uploaded text files
mkdir -p workflow
# (see run_demo.sh for full file contents)

# 3. Upload the custom workflow to the server
#    This registers the workflow under the name "custom-text-counter".
#    The server stores the workflow definition and step scripts so it can
#    execute them when data arrives.
umrs-client upload-workflow ./workflow --name "custom-text-counter"

# 4. Upload text data to the workflow's bucket
#    Files uploaded to this bucket will be processed by the custom workflow.
#    The server detects new objects and runs the "count_words" step on each
#    text file, producing a word_count_results.json output.
umrs-client upload_dir ../sample_data/texts user1-custom-text

# 5. Download the processed results
#    Retrieves all output files from the bucket to a local directory.
#    The main output is word_count_results.json with per-file statistics.
umrs-client download_results user1-custom-text ./results --force-overwrite
```

## Workflow Definition

The `workflow.json` defines a single-step pipeline:

```json
{
  "name": "custom-text-counter",
  "version": "1.0",
  "description": "Count words and characters in uploaded text files",
  "source": { "type": "uploaded", "path": "." },
  "execution_mode": "in_process",
  "steps": [
    {
      "name": "count_words",
      "entrypoint": "counter:WordCounter",
      "config": { "min_word_length": 3 }
    }
  ],
  "output": { "return_type": "file" }
}
```

The `WordCounter` step reads each `.txt` file, counts words (optionally
filtering by minimum length), and writes a JSON summary.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEXT_BUCKET` | `user1-custom-text` | S3 bucket alias for this workflow |

## Expected Output

```json
{
  "hpc_computing.txt": {
    "total_words": 245,
    "filtered_words": 198,
    "characters": 1523,
    "lines": 18
  },
  "ml_materials.txt": {
    "total_words": 312,
    "filtered_words": 271,
    "characters": 2041,
    "lines": 22
  }
}
```

## See Also

- [Demo 08: User-Defined Model](../08_user_model/README.md) — full Step API reference
- [Demo 12: Pipeline SDK](../12_pipeline_sdk/README.md) — 6-phase pipeline lifecycle
- [Demo 15: Complex DAG](../15_complex_dag/README.md) — DAG with parallel branches (local)
