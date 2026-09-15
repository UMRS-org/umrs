# Demo 16: Custom Server-Side Workflow

Upload a custom workflow (Python step scripts + `workflow.json`) to the
UMRS server, send data through it, and download processed results.

## Prerequisites

- UMRS server running and reachable (SSH tunnels if remote: ports 5555, 5556, 9000)
- `umrs-client` installed and in PATH

## Quick Start

```bash
cd demos/16_custom_workflow
# Follow the step-by-step commands in README.md
```

## What This Demo Does

1. **Pre-flight** — verifies server connectivity
2. **Prepare workflow files** — creates `workflow.json` (pipeline definition)
   and `counter.py` (custom word-counting step)
3. **Upload workflow** — sends the workflow package to the server via
   `umrs-client workflow upload`
4. **Upload sample data** — sends text files to the workflow's bucket
5. **Download results** — retrieves the processed output
   (`word_count_results.json`)

## Running Manually

```bash
# 1. Verify server connectivity
#    Sends a ZMQ ping to confirm the server is alive and your config is valid.
umrs-client ping

# 2. Prepare your workflow files
#    Create the uploaded workflow package locally.
mkdir -p workflow results

cat > workflow/workflow.json <<'JSONEOF'
{
  "name": "custom-text-counter",
  "version": "1.0",
  "description": "Count words and characters in uploaded text files",
  "source": {
    "type": "uploaded",
    "path": "."
  },
  "execution_mode": "in_process",
  "steps": [
    {
      "name": "count_words",
      "entrypoint": "counter:WordCounter",
      "config": {"min_word_length": 3}
    }
  ],
  "output": {
    "return_type": "file"
  }
}
JSONEOF

cat > workflow/counter.py <<'PYEOF'
"""Custom step: count words and characters in text files."""

from umrs.pipeline_sdk import Step, StepContext, StepResult


class WordCounter(Step):
    name = "count_words"

    def setup(self, config):
        self.min_word_length = config.get("min_word_length", 1)

    def teardown(self):
        pass

    def run(self, ctx: StepContext) -> StepResult:
        results = {}
        for key in ctx.input_keys:
            if key.endswith(".txt"):
                data = ctx.download_bytes(key)
                text = data.decode("utf-8", errors="replace")
                words = [w for w in text.split() if len(w) >= self.min_word_length]
                results[key] = {
                    "total_words": len(text.split()),
                    "filtered_words": len(words),
                    "characters": len(text),
                    "lines": text.count("\n") + 1,
                }

        import json

        ctx.upload_bytes(
            "word_count_results.json",
            json.dumps(results, indent=2).encode(),
            content_type="application/json",
        )

        return StepResult(
            output_keys=["word_count_results.json"],
            metadata=[{"summary": results}],
        )
PYEOF

# 3. Upload the custom workflow to the server
#    This registers the workflow under the name "custom-text-counter".
#    The server stores the workflow definition and step scripts so it can
#    execute them when data arrives.
umrs-client workflow upload ./workflow --name "custom-text-counter"

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

This demo shows the normal user path: upload the workflow bundle, upload data,
and collect outputs. If your site requires extra server-side activation, that is
an operator concern rather than part of the user demo.

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
