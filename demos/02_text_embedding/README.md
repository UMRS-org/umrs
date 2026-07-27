# Demo 02: Text Embedding

User-facing demo — upload text files, compute embeddings via the UMRS pipeline,
verify results in pgvector, and export.

## Prerequisites

- Services running
- UMRS client installed (see [demos/README.md](../README.md#prerequisites) — `uv pip install -e src/`)
- Server config JSON from your admin

## Quick Start

```bash
# Navigate from either repo level (auto-detects parent vs subdirectory)
cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/02_text_embedding"
# See the numbered steps below in this README
```

## What This Demo Does

1. **Ping** the UMRS server to verify connectivity
2. **Load the experience** — `experience.json` declares this demo's modality, model, bucket, and DB table. See the [User Manual](../USER_MANUAL.md#experiences) for the full schema.
3. **Verify the experience was added** — confirm it appears in your user config
4. **Reload** — triggers server-side bucket + DB provisioning
5. **Verify buckets** — confirm the new bucket exists before and after reload
6. **Verify or prepare** the embedding model (reuse a staged model, or stage it if your site exposes a writable model root)
7. **Upload** sample text files to the S3 bucket
8. **Wait** for the embedding pipeline to finish
9. **Verify** that embeddings are stored in pgvector
10. **Export** embeddings to a local CSV file

## What is `experience.json`?

Each demo ships an `experience.json` fragment that declares:

- **Modality** — the data type (text, image, movie, etc.)
- **Model** — the model name and type (e5-base-v2, sam3.1, etc.)
- **Bucket alias** — the S3 bucket alias that UMRS provisions
- **DB table** — the Postgres table name for storing embeddings/metadata

When you run `umrs experience add experience.json`, this fragment is merged
into your user config JSON. The next `umrs reload` instructs the server to
provision the declared MinIO buckets and Postgres tables.

## Running Manually

If you prefer to run steps individually:

```bash
# 1. Check server
umrs ping

# 2. Load the experience fragment into your user config
#    This registers the text embedding experience (model + bucket + table)
umrs experience add experience.json

# 3. Verify the experience was added successfully
umrs experience list
# Expected: "Text Embedding" should appear in the list

# 3b. (Optional) Validate the experience fragment
#    Checks the JSON structure and verifies model availability if server is reachable
umrs experience validate experience.json

# 4. Check buckets before reload
umrs bucket list
# Expected: only buckets from previous experiences (not demo-texts yet)

# 5. Reload to apply the new experience and provision buckets
umrs reload

# 6. Check buckets after reload — new buckets should appear
umrs bucket list
# Expected: demo-texts and demo-textfiles now visible

# 7. Verify the embedding model is visible to the server
umrs model list --scope system
umrs model list --scope user
# Use this to confirm intfloat/e5-base-v2 is registered

# Optional: only if your site gives you a writable, allowed staging root
umrs model download intfloat/e5-base-v2 --scope user --server-side \
    --staging-dir /path/to/site-user-model-root

# 8. Upload the bundled sample files
#
# IMPORTANT: The bucket alias depends on your username.
# The experience.json declares "demo-texts", but the actual bucket name
# on the server is prefixed by your username: <username>-<alias>.
# So if your user is "mnagaso3", the bucket is "mnagaso3-textfiles",
# NOT "demo-textfiles". Always use your username-prefixed alias.
umrs data dir-upload ../sample_data/texts "$USER-textfiles"

# 9. Monitor progress
umrs status process_objects

# 10. Once processing is done, export embeddings (default: minimal profile)
umrs export_embeddings "$USER-textfiles" ./results/embeddings.csv --format csv

# 11. Export all stored DB columns when you need the full row
umrs export_embeddings "$USER-textfiles" ./results/embeddings-full.csv --format csv --profile full

# 12. Export only the embedding vector itself
umrs export_embeddings "$USER-textfiles" ./results/embeddings-only.csv --format csv --profile embedding-only
```

## Test Data

Sample text files live in `../sample_data/texts/`:

| File | Topic |
|------|-------|
| `hpc_computing.txt` | High-performance computing |
| `ml_materials.txt` | Machine learning for materials science |
| `containers_hpc.txt` | Containers on HPC systems |
| `rivault_platform.txt` | Rivault platform description |
| `seismic_waves.txt` | Seismic wave analysis |

## Configuration

The demo script assumes a configured UMRS client and uses:
- Bucket alias: `demo-textfiles`
- Model: `intfloat/e5-base-v2` (768-d embeddings via vLLM in Mini-Rivault)
- Modality: `text`

On shared deployments, normal users should prefer an already staged model from
the visible `system` or `user` catalog. The optional server-side download step
only works when the deployment exposes a writable, allowed staging root.

If your site uses a different bucket alias, override it with:

```bash
TEXT_BUCKET=my-text-bucket
# Use any available bucket alias
```

## Expected Output

After a successful run:
- Embeddings stored in pgvector (`text` table in user database)
- Default CSV file at `./embeddings.csv` with essential metadata plus the embedding column
- Optional full export at `./embeddings-full.csv` with all stored DB columns
- Optional embedding-only export at `./embeddings-only.csv` with just the embedding vector
- Status shows all `process_objects` tasks as `Done`

---

## Custom Embedding with `user_script://`

Instead of using a built-in embedding model (e.g. `intfloat/e5-base-v2`), you
can provide your **own embedding pipeline** using the Pipeline SDK.

### Example: Word-Count Embedding

The `custom_embedding/` directory contains a complete working example:

| File | Description |
|------|-------------|
| `word_count_step.py` | `Step` that tokenizes text and produces 128-d word-count vectors |
| `workflow.json` | Pipeline definition pointing to the step |

### How to use it

1. Update the `source.path` in `workflow.json` to point to the directory
   containing your step files:

```json
{
    "source": {
        "type": "local_path",
        "path": "/home/user/custom_embedding"
    }
}
```

2. Set `model_type` in your config to reference the workflow:

```json
{
    "modality_name": "text",
    "model": {
        "model_name": "Word Count Embeddings",
        "model_type": "user_script:///home/user/custom_embedding/workflow.json",
        "vector_size": "128",
        "return_type": "vector"
    }
}
```

3. Upload text files and process as usual:

```bash
umrs data dir-upload ./test_data/ my-bucket
umrs status all
```

### Writing your own step

See [Demo 08: User-Defined Model](../08_user_model/README.md) for the full
`Step` API reference. In short:

```python
from umrs.pipeline_sdk import Step, StepContext, StepResult
import numpy as np

class MyEmbeddingStep(Step):
    name = "my_embedding"

    def setup(self, config: dict) -> None:
        # Load your model, tokenizer, etc.
        ...

    def run(self, ctx: StepContext) -> StepResult:
        vectors = []
        for key in ctx.input_keys:
            data = ctx.download_bytes(key)
            vec = self._encode(data)      # your encoding logic
            vectors.append(vec)
        return StepResult(
            output_keys=ctx.input_keys,
            embeddings=np.stack(vectors),  # shape (N, D)
        )
```

## See Also

- [User Guide](../../doc/user_guide.md) — full client documentation
- [Demo 03: SAM3](../03_sam3_segmentation/README.md) — image segmentation demo
- [Demo 08: User-Defined Model](../08_user_model/README.md) — full Step API reference
