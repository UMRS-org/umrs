# Demo 02: Text Embedding

User-facing demo — upload text files, compute embeddings via the UMRS pipeline,
verify results in pgvector, and export.

## Prerequisites

- Services running
- UMRS client installed (see [demos/README.md](../README.md#prerequisites) — `uv pip install -e src/`)
- Server config JSON from your admin

## Quick Start

```bash
cd demos/02_text_embedding
# Follow the step-by-step commands in README.md
```

## What This Demo Does

1. **Ping** the UMRS server to verify connectivity
2. **Verify or prepare** the embedding model (reuse a staged model, or stage it if your site exposes a writable model root)
3. **Upload** sample text files to the S3 bucket via `umrs-client`
4. **Wait** for the embedding pipeline to finish
5. **Verify** that embeddings are stored in pgvector
6. **Export** embeddings to a local CSV file

## Running Manually

If you prefer to run steps individually:

```bash
# 1. Check server
umrs-client ping

# 2. Ensure the embedding model is visible to the server
umrs-client model list --scope system
umrs-client model list --scope user

# Optional: only if your site gives you a writable, allowed staging root
umrs-client model download intfloat/e5-base-v2 --scope user --server-side \
    --staging-dir /path/to/site-user-model-root

# 3. Upload the bundled sample files
umrs-client data dir-upload ../sample_data/texts user1-textfiles

# 4. Monitor progress
umrs-client --output text status process_objects

# 5. Once processing is done, export embeddings (default: minimal profile)
umrs-client export_embeddings user1-textfiles ./results/embeddings.csv --format csv

# 6. Export all stored DB columns when you need the full row
umrs-client export_embeddings user1-textfiles ./results/embeddings-full.csv --format csv --profile full

# 7. Export only the embedding vector itself
umrs-client export_embeddings user1-textfiles ./results/embeddings-only.csv --format csv --profile embedding-only
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
- Bucket alias: `user1-textfiles`
- Model: `intfloat/e5-base-v2` (768-d embeddings via vLLM in Mini-Rivault)
- Modality: `text`

On shared deployments, normal users should prefer an already staged model from
the visible `system` or `user` catalog. The optional server-side download step
only works when the deployment exposes a writable, allowed staging root.

If your site uses a different bucket alias, override it with:

```bash
TEXT_BUCKET=my-text-bucket # Follow the step-by-step commands in README.md
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
umrs-client data dir-upload ./test_data/ my-bucket
umrs-client status all
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
