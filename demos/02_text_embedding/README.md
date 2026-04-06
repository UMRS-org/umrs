# Demo 02: Text Embedding

User-facing demo — upload text files, compute embeddings via the UMRS pipeline,
verify results in pgvector, and export.

## Prerequisites

- Services running
- UMRS client installed (`pip install .` from `src/`)
- Server config JSON from your admin

## Quick Start

```bash
cd demos/02_text_embedding
bash run_demo.sh
```

## What This Demo Does

1. **Ping** the UMRS server to verify connectivity
2. **Prepare** the embedding model (downloads to shared volume if needed)
3. **Upload** sample text files to the S3 bucket via `umrs-client`
4. **Wait** for the embedding pipeline to finish
5. **Verify** that embeddings are stored in pgvector
6. **Export** embeddings to a local CSV file

## Running Manually

If you prefer to run steps individually:

```bash
# 1. Check server
umrs-client -c conf_umrs.json ping

# 2. Upload text files
umrs-client -c conf_umrs.json upload_dir ./test_data/text/ rivault-demo-bucket

# 3. Monitor progress
umrs-client -c conf_umrs.json --output text status all

# 4. Once all tasks show "Done", export embeddings
umrs-client -c conf_umrs.json export_embeddings rivault-demo-bucket ./embeddings.csv
```

## Test Data

Sample text files in `test_data/text/` (symlinked from `demos/slurm-dev/test_data/text/`):

| File | Topic |
|------|-------|
| `hpc_computing.txt` | High-performance computing |
| `ml_geophysics.txt` | Machine learning in geophysics |
| `numerical_methods.txt` | Numerical methods |
| `rivault_platform.txt` | Rivault platform description |
| `seismic_waves.txt` | Seismic wave analysis |

## Configuration

The demo uses the Rivault demo config (`conf_umrs.json`), which defines:
- Username: `rivault_user`
- Bucket: `rivault-demo-bucket`
- Model: `intfloat/e5-base-v2` (768-d embeddings via vLLM in Mini-Rivault)
- Modality: `text`

## Expected Output

After a successful run:
- Embeddings stored in pgvector (`text` table in user database)
- CSV file at `./embeddings.csv` with columns: `id`, `s3_key`, `embedding`, etc.
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
umrs-client -c conf_umrs.json upload_dir ./test_data/ my-bucket
umrs-client -c conf_umrs.json status all
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
