# Demo 12 — Pipeline SDK Workflow

This demo showcases the UMRS Pipeline SDK by building a concrete `Pipeline`
subclass that runs through the full 6-phase lifecycle:

```
configure → ingest → compute → collect → verify → present
```

## What It Does

`TextAnalysisPipeline` reads `.txt` files, computes word-frequency statistics
(word count, unique words, top words, readability metrics), and outputs a
summary table. It uses `ComputeBackend.LOCAL` so **no external services are
needed**.

## Prerequisites

- Python 3.9+
- UMRS installed: `uv pip install -e src/`

## Quick Start

```bash
bash run_demo.sh
```

Or run the pipeline directly:

```bash
python3 text_analysis_pipeline.py --input-dir ../../data/file_of_texts
python3 text_analysis_pipeline.py --input-dir /your/text/dir --top-n 5
```

## Pipeline Phases Explained

| Phase | Method | What It Does |
|-------|--------|-------------|
| **1. Configure** | `configure()` | Returns a `PipelineConfig` with LOCAL backend and top-N setting |
| **2. Ingest** | `ingest(config)` | Scans the input directory for `.txt` files, creates `DataItem` per file |
| **3. Compute** | `compute(config, items)` | Analyses each text (word freq, sentence count, etc.), returns `TaskHandle`s |
| **4. Collect** | `collect(config, handles)` | Extracts results from handles → `Result` objects |
| **5. Verify** | `verify(config, results)` | Checks no failures and all files have word_count > 0 |
| **6. Present** | `present(config, results)` | Prints summary table + top words per file |

## Adapting for Real Workloads

To connect to actual UMRS/BentoML services, change the backend in `configure()`:

```python
def configure(self) -> PipelineConfig:
    return PipelineConfig(
        name="my-pipeline",
        compute_backend=ComputeBackend.UMRS_EMBEDDING,  # or BENTOML_TASK
        user_config_path="/path/to/user_rivault.json",
        umrs_server_ip="gpu-node-01",
        s3_endpoint="http://minio:9000",
        s3_access_key="...",
        s3_secret_key="...",
        s3_bucket="my-bucket",
    )
```

Then use the SDK connectors in `compute()` and `collect()`:

```python
from umrs.pipeline_sdk.connectors.umrs_client import UMRSClientConnector
from umrs.pipeline_sdk.connectors.s3 import S3Connector

def compute(self, config, items):
    s3 = S3Connector(config.s3_endpoint, config.s3_access_key,
                     config.s3_secret_key, config.s3_bucket)
    umrs = UMRSClientConnector(user_config=config.user_config_path,
                               server_ip=config.umrs_server_ip)
    # Upload data → trigger UMRS processing → return handles
    ...
```

## Available Connectors

| Connector | Module | Transport |
|-----------|--------|-----------|
| UMRS CLI | `connectors.umrs_client` | ZMQ (via subprocess) |
| BentoML | `connectors.bentoml` | HTTP REST |
| S3 / MinIO | `connectors.s3` | boto3 |
| pgvector | `connectors.pgvector` | psycopg2 |

## Related

- Pipeline SDK source: `src/umrs/pipeline_sdk/`
- JSON workflow example: `demos/16_custom_workflow/`
- DAG workflow example: `demos/15_complex_dag/`
- Model registry: `demos/07_model_registry/`
