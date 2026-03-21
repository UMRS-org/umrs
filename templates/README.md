# UMRS Pipeline Templates

> **Note**: The Python-first template approach has been replaced by
> self-contained demos and comprehensive guides that match the real
> UMRS deployment model (container-based, Mini-Rivault sandboxing).

## Where to Go

- **Getting started**: [User Guide](../doc/user_guide.md)
- **Deploying UMRS**: [Admin Guide](../doc/admin_guide.md)
- **Demos**: See [demos/README.md](../demos/README.md) for the full guide
  - [00 — Start Services](../demos/00_start_services/README.md) (admin)
  - [01 — Text Embedding](../demos/01_text_embedding/README.md) (basic)
  - [02 — SAM3 Segmentation](../demos/02_sam3_segmentation/README.md) (basic)
  - [03 — HMAE Inference](../demos/03_hmae_inference/README.md) (basic)
  - [07 — User-Defined Models](../demos/07_user_model/README.md) (advanced)

## Pipeline SDK (Optional)

The `src/umrs/pipeline_sdk/` package provides a Python SDK with abstract
`Pipeline` class and connectors (S3, BentoML, pgvector). This can be used
for building custom processing scripts that run *inside* containers.

See [pipeline_sdk/](../src/umrs/pipeline_sdk/) for details.
