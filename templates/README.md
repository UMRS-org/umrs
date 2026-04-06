# UMRS Pipeline Templates

> **Note**: The Python-first template approach has been replaced by
> self-contained demos and comprehensive guides that match the real
> UMRS deployment model (container-based, Mini-Rivault sandboxing).

## Where to Go

- **Getting started**: [User Guide](../doc/user_guide.md)
- **Deploying UMRS**: [Admin Guide](../doc/admin_guide.md)
- **Demos**: See [demos/README.md](../demos/README.md) for the full guide
  - [01 — Health Check](../demos/01_health_check/README.md) (basic)
  - [02 — Text Embedding](../demos/02_text_embedding/README.md) (basic)
  - [03 — SAM3 Segmentation](../demos/03_sam3_segmentation/README.md) (basic)
  - [08 — User-Defined Models](../demos/08_user_model/README.md) (advanced)

## Pipeline SDK (Optional)

The `src/umrs/pipeline_sdk/` package provides a Python SDK with abstract
`Pipeline` class and connectors (S3, BentoML, pgvector). This can be used
for building custom processing scripts that run *inside* containers.

See [pipeline_sdk/](../src/umrs/pipeline_sdk/) for details.
