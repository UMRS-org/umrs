# UMRS Demo Day Speaker Notes

## Opening
Describe UMRS as a governed execution platform, not merely an inference endpoint. Explain that a bucket alias represents an experience and data contract.

## Text RAG
Upload a small text directory to `mnagaso3-text`. Show the bucket list and model registry. Do not claim a new RAG pipeline exists unless a valid pipeline JSON has been submitted with `r4 rag ingest create --file`.

## SAM3.1
Use `demo-defect`, not the obsolete `demo-sam3` alias from older private documentation. The verified upload is `Cat_1.jpg`; wait for a real job or output object before showing results.

## HMAE
Use `demo-hmae` for reconstruction. `demo-hmae-mae` and `demo-hmae-features` are separate tasks. State clearly that the validation proved provisioning and S3 ingestion; current status did not show a launched GPU job.

## Close
Show `umrs-client doctor`. Explain that an idle server after upload needs an operator investigation of experience-to-runtime triggering before a GPU-output demo is promised.
