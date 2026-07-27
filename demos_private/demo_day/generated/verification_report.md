# Demo-Day Validation Report

Validation date: 2026-07-20

## Passed

- `experience add --update` succeeded for HMAE, Defect Analysis, HMAE full-pipeline aliases, and Opossum Analysis.
- `reload` succeeded and materialized `demo-defect`, `demo-hmae`, `demo-hmae-mae`, `demo-hmae-features`, and `demo-opossum` with `remote_status: synced`.
- The system model registry includes `hmae` and `sam3.1` vision registrations.
- `Cat_1.jpg` upload to `demo-hmae` and `demo-defect` completed with `status=ok` and image modality classification.

## Not yet demonstrated end-to-end

- Neither visual upload produced an observed active job or generated output before validation; `status all` reported the server idle.
- Download commands target output buckets and require existing local directories.
- No repository MP4 fixture exists for `demo-opossum`; it is provisioned but has not received a valid video workload.

## Demo-day decision

Use text ingestion live. Show SAM3.1 and HMAE uploads as validated transport demonstrations. Only present segmentation masks or reconstructions live after a real job and output artifact are observed. Treat missing execution as an operator runtime-routing follow-up, not an alias-provisioning failure.
