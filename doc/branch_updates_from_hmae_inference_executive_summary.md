# Executive Summary: Current Branch vs `HMAE_inference`

See the full technical breakdown in [branch_updates_from_hmae_inference.md](./branch_updates_from_hmae_inference.md).

## One-Paragraph Summary

Relative to `HMAE_inference`, the current branch is not just an incremental
workflow feature. It expands UMRS into a more complete platform by adding
site-driven deployment, durable job orchestration, endpoint registry support,
generic BentoML-based vision serving, broader model/workflow control-plane
features, a reorganized source tree, and a new canonical demo/documentation
set. HMAE support remains present, but it is now one part of a much more
general architecture.

## Change Scale

| Metric | Value |
|---|---:|
| Unique commits since `HMAE_inference` | 149 |
| Files changed | 476 |
| Insertions | 53,230 |
| Deletions | 16,576 |

## Five Key Messages

### 1. Deployment moved from ad hoc scripts to a site-driven model

- Main evidence:
  - [../src/umrs/orchestrator/site_config.py](../src/umrs/orchestrator/site_config.py)
  - [../deploy/rccs-cloud/site.yaml](../deploy/rccs-cloud/site.yaml)
  - [../deploy/hpbd-atom/site.yaml](../deploy/hpbd-atom/site.yaml)
- Meaning:
  - runtime behavior is now defined through typed `site.yaml` configuration,
    deploy wrappers, and site-specific start/stop/setup flows.

### 2. Provisioning and cleanup are now durable tracked jobs

- Main evidence:
  - [../src/umrs/jobs/service.py](../src/umrs/jobs/service.py)
  - [../src/umrs/server/durable_job_runtime.py](../src/umrs/server/durable_job_runtime.py)
  - [../src/umrs/orchestrator/endpoint_registry.py](../src/umrs/orchestrator/endpoint_registry.py)
- Meaning:
  - provisioning is no longer just a transient launch; it has persisted state,
    retries, monitoring, reconciliation, and endpoint lifecycle tracking.

### 3. The vision path became generic instead of HMAE-specific

- Main evidence:
  - [../src/umrs/vision_service/service.py](../src/umrs/vision_service/service.py)
  - [../src/umrs/vision_service/adapters/base.py](../src/umrs/vision_service/adapters/base.py)
  - [../src/umrs/vision_service/default_adapters/hmae_adapter.py](../src/umrs/vision_service/default_adapters/hmae_adapter.py)
  - [../src/umrs/vision_service/default_adapters/sam3_adapter.py](../src/umrs/vision_service/default_adapters/sam3_adapter.py)
- Meaning:
  - HMAE is still supported, but the service is now structured around generic
    adapters and a single BentoML inference surface.

### 4. UMRS gained a much broader control plane

- Main evidence:
  - [../src/client_umrs.py](../src/client_umrs.py)
  - [../src/umrs/api/models.py](../src/umrs/api/models.py)
  - [../src/umrs/server/workflow_registry_handler.py](../src/umrs/server/workflow_registry_handler.py)
  - [../src/umrs/tui/dashboard.py](../src/umrs/tui/dashboard.py)
- Meaning:
  - the branch adds model upload/download, durable job commands, workflow
    upload/registration, and a live job-monitor dashboard.

### 5. The branch repositions the repo as a platform, not a prototype

- Main evidence:
  - [../demos/README.md](../demos/README.md)
  - [../doc/admin_guide.md](../doc/admin_guide.md)
  - [../doc/demo_guide.md](../doc/demo_guide.md)
  - [../tests/test_site_config.py](../tests/test_site_config.py)
- Meaning:
  - canonical demos, deploy docs, admin/user guides, and broader tests now
    exist around the code, which signals a move toward a maintainable product
    surface.

## Practical Before/After Framing

| Area | `HMAE_inference` baseline | Current branch |
|---|---|---|
| Deployment | mostly hardcoded/env-driven behavior | typed `site.yaml` + deploy wrappers |
| Endpoint lifecycle | launch-oriented | registry-backed + durable job tracking |
| Vision serving | HMAE/SAM3-centric paths | generic BentoML service with adapters |
| Extensibility | limited | workflows, Pipeline SDK, plugins, workflow upload |
| Demos | older mixed/demo-dev structure | canonical demos `01` to `22` |

## What To Say In A Meeting

If you need a short verbal explanation:

"This branch started from the HMAE inference baseline, but it grew into a much
broader platform update. The main shift is that UMRS is now organized around
site-driven deployment, durable endpoint/job lifecycle management, a generic
vision service, and a larger model/workflow control plane. HMAE support remains,
but it is no longer the defining architectural concern."

## Best Follow-Up Reading

1. [branch_updates_from_hmae_inference.md](./branch_updates_from_hmae_inference.md)
2. [branch_updates_from_hmae_inference_pr_summary.md](./branch_updates_from_hmae_inference_pr_summary.md)
3. [../deploy/rccs-cloud/README.md](../deploy/rccs-cloud/README.md)
4. [../demos/README.md](../demos/README.md)
