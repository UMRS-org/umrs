## UMRS Future Plan: Deployment Focus

This document captures the near-term plan for turning the current embedding worker implementation into production-ready deployments using Docker and Kubernetes (with Kind for local development).

---

### 1. Docker Compose (Single-Node) – Baseline

Status: **Implemented and working**.

- Keep docker/docker-compose.yml as the reference deployment for:
	- umrs-server
	- umrs-embed-worker
	- postgres (pgvector)
	- minio
	- umrs-api-test
- Ensure the following are documented and stable:
	- Shared model cache volume (model_cache:/shared/models) between server and worker.
	- Recommended workflow (build, up, avoid down -v unless resetting state).
	- How to switch between local and remote embedding backends via configuration.

**Next small steps**

- Add short pointers from README.md to the new doc/embedding_worker_deployment.md.
- Optionally add a simple Makefile or helper script to run the standard compose workflow.

---

### 2. Kubernetes Deployment (Multi-Node)

Goal: Run UMRS server and embedding workers as separate Kubernetes deployments, using a shared model cache and externalized Postgres/MinIO where appropriate.

**Planned tasks**

- Define Kubernetes manifests (or Helm chart) for:
	- UMRS server Deployment + Service.
	- Embedding worker Deployment(s) + Service.
	- MinIO and Postgres (for environments that don’t provide managed equivalents).
- Model cache and data:
	- Design a PersistentVolume / PersistentVolumeClaim for /shared/models so server and worker pods share model caches.
	- Decide on PersistentVolume for Postgres data in non-managed setups.
- Configuration:
	- Map existing conf_umrs.json and user config files into Kubernetes via ConfigMaps/Secrets.
	- Expose embedding backend settings (remote address, timeouts) via ConfigMap values.
- Observability:
	- Add basic logging and readiness/liveness probes for server and worker pods.

**Acceptance criteria**

- Ability to deploy UMRS and a single worker into a Kubernetes cluster.
- umrs-api-test (or an equivalent job) completes successfully against the K8s deployment.

---

### 3. Local Kubernetes with Kind

Goal: Provide a reproducible, local K8s environment for development and testing, using Kind and a local Docker registry.

**Current building blocks (already present)**

- docker/kind-config.yaml – Kind cluster configuration.
- docker/setup_kind_k8s.sh – script to create a Kind cluster and local registry wiring.
- docker/rivault-deployment.yaml, docker/rivault-service.yaml – example of deploying a supporting service into Kind.

**Planned usage pattern**

1. Start Kind + registry:
	 - Run setup_kind_k8s.sh.
	 - Switch kubectl context to the Kind cluster.
2. Build and push images:
	 - Build UMRS server and worker images tagged into localhost:5001/....
	 - Push them so that Kind can pull from the local registry.
3. Apply K8s manifests:
	 - Deploy UMRS server, worker, Postgres, MinIO, and (optionally) RiVault.
4. Run tests:
	 - Trigger umrs-api-test as a Kubernetes Job or from outside the cluster.

**Acceptance criteria**

- A developer can spin up a full UMRS + worker stack in Kind using a small number of scripted commands.
- The behavior matches the Docker Compose deployment (same tests pass, same embedding flows work).

---

### 4. Later Enhancements (Optional)

Once Docker and K8s deployments are stable, we can revisit feature-oriented ideas such as:

- Multiple workers / horizontal scaling for heavy workloads.
- Metrics and dashboards for embedding throughput and latency.
- More advanced access control, audit logging, and cross-institution deployments.

These are intentionally kept out of the near-term plan to keep focus on a small, reliable deployment surface.

