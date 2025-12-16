# Embedding Worker & Remote Backend

This document describes the implemented embedding worker for UMRS, how it is wired into the server via the remote embedding backend, and how to run it with Docker Compose and (later) Kubernetes/Kind.

---

## 1. Overview

UMRS now supports running embedding computation in a separate process/container, while the UMRS server keeps ownership of orchestration and database writes.

- **Server responsibilities** (src/server_umrs.py):
  - List objects in S3/MinIO.
  - Decide which model and modality to use for each bucket.
  - Send embedding jobs to the worker when the remote backend is enabled.
  - Store embeddings and metadata into Postgres + pgvector.

- **Worker responsibilities** (src/embedding_worker.py):
  - Connect to S3/MinIO using its own S3 config.
  - Load models using umrs.models_loading.load_model, reusing the same model catalogue as the server (rivault_models.json).
  - Compute embeddings for images, text, and text files.
  - Return embeddings + checksums + metadata to the server via ZeroMQ.

- **Backend abstraction** (src/umrs/embedding_backend.py):
  - LocalEmbeddingBackend: runs compute_and_store_embedding in-process (legacy behavior).
  - RemoteEmbeddingBackend: sends jobs to the worker over ZeroMQ and writes results to the database.

The server chooses the backend at startup based on configuration.

---

## 2. Worker Protocol

The worker exposes a ZeroMQ REP socket. Each job from the server contains:

- job_id
- command (currently `compute_embeddings`)
- path_to_bucket (S3 bucket path)
- s3_keys (list of object keys)
- modality_name (e.g. `image`, `text`, `text file`)
- model_name (e.g. `clip-ViT-B-32`, `intfloat/e5-base-v2`, `Resnet 50`)

### 2.1 Text and Text File Modalities

For `text` and `text file` modalities:

- The underlying sentence-transformers / transformer encoder models produce **chunked** embeddings (lists of dicts with offsets, md5, and embedding vectors).
- These structures are not a simple 2D numeric array, so the worker returns them as JSON.
- Before sending, the worker converts all tensors/NumPy arrays to plain Python lists and scalars so that the payload is fully JSON-serializable.

The JSON reply has the form:

```json
{
  "job_id": "...",
  "status": "ok",
  "modality_name": "text" | "text file",
  "embeddings": [ ... chunk dicts ... ],
  "checksums": [ ... ],
  "metadata": [ ... ]
}
```

RemoteEmbeddingBackend accepts this JSON format and passes it directly to store_embeddings_in_db, which already knows how to handle chunked structures.

### 2.2 Image and Other Numeric Modalities

For purely numeric modalities (e.g. `image`, `tomography`):

- The worker converts the embeddings to a contiguous 2D NumPy array of dtype float32.
- It sends a **4-part multipart** ZeroMQ message:
  1. Header JSON (job_id, status, modality_name, shape, dtype)
  2. Raw binary bytes of the embeddings array
  3. Checksums as JSON
  4. Metadata as JSON

The server reconstructs the array, validates lengths, and stores embeddings in the database.

---

## 3. Model Loading and Caching

The worker uses the same model catalogue as the server: [src/rivault_models.json](../src/rivault_models.json).

### 3.1 HF-Based Models (SentenceTransformer, TransformerEncoder)

- Examples: `clip-ViT-B-32`, `all-mpnet-base-v2`, `intfloat/e5-base-v2`.
- These models are expected to be cached under a shared model cache directory.
- The Docker stack mounts `model_cache:/shared/models` into both server and worker, and environment variables direct HF/SentenceTransformers caches there:
  - SENTENCE_TRANSFORMERS_HOME=/shared/models/sentence_transformers
  - HF_HOME=/shared/models/hf
- The helper umrs.cache_paths.find_model_cache(model_name) looks for the model in those caches, supporting common HF and sentence-transformers naming patterns.
- On the **worker**, HF-based models are only loaded if the cache is already present. The server is responsible for downloading them (e.g. during initial configuration), after which the worker reuses the shared cache.

### 3.2 Non-HF Models (Resnet 50, MAE)

- Example: `Resnet 50` is configured with a checkpoint path in rivault_models.json:
  - "/shared/models/nvidia_resnet50_200821.pth.tar".
- The worker does **not** require a HF cache for these.
- If a configured checkpoint path is missing, the worker logs a warning and falls back to default weights (e.g. ResNet50_Weights.DEFAULT), instead of failing with a cache error.

---

## 4. Docker Compose Deployment

The reference single-node deployment is defined in [docker/docker-compose.yml](../docker/docker-compose.yml).

Key points:

- Services:
  - umrs-server: main UMRS service, uses RemoteEmbeddingBackend by default in this setup.
  - umrs-embed-worker: embedding worker container.
  - minio: S3-compatible object store.
  - postgres: Postgres + pgvector.
  - umrs-api-test: integration test client.
- Shared volumes:
  - model_cache:/shared/models (mounted into server and worker for model caches and checkpoints).

### 4.1 Typical Workflow

From the docker/ directory:

1. Build the images:

   ```bash
   docker compose build umrs-server umrs-embed-worker
   ```

2. Start the stack and run the integration test:

   ```bash
   docker compose up
   ```

3. On the first run:
   - The server may download HF models (e.g. clip-ViT-B-32, all-mpnet-base-v2, intfloat/e5-base-v2) into the shared model_cache volume.
   - While caches are empty, the worker may initially fall back to server-side computation.

4. Subsequent runs **without** `docker compose down -v` reuse the warmed model_cache, and the worker loads models directly from the cache.

### 4.2 Notes

- Avoid `docker compose down -v` unless you deliberately want to clear Postgres and model_cache; otherwise, caches will be reused by the worker.
- The embedding worker is configured to use CUDA if available (`EMBEDDING_DEVICE=cuda`); adjust to `cpu` if you run on a machine without a GPU.

---

## 5. Kubernetes & Kind (Future/External Deployment)

The current implementation is production-ready for Docker Compose on a single node. For multi-node or cloud environments, the recommended next step is to run UMRS and the embedding worker in Kubernetes, keeping the same separation of concerns:

- A UMRS server deployment (with Postgres and MinIO, or binding to managed equivalents).
- One or more embedding worker deployments, each with access to the same S3 and model cache volumes.

For local Kubernetes development, a Kind-based setup is available under docker/ (kind-config.yaml, setup_kind_k8s.sh, rivault-deployment.yaml, rivault-service.yaml). The intended pattern is:

1. Start a Kind cluster with a local Docker registry.
2. Build and push UMRS and worker images into the local registry.
3. Deploy RiVault (if used), then UMRS server and worker pods referencing the same model cache PersistentVolume.
4. Configure UMRS to use RemoteEmbeddingBackend pointing at the worker service address.

These steps can be refined as we move from prototype to full Kubernetes deployment, but they reuse the same worker binary and embedding backend design described above.
