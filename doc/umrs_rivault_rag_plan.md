# UMRS × RiVault × internal_genAI – RAG Integration Plan
# demos/slurm-dev: Integration Alignment

The `demos/slurm-dev` directory is designed as a practical testbed for the UMRS × RiVault × internal_genAI integration. It provides:
  - Example configuration files (e.g., `conf_umrs.json`, `docker-compose.yaml`) for orchestrating UMRS, RiVault, and related services.
  - Dockerfiles and entrypoints for launching UMRS server, embedding worker, and RiVault mock/real registry.
  - Scripts for initializing databases, MinIO, and populating demo data.
  - Test data and user files for validating end-to-end flows (model registry, embedding, retrieval, RAG API).
  - A reference for how to wire UMRS to RiVault endpoints and test retrieval via `/rag/search`.

This directory is the recommended starting point for local and CI testing of the integration plan described below.


## 1. Context & Goals

We have two main systems:

- **UMRS** (this repo):
  - Ingests data (S3/MinIO), computes embeddings, stores them in **Postgres + pgvector**.
  - Supports **remote embedding** via `embedding_worker.py`.
  - Already integrates with a **RiVault-like model registry** via `RIVAULT_ENDPOINT` and `/api/models/**` HTTP calls.

- **internal_genAI**:
  - Runs a **GenAI stack** with:
    - `nginx` reverse proxy.
    - `openwebui` chat UI with built-in RAG support.
    - LLM backends: **vLLM** (OpenAI-compatible API) and **Ollama**.
    - Optional **RagFlow** stack for full-featured RAG (Elasticsearch, MinIO, MySQL, Redis).

**High-level goals:**

1. Replace/augment UMRS’s local **RiVault mock** with a minimal, real RiVault setup based on internal_genAI.
2. Keep UMRS’s **model loading and weight distribution** working via HTTP (`/api/models`, `/api/models/{model_name}/weights`).
3. Expose UMRS’s **pgvector embeddings** through a simple HTTP retrieval API.
4. Make **OpenWebUI** (and optionally RagFlow) consume that retrieval API for RAG.
5. Provide a clear test and rollout path.

---

## 2. Minimal RiVault Infrastructure (Based on internal_genAI)

### 2.1 Components

Use internal_genAI’s existing services:

- **Reverse proxy**: `nginx` (`genai-docker-compose.yaml`, `nginx_fs/nginx.conf`).
- **UI & Orchestration**: `openwebui`.
- **LLM backends**:
  - `vllm` (HTTP OpenAI-like API on port 8000).
  - `ollamaweb`/`ollamacli` (Ollama API on port 11434).
- **RAG ecosystem (optional but present)**:
  - `ragflow` stack (`ragflow.yaml`), accessed via `nginx` on port 8080.

### 2.2 Model Registry API (“RiVault”)

UMRS expects:

- `GET /api/models/{model_name}` → JSON:
  ```json
  {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "config": { "model_type": "...", "vector_size": ..., ... }
  }
  ```
- `GET /api/models/{model_name}/weights` → tarball of HF cache / checkpoint files.
- `PUT /api/models/{model_name}/weights` → upload tarball (best-effort).

**Plan:**

- **Option A (simple and aligned with current code)**:
  - Package existing UMRS `rivault_service.py` as a container inside internal_genAI.
  - Mount:
    - A catalog JSON (e.g., `rivault_models.json`).
    - A directory for weights (`rivault_models_store` or similar).
  - Expose it internally as `rivault:8000` (or similar).
  - Add `nginx` routing:
    - `location /api/models/ { proxy_pass http://rivault:8000; ... }`.

- **Option B (if internal_genAI already has a true RiVault)**:
  - Ensure that implementation exposes the **same** URL shapes and JSON payloads as `rivault_service.py`.
  - Adjust UMRS client code only if necessary (prefer not to).

---

## 3. UMRS ↔ RiVault Integration

### 3.1 Environment & Network

- In UMRS:
  - For **Docker Compose**:
    - Replace `RIVAULT_ENDPOINT=http://rivault-mock:8000` with the real endpoint, e.g.:
      - `RIVAULT_ENDPOINT=http://nginx:8000` (if `/api/models` proxied there), or
      - `RIVAULT_ENDPOINT=http://rivault:8000` (direct service).
  - For **Kubernetes**:
    - `docker/k8s/configmaps.yaml` currently sets:
      - `RIVAULT_ENDPOINT: http://rivault:8000`
    - Ensure `rivault` service points to real registry API.

- Ensure **embedding_worker** can reach `RIVAULT_ENDPOINT`:
  - It uses `--rivault-endpoint` / `RIVAULT_ENDPOINT` in `embedding_worker.py`.

### 3.2 Model Catalog & Configs

- Define a set of models for embeddings:
  - **Text**: `sentence-transformers/all-MiniLM-L6-v2` (already used by OpenWebUI).
  - Optionally: other text/vision models (TransformerEncoder, MAE, ResNet-50).

- For each model, define a RiVault catalog entry (compatible with `umrs.models_loading.load_model`):
  - `model_name`
  - `model_type` (`SentenceTransformer`, `TransformerEncoder`, `MAE`, `Resnet-50`)
  - `vector_size`
  - Optional: `model_cfg_file`, `checkpoint_path`, `spacy_model`, etc.

### 3.3 Weight Distribution Flow

- **Server side (UMRS)** – in `server_umrs.py`:
  - During user modality configuration, `configure_user` already calls:
    - `_upload_weights_to_rivault(model_cfg.get("model_name"), tmp_dir_or_user_conf_dir)`
  - That:
    - Tar-balls the **local HF cache** for the model (`find_model_cache`).
    - Uploads to `PUT /api/models/{safe_model_name}/weights` on RiVault.

- **Worker side** – in `embedding_worker.py`:
  - On first `model_name` use:
    - `_ensure_weights_from_rivault(endpoint, model_name)`:
      - Downloads `GET /api/models/{model_name}/weights`.
      - Extracts into HF cache roots with `_extract_weights_archive`.
    - `_fetch_model_cfg(endpoint, model_name)`:
      - Calls `GET /api/models/{model_name}` for config.
    - `load_model(model_cfg, modality_name)`:
      - Loads the model using UMRS’s normal loaders.

**Success criteria:** embedding_worker loads and uses models without hitting the Internet, using only RiVault for both config and weights.

---

## 4. RAG Retrieval Layer on UMRS

### 4.1 Retrieval API Design

Goal: expose UMRS’s pgvector embeddings for RAG:

- New service (FastAPI or Flask) running alongside UMRS, e.g. `umrs-rag-api`.

**Minimal endpoints:**

- `POST /rag/search`
  - Request:
    ```json
    {
      "query": "user question",
      "user": "optional-umrs-user",
      "modality": "text",
      "top_k": 5,
      "filters": {
        "bucket_alias": "...",
        "xp_name": "..."
      }
    }
    ```
  - Behaviour:
    1. Encode `query` with **same embedding model** as used for corpus:
       - Either:
         - Call UMRS/embedding_worker’s `RemoteEmbeddingBackend` with a “query-only” mode, or
         - Use the same SentenceTransformers model directly in this API (provided config matches UMRS).
    2. Query Postgres+pgvector:
       - `SELECT ... ORDER BY embedding <-> :query_embedding LIMIT :top_k`.
       - Filter by user / modality / bucket as needed.
    3. Return matched rows + metadata (e.g., text, titles, file IDs).

  - Response:
    ```json
    {
      "results": [
        {
          "id": "row-id",
          "score": 0.12,
          "content": "text snippet or full doc",
          "metadata": {
            "bucket": "...",
            "s3_key": "...",
            "modality": "text",
            "user": "..."
          }
        }
      ]
    }
    ```

### 4.2 Implementation Notes

- Prefer to **reuse UMRS config** to locate:
  - `db_model_store` mappings.
  - Exact table names and vector column names.
- Use direct SQL (via psycopg2/sqlalchemy) or a small ORM wrapper; keep it minimal:
  - UMRS already handles data ingestion and embedding; retrieval layer should just read.

- Optionally wrap this in a small library/layer if you later want to plug into:
  - **LangChain**’s `PGVector` integration, or
  - **LlamaIndex**.

### 4.3 Concrete LangChain-based Implementation (Chosen Option A)

- Implementation lives in the UMRS repo as a small FastAPI app:
  - Service entrypoint: `src/umrs/api/rag_service.py`.
  - Uses **LangChain** (`langchain`, `langchain-community`) and **SentenceTransformers** for query embeddings.
- Startup behaviour:
  - Resolves the UMRS server config JSON similar to `server_umrs.py`:
    - Uses `UMRS_CONFIG` (if set) or falls back to `~/.config/umrs/conf_umrs.json`.
  - Calls `load_server_cfg_file` and `server_umrs.configure_user` for each user to rebuild `cfg_store`.
  - Reads `cfg_store['db_model_store']` to discover available embedding tables and their `data_base`/`table_name`/`modality_name`/`bucket_alias`/`xp_name`.
  - Initializes a LangChain `SentenceTransformerEmbeddings` with model name:
    - `UMRS_RAG_EMBEDDING_MODEL` env var, defaulting to `sentence-transformers/all-MiniLM-L6-v2`.
- Custom retriever:
  - Class: `UMRSPgVectorRetriever(BaseRetriever)`.
  - Filters `db_model_store` entries by:
    - `user` (UMRS `s3_username`), `xp_name`, `bucket_alias`, `modality` (e.g. `"text"`).
    - Requires exactly **one** matching table with `store_embeddings_flag = True`.
  - For a given table, computes query embeddings via LangChain and runs a pgvector similarity SQL query:
    - `SELECT id, s3_object, embedding <-> :query_vec AS distance, _content, chunk_* ... FROM table WHERE embedding IS NOT NULL ORDER BY embedding <-> :query_vec LIMIT :top_k;`
  - Wraps each row as a LangChain `Document` with:
    - `page_content` = `_content` for text modalities (or empty string for non-text).
    - `metadata` including `id`, `s3_object`, `distance`, `checksum`, `model`, `chunk_start/end`, `chunk_md5`, `user`, `bucket_alias`, `xp_name`.
- `/rag/search` endpoint (FastAPI):
  - Path: `POST /rag/search` (served by `src/umrs/api/rag_service.py`).
  - Request body:
    ```json
    {
      "query": "user question",
      "top_k": 5,
      "user": "optional-umrs-user",
      "modality": "text",
      "bucket_alias": "optional-bucket-alias",
      "xp_name": "optional-experience-name"
    }
    ```
  - Behaviour:
    1. Builds a `UMRSPgVectorRetriever` with the provided filters and `top_k`.
    2. Calls `get_relevant_documents(query)`.
    3. Converts `Document` objects into JSON with a score computed from distance (`1/(1+distance)`).
  - Response:
    ```json
    {
      "results": [
        {
          "id": 123,
          "score": 0.92,
          "content": "text snippet or full chunk",
          "metadata": {
            "s3_object": "s3://bucket/path/file.txt",
            "bucket_alias": "...",
            "xp_name": "...",
            "user": "...",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_start": 0,
            "chunk_end": 512,
            "distance": 0.086
          }
        }
      ]
    }
    ```

**Note:** For now, the RAG service computes **query embeddings locally** (via `SentenceTransformerEmbeddings`). A later enhancement can replace this with a thin `Embeddings` wrapper that calls UMRS's `embedding_worker` so that both corpus and queries use weights delivered via RiVault only.

---

## 5. OpenWebUI RAG Integration

### 5.1 OpenWebUI Capabilities (from internal_genAI)

- `openwebui` is already configured with:
  - `RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`.
  - `OLLAMA_BASE_URL` and `OPENAI_API_BASE_URLS` pointing at Ollama and vLLM.
  - It can:
    - Do its own embeddings.
    - Use built-in vector stores.
    - Use external tools / HTTP endpoints.

### 5.2 Strategy: UMRS as External Retriever

Instead of letting OpenWebUI embed+store everything itself, we:

- Keep **UMRS as the embedding + storage engine**.
- Expose `/rag/search` as described above.
- Integrate UMRS with OpenWebUI via:

**Option A – Tools / function calling:**

- Define a custom tool in OpenWebUI (via admin UI or config file) that:
  - Calls `POST http://umrs-rag-api:PORT/rag/search`.
  - Returns documents as a JSON array.
- Update the system prompt and/or tool description so the LLM:
  - Calls the tool when it needs context.
  - Uses the `content` fields in the final answer.

**Option B – Custom RAG backend:**

- If your OpenWebUI version supports specifying an external RAG backend:
  - Configure it to use UMRS’s `/rag/search` endpoint as its retrieval engine.
  - Let OpenWebUI still handle prompt composition and answer generation.

Start with **Option A** (tools), as it:
- Leaves UMRS independent.
- Requires minimal changes to OpenWebUI.
- Is easy to test and iterate.

---

## 6. Testing & Rollout Plan

### 6.1 Local End-to-End Test

1. **Bring up internal_genAI stack** (`genai-docker-compose.yaml`):
   - `nginx`, `openwebui`, `ollama`, `vllm`, (optionally `ragflow`).
2. **Bring up UMRS stack**:
   - `minio`, `postgres`, `umrs-server`, `umrs-embed-worker`.
   - Set `RIVAULT_ENDPOINT` to the real RiVault registry endpoint (through nginx or directly).
3. **Register models in RiVault**:
   - Add catalog entries for at least one text model (e.g. `all-MiniLM-L6-v2`).
   - Run UMRS configuration so `_upload_weights_to_rivault` populates registry weights.
4. **Ingest sample data** via UMRS:
   - Text documents in S3/MinIO.
   - Check that embeddings are computed by `embedding_worker.py` and stored in pgvector.
5. **Test retrieval API**:
   - Run UMRS RAG API (`/rag/search`) locally.
   - Use `curl`/Postman to verify good nearest-neighbor results.
6. **Wire to OpenWebUI**:
   - Define a tool that calls `/rag/search`.
   - Ask a question in UI; verify it pulls context from UMRS and answers based on it.

---

## 7. Open Questions / Decisions to Make

2. **Query Embedding Path**
   - For `/rag/search`, should we:
     - Call `embedding_worker` for query embeddings, or
     - Embed directly inside RAG API using the same HF model?

3. **User Isolation**
   - How should OpenWebUI users map to UMRS users/buckets?
   - For first iteration, a single shared UMRS user may be sufficient.

4. **RagFlow Usage**
   - Do we want RagFlow in the loop, or is UMRS + OpenWebUI enough?
   - If yes, RagFlow could consume UMRS as a data source or vice versa.
