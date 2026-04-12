# UMRS Project Overview

This document explains what the UMRS project does, how it is structured, how to run it with Docker, and how to understand and extend the implementation.

## What is UMRS?

UMRS is a small, config-driven system that:

- Ingests multimodal data into S3-compatible object storage (e.g., MinIO)
- Computes embeddings for those objects using configurable models
- Stores the embeddings and metadata in PostgreSQL with the pgvector extension
- Exposes a client CLI to upload objects and trigger server-side operations via a ZeroMQ control channel

It supports multiple users and “experiences” (projects), each with multiple modalities (e.g., `image`, `text`, `text file`) configured in JSON.

## High-level architecture

- S3-compatible storage: Objects are written to S3 (MinIO in Docker) under per-user, per-experience, per-bucket naming.
- Server: `src/server_umrs.py` runs a long-lived process that listens on ZeroMQ for notifications of new objects, then pulls objects from S3, computes embeddings, and writes them to PostgreSQL tables.
- Client: `src/client_umrs.py` is a CLI that uploads data to S3 and informs the server via ZeroMQ. It also exposes convenience actions like `ping`, `list_buckets`, and maintenance triggers (`recompute`, `update`, `check`).
- Database: PostgreSQL with pgvector stores metadata and embeddings; tables are created/validated from configuration.

## Repository layout (relevant parts)

- `src/server_umrs.py` – server entry point
- `src/client_umrs.py` – client CLI entry point
- `src/conf_umrs.json` – server configuration referencing per-user JSONs
- `src/umrs/` – core modules: S3, SQL, models, message queue, utilities, etc.
- `docker/` – Dockerfiles, compose file, example S3 configs, example user JSONs, and a test script
- `data/` – sample images and models (e.g., ResNet-50 checkpoint)
- `doc/` – documentation (this file and others)

## Configuration model

1) Server config `src/conf_umrs.json` (used by the server image) specifies:

- `admin_s3_configuration_file`: S3 admin credentials (e.g., MinIO root user)
- `user_files_dir`: canonical directory for server-managed per-user JSON files
- `users`: array of paths to per-user configuration files (e.g., `docker/user_files/user1.json`)

2) User config `docker/user_files/userX.json` defines:

- `username`, `fullname`, `s3_configuration_file`
- `experiences`: list with `xp_name`, `sql_username`, and `modalities`
- For each modality:
  - `modality_name`: `image`, `text`, or `text file`
  - `s3.bucket_alias`: logical bucket name
  - `model`: model type and parameters
  - `sql`: whether to store embeddings, table name, and user-defined metadata columns

The server will:

- Create Postgres DBs and tables for each `sql_username`/experience
- Create S3 users/buckets when using MinIO, and write user-specific S3 config files if missing
- Load models according to per-modality configuration

## Data flow at a glance

1. Client resolves the target bucket using its config, connects to the server’s ZeroMQ control plane, and allocates a ZeroMQ data channel.
2. Client uploads the object to S3 (MinIO) and notifies the server:
   - For text: inline content or a text file
   - For images: local file(s)
3. Server receives the notification, groups S3 keys per bucket, downloads the objects, computes embeddings in batch, and stores results in Postgres.
4. For text modalities, the server chunks text (by tokens/sentences), computes embeddings per chunk, and stores chunk metadata and hashes.

## Database schema (per modality)

Tables are created or validated from configuration, with core columns:

- `id SERIAL PRIMARY KEY`
- `s3_object TEXT`
- `checksum TEXT` (S3 ETag)
- `date DATE`, `time TIME`
- If `store_embeddings=yes`: `embedding VECTOR(d)`, `model TEXT`
- User-defined metadata columns as declared in JSON (`name`, `type`)

Text and text file modalities add chunking columns and a uniqueness constraint:

- `_content TEXT`, `chunk_start BIGINT`, `chunk_end BIGINT`, `chunk_md5 TEXT`
- `UNIQUE(s3_object, chunk_start)`

Vector size validation: when storing embeddings, the server validates that the configured model vector size matches the existing table’s vector dimension.

## Models supported

- `SentenceTransformer` (e.g., `all-mpnet-base-v2`, `clip-ViT-B-32`)
- `Resnet-50` (optionally with a checkpoint path)
- `TransformerEncoder` (Hugging Face `AutoModel` + mean pooling)

SpaCy can optionally be used for sentence splitting in text chunking (`spacy_model` setting).

## Running with Docker

The project includes a complete compose stack for local development and testing.

### Services

- `umrs-server`: The Python server, built from `docker/Dockerfile` with the project’s `src/` and `doc/` copied in
- `minio`: S3-compatible object storage with a web console
- `postgres`: PostgreSQL with the pgvector extension
- `umrs-api-test`: A helper container that can run scripted API/client tests

### Start the stack

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Watch the server logs:

```bash
docker logs -f umrs-server
```

MinIO console will be on `http://localhost:9001`, S3 API on `http://localhost:9000`.

### Server configuration inside the container

The server image copies `docker/conf_umrs.json` to `/root/conf_umrs.json` and includes user files from `docker/user_files/` under `/root/user_files`. The server config declares `"user_files_dir": "/root/user_files"`, so uploaded user configs and reloads remain anchored to that directory. It uses `~/.s3cfg` for the admin S3 config, which is mounted from `docker/s3cfg` in compose.

Environment variables used by the server (see `docker/docker-compose.yml`):

- `DB_HOST=postgres`
- `DB_SUPERUSER_PASSWORD=postgres`
- `DB_USER_PASSWORD=ejeannot` (password for created SQL users)
- `UMRS_SERVER_HOST=umrs-server` (used by clients to reach control plane)

### Try the client inside the server container

Check server readiness:

```bash
docker exec -it umrs-server python src/client_umrs.py ping
```

List configured buckets for a user:

```bash
docker exec -it umrs-server python src/client_umrs.py \
   --conf /root/conf_umrs.json \
  list_buckets user1
```

Upload inline text to a `text` bucket alias:

```bash
docker exec -it umrs-server python src/client_umrs.py \
   --conf /root/conf_umrs.json \
  upload_text "I eat apple at breakfast" bucket-2 user1 --meta keywords:fruit
```

Upload an image file to an `image` bucket alias (example data path):

```bash
docker exec -it umrs-server python src/client_umrs.py \
   --conf /root/conf_umrs.json \
   upload_image /root/data/img/Cat_1.jpg bucket-1 user1 --meta keywords:cat
```

Recompute or update embeddings server-side:

```bash
docker exec -it umrs-server python src/client_umrs.py --conf /root/conf_umrs.json recompute
docker exec -it umrs-server python src/client_umrs.py --conf /root/conf_umrs.json update
docker exec -it umrs-server python src/client_umrs.py --conf /root/conf_umrs.json check
```

Alternatively, you can start the `umrs-api-test` container which runs `docker/test_umrs_api.sh` automatically when the stack is up.

## How to understand the codebase

Recommended reading order and roles of key modules:

1. `src/server_umrs.py` – Main server loop
   - Parses `conf_umrs.json`, loads per-user configs
   - Creates S3 users/buckets (MinIO) and DBs/tables as needed
   - Starts a ZeroMQ REP socket (`MessageQueue`) for the control plane
   - On notifications, groups S3 objects per bucket, loads them, computes embeddings, and stores them
   - Supports maintenance operations: `--force-recompute`, `--force-update`, `--check-objects`

2. `src/client_umrs.py` – Client CLI
   - Subcommands: `ping`, `list_buckets`, `upload_text`, `upload_textfile`, `upload_image`, `upload_text_batch`, `recompute`, `update`, `check`
   - Resolves bucket aliases from the user config
   - Establishes a control channel and allocates a per-client data channel (ZeroMQ REQ/REP + PUSH/PULL)
   - Uploads to S3 and notifies the server of new objects

3. `src/umrs/MessageQueue.py` – ZeroMQ wiring
   - Control channel: server uses `REP`, clients use `REQ`
   - Data channel: server `PULL`, client `PUSH`
   - Control commands: `READY?`, `CREATE_DATA_QUEUE`, `OBJECT`, `FILE`, `RECOMPUTE`, `UPDATE`, `CHECK`

4. `src/umrs/s3.py` and `src/umrs/minio.py` – Object storage helpers
   - Connect to S3 via `boto3`
   - Check/create buckets; upload objects; list and fetch objects
   - For MinIO, create users and attach per-bucket policies via `mc`

5. `src/umrs/sql.py` – Database helpers
   - Create databases for each `sql_username` and experience
   - Create/validate tables with pgvector columns and metadata columns
   - Store embeddings per image, and per text chunk with uniqueness on `(s3_object, chunk_start)`

6. `src/umrs/models.py` – Model loading and embedding computation
   - SentenceTransformers, ResNet-50, and generic Transformer encoder
   - Text chunking pipeline via `src/umrs/utils.py`

7. Other utilities
   - `src/umrs/metadata.py` – parse and validate CLI metadata
   - `src/umrs/db_store.py` – associate buckets with DB/models in memory
   - `src/umrs/pretty_print.py` – tabular summaries for configuration and maintenance outputs

## Extending UMRS

- Add a new modality: Update `src/umrs/modality.py` and implement S3 load/store and SQL schema rules analogous to existing modalities.
- Add a new model: Extend `load_model` in `src/umrs/models.py` with the loading code and a compatible `compute_embeddings_*` function that returns either a list (images) or a list of chunk dicts (text).
- Extra metadata: Declare additional columns in user JSON under `sql.meta_data`. The server will validate the table and update rows accordingly when storing embeddings.

## Maintenance and consistency

- `check`: compares S3 objects with DB checksums and reports matches/mismatches/not found
- `update`: recomputes embeddings only for missing/mismatched entries
- `recompute`: recomputes for all objects and overwrites existing entries

## Troubleshooting

- Server not ready: ensure ZeroMQ control port `5555` is open; `docker logs -f umrs-server` to see readiness messages
- Postgres pgvector: pgvector is initialized by `docker/init_pgvector.sql` and by runtime checks; if you run Postgres outside Docker, ensure the extension is installed
- Embedding vector size mismatch: the server will exit if a table’s vector dimension doesn’t match the configured model size; recreate the table or adjust config
- S3 credentials: in Docker the server mounts S3 configs from `docker/s3cfg` and user-specific files from `docker/s3cfg_user*`; ensure paths in JSON resolve inside the container
- Empty text objects: handled by zeroing the embedding and storing chunk metadata with length 0

## Related docs

- `docker/README.md` – quick Docker tips and examples
- `doc/HOWTO.md`, `doc/set-up_pgvector.md` – additional setup information
