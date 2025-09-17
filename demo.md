# U.M.R.S. Demo and Usage Guide

This guide shows how to run the UMRS server and client locally or remotely, how to use the API-oriented client commands, and how to verify data in PostgreSQL.

UMRS stores raw objects in S3 (e.g., MinIO), computes embeddings, and writes rows into Postgres tables (with pgvector). Buckets and tables are configured per user, experience, and modality (image, text, text file).

Note on metadata: Metadata keys must be declared in the configuration. Unknown keys are rejected. See the “Metadata” section below.

## 1) Prerequisites

- Python 3.10+ (for local/bare-metal)
- PostgreSQL 15+ with pgvector extension
- S3-compatible storage (e.g., MinIO) and s3cfg files for admin and users
- Repo dependencies installed (see `requirements.txt`)

Optional: Docker and Docker Compose (for an all-in-one local stack).

## 2) Local (bare-metal) quick start

We recommend running from the `src/` directory so script paths and config defaults work out-of-the-box.

```
cd src
```

### 2.1 Start the UMRS server

Server reads the server configuration (which references user configs) and initializes S3 buckets, DBs, and tables as needed.

```
./server_umrs.py -c ./conf_umrs.json
```

Wait until you see “Server configuration done!” and “Waiting for new objects…”. Keep it running.

### 2.2 Use the client (API mode)

The client uses a per-user config (`-c users_json/user1.json`) that defines the user’s experiences, modalities, and bucket aliases.

- Ping the server (no config required):

```
./client_umrs.py ping
```

- List the configured buckets for a user (username inferred from `-c` when omitted):

```
./client_umrs.py -c users_json/user1.json --output text list_buckets
```

- Upload a single inline text to the text bucket (user1’s `bucket-2`):

```
./client_umrs.py -c users_json/user1.json --output text -k keywords:greeting \
	upload_text "Hello world" bucket-2
```

- Upload multiple inline texts at once:

```
./client_umrs.py -c users_json/user1.json --output text -k keywords:batch \
	upload_text_batch bucket-2 --texts "alpha one" "beta two" "gamma three"
```

- Upload a text file as a single “text file” object (user1’s `bucket-5`):

```
./client_umrs.py -c users_json/user1.json --output text -k keywords:beast \
	upload_textfile ../data/texts/beast_in_the_jungle.txt bucket-5
```

- Upload an image (user1’s `bucket-1` expects only `keywords`):

```
./client_umrs.py -c users_json/user1.json --output text -k keywords:demo \
	upload_image ../data/img_test/cat.jpeg bucket-1
```

- Upload an image to the “rich metadata” bucket (user1’s `bucket-3`):

```
./client_umrs.py -c users_json/user1.json --output text \
	-k keywords:cat -k author:demo -k x_resolution:256 -k y_resolution:512 \
	upload_image ../data/img_test/cat.jpeg bucket-3
```

Notes
- Legacy mode is still available (positional `data bucket`), but API mode is recommended.
- If your server runs on another host, set `UMRS_SERVER_HOST=<server-hostname>` or pass `--server-ip <host>`.

### 2.3 Verify data in PostgreSQL

User1’s “Text and Images” experience uses the database `text_and_images_usr1_2`. Example queries:

```
psql -d text_and_images_usr1_2 usr1_2
SELECT s3_object, _content, keywords, date, time FROM text ORDER BY time;
```

Check isolation: the wrong user should not be able to read the DB:

```
psql -d text_and_images_usr1_2 usr1_1
SELECT s3_object, _content, keywords, date, time FROM text ORDER BY time;
```

For images in the same experience:

```
psql -d text_and_images_usr1_2 usr1_2
SELECT s3_object, date, time FROM image ORDER BY time;
```

## 3) Local via Docker Compose (optional)

There’s a ready-to-run stack under `docker/` that brings up Postgres, MinIO, and the UMRS server, and runs an integration test container.

From the repo root:

```
docker compose -f docker/docker-compose.yml up --build --abort-on-container-exit --renew-anon-volumes
```

This will:
- Configure users, buckets, DBs/tables
- Run a test script that exercises ping, list_buckets, uploads, and server-side check/update/recompute

To run interactively against the compose server, you can exec into the test container or run a shell container with `/app/src/client_umrs.py` using `UMRS_SERVER_HOST=umrs-server`.

## 4) Remote deployment (API)

Run the server on a remote host where Postgres and S3 are accessible from clients.

On the server host:

```
cd /path/to/umrs/src
./server_umrs.py -c ./conf_umrs.json
```

Ensure TCP port 5555 is reachable by clients.

On the client host:

```
cd /path/to/umrs/src
./client_umrs.py --server-ip <server-ip-or-dns> ping
./client_umrs.py -c users_json/user1.json --server-ip <server-ip-or-dns> \
	--output text list_buckets
```

Then use the same upload commands as in the local section (add `--server-ip` or set `UMRS_SERVER_HOST`).

## 5) Metadata

Metadata keys must be declared in the configuration for the target table/modality. Unknown keys are rejected.

For the default user1 config:
- Text (bucket-2 → table `text`): `keywords`
- Text file (bucket-5 → table `text_file`): `keywords`
- Image (bucket-1 → table `image`): `keywords`
- Image (bucket-3 → table `image`): `keywords`, `author`, `x_resolution`, `y_resolution`

Example (image rich metadata):

```
./client_umrs.py -c users_json/user1.json --output text \
	-k keywords:cat -k author:demo -k x_resolution:256 -k y_resolution:512 \
	upload_image ../data/img_test/cat.jpeg bucket-3
```

## 6) Extras: clustering examples

To experiment with clustering using pgvector-embedded data:

```
../tests/test_embeddings/cluster_txt.py
../tests/test_embeddings/cluster_img.py
```

## 7) Troubleshooting

- “Unknown metadata key(s)…”: Add the key(s) to the `sql.meta_data` list for the modality in the config and restart the server.
- “No bucket … for modality … and user …”: Verify the bucket alias and choose an alias that matches the modality for that user.
- Client can’t reach server: set `UMRS_SERVER_HOST` or `--server-ip` to the server hostname/IP; ensure TCP 5555 is reachable.
- DB query errors: confirm the DB name pattern is `<xp_name>_<sql_username>` lowercased (e.g., `text_and_images_usr1_2`).

# U.M.R.S. Demo Guide

## 1. Prepare the Environment

Before running commands, switch to the source directory so paths and defaults match the code layout.

```
cd src
```

### ➤ Switch to the demo branch

`git checkout demo`

### ➤ [Optional] Launch the local S3 service (e.g., MinIO)

`minio server /data --console-address ":9001"`

### ➤ Delete minIO users
`mc admin user rm myminio user1 --insecure`
`mc admin user rm myminio user2 --insecure`


## 2. Terminal Windows Setup

### Window 1 — PostgreSQL

1. Connect to PostgreSQL and drop existing demo databases and delete users:

```
psql -U postgres
\t 
drop database images_and_texts_usr1_1;
drop database images_and_texts_usr2_1;
drop database text_and_images_usr1_2;
\du 
drop user usr1_1;
drop user usr1_2;
drop user usr2_1;
```

### Window 2 — Server

1. Launch the U.M.R.S. server:

`./server_umrs.py`

Wait for the server to finish setup and listen for incoming requests.

### Window 3 — Client

The client supports two modes that are both available in the current implementation:

- Legacy mode (positional arguments): convenient for quick uploads
- API mode (subcommands): explicit, structured operations (also used by Docker tests)

Basic usage examples for both are shown below.

## 3. Upload and Query Data

### 3.0 Basic client usage

- Check server readiness (API mode, no config required):

```
./client_umrs.py ping
```

- If your server runs on another host/container, either set the environment variable `UMRS_SERVER_HOST` or pass `--server-ip <host>` to the client.

- List configured buckets for a given user (API mode):

```
./client_umrs.py -c users_json/user1.json --output text list_buckets user1
```

- Upload inline text (choose the text bucket alias from list_buckets):
	- Legacy mode:

```
./client_umrs.py -c users_json/user1.json "Hello world" bucket-2
```

		- API mode (metadata via -k; username inferred from config):

```
	./client_umrs.py -c users_json/user1.json --output text -k keywords:greeting upload_text "Hello world" bucket-2
```

- Upload an image file (API mode):

```
./client_umrs.py -c users_json/user1.json --output text -k keywords:demo upload_image ../data/img_test/cat.jpeg bucket-1
```

- Upload a full text file to the "text file" modality with chunking (API mode):

```
./client_umrs.py -c users_json/user1.json --output text upload_textfile ../data/texts/beast_in_the_jungle.txt bucket-5
```

### 3.1 Submit text examples
```
./client_umrs.py -c users_json/user1.json "I eat an apple at breakfast" bucket-2
```

### 3.2 Query SQL to verify

```
psql -d text_and_images_usr1_2 usr1_2
SELECT s3_object, _content, keywords, date, time FROM text ORDER BY time;
```

Check that credential isolation works and that you cannot read the database if you are not the right user: 

```
psql -d text_and_images_usr1_2 usr1_1
SELECT s3_object, _content, keywords, date, time FROM text ORDER BY time;
```
### 3.3 Upload a full text file

Note: For the "text" modality, providing a filename will read and upload the file line-by-line as separate text objects.

`./client_umrs.py -c users_json/user1.json apple.txt bucket-2`

### 3.4 Check results in DB

```
psql -d text_and_images_usr1_2 usr1_2
SELECT s3_object, _content, date, time FROM text ORDER BY time;
```

### 3.5 Cluster texts using PGVector

`../tests/test_embeddings/cluster_txt.py`

## 4. Work with Images

### 4.1 Upload an entire image directory

` ./client_umrs.py -c users_json/user1.json ../data/img bucket-1`

### 4.2 Query the image table

```
psql -d text_and_images_usr1_2 usr1_2
SELECT s3_object, date, time FROM image ORDER BY time;
```

### 4.3 Cluster images using PGVector

`../tests/test_embeddings/cluster_img.py`

## 5. Metadata & Keywords

### Valid metadata keys per modality/bucket

Metadata provided with -k must match the configured schema per bucket/modality. Unknown keys are rejected as errors—add them to the config first.

- Text (user1, bucket-2 → table text):
	- keywords
- Text file (user1, bucket-5 → table text_file):
	- keywords
- Image (user1, bucket-1 → table image):
	- keywords
- Image (user1, bucket-3 → table image):
	- keywords, author, x_resolution, y_resolution

### Upload images with/without keywords

```
./client_umrs.py -c users_json/user2.json ../data/img/Cat_2.jpeg bucket-1
./client_umrs.py -c users_json/user2.json ../data/img/Cat_1.jpeg bucket-1 -k keywords:cat -k author:internet -k x_resolution:256 -k y_resolution:512
```

### Query metadata

```
psql -d images_and_texts_usr2_1  usr2_1
SELECT id, s3_object, date, time, keywords, author, x_resolution, y_resolution FROM image ORDER BY time;
```

## 6. Upload a Full Text File as a Document

### Submit a large file

`./client_umrs.py -c users_json/user1.json ../data/texts/beast_in_the_jungle.txt bucket-5`
### Inspect chunking results

```
psql -d images_and_texts_usr1_1 usr1_1
```

```
SELECT _content, chunk_start, chunk_end, chunk_md5
FROM text_file
WHERE s3_object ~* 'beast'
ORDER BY chunk_start;
```


## 7. Wrap Up

### Return to the main branch

`git checkout main`
