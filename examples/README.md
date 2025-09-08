# UMRS Examples

This folder contains a minimal ZeroMQ JSON-RPC client example and instructions to run a local smoke test.

Requirements
- Python 3.8+
- Optional: Docker + Docker Compose for full environment tests
- If running locally (without Docker) install runtime dependencies:
  ```bash
  pip install pyzmq boto3 torch psycopg2-binary
  ```

Run ZeroMQ API server (local, not Docker)
1. Start the server in API mode (this will call into the ZeroMQ API implementation in `src/umrs/api.py`):
   ```bash
   python3 src/server_umrs.py --api-mode --conf docker/conf_umrs_users.json --host 127.0.0.1 --port 8000
   ```

2. In another shell run the example client to call health/status or upload endpoints:
   ```bash
   python3 examples/zeromq_client.py --host 127.0.0.1 --port 8000 health
   python3 examples/zeromq_client.py --host 127.0.0.1 --port 8000 status
   python3 examples/zeromq_client.py --host 127.0.0.1 --port 8000 upload_text --username user1 --bucket_alias mybucket --text "Hello"
   ```

Test with Docker Compose (recommended for full-stack)
1. From the repository root run the provided task which builds and starts the services with Docker Compose.
   ```bash
   # from repository root
   docker compose up --build
   ```

2. Once the stack is up, you can run the ZeroMQ client (if the API service is exposed on a TCP port) or use the included API test scripts in `docker/`.

Notes
- The ZeroMQ API listens on a TCP port when started with `--api-mode`; the client example uses a REQ socket to send JSON payloads like:
  ```json
  {"action": "upload_text", "params": {"username": "user1", "bucket_alias": "mybucket", "text": "Hello"}}
  ```
- Administrative actions (create_user, create_buckets, recompute, update, check) require the server-side `server_umrs` functions; if the server cannot import `server_umrs` those actions will return an error.

If you want me to run the Docker Compose test now, tell me to proceed and I will run the configured task and report the output.
