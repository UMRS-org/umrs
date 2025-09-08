# Merge / ZeroMQ transition — Memo & TODO

Date: 2025-09-04

Purpose
-------
This memo summarizes the goals of the `main` -> `multi_users` src-only merge and the transition of the API implementation from FastAPI to a ZeroMQ control/data-plane. It records what has been completed, remaining tasks, and suggested next steps.

Objectives
----------
- Merge `origin/main` updates for the `src/` subtree into branch `multi_users`, preserving multi-user behavior.
- Replace the FastAPI+uvicorn API with a ZeroMQ-based API implementation while preserving the lifecycle contract:
  - `create_api_server().initialize_server(configure_file)`
  - `run(host, port, ssl_keyfile, ssl_certfile)`
- Preserve upload flow so that uploads still call `MessageQueue.notify_s3(...)` after writing to S3.
- Integrate selective ZeroMQ control/data-plane improvements from `origin/main` (file-transfer helpers, control-plane messages) while keeping existing IPC backends (posix/sysv) compatible where practical.

Work status (what's done)
-------------------------
- Performed a cautious, src-only merge and resolved many conflict markers in API, SQL, MinIO and other modules.
- Replaced the FastAPI API with a ZeroMQ JSON-RPC server implementation preserving `initialize_server()` and `run()` semantics.
- Updated `src/umrs/s3.py` to support verify with the system CA bundle and fixed a couple of `sys.exit` message bugs.
- Added ZeroMQ file-transfer helpers to `src/umrs/MessageQueue.py`: `send_file()` and `recv_file()` (ZeroMQ-only).
- Added a control-plane compatibility wrapper: `server_control_queue_loop()` in `MessageQueue` and updated `src/server_umrs.py` to handle `HANDSHAKE` and `OBJECT` flows.
- Added a `client` helper `send_files_to_server()` in `src/client_umrs.py` to demonstrate client-side use of the dataplane helpers.
- Created an initial smoke-test `tests/test_zmq_smoke.py` (receives a file on the server side and compares contents).
- Committed the above changes to branch `multi_users` (commits include the ZeroMQ helpers and control-plane compatibility wrapper).

Open issues / problems encountered
---------------------------------
- Docker Compose integration tests failed initially due to S3 configuration and later MinIO TLS verification errors (self-signed certs). `s3.connect_to_s3()` was updated to use the system CA bundle but Docker environment validation is still required.
- `origin/main` introduced richer control-plane commands (CREATE_DATA_QUEUE, more structured object/file commands). These were examined and partially integrated in a compatibility-preserving fashion, but the full `main` control-plane command set was not adopted wholesale.
- Some example/demo/test files outside `src/` were left unmerged intentionally to avoid breaking changes.

Todo (short-term / high priority)
--------------------------------
```markdown
- [ ] Run the local ZeroMQ smoke test (`tests/test_zmq_smoke.py`) and confirm send_file/recv_file and handshake flows work outside Docker.
- [ ] Re-run Docker Compose stack and validate MinIO TLS/CA trust: confirm `s3.connect_to_s3()` + Dockerfile CA installation resolves `CERTIFICATE_VERIFY_FAILED`.
- [ ] If TLS still fails in Docker, choose one of:
  - Mount a proper CA bundle into the container and pass its path to `boto3`'s `verify` parameter (preferred), or
  - Temporarily disable SSL verification for local integration testing (set `check_ssl_certificate=false` in `docker/s3cfg` only for dev), or
  - Create a CA-signed certificate (dev CA) and add its CA cert to the container trust store.
- [ ] Add unit/integration tests for control-plane commands: `HANDSHAKE`, `FILE`, `OBJECT`, and (optionally) `CREATE_DATA_QUEUE`.
- [ ] Implement the full `origin/main` control-plane commands in a ZeroMQ-optional way (i.e., enable when `queue_type=='zeromq'`) and keep POSIX/SYSV behavior unchanged.
- [ ] Update client examples and documentation to include the ZeroMQ client helpers (e.g., `send_files_to_server`) and how to invoke them.
```

Possible additions / medium-term
--------------------------------
- Implement a small JSON-RPC compatibility layer so existing client scripts that expected HTTP/REST can be adapted more easily to ZeroMQ (adapter pattern).
- Add end-to-end CI that runs the Docker Compose stack on a runner with a configured CA to validate TLS flows automatically.
- Add a small `zmq`-based test harness for control-plane command sequences (handshake, send file, ack flow) and include it in `tests/`.
- Add logging and metrics around file-transfer progress and control-plane command latency for observability.

Quick how-to (smoke test)
-------------------------
Run the local smoke test (no Docker required):

```bash
PYTHONPATH=src python3 tests/test_zmq_smoke.py
```

Expected result: `SMOKE TEST PASS: received file matches` (exit code 0). If it fails, collect printed output and logs; common causes are port binding timing or missing `pyzmq` in the running Python environment.

Requirements coverage mapping (high level)
---------------------------------------
- Merge only `src/` contents from `main`: Done (selective, conflict-resolved).
- Replace FastAPI with ZeroMQ while preserving lifecycle: Done (ZeroMQ server implements `initialize_server()` and `run()`).
- Preserve upload flow (`MessageQueue.notify_s3`): Done (s3 store functions call `mq.notify_s3`).
- Keep multi-backend compatibility: Partially done (ZeroMQ helpers added without removing posix/sysv implementations).

Next steps (what I will do if you want me to continue)
-----------------------------------------------------
1. Run the `tests/test_zmq_smoke.py` locally and report results (quick).  
2. If the smoke test passes, re-run `docker compose up --build` and collect logs to confirm the MinIO TLS issue is resolved.  
3. Integrate additional control-plane commands from `origin/main` in a backward-compatible way and add tests.

Contact / Notes
---------------
If you want I can run the smoke test now, then proceed to rebuild the Docker stack and iterate on the TLS issue — tell me to proceed and I'll run each step and report back.
