## merge_multi_users — short-term todo (updated)

Completed:

- [x] Run the local ZeroMQ smoke test (`tests/test_zmq_smoke.py`) and confirm send_file/recv_file and handshake flows work outside Docker.

High priority next steps:

- [ ] Re-run Docker Compose stack and validate MinIO TLS/CA trust: confirm `s3.connect_to_s3()` + Dockerfile CA installation resolves `CERTIFICATE_VERIFY_FAILED`.
- [ ] If TLS still fails in Docker, choose one of:
  - [ ] Mount a proper CA bundle into the container and pass its path to `boto3`'s `verify` parameter (preferred).
  - [ ] Temporarily disable SSL verification for local integration testing (set `check_ssl_certificate=false` in `docker/s3cfg` only for dev).
  - [ ] Create a CA-signed certificate (dev CA) and add its CA cert to the container trust store.
- [ ] Add unit/integration tests for control-plane commands: `HANDSHAKE`, `FILE`, `OBJECT`, and (optionally) `CREATE_DATA_QUEUE`.
- [ ] Implement the full `origin/main` control-plane commands in a ZeroMQ-optional way (i.e., enable when `queue_type=='zeromq'`) and keep POSIX/SYSV behavior unchanged.
- [ ] Update client examples and documentation to include the ZeroMQ client helpers (e.g., `send_files_to_server`) and how to invoke them.
