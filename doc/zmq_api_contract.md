# UMRS ZeroMQ API Contract

This document defines the ZeroMQ-only control and data planes between client(s) and server.

## Channels
- Control channel: `REQ` (client) ↔ `REP` (server) on `tcp://<server-host>:5555`.
- Data channel: Per-client `PUSH` (client) → `PULL` (server) on dynamically assigned ports `> 5555`.

## Environment
- `UMRS_SERVER_HOST`: When set on the server process, it is used to advertise the routable host in data channel endpoints returned to clients (e.g., container name `umrs-server`).

## Control Commands
1) READY?
- Request: `READY?`
- Response: `READY!`
- Purpose: Liveness/handshake; client retries once on timeout.

2) CREATE_DATA_QUEUE
- Request: `CREATE_DATA_QUEUE:<username>`
- Response: `tcp://<server-host>:<port>`
- Behavior: Server allocates or reuses a dedicated PULL socket bound on `<port>` and maps it to `<username>`; returns a connectable endpoint.

3) OBJECT
- Request: `OBJECT:<username>:<count>`
- Response: `ACKNOWLEDGE` | `NO_DATA_QUEUE`
- Behavior: Server reads `<count>` text messages from the per-client data socket (each is `s3://<bucket>/<key>`), then acknowledges. On missing data queue, responds `NO_DATA_QUEUE`.

4) FILE
- Request: `FILE:<username>:<count>`
- Response: `ACKNOWLEDGE` | `NO_DATA_QUEUE`
- Behavior: Server receives `<count>` files over the per-client data socket using chunked transfer terminated by `__END_OF_FILE__`, writes them under `./tmp`, then acknowledges.

## Data Channel Payloads
- S3 event notification: `notify_s3(s3_object)` sends a single frame string like `s3://<bucket>/<key>`.
- File transfer: `send_file(path)` sends binary chunks followed by `__END_OF_FILE__` sentinel.

## Timeouts and Reliability
- Client REQ socket uses send/recv timeouts and retries once for `READY?`.
- Server REP/ PULL sockets use blocking receive on control, and configured timeouts on data sockets.
- HWM: Client `PUSH` uses `SNDHWM`, server `PULL` uses `RCVHWM`, default `10240` messages.

## Error Handling
- `NO_DATA_QUEUE`: Server couldn’t find an existing data queue mapping for the username.
- Unexpected replies raise runtime errors on the client.
- File not found or wrong modality errors are surfaced at the client CLI and do not send notifications.

## Versioning
- Base port: `5555`.
- Protocol strings are stable: `READY?`, `READY!`, `CREATE_DATA_QUEUE`, `OBJECT`, `FILE`, `ACKNOWLEDGE`, `NO_DATA_QUEUE`, `EXPORT_EMBEDDINGS`.

## Additional Control Commands

5) STATUS
- Request: `STATUS:<username>:<key>`
- Response: JSON payload with task summaries
- Behavior: Server returns status overview for the given user/key. Synchronous request/response on the control channel.

6) RECOMPUTE / UPDATE / CHECK / RELOAD
- Request: `<CMD>:<username>:0`
- Response: `ACKNOWLEDGE`
- Behavior: Server schedules the maintenance command and acknowledges immediately. The operation runs in a background thread.

7) WAIT_OUTPUT
- Request: `WAIT_OUTPUT:<username>:<bucket_name>;<timeout>`
- Response: Status message (blocking until done or timeout)
- Behavior: Server waits for DB table modifications on the bucket's table and replies when new data is detected.

8) EXPORT_EMBEDDINGS
- Request: `EXPORT_EMBEDDINGS:<username>:<path_to_bucket>;<format>`
- Response: JSON payload `{"status": "ok", "database": "...", "table": "...", "columns": [...], "rows": [[...], ...], "rows_exported": N, "format": "csv|jsonl"}`
- Behavior: Server queries the pgvector database for all embeddings in the table associated with the given bucket path. Results are serialised as JSON and returned inline on the control channel. The client writes the data to a local file. **The client never connects directly to the database** — the server mediates all DB access (Client → Server → Database).
- Error: `{"status": "error", "message": "..."}`
