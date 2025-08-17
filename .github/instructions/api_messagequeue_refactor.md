---
applyTo: '**'
---

# MessageQueue API Refactor Plan

## Goal
Refactor the FastAPI-based UMRS API to use MessageQueue for embedding computation, making uploads asynchronous and offloading heavy processing to the queue.

## Steps
1. Import and initialize MessageQueue (client mode) in the API server class.
2. Refactor upload endpoints (`/upload/text`, `/upload/image`, `/upload/textfile`) to enqueue S3 object info for embedding computation instead of calling `_process_single_object` directly.
3. Ensure the API returns a response immediately after upload, indicating embedding computation is pending.
4. Add error handling for queue operations.
5. Make queue type configurable (default to zeromq).
6. Update memory and documentation to reflect this change.

## Implementation Notes
- Use `MessageQueue(queue_type='zeromq', role='client')` for API.
- Use `notify_s3(s3_object)` to enqueue embedding computation tasks.
- Remove or comment out direct calls to `_process_single_object` in upload endpoints.
- Optionally, add a status endpoint to check embedding computation progress (future work).

## Benefits
- Decouples upload from embedding computation.
- Improves API responsiveness and scalability.
- Unifies processing logic with non-API mode.

---
