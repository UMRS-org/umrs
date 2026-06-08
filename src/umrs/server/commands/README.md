# `umrs/server/commands` — Developer Guide

This package implements the **Command Handler pattern** for the UMRS server
event loop.  Every ZeroMQ command (`TaskCmd`) is handled by a dedicated class
instead of an `if/elif` chain in `server_umrs.py`.

---

## Why this pattern?

Before this refactoring, `server_event_loop()` in `server_umrs.py` contained
a single `if/elif` block with ~350 lines that grew with every new command.
Adding a command meant editing the core loop, risking regressions across
unrelated commands, and duplicating authorization boilerplate.

With the Command Handler pattern:

- **Each command is isolated** in its own file.
- **Authorization is centralized** in `CommandRegistry.dispatch()` — handlers
  never check permissions themselves.
- **Adding a command = one new file + one line** in `build_registry()`.
- **`task_id` is always available** in every handler, enabling uniform
  tracking, logging, and future observability even for commands that do not
  launch a background thread today.

---

## Package structure

```
commands/
├── README.md               ← this file
├── __init__.py             ← re-exports build_registry for external imports
├── base.py                 ← HandlerContext, CommandHandler ABC
├── registry.py             ← CommandRegistry (authorization + dispatch)
├── registry_builder.py     ← build_registry(): maps every TaskCmd to its handler
│
├── process_objects.py      ← PROCESS_OBJECTS
├── maintenance.py          ← RECOMPUTE, UPDATE, CHECK, RELOAD
├── wait_output.py          ← WAIT_OUTPUT
├── export_embeddings.py    ← EXPORT_EMBEDDINGS
├── file_upload.py          ← FILE
├── enroll_user.py          ← ENROLL_USER
├── upload_workflow.py      ← UPLOAD_WORKFLOW
├── workflow_commands.py    ← LIST_WORKFLOWS, RUN_WORKFLOW
├── model.py                ← MODEL_REGISTER, MODEL_LIST, MODEL_DOWNLOAD
└── job.py                  ← JOB_SUBMIT, JOB_CANCEL, JOB_LIST
```

---

## Core abstractions

### `HandlerContext` (`base.py`)

A dataclass injected into every `execute()` call. It carries **all
server-level dependencies** so handlers never access global state.

```python
@dataclass
class HandlerContext:
    task_manager:          Any      # thread pool + task status store
    control_mq:            Any      # ZeroMQ MessageQueue (send replies)
    server_cfg:            Any      # ServerCfg (buckets, user map)
    device:                str      # "cuda" or "cpu"
    embedding_backend:     Any      # LocalEmbeddingBackend instance
    configure_file:        str      # path to conf_umrs.json
    job_command_backend:   Any      # durable job backend (may be None)
    model_command_backend: Any      # model registry backend (may be None)

    # convenience properties
    ctx.buckets_cfg       # → server_cfg.buckets_cfg
    ctx.user_config_map   # → server_cfg.user_config_map
```

### `CommandHandler` ABC (`base.py`)

Every handler subclasses `CommandHandler` and implements one method:

```python
class CommandHandler(ABC):
    def execute(
        self,
        payload:       Any,             # raw ZeroMQ payload
        update_status: StatusCallback,  # call with TaskStatus.*
        ctx:           HandlerContext,  # all server dependencies
        task_id:       int,             # task identifier (always present)
    ) -> None: ...
```

**Rules for `execute()`:**

| Rule | Detail |
|------|--------|
| Never check authorization | The `CommandRegistry` does it before calling `execute()`. |
| Always call `update_status` | Call exactly once with `COMPLETED`, `ERROR`, or `NO_SUCH_BUCKET`. For async commands (thread launched), the thread calls it when it finishes — do not call it again in `execute()`. |
| Never raise | Catch exceptions internally; the registry catches unhandled ones as a safety net and marks the task `ERROR`. |
| Use `task_id` for thread association | Call `ctx.task_manager.associate_thread_with_task(thread_ident, task_id)` when a background thread is launched. |
| Use `logger`, not `print` | All handlers use `logging.getLogger(__name__)`. |

### `CommandRegistry` (`registry.py`)

Holds the `cmd → handler` mapping.  The event loop calls:

```python
registry.dispatch(cmd, payload, update_status, ctx, task_id)
```

`dispatch()` performs:
1. Look up the handler — mark `UNKNOWN` if not found.
2. Extract `client_id` (first element of payload).
3. Call `control_mq.get_auth_context(client_id).can_execute_command(cmd)`.
4. On failure: send `UNAUTHORIZED:<cmd>` via ZeroMQ (if a sender id is present) and mark `ERROR`.
5. On success: call `handler.execute(payload, update_status, ctx, task_id)`.

Authorization rules live in `umrs/shared/auth.py` (`ADMIN_ONLY_COMMANDS`,
`USER_ALLOWED_COMMANDS`).  Do **not** duplicate them in handlers.

---

## Payload conventions

ZeroMQ payloads follow two shapes depending on whether the command needs to
send a reply:

| Shape | Example commands | Description |
|-------|-----------------|-------------|
| `str` | `RECOMPUTE`, `UPDATE`, `CHECK`, `RELOAD` | The payload **is** the `client_id`. |
| `(client_id, data, zmq_sender_id)` | `MODEL_*`, `JOB_*`, `WAIT_OUTPUT`, `EXPORT_EMBEDDINGS` | Three-tuple; `zmq_sender_id` is used to route the ZeroMQ reply back to the right client. |
| `(client_id, files, ...)` | `FILE`, `ENROLL_USER`, `UPLOAD_WORKFLOW` | Tuple with a file list. |

The `CommandRegistry` extracts `client_id` automatically for authorization.
The handler is responsible for unpacking the rest of the payload.

---

## Thread return values from `handle_maintenance_command`

`maintenance.py` calls `handle_maintenance_command()` which returns one of:

| Return value | Meaning |
|---|---|
| `int` | A background thread was launched (client-triggered, async). Associate with `task_id`. |
| `None` | Ran synchronously and succeeded (server bootstrap context, blocking by design). Mark `COMPLETED`. |
| `False` | Ran synchronously but encountered an error. Mark `ERROR`. |

---

## How to add a new command

### Step 1 — Declare the command string

Add the new constant to `TaskCmd` in `umrs/core/TaskCmdStatus.py`:

```python
class TaskCmd:
    ...
    MY_NEW_CMD = "MY_NEW_CMD"
```

Add it to `USER_ALLOWED_COMMANDS` or `ADMIN_ONLY_COMMANDS` in
`umrs/shared/auth.py` depending on the required permission level.

### Step 2 — Create the handler file

Create `umrs/server/commands/my_new_cmd.py`:

```python
"""Handler for MY_NEW_CMD.

<One paragraph explaining what this command does, when it is called,
and any non-obvious behaviour (async vs sync, reply protocol, etc.)>
"""
from __future__ import annotations
import logging
from typing import Any
from umrs.core.TaskCmdStatus import TaskStatus
from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)


class MyNewCmdHandler(CommandHandler):
    """Handle MY_NEW_CMD — <short description>."""

    def execute(
        self,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        client_id, my_data = payload   # unpack according to your payload shape

        # --- your logic here ---

        update_status(TaskStatus.COMPLETED)
```

**Common patterns to copy:**

- **Send a ZeroMQ reply** (like `model.py`):
  ```python
  client_id, payload_str, zmq_sender_id = payload
  result_json = my_backend_function(client_id, payload_str)
  try:
      ctx.control_mq.send_string(result_json, zmq_sender_id)
  except RuntimeError as exc:
      logger.error("Error sending MY_NEW_CMD response: %s", exc)
  update_status(TaskStatus.COMPLETED)
  ```

- **Launch a background thread** (like `process_objects.py`):
  ```python
  from umrs.core.thread_pool import ServerOverloadedError
  import contextlib
  try:
      thread_ident = ctx.task_manager.active_threads.launch_my_worker(...)
      ctx.task_manager.associate_thread_with_task(thread_ident, task_id)
  except ServerOverloadedError as exc:
      logger.warning("Rejecting MY_NEW_CMD task %s: %s", task_id, exc)
      update_status(TaskStatus.ERROR)
      with contextlib.suppress(RuntimeError):      # if a reply is expected
          ctx.control_mq.send_string("SERVER_OVERLOADED", zmq_sender_id)
  ```

### Step 3 — Register the handler

Open `umrs/server/commands/registry_builder.py` and add two lines:

```python
# 1. import at the top of the file
from .my_new_cmd import MyNewCmdHandler

# 2. one register() call inside build_registry(), in the relevant section
r.register(TaskCmd.MY_NEW_CMD, MyNewCmdHandler())
```

`registry_builder.py` is the **only file to edit** when adding a command.
`server_umrs.py` and the event loop are never touched.

### Step 4 — Verify

```bash
python -m py_compile umrs/server/commands/my_new_cmd.py
python -c "from umrs.server.commands import build_registry; print('OK')"
```

That is all.  The event loop in `server_umrs.py` does not need to be touched.

---

## What does NOT belong in a handler

| Concern | Where it lives |
|---------|---------------|
| Authorization / role check | `umrs/shared/auth.py` + `CommandRegistry._authorize()` |
| Task creation (`create_task`) | `server_umrs.server_event_loop()` |
| Thread cleanup (`clean_finished_threads`) | `server_umrs.server_event_loop()` |
| SIGHUP hot-reload logic | `server_umrs.server_event_loop()` |
| Pool metrics logging | `server_umrs.server_event_loop()` |
| Per-user status summaries | `server_umrs.server_event_loop()` |
lt
