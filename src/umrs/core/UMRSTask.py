import json
import logging
import textwrap
import time
import traceback

import psycopg2
from tabulate import tabulate

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.shared.utils import utc_to_local

logger = logging.getLogger(__name__)


class UMRSTask:
    def __init__(self, db_pool, max_attempts=3, reset=False):
        self.db_pool = db_pool
        self.max_attempts = max_attempts
        self._init_db()
        if reset:
            self._reset_table()
        self.mark_running_tasks_aborted()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self):
        return self.db_pool.getconn()

    def _put_conn(self, conn):
        self.db_pool.putconn(conn)

    def _init_db(self):
        conn = self._get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    # The table uses UTC internally thanks to TIMESTAMPTZ.
                    # payload_hash is a GENERATED ALWAYS column (md5 of payload)
                    # so the unique index can use a fixed-size key instead of
                    # the raw payload, which may exceed PostgreSQL's 8191-byte
                    # index row limit for large payloads (e.g. spacy text chunks).
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS tasks (
                            id           SERIAL PRIMARY KEY,
                            client_id    TEXT,
                            cmd          TEXT,
                            payload      TEXT,
                            payload_hash TEXT GENERATED ALWAYS AS (md5(coalesce(payload, ''))) STORED,
                            status       TEXT,
                            submitted_at TIMESTAMPTZ DEFAULT now(),
                            updated_at   TIMESTAMPTZ DEFAULT now()
                        )
                    """)
                    # Migrate pre-existing tables (created before payload_hash
                    # was introduced): add the generated column if missing.
                    # CREATE TABLE IF NOT EXISTS above is a no-op on an existing
                    # table, so without this the dedup index below would fail
                    # with 'column "payload_hash" does not exist'. Idempotent:
                    # a no-op once the column exists.
                    cur.execute("""
                        ALTER TABLE tasks
                        ADD COLUMN IF NOT EXISTS payload_hash TEXT
                        GENERATED ALWAYS AS (md5(coalesce(payload, ''))) STORED
                    """)
                    # Partial unique index on payload_hash (md5 of payload) rather
                    # than payload itself: PostgreSQL limits index row size to 8191
                    # bytes, which large payloads exceed. payload_hash is a
                    # generated column so the index is still race-free and atomic
                    # via ON CONFLICT in add_task().
                    #
                    # Migrate a stale index from older server versions that
                    # indexed the raw `payload` column. CREATE UNIQUE INDEX
                    # IF NOT EXISTS is a no-op when an index of the same *name*
                    # already exists — so a pre-existing index with the wrong
                    # definition would silently persist and break add_task's
                    # ON CONFLICT (…payload_hash…), raising
                    # "no unique or exclusion constraint matching the ON CONFLICT
                    # specification" and crashing every task-creating command.
                    # Detect the mismatch by its definition and drop it so the
                    # correct index below is (re)created.
                    cur.execute(
                        """
                        SELECT indexdef FROM pg_indexes
                        WHERE tablename = 'tasks'
                          AND indexname = 'tasks_active_dedup'
                        """
                    )
                    existing_index = cur.fetchone()
                    if existing_index is not None and "payload_hash" not in existing_index[0]:
                        cur.execute("DROP INDEX IF EXISTS tasks_active_dedup")
                    cur.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS tasks_active_dedup
                        ON tasks (client_id, cmd, payload_hash)
                        WHERE status IN ('Submitted', 'Running')
                    """)
        finally:
            self._put_conn(conn)

    def _reset_table(self):
        """Truncate the tasks table and restart the ID sequence.

        Only safe to call at startup before any tasks are submitted.
        Raises RuntimeError if active tasks (Submitted/Running) are found —
        resetting under live traffic would orphan in-flight tasks.
        """
        conn = self._get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM tasks WHERE status IN (%s, %s)",
                        (TaskStatus.SUBMITTED, TaskStatus.RUNNING),
                    )
                    active_count = cur.fetchone()[0]
                    if active_count > 0:
                        raise RuntimeError(
                            f"Cannot reset tasks table: {active_count} active task(s) found. "
                            "Stop all clients before resetting."
                        )
                    # TRUNCATE is faster than DELETE and resets the SERIAL sequence
                    # RESTART IDENTITY resets id back to 1
                    # CASCADE handles any future foreign keys referencing tasks
                    cur.execute("TRUNCATE TABLE tasks RESTART IDENTITY CASCADE")
            logger.info(
                "==> tasks table reset: all rows deleted, ID sequence restarted."
            )
        finally:
            self._put_conn(conn)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def list_incomplete_tasks(self, mark_override=False, verbose=False):
        """List unique tasks not yet completed, deduplicated by (client_id, cmd, payload).

        If mark_override=True, earlier incomplete tasks superseded by a newer
        completed one are marked OVERRIDDEN.
        """
        if verbose:
            traceback.print_stack()

        conn = self._get_conn()
        try:
            with conn:
                # Race condition: without locking, two threads could both read the
                # same rows, both decide to mark a task OVERRIDDEN, and issue
                # concurrent UPDATEs — leading to double-writes or missed updates.
                #
                # Fix: when mark_override=True, add FOR UPDATE to the SELECT.
                # PostgreSQL locks each fetched row for the duration of the
                # transaction, blocking any other thread that tries to SELECT FOR
                # UPDATE or UPDATE the same rows until this transaction commits.
                # Read-only callers (mark_override=False) skip the lock to avoid
                # unnecessary contention.
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, client_id, cmd, payload, status, submitted_at "
                        "FROM tasks ORDER BY submitted_at DESC "
                        + ("FOR UPDATE" if mark_override else "")
                    )
                    rows = cur.fetchall()

                seen_tasks = {}
                for task_id, client_id, cmd, payload, status, submitted_at in rows:
                    key = (client_id, cmd, payload)
                    if key not in seen_tasks:
                        seen_tasks[key] = {
                            "id": task_id,
                            "status": status,
                            "submitted_at": submitted_at,
                        }
                    elif mark_override:
                        current_status = seen_tasks[key]["status"]
                        if (
                            current_status == TaskStatus.COMPLETED
                            and status != TaskStatus.COMPLETED
                        ):
                            with conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE tasks SET status = %s, updated_at = now() WHERE id = %s",
                                    (TaskStatus.OVERRIDDEN, task_id),
                                )
        finally:
            self._put_conn(conn)

        return {
            key: {"id": info["id"], "status": info["status"]}
            for key, info in seen_tasks.items()
            if info["status"] != TaskStatus.COMPLETED
        }

    def mark_running_tasks_aborted(self):
        """Set non-terminal tasks to Aborted on server restart/recovery."""
        conn = self._get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE tasks SET status = %s, updated_at = now()
                        WHERE status IS NULL OR status NOT IN (%s, %s, %s)
                        """,
                        (
                            TaskStatus.ABORTED,
                            TaskStatus.COMPLETED,
                            TaskStatus.OVERRIDDEN,
                            TaskStatus.NO_SUCH_BUCKET,
                        ),
                    )
        finally:
            self._put_conn(conn)

    def add_task(self, client_id: str, cmd: str, payload: str, status: str) -> int:
        """Insert a task and return its integer ID.

        Deduplicates: if an identical (client_id, cmd, payload) task already
        exists with status Submitted or Running, returns the existing ID.
        """
        for attempt in range(1, self.max_attempts + 1):
            conn = self._get_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        # Race condition: a naive SELECT-then-INSERT pattern is a
                        # TOCTOU (time-of-check / time-of-use) bug. Two threads
                        # submitting the same (client_id, cmd, payload) concurrently
                        # could both see "no active task" and both INSERT, creating
                        # duplicate rows.
                        #
                        # Fix: one atomic statement using the partial unique index
                        # tasks_active_dedup (defined in _init_db). PostgreSQL
                        # evaluates the INSERT and the uniqueness check atomically
                        # under a row-level lock:
                        #   - if no active task exists → INSERT succeeds, RETURNING
                        #     gives us the new id immediately.
                        #   - if an active task already exists → ON CONFLICT DO
                        #     NOTHING silently skips the INSERT and returns no row,
                        #     so we fall through to SELECT to fetch the existing id.
                        # No window exists between the two operations where another
                        # thread can sneak in a duplicate.
                        cur.execute(
                            "INSERT INTO tasks (client_id, cmd, payload, status) "
                            "VALUES (%s, %s, %s, %s) "
                            "ON CONFLICT (client_id, cmd, payload_hash) "
                            "WHERE status IN ('Submitted', 'Running') DO NOTHING "
                            "RETURNING id",
                            (client_id, cmd, payload, status),
                        )
                        row = cur.fetchone()
                        if row is not None:
                            return row[0]

                        # Conflict path: another thread already holds an active task
                        # for this (client_id, cmd, payload) — reuse it.
                        cur.execute(
                            "SELECT id FROM tasks "
                            "WHERE client_id = %s AND cmd = %s "
                            "AND payload_hash = md5(%s) "
                            "AND status IN (%s, %s)",
                            (
                                client_id,
                                cmd,
                                payload,
                                TaskStatus.SUBMITTED,
                                TaskStatus.RUNNING,
                            ),
                        )
                        row = cur.fetchone()
                        if row is not None:
                            logger.info(
                                f"Task dedup: reusing existing task {row[0]} for ({client_id}, {cmd})"
                            )
                            return row[0]
                        # Should not happen: conflict fired but no active task found.
                        # The task likely transitioned to a terminal state between
                        # the INSERT conflict and the SELECT — retry the whole block.
                        raise psycopg2.OperationalError(
                            "dedup conflict but no active task found"
                        )
            except psycopg2.Error as e:
                if attempt < self.max_attempts:
                    time.sleep(0.5)
                else:
                    logger.info(
                        f"PostgreSQL error in add_task: {type(e).__name__}: {e}"
                    )
                    traceback.print_exc()
                    raise
            finally:
                self._put_conn(conn)

    def update_status(self, task_id: int, new_status: str, verbose=False):
        """Update the status of a single task."""
        if task_id is None:
            return
        if verbose:
            traceback.print_stack()

        for attempt in range(1, self.max_attempts + 1):
            conn = self._get_conn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE tasks SET status = %s, updated_at = now() WHERE id = %s",
                            (new_status, task_id),
                        )
                break
            except psycopg2.Error as e:
                if attempt < self.max_attempts:
                    time.sleep(0.5)
                else:
                    logger.info(
                        f"Critical PostgreSQL error while updating task {task_id}: {type(e).__name__}: {e}"
                    )
                    traceback.print_exc()
                    raise
            finally:
                self._put_conn(conn)

        self.pretty_print_tasks(limit=20)

    # ------------------------------------------------------------------
    # Task creation (cmd dispatch)
    # ------------------------------------------------------------------

    def create_task(self, cmd, full_payload, verbose=False):
        if verbose:
            traceback.print_stack()

        if cmd == TaskCmd.PROCESS_OBJECTS:
            client_id, s3_objects = full_payload
            str_payload = json.dumps(s3_objects)

        elif cmd == TaskCmd.WAIT_OUTPUT:
            client_id, output_path_to_bucket, timeout, zmq_sender_id = full_payload
            str_payload = json.dumps(output_path_to_bucket)

        elif cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE, TaskCmd.CHECK, TaskCmd.RELOAD):
            client_id, _, _zmq_sender_id = full_payload
            str_payload = json.dumps("")

        elif cmd == TaskCmd.FILE:
            client_id, files = (
                full_payload if isinstance(full_payload, tuple) else ("", full_payload)
            )
            str_payload = json.dumps(files)

        elif cmd == TaskCmd.EXPORT_EMBEDDINGS:
            client_id, export_args, _ = full_payload
            str_payload = json.dumps(export_args)

        elif cmd in (
            TaskCmd.MODEL_REGISTER,
            TaskCmd.MODEL_LIST,
            TaskCmd.MODEL_DOWNLOAD,
            TaskCmd.LIST_WORKFLOWS,
            TaskCmd.RUN_WORKFLOW,
            TaskCmd.JOB_SUBMIT,
            TaskCmd.JOB_CANCEL,
            TaskCmd.PURGE_BUCKETS,
        ):
            client_id, payload_str, _zmq_sender_id = full_payload
            str_payload = payload_str

        elif cmd == TaskCmd.UPLOAD_WORKFLOW:
            client_id, files, workflow_name = full_payload
            str_payload = json.dumps({"files": files, "workflow_name": workflow_name})

        elif cmd in (
            TaskCmd.HANDSHAKE,
            TaskCmd.CREATE_DATA_QUEUE,
            TaskCmd.STATUS,
            TaskCmd.JOB_LIST,
            TaskCmd.WORKFLOW_STATUS,
            TaskCmd.WORKFLOW_RESULTS,
            TaskCmd.WORKFLOW_CANCEL,
            TaskCmd.WORKFLOW_ERRORS,
            TaskCmd.WORKFLOW_EXECUTIONS,
            TaskCmd.WORKFLOW_LOGS,
        ):
            logger.info(f"We do not create task_id for {cmd=}")
            return None

        else:
            raise ValueError(f"Unknown TaskCmd: {cmd}")

        return self.add_task(client_id, cmd, str_payload, TaskStatus.SUBMITTED)

    # ------------------------------------------------------------------
    # Query / display
    # ------------------------------------------------------------------

    def get_tasks(
        self,
        status: str | None = None,
        client_id: str | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        if verbose:
            traceback.print_stack()

        query = (
            "SELECT id, client_id, cmd, payload, status, submitted_at, updated_at "
            "FROM tasks WHERE 1=1"
        )
        params = []
        if status:
            query += " AND status = %s"
            params.append(status)
        if client_id:
            query += " AND client_id = %s"
            params.append(client_id)

        conn = self._get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()
        finally:
            self._put_conn(conn)

        return [
            {
                "id": r[0],
                "client_id": r[1],
                "cmd": r[2],
                "payload": r[3],
                "status": r[4],
                "submitted_at": str(r[5]) if r[5] else "-",
                "updated_at": str(r[6]) if r[6] else "-",
            }
            for r in rows
        ]

    def all_uncompleted_task(self, client_id, status_key):
        tasks = self.get_tasks(client_id=client_id)
        return [
            t
            for t in tasks
            if t["status"] not in (TaskStatus.COMPLETED, TaskStatus.OVERRIDDEN)
            and (status_key in (TaskCmd.ALL, "*") or t["cmd"] == status_key)
        ]

    def pretty_print_tasks(
        self,
        status_filter=None,
        client_id_filter=None,
        print_output=True,
        verbose=False,
        limit=None,
    ):
        if verbose:
            traceback.print_stack()

        query = (
            "SELECT id, client_id, cmd, payload, status, submitted_at, updated_at "
            "FROM tasks WHERE 1=1"
        )
        params = []
        if status_filter:
            query += " AND status = %s"
            params.append(status_filter)
        if client_id_filter:
            query += " AND client_id = %s"
            params.append(client_id_filter)
        query += " ORDER BY id DESC"
        if limit:
            query += " LIMIT %s"
            params.append(limit)

        conn = self._get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    if limit:
                        count_q = "SELECT COUNT(*) FROM tasks WHERE 1=1"
                        count_p = []
                        if status_filter:
                            count_q += " AND status = %s"
                            count_p.append(status_filter)
                        if client_id_filter:
                            count_q += " AND client_id = %s"
                            count_p.append(client_id_filter)
                        cur.execute(count_q, count_p)
                        total_count = cur.fetchone()[0]
                    else:
                        total_count = 0

                    cur.execute(query, params)
                    rows = cur.fetchall()
        finally:
            self._put_conn(conn)

        rows = list(reversed(rows))
        wrapped_rows = []
        for row in rows:
            row = list(row)
            if row[3] is not None:
                lines = textwrap.wrap(str(row[3]), width=70)
                row[3] = "\n".join(lines[:2]) + ("…" if len(lines) > 2 else "")
            row[5] = utc_to_local(str(row[5]) if row[5] else "-")
            row[6] = utc_to_local(str(row[6]) if row[6] else "-")
            wrapped_rows.append(row)

        if not rows:
            message = "No tasks found"
        else:
            headers = [
                "ID",
                "Client ID",
                "Command",
                "Payload",
                "Status",
                "Submitted At",
                "Updated At",
            ]
            table = tabulate(wrapped_rows, headers, tablefmt="fancy_grid")
            if limit and total_count > limit:
                message = (
                    f"Tasks in DB (showing {limit} of {total_count} most recent):\n"
                    + table
                )
            else:
                message = "Tasks in DB:\n" + table

        if print_output:
            logger.info(message)
        return message

    def close(self, verbose=False):
        if verbose:
            traceback.print_stack()
        self.db_pool.closeall()
