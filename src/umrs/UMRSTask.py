import sqlite3
from datetime import datetime
from typing import List, Optional
import json
import os
from tabulate import tabulate
from umrs.TaskCmdStatus import TaskCmd, TaskStatus
import textwrap
import threading
verbose = False
if verbose:
    import traceback


class UMRSTask:
    def __init__(self, db_path: str = "tasks.db", max_attempts = 3):
        self.db_path = db_path
         # Ensure parent directory exists
        dir_path = os.path.dirname(self.db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn_lock = threading.Lock()  # To ensure thread-safe commits
        self._init_db()
        self.mark_running_tasks_aborted()  # They were running it crashes
        self.max_attempts = max_attempts

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT,
                cmd TEXT,            
                payload TEXT,
                status TEXT,
                submitted_at TEXT,
                updated_at TEXT
            )
            """)

    def list_incomplete_tasks(self, mark_override=False):
        """
        List unique tasks that are not completed, deduplicated by (client_id, cmd, payload).

        Rules:
        - Tasks are ordered by submitted_at descending (newest first).
        - For each unique task, only the latest status is considered.
        - If mark_override=True:
            * Earlier incomplete tasks that were later not completed are marked as OVERRIDDEN.
            * Completed tasks remain as COMPLETED.
        
        Returns:
        dict with key = (client_id, cmd, payload) and value = status (or task dict)
        """
        (verbose and traceback.print_stack())
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, client_id, cmd, payload, status, submitted_at FROM tasks ORDER BY submitted_at DESC")
        rows = cursor.fetchall()

        seen_tasks = {}  # key = (client_id, cmd, payload), value = dict(id, status, submitted_at)
        
        for row in rows:
            task_id, client_id, cmd, payload, status, submitted_at = row
            key = (client_id, cmd, payload)

            # If task is not yet seen, store it
            if key not in seen_tasks:
                seen_tasks[key] = {
                    "id": task_id,
                    "status": status,
                    "submitted_at": submitted_at
                }
            else:
                # Task already seen
                if mark_override:
                    # Only consider overriding earlier incomplete tasks
                    current_status = seen_tasks[key]["status"]
                    if current_status == TaskStatus.COMPLETED and status != TaskStatus.COMPLETED:
                        # Mark older incomplete task as OVERRIDDEN
                        with self.conn_lock:
                            with self.conn:
                                self.conn.execute(
                                    "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                                    (TaskStatus.OVERRIDDEN, datetime.utcnow().isoformat(), task_id)
                                )
        

        # Return dict of unique tasks not completed
        result = {
            key: {"id": info["id"], "status": info["status"]}
            for key, info in seen_tasks.items()
            if info["status"] != TaskStatus.COMPLETED
        }
        return result

    def mark_running_tasks_aborted(self):
        """Set all tasks to 'Aborted' except those already in terminal states (COMPLETED, OVERRIDDEN, NO_SUCH_BUCKET); used for restart/recovery."""
        (verbose and traceback.print_stack())
        with self.conn_lock:
            now = datetime.utcnow().isoformat()
            with self.conn:
                self.conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, updated_at = ?
                    WHERE status IS NULL OR status NOT IN (?,?,?)
                    """,
                    (TaskStatus.ABORTED, now, TaskStatus.COMPLETED, TaskStatus.OVERRIDDEN, TaskStatus.NO_SUCH_BUCKET)
                )
 
    def pretty_print_tasks(self, status_filter=None, client_id_filter=None, print_output=True):
        """
        Pretty-print tasks from the database.
        Can filter by status or client_id.
        """
        (verbose and traceback.print_stack())
        query = "SELECT id, client_id, cmd, payload, status, submitted_at, updated_at FROM tasks WHERE 1=1"
        params = []

        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)
        if client_id_filter:
            query += " AND client_id = ?"
            params.append(client_id_filter)

        cur = self.conn.execute(query, params)
        rows = cur.fetchall()


        wrapped_rows = []
        for row in rows:
            row = list(row)
            if row[3] is not None:  # payload column
                row[3] = "\n".join(
                    textwrap.wrap(str(row[3]), width=70)
                )
            wrapped_rows.append(row)

        if not rows:
            message = "No tasks found"
        else:
            headers = ["ID", "Client ID", "Command", "Payload", "Status", "Submitted At", "Updated At"]
            table = tabulate(wrapped_rows, headers, tablefmt="fancy_grid")
            message = "Tasks in DB:\n" + table

        if print_output:
            print(message)
        return message
   
    def create_task(self, cmd, full_payload):
        (verbose and traceback.print_stack())
        if cmd == TaskCmd.PROCESS_OBJECTS:
            client_id, s3_objects = full_payload
            str_payload = json.dumps(s3_objects)

        elif cmd == TaskCmd.WAIT_OUTPUT:
            client_id, output_path_to_bucket, timeout, zmq_sender_id = full_payload
            str_payload = json.dumps(output_path_to_bucket)

        elif cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE, TaskCmd.CHECK, TaskCmd.RELOAD):
            client_id = full_payload
            str_payload = json.dumps("")

        elif cmd == TaskCmd.FILE:
            client_id, files = full_payload if isinstance(full_payload, tuple) else ("", full_payload)
            str_payload = json.dumps(files)

        elif cmd in (TaskCmd.HAND_CHECK, TaskCmd.CREATE_DATA_QUEUE, TaskCmd.STATUS):
            print(f"We do not create task_id for {cmd=}")
            return None

        else:
            raise ValueError(f"Unknown TaskCmd: {cmd}")

        # Returns the task ID as int
        return self.add_task(client_id, cmd, str_payload, TaskStatus.SUBMITTED)


    def add_task(self, client_id: str, cmd: str, payload: str, status: str) -> int:
        """Insert a task in the DB and return its integer ID."""
        (verbose and traceback.print_stack())
        for attempt in range(1, self.max_attempts + 1):
            try:
                with self.conn_lock:
                    now = datetime.utcnow().isoformat()
                    with self.conn:
                        cur = self.conn.execute(
                            "INSERT INTO tasks (client_id, cmd, payload, status, submitted_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (client_id, cmd, payload, status, now, now)
                        )
                # Success -> leave the loop
                break
            except sqlite3.Error as e:
                if attempt < self.max_attempts:
                    # first failures sleep and retry
                    time.sleep(0.5)
                else:
                    # last failure Print and raise error properly
                    print(f"Erreur SQLite critique lors de la mise à jour de la tâche {task_id}:")
                    print(f"  {type(e).__name__}: {e}")
                    traceback.print_exc()
                    raise
                        
        return cur.lastrowid

    def update_status(self, task_id: int, new_status: str, verbose=False):
        """Update the status of a single task, retry once on SQLite error."""
        if verbose:
            traceback.print_stack()

        for attempt in range(1, self.max_attempts + 1):
            try:
                with self.conn_lock:
                    now = datetime.utcnow().isoformat()
                    with self.conn:
                        self.conn.execute(
                            "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                            (new_status, now, task_id)
                        )
                # Success -> leave the loop
                break
            except sqlite3.Error as e:
                if attempt < self.max_attempts:
                    # first failures sleep and retry
                    time.sleep(0.5)
                else:
                    # last failure Print and raise error properly
                    print(f"Erreur SQLite critique lors de la mise à jour de la tâche {task_id}:")
                    print(f"  {type(e).__name__}: {e}")
                    traceback.print_exc()
                    raise

        # Print tasks
        self.pretty_print_tasks()
    
    def get_tasks(
        self,
        status: Optional[str] = None,
        client_id: Optional[str] = None
    ) -> List[dict]:

        query = """
        SELECT id, client_id, cmd, payload, status, submitted_at, updated_at
        FROM tasks
        WHERE 1=1
        """
        (verbose and traceback.print_stack())
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)

        cur = self.conn.execute(query, params)
        rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "client_id": r[1],
                "cmd": r[2],
                "payload": r[3],
                "status": r[4],
                "submitted_at": r[5],
                "updated_at": r[6],
            }
            for r in rows
        ]

    def close(self):
        (verbose and traceback.print_stack())
        self.conn.close()