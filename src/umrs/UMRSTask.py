import sqlite3
from datetime import datetime
from typing import List, Optional
import json
import os
from tabulate import tabulate
from umrs.TaskCmdStatus import TaskCmd, TaskStatus
import textwrap
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
verbose = False
if verbose:
    import traceback


class UMRSTask:
    def __init__(self, db_path: str = "tasks.db", max_attempts = 3):
        self.db_path = db_path
        # Configure logging to file
        logging.basicConfig(
            filename=os.path.join(os.path.dirname(self.db_path), 'umrs.log'),
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
         # Ensure parent directory exists
        dir_path = os.path.dirname(self.db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout = 5000")  # 5 seconds timeout for locks
        self.conn_lock = threading.Lock()  # To ensure thread-safe commits
        self.db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DBWorker")  # Single-threaded DB executor
        self._init_db()
        self.mark_running_tasks_aborted()  # They were running it crashes
        self.max_attempts = max_attempts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db_executor.shutdown(wait=True)
        self.close()

    def _init_db(self):
        with self.conn_lock:
            with self.conn:
                # Get all possible statuses dynamically
                statuses = TaskStatus.all_status()
                status_list = ', '.join(f"'{s}'" for s in statuses)
                
                # Create metadata table to track schema
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """)
                
                # Check if tasks table needs migration
                cur = self.conn.cursor()
                cur.execute("SELECT value FROM schema_metadata WHERE key = 'status_list'")
                stored_status_list = cur.fetchone()
                
                if stored_status_list and stored_status_list[0] != status_list:
                    # Statuses changed; migrate
                    self._migrate_table(status_list)
                
                # Create or update tasks table
                self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    cmd TEXT NOT NULL,            
                    payload TEXT,
                    status TEXT NOT NULL CHECK (status IN ({status_list})),
                    submitted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)
                
                # Store current status_list in metadata
                self.conn.execute("INSERT OR REPLACE INTO schema_metadata (key, value) VALUES (?, ?)", ("status_list", status_list))
                
                # Add indexes for performance
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_client_cmd ON tasks(client_id, cmd)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_submitted ON tasks(submitted_at DESC)")
                # Enable WAL for better concurrency
                self.conn.execute("PRAGMA journal_mode=WAL")

    def _migrate_table(self, status_list):
        """Migrate the tasks table to update the CHECK constraint."""
        # Rename old table
        self.conn.execute("ALTER TABLE tasks RENAME TO tasks_old")
        
        print(f"Migrating UMRS Tasks database to {status_list=}")

        # Create new table with updated CHECK
        self.conn.execute(f"""
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            cmd TEXT NOT NULL,            
            payload TEXT,
            status TEXT NOT NULL CHECK (status IN ({status_list})),
            submitted_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
        
        # Copy data from old table
        self.conn.execute("""
        INSERT INTO tasks (id, client_id, cmd, payload, status, submitted_at, updated_at)
        SELECT id, client_id, cmd, payload, status, submitted_at, updated_at FROM tasks_old
        """)
        
        # Drop old table
        self.conn.execute("DROP TABLE tasks_old")

    def mark_uncompleted_tasks(self, user_id=None, cmd=TaskCmd.PROCESS_OBJECTS, new_status=TaskStatus.OVERRIDDEN):
        """
        Change the status of all uncompleted tasks for a given cmd and user_id
        to new_status.

        If user_id is None, apply to all users.
        """
        with self.conn_lock:
            now = datetime.utcnow().isoformat()
            query = """
                UPDATE tasks
                SET status = ?, updated_at = ?
                WHERE cmd = ?
                AND status != ?
            """

            params = [new_status, now, cmd, TaskStatus.COMPLETED]

            if user_id is not None:
                query += " AND client_id = ?"
                params.append(user_id)

            with self.conn:
                self.conn.execute(query, params)

    def last_uncompleted_task(self, client_id: str, cmd: str):
        """
        Return the payload of the last (chronological order) task with 
        status != 'COMPLETED' and not OVERRIDDEN  for the given cmd and client_id.
        Returns None if not found.
        """
        with self.conn_lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT payload 
                FROM tasks
                WHERE client_id = ? AND cmd = ?
                AND status NOT IN (?, ?)
                ORDER BY submitted_at DESC
                LIMIT 1
            """, (client_id, cmd, TaskStatus.COMPLETED, TaskStatus.OVERRIDDEN))
            row = cur.fetchone()
        return row[0] if row else None


    def all_uncompleted_task(self, client_id: str, cmd: str):
        """
        Return a list of unique payloads for all tasks with 
        status != COMPLETED and not OVERRIDDEN for the given cmd and client_id,
        ordered from newest to oldest, removing duplicates.
        """
        with self.conn_lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT payload 
                FROM tasks
                WHERE client_id = ? AND cmd = ? 
                AND status NOT IN (?, ?)
                ORDER BY submitted_at DESC
            """, (client_id, cmd, TaskStatus.COMPLETED, TaskStatus.OVERRIDDEN))
            rows = cur.fetchall()

        seen = set()
        unique_payloads = []
        for row in rows:
            payload = row[0]
            if payload not in seen:
                seen.add(payload)
                unique_payloads.append(payload)

        return unique_payloads
   
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
        with self.conn_lock:
            with self.conn:  # Start transaction
                cursor = self.conn.cursor()
                cursor.execute("SELECT id, client_id, cmd, payload, status, submitted_at FROM tasks ORDER BY submitted_at DESC")
                rows = cursor.fetchall()

                seen_tasks = {}  # key = (client_id, cmd, payload), value = dict(id, status, submitted_at)
                
                tasks_to_override = []  # Collect IDs to update
                
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
                                tasks_to_override.append(task_id)
                
                # Perform all updates in the same transaction
                if tasks_to_override:
                    now = datetime.utcnow().isoformat()
                    placeholders = ','.join('?' for _ in tasks_to_override)
                    self.conn.execute(
                        f"UPDATE tasks SET status = ?, updated_at = ? WHERE id IN ({placeholders})",
                        [TaskStatus.OVERRIDDEN, now] + tasks_to_override
                    )
                # Transaction commits here

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
 
    def pretty_print_tasks(self, status_filter=None, client_id_filter=None, task_id_filter=None, print_output=True):
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
        if task_id_filter:
            query += " AND id = ?"
            params.append(task_id_filter)

        with self.conn_lock:
            with self.conn:
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
            message = "Task(s) with updated status:\n" + table

        if print_output:
            print(message)
        return message
   
    def create_task(self, cmd, full_payload):
        (verbose and traceback.print_stack())
        if cmd == TaskCmd.PROCESS_OBJECTS:
            if not isinstance(full_payload, tuple) or len(full_payload) != 2:
                raise ValueError("For PROCESS_OBJECTS, full_payload must be a tuple (client_id, s3_objects)")
            client_id, s3_objects = full_payload
            try:
                str_payload = json.dumps(s3_objects)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid s3_objects for JSON serialization: {e}")

        elif cmd == TaskCmd.WAIT_OUTPUT:
            if not isinstance(full_payload, tuple) or len(full_payload) != 4:
                raise ValueError("For WAIT_OUTPUT, full_payload must be a tuple (client_id, output_path_to_bucket, timeout, zmq_sender_id)")
            client_id, output_path_to_bucket, timeout, zmq_sender_id = full_payload
            try:
                str_payload = json.dumps(output_path_to_bucket)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid output_path_to_bucket for JSON serialization: {e}")

        elif cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE, TaskCmd.CHECK, TaskCmd.RELOAD):
            if not isinstance(full_payload, str):
                raise ValueError(f"For {cmd}, full_payload must be a string (client_id)")
            client_id = full_payload
            str_payload = json.dumps("")

        elif cmd == TaskCmd.FILE:
            if not isinstance(full_payload, tuple) or len(full_payload) not in (1, 2):
                raise ValueError("For FILE, full_payload must be a tuple (client_id, files) or (files,)")
            if len(full_payload) == 2:
                client_id, files = full_payload
            else:
                client_id, files = "", full_payload[0]
            try:
                str_payload = json.dumps(files)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid files for JSON serialization: {e}")

        elif cmd in (TaskCmd.HANDSHAKE, TaskCmd.CREATE_DATA_QUEUE, TaskCmd.STATUS):
            self.logger.info(f"We do not create task_id for {cmd=}")
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
                    self.logger.warning(f"SQLite error on attempt {attempt} for add_task: {e}")
                else:
                    # last failure
                    self.logger.error(f"Critical SQLite error in add_task after {self.max_attempts} attempts: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    raise
                        
        return cur.lastrowid

    def update_status(self, task_id: int, new_status: str, verbose=False):
        """Update the status of a single task, retry once on SQLite error."""
        if verbose:
            import traceback
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
                    self.logger.warning(f"SQLite error on attempt {attempt} for update_status: {e}")
                else:
                    # last failure
                    self.logger.error(f"Critical SQLite error in update_status after {self.max_attempts} attempts: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    raise

        # Print tasks
        self.pretty_print_tasks(task_id_filter=task_id)
    
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

        try:
            with self.conn_lock:
                with self.conn:
                    cur = self.conn.execute(query, params)
                    rows = cur.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"Error in get_tasks: {e}")
            raise

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
        self.db_executor.shutdown(wait=True)

    def health_check(self) -> bool:
        """Perform a basic health check on the database."""
        try:
            with self.conn_lock:
                with self.conn:
                    cur = self.conn.execute("SELECT COUNT(*) FROM tasks")
                    count = cur.fetchone()[0]
                    self.logger.info(f"Health check passed: {count} tasks in DB")
                    return True
        except sqlite3.Error as e:
            self.logger.error(f"Health check failed: {e}")
            return False