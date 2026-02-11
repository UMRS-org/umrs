import sqlite3
from datetime import datetime
from typing import List, Optional
import json

class TaskStatus:
    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

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

    def create_task(self, cmd, full_paylaod):
        def create_task(self, cmd, full_payload):

        if cmd == TaskCmd.PROCESS_OBJECTS:
            client_id, s3_objects = payload
            new_payload = json.dumps(s3_objects)
    
        elif cmd == TaskCmd.WAIT_OUTPUT:
            client_id, output_path_to_bucket, timeout, zmq_sender_id = payload
            new_payload = json.dumps(output_path_to_bucket)

        elif cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE, TaskCmd.CHECK, TaskCmd.RELOAD):
            client_id = payload
            new_payload = ""

        elif cmd == TaskCmd.FILE:
            client_id, files = payload if isinstance(payload, tuple) else (None, payload)
            new_payload = json.dumps(files)

        elif cmd == TaskCmd.HAND_CHECK:
            print (f"We do not create task_id for {cmd=}")
            return None

        elif cmd == TaskCmd.CREATE_DATA_QUEUE:
            print (f"We do not create task_id for {cmd=}")
            return None

        elif cmd == TaskCmd.STATUS:
            print (f"We do not create task_id for {cmd=}")
            return None

        else:
            raise ValueError(f"Unknown TaskCmd: {cmd}")
        
        self.add_task(client_id, cmd, new_payload, Status.SUBMITTED)

    def add_task(self, client_id: str, cmd: str, payload: str, status) -> int:
        now = datetime.utcnow().isoformat()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO tasks (client_id, cmd, payload, status, submitted_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (client_id, cmd, payload, status, now, now)
            )
            return cur.lastrowid

    def update_status(self, task_id: int, new_status: str):
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, now, task_id)
            )

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
        self.conn.close()