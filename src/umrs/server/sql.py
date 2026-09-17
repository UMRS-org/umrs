#!/usr/bin/python3
import contextlib
import json
import logging
import sys
import threading
import time
from datetime import datetime

import psycopg2

logger = logging.getLogger(__name__)
from psycopg2 import pool as pg_pool
from psycopg2 import sql

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.server.config import (
    get_db_config,
    get_umrs_server_db_name,
    get_umrs_server_db_user,
)
from umrs.shared.buckets import BucketsCfg
from umrs.shared.pg_settings import pg_connect_kwargs
from umrs.shared.sql_helpers import create_db_name, metadata_dict

# Per-process cache of per-experience DataBase/connection-pool objects, keyed
# by db_name. create_db() is called on every server start AND on every
# configuration reload (once per experience per user). Without this cache,
# each call built a brand-new psycopg2 SimpleConnectionPool(minconn=1, ...)
# that eagerly opens a connection which is never closed when the previous
# pool object is superseded (buckets_cfg.store() simply overwrites the old
# bucket_entry). Over repeated reloads this silently leaked one Postgres
# connection per experience per reload until max_connections was exhausted.
_db_pool_cache: dict[str, "DataBase"] = {}
_db_pool_cache_lock = threading.Lock()


class DataBase:
    def __init__(self, db_pool, db_name, username):
        self.db_pool = db_pool
        self.db_name = db_name
        self.username = username
        self.tables = {}

    def put_conn(self, conn, close=False):
        """
        Return a connection to the pool. Optionally close it.

        Args:
            conn: The connection object to return.
            close (bool): If True, the connection will be closed instead of returned to the pool.
        """
        try:
            self.db_pool.putconn(conn, close=close)
        except Exception as e:
            logger.warning("Error returning connection to pool: %s", e)

    def get_conn(self):
        """
        Retrieve a live connection from the pool.
        Raises RuntimeError if it is not possible to obtain a valid connection.
        """
        try:
            conn = self.db_pool.getconn()
            if conn.closed:
                # connexion dead → close and get a new one
                self.db_pool.putconn(conn, close=True)
                conn = self.db_pool.getconn()
                if conn.closed:
                    raise RuntimeError("Unable to get a live connection from the pool")
            return conn
        except Exception as e:
            raise RuntimeError(f"Failed to get connection: {e}") from e

    def add_table(self, table_name, columns):
        """
        Adds a table with its columns and types.

        Args:
            table_name (str): The name of the table.
            columns (dict): A dictionary {column_name: data_type}.
        """
        self.tables[table_name] = columns

    def has_column(self, table_name, column_name):
        """
        Checks if a specific column exists in a table.

        Returns:
            bool: True if column exists, False otherwise.
        """
        if table_name not in self.tables:
            logger.info(
                "Table '%s' not found in the database '%s'.", table_name, self.db_name
            )
            return False
        return column_name in self.tables[table_name]

    def get_column_type(self, table_name, column_name):
        """
        Returns the data type of a given column.

        Returns:
            str or None: The data type if column exists, else None.
        """
        if self.has_column(table_name, column_name):
            return self.tables[table_name][column_name]
        else:
            logger.info("Column '%s' not found in table '%s'.", column_name, table_name)
            return None

    def table_exists(self, conn, table_name: str) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                );
            """,
                (table_name,),
            )
            return cur.fetchone()[0]

    def wait_for_table_update(self, table_name, timeout=None):
        import select

        """
        Blocks until the given table is updated (via NOTIFY).

        Args:
            table_name (str): Table to listen to
            timeout (float or None): Seconds before returning None

        Returns:
            str or None: Notification payload
        """
        conn = self.get_conn()
        try:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

            # 1️⃣ Check the table exist
            if not self.table_exists(conn, table_name):
                raise ValueError(
                    f"Table '{table_name}' does not exist in database '{self.db_name}'"
                )

            channel = f"{table_name}_update"

            with conn.cursor() as cur:
                cur.execute(sql.SQL("LISTEN {};").format(sql.Identifier(channel)))

            # 2️⃣ Block until notification

            if select.select([conn.fileno()], [], [], timeout) == ([], [], []):
                return None  # timeout

            conn.poll()
            if not conn.notifies:
                return None  # spurious wakeup
            notify = conn.notifies.pop(0)
            conn.notifies.clear()

            return notify.payload

        finally:
            self.db_pool.putconn(conn)

    def count_s3_objects(self, table_name: str) -> int:
        """
        Count the number of unique S3 objects stored in the given table.

        Assumes the table contains a TEXT column named 's3_object'.

        Args:
            table_name (str): Name of the table to query.

        Returns:
            int: Number of distinct s3_object values.
        """
        # 1️⃣ Vérifications de cohérence côté méta-modèle
        if not self.has_column(table_name, "s3_object"):
            raise ValueError(f"Table '{table_name}' has no 's3_object' column.")

        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("""
                    SELECT COUNT(DISTINCT s3_object)
                    FROM {}
                    WHERE s3_object IS NOT NULL;
                    """).format(sql.Identifier(table_name))
                )
                return cur.fetchone()[0]

        finally:
            self.db_pool.putconn(conn)


def export_embeddings(
    active_threads,
    user,
    export_args,
    buckets_cfg,
    control_mq,
    zmq_sender_id,
    key,
    update_status_callback=None,
):
    """Query the DB for embeddings and return a JSON-serialised response.

    ``export_args`` is ``"path_to_bucket;format;profile"`` where *format* is
    ``csv`` or ``jsonl`` and *profile* controls which columns are exported.
    The response is a JSON string with columns + rows so the client can write
    them to a local file without ever touching the DB.
    """

    status = TaskStatus.RUNNING
    if update_status_callback:
        update_status_callback(status)

    parts = export_args.split(";")
    path_to_bucket = parts[0]
    export_format = parts[1] if len(parts) > 1 else "csv"
    export_profile = parts[2] if len(parts) > 2 else "minimal"

    def _select_export_columns(columns):
        if export_profile == "embedding-only":
            return ["embedding"] if "embedding" in columns else columns
        if export_profile != "minimal":
            return columns
        preferred = [name for name in ("s3_object", "embedding") if name in columns]
        if preferred:
            return preferred
        if "embedding" in columns:
            return ["embedding"]
        return columns

    entry = buckets_cfg.get(path_to_bucket)
    if entry is None:
        results = json.dumps(
            {
                "status": "error",
                "message": f"No bucket entry found for path: {path_to_bucket}",
            },
            ensure_ascii=False,
        )
        status = TaskStatus.ERROR
    else:
        data_base = entry.data_base
        table_name = entry.table_name
        db_name = entry.db_name
        rows = []
        selected_columns = []

        conn = data_base.get_conn()
        try:
            with conn.cursor() as cur:
                # Discover columns
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position",
                    (table_name,),
                )
                columns = [row[0] for row in cur.fetchall()]
                if not columns:
                    results = json.dumps(
                        {
                            "status": "error",
                            "message": f"Table {table_name} not found in database {db_name}",
                        },
                        ensure_ascii=False,
                    )
                    status = TaskStatus.ERROR
                elif "embedding" not in columns:
                    results = json.dumps(
                        {
                            "status": "error",
                            "message": f"Table {table_name} has no embedding column",
                        },
                        ensure_ascii=False,
                    )
                    status = TaskStatus.ERROR
                else:
                    selected_columns = _select_export_columns(columns)
                    # Fetch all rows
                    cur.execute(
                        sql.SQL("SELECT {} FROM {}").format(
                            sql.SQL(", ").join(
                                sql.Identifier(c) for c in selected_columns
                            ),
                            sql.Identifier(table_name),
                        )
                    )
                    rows = cur.fetchall()
        except Exception as e:
            results = json.dumps(
                {"status": "error", "message": str(e)}, ensure_ascii=False
            )
            status = TaskStatus.ERROR
        finally:
            data_base.put_conn(conn)

        if status != TaskStatus.ERROR:
            # Serialise rows — convert non-JSON-serialisable types to strings
            serialised_rows = []
            for row in rows:
                serialised_rows.append(
                    [
                        v
                        if v is None or isinstance(v, (int, float, bool, str))
                        else str(v)
                        for v in row
                    ]
                )

            results = json.dumps(
                {
                    "status": "ok",
                    "database": db_name,
                    "table": table_name,
                    "columns": selected_columns,
                    "rows": serialised_rows,
                    "rows_exported": len(serialised_rows),
                    "format": export_format,
                    "profile": export_profile,
                },
                ensure_ascii=False,
            )

            status = TaskStatus.COMPLETED

    with contextlib.suppress(RuntimeError):
        control_mq.send_string(results, zmq_sender_id)

    active_threads.update_status_to_done()
    active_threads.record_status_for_users_by_key([user], key)
    active_threads.remove_from_running_list()

    if update_status_callback:
        update_status_callback(status)


def wait_table_modification_on_db(
    active_threads,
    user,
    key,
    data_base,
    table_name,
    timeout,
    control_mq,
    zmq_sender_id,
    update_status_callback=None,
):
    from umrs.server.config import ENABLE_PRETTY_PRINT
    from umrs.shared.pretty_print import pretty_print_thread_status

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    try:
        res = data_base.wait_for_table_update(table_name, timeout=timeout)
    except Exception as e:
        raise RuntimeError(
            f"Error while waiting for table '{table_name}' update: {e}"
        ) from e

    msg = "TIMEOUT" if res is None else res

    control_mq.send_string(msg, zmq_sender_id)

    active_threads.update_status_to_done()
    # print(f"wait is done for {user=}")
    active_threads.record_status_for_users_by_key([user], key)
    active_threads.remove_from_running_list()

    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)

    with active_threads.lock:
        pretty_print_thread_status(
            active_threads.thread_dict,
            running_only=False,
            print_output=ENABLE_PRETTY_PRINT,
        )
    logger.info(
        "####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######"
    )


def db_user_exists(conn, username):
    with conn.cursor() as cur:
        # Check if the username exists
        cur.execute("SELECT 1 FROM pg_roles where rolname = %s;", (username,))
        exists = cur.fetchone()
        return exists


def create_db_user(username):
    cfg = get_db_config()
    conn = psycopg2.connect(
        dbname="postgres",
        user=cfg["superuser"],
        password=cfg["superuser_password"],
        host=cfg["host"],
        port=cfg["port"],
        **pg_connect_kwargs("admin"),
    )
    conn.autocommit = True

    DB_USER_PASSWORD = cfg["user_password"]
    cur = conn.cursor()
    if db_user_exists(conn, username):
        logger.info(f"==> User '{username}' exist in DB")
        cur.execute(
            sql.SQL("ALTER USER {} WITH PASSWORD %s;").format(sql.Identifier(username)),
            (DB_USER_PASSWORD,),
        )
        conn.commit()
        cur.close()
        conn.close()
        return

    cur.execute(
        sql.SQL("CREATE USER {} WITH PASSWORD %s;").format(sql.Identifier(username)),
        (DB_USER_PASSWORD,),
    )
    logger.info(f"==> User '{username}' created in DB")
    cur.close()
    conn.close()


def _adopt_legacy_database(cur, db_name: str, legacy_db_name: str) -> bool:
    """One-time rename of an un-namespaced legacy database onto ``db_name``.

    Best-effort: on any failure the caller falls through and creates a fresh
    database, leaving the legacy one untouched for manual migration.
    Returns True when the legacy database was adopted.
    """
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (legacy_db_name,))
    if not cur.fetchone():
        return False
    try:
        cur.execute(
            sql.SQL("ALTER DATABASE {} RENAME TO {};").format(
                sql.Identifier(legacy_db_name), sql.Identifier(db_name)
            )
        )
    except Exception as e:
        logger.warning(
            f"\t\t-->  Could not migrate legacy database "
            f"'{legacy_db_name}' -> '{db_name}': {e}. Creating a fresh "
            f"database; legacy data left in place for manual migration."
        )
        return False
    logger.info(f"\t\t-->  Migrated legacy database '{legacy_db_name}' -> '{db_name}'.")
    return True


def _create_database_if_missing(cur, db_name: str, db_user: str, exists) -> None:
    """CREATE DATABASE unless it exists; a concurrent creator counts as success."""
    if exists:
        logger.info(f"\t\t-->  Database '{db_name}' already exists.")
        return
    # In concurrent startup/reload flows another process can create it
    # between the existence SELECT and CREATE DATABASE.
    try:
        cur.execute(
            sql.SQL("CREATE DATABASE {} OWNER {};").format(
                sql.Identifier(db_name), sql.Identifier(db_user)
            )
        )
        logger.info(f"\t\t-->  Database '{db_name}' created successfully.")
    except (psycopg2.errors.DuplicateDatabase, psycopg2.errors.UniqueViolation):
        logger.info(f"\t\t-->  Database '{db_name}' already exists (race detected).")


def _enable_pgvector(cfg: dict, db_name: str) -> None:
    """Always ensure the pgvector extension exists in the target DB."""
    conn_new = psycopg2.connect(
        dbname=db_name,
        user=cfg["superuser"],
        password=cfg["superuser_password"],
        host=cfg["host"],
        port=cfg["port"],
        **pg_connect_kwargs("admin"),
    )
    conn_new.autocommit = True
    cur_new = conn_new.cursor()
    try:
        cur_new.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info(f"\t\t-->  pgvector extension enabled in '{db_name}'.")
    except Exception as e:
        logger.info(f"\t\t-->  Error enabling pgvector extension in '{db_name}': {e}")
    finally:
        cur_new.close()
        conn_new.close()


def create_db(sql_username, xp_name, username=None):
    db_name = create_db_name(sql_username, xp_name, username)

    # Reuse an already-open pool for this database within this process
    # instead of opening a brand-new SimpleConnectionPool on every call.
    # configure_user() calls create_db() once per experience on EVERY server
    # start AND every configuration reload; without this cache each reload
    # leaked at least one idle Postgres connection per experience (the old
    # pool object was simply discarded when buckets_cfg overwrote the bucket
    # entry, never calling closeall()), eventually exhausting max_connections.
    with _db_pool_cache_lock:
        cached = _db_pool_cache.get(db_name)
    if cached is not None:
        logger.info(f"\t\t-->  Reusing existing connection pool for '{db_name}'.")
        return cached, db_name

    legacy_db_name = create_db_name(sql_username, xp_name)
    logger.info(f"\t\t-->  Configuring database '{db_name}'...")
    cfg = get_db_config()
    DB_NAME = db_name  # Replace with the target database
    DB_USER = sql_username  # Replace with the target user

    # Connect to PostgreSQL default database (`postgres`)
    conn = psycopg2.connect(
        dbname="postgres",
        user=cfg["superuser"],
        password=cfg["superuser_password"],
        host=cfg["host"],
        port=cfg["port"],
        **pg_connect_kwargs("admin"),
    )
    conn.autocommit = True  # Enable auto-commit mode
    cur = conn.cursor()

    # Check if the database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    exists = cur.fetchone()

    # One-time migration to per-user namespaced databases: if the new
    # (username-namespaced) database does not yet exist but a legacy
    # un-namespaced database does, adopt it by renaming (best-effort).
    if not exists and username and legacy_db_name != db_name:
        exists = _adopt_legacy_database(cur, db_name, legacy_db_name)

    _create_database_if_missing(cur, DB_NAME, DB_USER, exists)
    _enable_pgvector(cfg, DB_NAME)
    # Clean up and then create a pool to the target DB as the application user
    cur.close()
    conn.close()

    DB_USER_PASSWORD = cfg["user_password"]
    db_pool = pg_pool.ThreadedConnectionPool(  # thread-safe (ejeannot exhaustion fix)
        minconn=1,
        maxconn=5,
        dbname=db_name,
        user=sql_username,
        password=DB_USER_PASSWORD,
        host=cfg["host"],
        port=cfg["port"],
        **pg_connect_kwargs("experience-pool"),
    )
    data_base = DataBase(db_pool, db_name, sql_username)
    with _db_pool_cache_lock:
        # Another thread may have created and cached a pool for this
        # db_name while we were doing the setup above; prefer that one and
        # close ours to avoid leaking the connection(s) we just opened.
        winner = _db_pool_cache.setdefault(db_name, data_base)
        if winner is not data_base:
            db_pool.closeall()
    return winner, db_name


def _ensure_umrs_db(superuser_conn, db_name: str) -> None:
    """Create the umrs database if it does not already exist.

    Must be called with a superuser connection in autocommit mode.
    CREATE DATABASE cannot run inside a transaction block.
    """
    with superuser_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
        if cur.fetchone():
            logger.info(f"==> Database '{db_name}' already exists.")
            return
    try:
        with superuser_conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(db_name)))
        logger.info(f"==> Database '{db_name}' created.")
    except psycopg2.errors.DuplicateDatabase:
        logger.info(f"==> Database '{db_name}' already exists (race detected).")


def create_umrs_server_db_user():
    """Create the umrs_server_db_user PostgreSQL role (idempotent).

    Reads credentials from env vars:
      DB_HOST, DB_PORT
      DB_SUPERUSER / POSTGRES_USER   (superuser to run CREATE USER)
      DB_SUPERUSER_PASSWORD / POSTGRES_PASSWORD
      DB_UMRS_SERVER_PASSWORD        (password for umrs_server_db_user)
    """
    cfg = get_db_config()
    DB_HOST = cfg["host"]
    DB_PORT = cfg["port"]
    DB_SUPERUSER = cfg["superuser"]
    DB_SUPERUSER_PASSWORD = cfg["superuser_password"]
    DB_UMRS_SERVER_PASSWORD = cfg["umrs_server_password"]

    DB_NAME = get_umrs_server_db_name()
    DB_USER = get_umrs_server_db_user()

    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_SUPERUSER,
        password=DB_SUPERUSER_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        **pg_connect_kwargs("admin"),
    )
    conn.autocommit = True
    try:
        # Create the umrs database before any GRANT (connection to it would fail otherwise)
        _ensure_umrs_db(conn, DB_NAME)

        if db_user_exists(conn, DB_USER):
            logger.info(f"==> DB user '{DB_USER}' already exists — updating password.")
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("ALTER USER {} WITH PASSWORD %s;").format(
                        sql.Identifier(DB_USER)
                    ),
                    (DB_UMRS_SERVER_PASSWORD,),
                )
        else:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE USER {} WITH PASSWORD %s;").format(
                        sql.Identifier(DB_USER)
                    ),
                    (DB_UMRS_SERVER_PASSWORD,),
                )
            logger.info(f"==> DB user '{DB_USER}' created.")

        # Grant permission to create tables in the umrs database
        with psycopg2.connect(
            dbname=DB_NAME,
            user=DB_SUPERUSER,
            password=DB_SUPERUSER_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            **pg_connect_kwargs("admin"),
        ) as conn_umrs:
            conn_umrs.autocommit = True
            with conn_umrs.cursor() as cur:
                cur.execute(
                    sql.SQL("GRANT CREATE ON SCHEMA public TO {};").format(
                        sql.Identifier(DB_USER)
                    )
                )
            logger.info(
                f"==> GRANT CREATE ON SCHEMA public → '{DB_USER}' in '{DB_NAME}'."
            )
    finally:
        conn.close()


def create_server_db_pool(pool_size: int = 10) -> pg_pool.ThreadedConnectionPool:
    """Create a connection pool for umrs_server_db_user on the umrs database.

    The pool is used for server-level tables (e.g. tasks).
    Reads credentials from env vars:
      DB_HOST, DB_PORT
      DB_UMRS_NAME              (default: "umrs")
      DB_UMRS_SERVER_PASSWORD   (password for umrs_server_db_user)
    """
    cfg = get_db_config()
    DB_HOST = cfg["host"]
    DB_PORT = cfg["port"]
    DB_NAME = get_umrs_server_db_name()
    DB_USER = get_umrs_server_db_user()
    DB_UMRS_SERVER_PASSWORD = cfg["umrs_server_password"]

    # ThreadedConnectionPool (not SimpleConnectionPool): SimpleConnectionPool
    # is NOT thread-safe — concurrent getconn()/putconn() calls from different
    # threads can corrupt its internal free-list. ThreadedConnectionPool wraps
    # the same logic with a threading.Lock(), making it safe for multi-threaded
    # servers where UMRSTask is called from multiple futures/threads in parallel.
    # R-14 session hygiene (application_name + idle-in-transaction reaper)
    # comes from umrs.shared.pg_settings — one place for every pool.
    return pg_pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=pool_size,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_UMRS_SERVER_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        **pg_connect_kwargs("server-pool"),
    )


def get_table_columns_and_types(conn, table_name, schema="public"):
    """
    Get column names and their data types from a PostgreSQL table.

    Args:
        conn: psycopg2 connection object.
        table_name (str): The name of the table.
        schema (str): The schema where the table exists (default is 'public').

    Returns:
        dict: A dictionary with column names as keys and data types as values.
    """
    column_info = {}
    try:
        with conn.cursor() as cur:
            query = """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s;
            """
            cur.execute(query, (schema, table_name))
            rows = cur.fetchall()

            # Populate dictionary with column names and types
            column_info = dict(rows)

    except Exception as e:
        logger.info(f"Error fetching column info: {e}")

    return column_info


from umrs.shared.pgvector_index import (  # noqa: F401 — phase-4 transition re-export (ADR-053)
    ensure_pgvector_enabled,
    maintain_pgvector_index,
)


def get_vector_size(cursor, table_name):
    """
    Get the size (dimension) of the 'embedding' vector column in a PostgreSQL table.

    Args:
        cursor: psycopg2 cursor object
        table_name: Name of the table to check

    Returns:
        int: Vector size (dimension) if found, else None
    """
    query = """
    SELECT atttypmod AS vector_size
    FROM pg_attribute
    WHERE attrelid = %s::regclass AND attname = 'embedding';
    """

    cursor.execute(query, (table_name,))
    result = cursor.fetchone()

    return result[0] if result else None  # Return vector size or None


def enable_table_notifications(conn, table_name: str):
    """
    Enable PostgreSQL NOTIFY on INSERT/UPDATE/DELETE for a given table.
    Creates a trigger and trigger function if they do not exist.

    Uses a PostgreSQL advisory lock to prevent concurrent DDL from
    multiple processes (e.g. RAG service + main server starting together).
    """
    func_name = f"notify_{table_name}_update"
    trigger_name = f"{table_name}_notify_trigger"
    channel = f"{table_name}_update"

    # Derive a stable advisory lock key from the table name
    lock_key = hash(table_name) & 0x7FFFFFFF  # positive 32-bit int

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with conn.cursor() as cur:
                # Acquire session-level advisory lock to serialize DDL
                cur.execute("SELECT pg_advisory_lock(%s);", (lock_key,))

                try:
                    # 1️⃣ Create trigger function
                    cur.execute(
                        sql.SQL("""
                        CREATE OR REPLACE FUNCTION {}()
                        RETURNS trigger AS $$
                        BEGIN
                            PERFORM pg_notify({}, TG_OP);
                            RETURN NEW;
                        END;
                        $$ LANGUAGE plpgsql;
                        """).format(sql.Identifier(func_name), sql.Literal(channel))
                    )

                    # 2️⃣ Drop trigger if exists (idempotent)
                    cur.execute(
                        sql.SQL("DROP TRIGGER IF EXISTS {} ON {};").format(
                            sql.Identifier(trigger_name), sql.Identifier(table_name)
                        )
                    )

                    # 3️⃣ Create trigger
                    cur.execute(
                        sql.SQL("""
                        CREATE TRIGGER {}
                        AFTER INSERT OR UPDATE OR DELETE ON {}
                        FOR EACH STATEMENT
                        EXECUTE FUNCTION {}();
                        """).format(
                            sql.Identifier(trigger_name),
                            sql.Identifier(table_name),
                            sql.Identifier(func_name),
                        )
                    )
                finally:
                    cur.execute("SELECT pg_advisory_unlock(%s);", (lock_key,))

            conn.commit()
            return  # success
        except psycopg2.errors.InternalError_ as exc:
            conn.rollback()
            if "tuple concurrently updated" in str(exc) and attempt < max_retries:
                logger.info(
                    f"\t\t\t--> Concurrent DDL detected on '{table_name}', retrying ({attempt}/{max_retries})..."
                )
                time.sleep(0.5 * attempt)
            else:
                raise


def configure_table(data_base, sql_cfg, modality_name, model):
    conn = data_base.get_conn()

    table_name = sql_cfg["table_name"].lower()
    (f"\t\t\t--> Configuring table '{table_name}'...")
    table_columns = []
    column = {"name": "id", "type": "SERIAL PRIMARY KEY"}
    table_columns.append(column)
    column = {"name": "s3_object", "type": "TEXT"}
    table_columns.append(column)
    column = {"name": "checksum", "type": "TEXT"}
    table_columns.append(column)
    store_embeddings_flag = False
    column = {"name": "date", "type": "DATE"}
    table_columns.append(column)
    column = {"name": "time", "type": "TIME"}
    table_columns.append(column)
    if sql_cfg["store_embeddings"] == "yes":
        store_embeddings_flag = True
        vector_size = model.embedding_vector_size
        column = {"name": "embedding", "type": f"VECTOR({vector_size})"}
        table_columns.append(column)
        column = {"name": "model", "type": "TEXT"}
        table_columns.append(column)

    if model.return_type is not None:
        if model.return_type == "file":
            column = {"name": "saved_s3_object", "type": "TEXT"}
            table_columns.append(column)
        elif model.return_type in ("embedding", "raw"):
            pass  # no extra column needed
        else:
            logger.info(
                f"WARNING: Unknown return type '{model.return_type}' — no extra columns added"
            )

    for column in sql_cfg["meta_data"]:
        table_columns.append(column)

    if modality_name in ("text", "text_file", "text file"):
        columns_to_add = [
            {"name": "_content", "type": "TEXT"},
            {"name": "chunk_start", "type": "BIGINT"},
            {"name": "chunk_end", "type": "BIGINT"},
            {"name": "chunk_md5", "type": "TEXT"},
        ]
        table_columns.extend(columns_to_add)
        table_columns.append({"name": "UNIQUE(s3_object, chunk_start)", "type": ""})
    else:
        table_columns.append({"name": "UNIQUE(s3_object)", "type": ""})

    metadata = metadata_dict(sql_cfg)

    ensure_pgvector_enabled(conn)
    cur = conn.cursor()

    # Check if table exists
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);",
        (table_name,),
    )
    table_exists = cur.fetchone()[0]

    if not table_exists:
        # Create table if it doesn't exist
        column_defs = ", ".join(
            [f"{column['name']} {column['type']}" for column in table_columns]
        )
        create_query = sql.SQL("CREATE TABLE {} ({});").format(
            sql.Identifier(table_name), sql.SQL(column_defs)
        )
        try:
            cur.execute(create_query)
            conn.commit()
            logger.info(f"\t\t\t--> Table '{table_name}' created successfully.")
        except (psycopg2.errors.UniqueViolation, psycopg2.errors.DuplicateTable):
            # Sequence or table already exists (leftover from partial creation or race)
            conn.rollback()
            logger.info(
                f"\t\t\t--> Table '{table_name}' already exists (concurrent creation)."
            )
            table_exists = True
        # Create helpful indexes (IF NOT EXISTS is idempotent; safe after race too)
        try:
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (s3_object);").format(
                    sql.Identifier(f"idx_{table_name}_s3_object"),
                    sql.Identifier(table_name),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (checksum);").format(
                    sql.Identifier(f"idx_{table_name}_checksum"),
                    sql.Identifier(table_name),
                )
            )
            if store_embeddings_flag:
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
                    ).format(
                        sql.Identifier(f"idx_{table_name}_embedding_hnsw"),
                        sql.Identifier(table_name),
                    )
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.info(
                f"\t\t\t--> Warning: failed creating indexes on '{table_name}': {e}"
            )

    # enable notification for table that generate something so the server can listen to such notifications
    if model.return_type is not None:
        if model.return_type == "file":
            enable_table_notifications(conn, table_name)
        elif model.return_type in ("embedding", "raw"):
            pass  # no notifications needed
        else:
            logger.info(
                f"WARNING: Unknown return type '{model.return_type}' — skipping notifications"
            )

    else:
        # Validate existing table schema
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;",
            (table_name,),
        )
        existing_columns = {row[0]: row[1] for row in cur.fetchall()}

        # Remove UNIQUE constraint rows
        real_columns = [
            col
            for col in table_columns
            if not col["name"].upper().startswith("UNIQUE(")
        ]

        # Verify required columns exist in the table (metadata columns must also exist)
        core_required = real_columns
        for column in core_required:
            if column["name"] not in existing_columns:
                sys.exit(
                    f"Missing column: {column['name']} ({column['type']}) in '{table_name}'.\nExisting columns: {list(existing_columns.keys())}"
                )

        table_vector_size = get_vector_size(cur, table_name)
        if sql_cfg.get("store_embeddings") == "yes":
            expected_size = getattr(model, "embedding_vector_size", None)
            if expected_size is not None and table_vector_size != expected_size:
                sys.exit(
                    f"Error: SQL configuration mismatch!\nExpected vector size {expected_size} per model, found {table_vector_size} in database for table '{table_name}'."
                )
            msg_size = expected_size if expected_size is not None else table_vector_size
            logger.info(
                f"\t\t\t--> Table '{table_name}' already exists with {len(existing_columns)} columns. Embedding vector size: {msg_size}"
            )
            # Ensure HNSW index exists (handles upgrades of existing deployments)
            hnsw_idx_name = f"idx_{table_name}_embedding_hnsw"
            cur.execute(
                "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexname = %s;",
                (table_name, hnsw_idx_name),
            )
            if cur.fetchone() is None:
                try:
                    cur.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
                        ).format(
                            sql.Identifier(hnsw_idx_name),
                            sql.Identifier(table_name),
                        )
                    )
                    conn.commit()
                    logger.info(
                        f"\t\t\t--> Created HNSW index '{hnsw_idx_name}' on '{table_name}'."
                    )
                except Exception as e:
                    conn.rollback()
                    logger.info(
                        f"\t\t\t--> Warning: failed creating HNSW index on '{table_name}': {e}"
                    )
        else:
            logger.info(
                f"\t\t\t--> Table '{table_name}' already exists with {len(existing_columns)} columns. Embeddings not stored."
            )

    columns = get_table_columns_and_types(conn, table_name)
    data_base.add_table(table_name, columns)

    # Cleanup
    cur.close()
    data_base.put_conn(conn)
    return table_name, store_embeddings_flag, metadata


def is_string_type(sql_type):
    """Check if the given SQL column type is a string or equivalent."""
    string_types = {"CHAR", "VARCHAR", "TEXT", "CLOB", "NVARCHAR", "NCHAR"}
    return sql_type.upper() in string_types


def store_one_embedding_in_db(
    data_base,
    embedding,
    table_name,
    bucket_name,
    s3_key,
    saved_s3_object=None,
    checksum=None,
    content=None,
    start=None,
    end=None,
    md5=None,
    model_name=None,
    metadata=None,
    silent=False,
):
    """
    Stores embeddings into the PostgreSQL table safely. Updates if `s3_object` exists.

    Args:
        data_base: data base object
        embedding: Vector embedding (list of floats)
        table_name: Name of the table to store embeddings
        bucket_name: Name of the S3 bucket
        s3_key: S3 object key
    """
    date = datetime.today().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    conn = data_base.get_conn()
    cur = None
    try:
        cur = conn.cursor()

        # Ensure embedding is a list before inserting
        embedding = embedding.tolist() if hasattr(embedding, "tolist") else embedding

        s3_object = f"{bucket_name}/{s3_key}"  # Full S3 path

        # INSERT and UPDATE if s3_object already exists
        # Start building the INSERT query dynamically for the given table
        safe_table = sql.Identifier(table_name).as_string(conn)
        has_embedding_col = data_base.has_column(table_name, "embedding")

        if has_embedding_col:
            query = f"""
        INSERT INTO {safe_table} (
            s3_object, embedding, date, time, checksum, model
        """
            # Initialize the list of values corresponding to the columns
            values = [s3_object, embedding, date, time, checksum, model_name]
        else:
            query = f"""
        INSERT INTO {safe_table} (
            s3_object, date, time, checksum
        """
            values = [s3_object, date, time, checksum]

        # If the table has the "_content" column, add related columns to the query
        if data_base.has_column(table_name, "_content"):
            query += ", _content, chunk_start, chunk_end, chunk_md5"
            # Add the corresponding values for these columns
            values.extend([content, start, end, md5])

        # If a saved S3 object key is provided, add the column and its value
        if saved_s3_object is not None:
            query += ", saved_s3_object"
            values.append(saved_s3_object)

        # Close the column list and create the VALUES placeholders dynamically
        query += ") VALUES ({})".format(", ".join(["%s"] * len(values)))

        # Determine the ON CONFLICT clause depending on whether the "_content" column exists
        if data_base.has_column(table_name, "_content"):
            # Use a compound unique key for conflicts when storing chunked content
            query += " ON CONFLICT (s3_object, chunk_start) DO UPDATE SET"
        else:
            # Use only the s3_object key for conflicts in simpler tables
            query += " ON CONFLICT (s3_object) DO UPDATE SET"

        # Specify the columns to update in case of conflict (always update these)
        if has_embedding_col:
            query += """
            embedding = EXCLUDED.embedding,
            date = EXCLUDED.date,
            time = EXCLUDED.time,
            checksum = EXCLUDED.checksum,
            model = EXCLUDED.model
        """
        else:
            query += """
            date = EXCLUDED.date,
            time = EXCLUDED.time,
            checksum = EXCLUDED.checksum
        """

        # If the "_content" column exists, also update all content-related fields
        if data_base.has_column(table_name, "_content"):
            query += """,
            _content = EXCLUDED._content,
            chunk_start = EXCLUDED.chunk_start,
            chunk_end = EXCLUDED.chunk_end,
            chunk_md5 = EXCLUDED.chunk_md5
            """

        # If a saved S3 object key is provided, update it as well
        if saved_s3_object is not None:
            query += ", saved_s3_object = EXCLUDED.saved_s3_object"

        # Finish the query by returning the ID of the inserted or updated row
        query += " RETURNING id;"

        # Execute the query with the prepared values
        cur.execute(query, values)

        # Fetch the inserted/updated ID
        inserted_id = cur.fetchone()[0]  # Fetch the first column of the result

        # Now manage metadata if any
        if metadata:
            unknown_keys = []
            for key, value in metadata.items():
                sql_type = data_base.get_column_type(table_name, key)
                if sql_type is None:
                    unknown_keys.append(key)
                    continue

                if sql_type.upper() in ["INT", "INTEGER"]:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Metadata '{key}' must be INT but value '{value}' is not."
                        ) from None
                elif not is_string_type(sql_type):
                    # For non-string/non-int types, rely on DB to validate
                    pass

                # Execute the UPDATE query to store metadata in the row with the matching ID
                upd = sql.SQL("UPDATE {} SET {} = %s WHERE id = %s;").format(
                    sql.Identifier(table_name), sql.Identifier(key)
                )
                cur.execute(upd, (value, inserted_id))

            if unknown_keys:
                raise ValueError(
                    f"Unknown metadata key(s) for table '{table_name}': {', '.join(unknown_keys)}"
                )

        conn.commit()

        if not silent:
            logger.info(
                f"\t--> Stored/Updated embedding in '{table_name}' for '{s3_object}'"
            )
            if metadata:
                logger.info("\t--> metadata have been added/updated'")
            if saved_s3_object:
                logger.info(f"\t--> Output saved in {saved_s3_object}'")

    except Exception as e:
        try:
            if conn:
                conn.rollback()
        except Exception as rollback_exc:
            logger.info(f"\t--> Error during rollback: {rollback_exc}")
        logger.info(f"\t--> Error storing embedding for query: {e}")

    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception as close_err:
                logger.info(f"Warning: Failed to close cursor: {close_err}")
        data_base.put_conn(conn)


def store_embeddings_in_db(
    data_base,
    embeddings,
    table_name,
    bucket_name,
    s3_keys,
    saved_s3_objects,
    modality_name,
    checksums,
    model_name,
    metadata_list=None,
):
    """
    Store embeddings in the database depending on the data modality.

    Parameters:
    - data_base: Database manager object
    - embeddings: List of embeddings (or list of list of chunk embeddings for text)
    - table_name: Name of the SQL table
    - bucket_name: Name of the S3 bucket
    - s3_keys: List of S3 object keys corresponding to the data
    - modality_name: Either 'image', 'text', or 'text_file'
    - metadata_list: Optional list of metadata dictionaries, one per S3 object
    """
    if metadata_list is None:
        metadata_list = [{} for _ in s3_keys]

    if modality_name in ("image", "tomography", "volume"):
        # One embedding per image file
        for embedding, s3_key, saved_s3_object, metadata, checksum in zip(
            embeddings,
            s3_keys,
            saved_s3_objects,
            metadata_list,
            checksums,
            strict=False,
        ):
            # print(f"storing embedding for {s3_key}... ")
            store_one_embedding_in_db(
                data_base,
                embedding=embedding,
                table_name=table_name,
                bucket_name=bucket_name,
                s3_key=s3_key,
                saved_s3_object=saved_s3_object,
                content=None,
                metadata=metadata,
                model_name=model_name,
                checksum=checksum,
                silent=False,
            )

    elif modality_name in ("text", "text_file", "text file"):
        # Multiple embeddings per file (one per text chunk)
        for chunked_embeddings, s3_key, saved_s3_object, metadata, checksum in zip(
            embeddings,
            s3_keys,
            saved_s3_objects,
            metadata_list,
            checksums,
            strict=False,
        ):
            sorted_chunks = sorted(chunked_embeddings, key=lambda chunk: chunk["start"])
            for chunk in sorted_chunks:
                store_one_embedding_in_db(
                    data_base,
                    embedding=chunk["embedding"],
                    table_name=table_name,
                    bucket_name=bucket_name,
                    s3_key=s3_key,
                    saved_s3_object=saved_s3_object,
                    content=chunk["text"],
                    start=chunk["start"],
                    end=chunk["end"],
                    md5=chunk["md5"],
                    metadata=metadata,
                    model_name=model_name,
                    checksum=checksum,
                )

    else:
        raise ValueError(
            f"Unsupported modality '{modality_name}' for embedding storage."
        )


def check_status_db(bucket_name, s3_key, checksum, data_base, table_name):
    conn = data_base.get_conn()
    cur = None
    try:
        cur = conn.cursor()
        s3_object = f"{bucket_name}/{s3_key}"  # Full S3 path

        # Query to get the existing checksum for this S3 object
        query = sql.SQL("SELECT checksum FROM {} WHERE s3_object = %s;").format(
            sql.Identifier(table_name)
        )
        cur.execute(query, (s3_object,))
        results = cur.fetchall()

        # If no rows exist yet for this object, report as not_found (not mismatch)
        if not results:
            return {
                "status": "not_found",
                "message": f"No entry found in table '{table_name}' for key: {s3_object}",
            }

        # Extract checksums into a list (flatten from rows)
        db_checksums = [row[0] for row in results]

        # print(f"{checksum=} <=> {db_checksums=}")

        if checksum in db_checksums:
            return {"status": "match", "message": f"Checksum match for '{s3_object}'."}
        else:
            return {
                "status": "mismatch",
                "message": f"Checksum mismatch for '{s3_object}': expected '{checksum}', found {db_checksums}",
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception occurred during DB check: {e}",
        }
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception as close_err:
                logger.info(f"Warning: failed to close cursor: {close_err}")
        data_base.put_conn(conn)


def check_S3_object_status_in_db(
    path_to_bucket, s3_objects, checksums, buckets_cfg: BucketsCfg
):

    selected_fields = ["data_base", "table_name"]
    data_base, table_name = buckets_cfg.get(path_to_bucket, fields=selected_fields)
    status = []
    for i, s3_object in enumerate(s3_objects):
        _, s3_key = s3_object.replace("s3://", "").split("/", 1)
        status.append(
            check_status_db(path_to_bucket, s3_key, checksums[i], data_base, table_name)
        )
    return status
