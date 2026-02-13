#!/usr/bin/python3
import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2 import pool as pg_pool
from datetime import datetime
from umrs.BucketsCfg import BucketsCfg
from umrs.TaskCmdStatus import TaskStatus

class DataBase:
    def __init__(self, db_pool, db_name, username):
        self.db_pool = db_pool
        self.db_name = db_name
        self.username = username
        self.tables = dict()

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
            print(f"Table '{table_name}' not found in the database '{self.db_name}'.")
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
            print(f"Column '{column_name}' not found in table '{table_name}'.")
            return None
        
    def table_exists(self, conn, table_name: str) -> bool:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                );
            """, (table_name,))
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
        conn = self.db_pool.getconn()
        try:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

            # 1️⃣ Check the table exist
            if not self.table_exists(conn, table_name):
                raise ValueError(
                    f"Table '{table_name}' does not exist in database '{self.db_name}'"
                )

            channel = f"{table_name}_update"

            with conn.cursor() as cur:
                cur.execute(f"LISTEN {channel};")

            # 2️⃣ Block until notification

            if select.select([conn.fileno()], [], [], timeout) == ([], [], []):
                return None  # timeout

            conn.poll()
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
            raise ValueError(
                f"Table '{table_name}' has no 's3_object' column."
            )

        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT COUNT(DISTINCT s3_object)
                    FROM {table_name}
                    WHERE s3_object IS NOT NULL;
                    """
                )
                return cur.fetchone()[0]

        finally:
            self.db_pool.putconn(conn)

def wait_table_modification_on_db(active_threads, user, key, data_base, table_name, timeout, control_mq, zmq_sender_id, on_finish=None):
    from server_umrs import ENABLE_PRETTY_PRINT
    from umrs.pretty_print import pretty_print_thread_status

    try:
        res = data_base.wait_for_table_update(
            table_name,
            timeout=timeout
        )
    except Exception as e:
        raise RuntimeError(
            f"Error while waiting for table '{table_name}' update: {e}"
        ) from e
    
    if res is None : 
        msg = "TIMEOUT"
    else : 
        msg = res 
    
    control_mq.send_string(msg, zmq_sender_id)

    active_threads.update_status_to_done()
    # print(f"wait is done for {user=}")
    active_threads.record_status_for_users_by_key([user], key)
    active_threads.remove_from_running_list()        

    # execute callback to update task status to completed
    if on_finish:
        on_finish(TaskStatus.COMPLETED)

    with active_threads.lock:
        pretty_print_thread_status(active_threads.thread_dict, running_only=False, print_output=ENABLE_PRETTY_PRINT)
    print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")




def create_db_name(sql_username, xp_name):
    db_name = xp_name+"_"+sql_username
    db_name = db_name.replace(" ","_").lower()
    return db_name

def db_user_exists(conn, username):
    with conn.cursor() as cur:
        # Check if the username exists
        cur.execute("SELECT 1 FROM pg_roles where rolname = %s;", (username,))
        exists = cur.fetchone()
        return exists


def create_db_user(username):
    # Database credentials
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = "5432"
    DB_SUPERUSER = "postgres"  # Use a superuser or admin account
    DB_SUPERUSER_PASSWORD = os.environ.get("DB_SUPERUSER_PASSWORD", "postgres")

    conn = psycopg2.connect(dbname="postgres", user=DB_SUPERUSER, password=DB_SUPERUSER_PASSWORD, host=DB_HOST, port=DB_PORT)
    conn.autocommit = True  # Enable auto-commit mode

    DB_USER_PASSWORD = os.environ.get("DB_USER_PASSWORD", "ejeannot")
    cur = conn.cursor()
    if db_user_exists(conn, username):
        print(f"==> User '{username}' exist in DB")
        # Ensure password matches the configured one
        cur.execute(sql.SQL("ALTER USER {} WITH PASSWORD %s;").format(sql.Identifier(username)), (DB_USER_PASSWORD,))
        conn.commit()
        cur.close()
        conn.close()
        return

    cur.execute(
        sql.SQL("CREATE USER {} WITH PASSWORD %s;").format(sql.Identifier(username)),
        (DB_USER_PASSWORD,)
    )
    print(f"==> User '{username}' created in DB")
    cur.close()
    conn.close()


def create_db(sql_username, xp_name):
    db_name = create_db_name(sql_username, xp_name)
    print(f"\t\t-->  Configuring database '{db_name}'...")
    # Database credentials
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = "5432"
    DB_SUPERUSER = "postgres"  # Use a superuser or admin account
    DB_SUPERUSER_PASSWORD = os.environ.get("DB_SUPERUSER_PASSWORD", "postgres")
    DB_NAME = db_name  # Replace with the target database
    DB_USER = sql_username  # Replace with the target user

    # Connect to PostgreSQL default database (`postgres`)
    conn = psycopg2.connect(dbname="postgres", user=DB_SUPERUSER, password=DB_SUPERUSER_PASSWORD, host=DB_HOST, port=DB_PORT)
    conn.autocommit = True  # Enable auto-commit mode
    cur = conn.cursor()

    # Check if the database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    exists = cur.fetchone()

    if exists:
        print(f"\t\t-->  Database '{db_name}' already exists.")
    else:
        # Create the database if it does not exist
        cur.execute(sql.SQL("CREATE DATABASE {} OWNER {};" ).format(
            sql.Identifier(DB_NAME), sql.Identifier(DB_USER)
        ))
        print(f"\t\t-->  Database '{db_name}' created successfully.")

        # Connect to the new database as superuser and create the extension
        conn_new = psycopg2.connect(dbname=DB_NAME, user=DB_SUPERUSER, password=DB_SUPERUSER_PASSWORD, host=DB_HOST, port=DB_PORT)
        conn_new.autocommit = True
        cur_new = conn_new.cursor()
        try:
            cur_new.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print(f"\t\t-->  pgvector extension enabled in '{db_name}'.")
        except Exception as e:
            print(f"\t\t-->  Error enabling pgvector extension in '{db_name}': {e}")
        finally:
            cur_new.close()
            conn_new.close()
    # Clean up and then create a pool to the target DB as the application user
    cur.close()
    conn.close()

    DB_USER_PASSWORD = os.environ.get("DB_USER_PASSWORD", "ejeannot")
    db_pool = pg_pool.SimpleConnectionPool(
        minconn=1, maxconn=50,  # Adjust based on your needs
        dbname=db_name,
        user=sql_username,
        password=DB_USER_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return DataBase(db_pool, db_name, sql_username), db_name


def get_table_columns_and_types(conn, table_name, schema='public'):
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
            column_info = {column: data_type for column, data_type in rows}

    except Exception as e:
        print(f"Error fetching column info: {e}")

    return column_info


def ensure_pgvector_enabled(conn):
    """Ensure pgvector extension is installed."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

def get_vector_size(cursor, table_name):
    """
    Get the size (dimension) of the 'embedding' vector column in a PostgreSQL table.

    Args:
        cursor: psycopg2 cursor object
        table_name: Name of the table to check

    Returns:
        int: Vector size (dimension) if found, else None
    """
    query = f"""
    SELECT atttypmod AS vector_size
    FROM pg_attribute
    WHERE attrelid = %s::regclass AND attname = 'embedding';
    """

    cursor.execute(query, (table_name,))
    result = cursor.fetchone()

    return result[0] if result else None  # Return vector size or None

def metadata_dict(sql_cfg):
    metadata = {}
    for i,column in enumerate(sql_cfg["meta_data"]):
        metadata[f"key{i}"] = column

    return metadata

def enable_table_notifications(conn, table_name: str):
    """
    Enable PostgreSQL NOTIFY on INSERT/UPDATE/DELETE for a given table.
    Creates a trigger and trigger function if they do not exist.
    """
    func_name = f"notify_{table_name}_update"
    trigger_name = f"{table_name}_notify_trigger"
    channel = f"{table_name}_update"

    with conn.cursor() as cur:
        # 1️⃣ Create trigger function
        cur.execute(f"""
        CREATE OR REPLACE FUNCTION {func_name}()
        RETURNS trigger AS $$
        BEGIN
            PERFORM pg_notify('{channel}', TG_OP);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """)

        # 2️⃣ Drop trigger if exists (idempotent)
        cur.execute(f"""
        DROP TRIGGER IF EXISTS {trigger_name} ON {table_name};
        """)

        # 3️⃣ Create trigger
        cur.execute(f"""
        CREATE TRIGGER {trigger_name}
        AFTER INSERT OR UPDATE OR DELETE ON {table_name}
        FOR EACH STATEMENT
        EXECUTE FUNCTION {func_name}();
        """)

    conn.commit()

def configure_table(data_base, sql_cfg,  modality_name, model):

    db_pool = data_base.db_pool;
    table_name = sql_cfg['table_name'].lower()
    (f"\t\t\t--> Configuring table '{table_name}'...")
    table_columns = [];
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
        else:
            sys.exit(f"Do not know how to manage return type '{model.return_type}' in cfg file")

    for column in sql_cfg["meta_data"]:
        table_columns.append(column)

    if modality_name == "text" or modality_name == "text file":
        columns_to_add = [
            {"name": "_content", "type": "TEXT"},
            {"name": "chunk_start", "type": "BIGINT"},
            {"name": "chunk_end", "type": "BIGINT"},
            {"name": "chunk_md5", "type": "TEXT"}
        ]
        table_columns.extend(columns_to_add)
        table_columns.append({"name": "UNIQUE(s3_object, chunk_start)", "type": ""})
    else:
        table_columns.append({"name": "UNIQUE(s3_object)", "type": ""})

    metadata = metadata_dict(sql_cfg)

    conn = db_pool.getconn()
    ensure_pgvector_enabled(conn)
    cur = conn.cursor()

    # Check if table exists
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);", (table_name,))
    table_exists = cur.fetchone()[0]

    if not table_exists:
        # Create table if it doesn't exist
        column_defs = ", ".join([f"{column['name']} {column['type']}" for column in table_columns])
        create_query = sql.SQL("CREATE TABLE {} ({});").format(
            sql.Identifier(table_name), sql.SQL(column_defs)
        )
        cur.execute(create_query)
        conn.commit()
        print(f"\t\t\t--> Table '{table_name}' created successfully.")
        # Create helpful indexes
        try:
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (s3_object);").format(
                sql.Identifier(f"idx_{table_name}_s3_object"), sql.Identifier(table_name)))
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (checksum);").format(
                sql.Identifier(f"idx_{table_name}_checksum"), sql.Identifier(table_name)))
            conn.commit()
        except Exception as e:
            print(f"\t\t\t--> Warning: failed creating indexes on '{table_name}': {e}")

    # enble notification for table that generate something so the server can listen to such notifications
    if model.return_type is not None:
        if model.return_type == "file":
            enable_table_notifications(conn, table_name)
        else:
            sys.exit(f"Do not know how to manage return type '{model.return_type}' in cfg file")

        

    else:
        # Validate existing table schema
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;", (table_name,))
        existing_columns = {row[0]: row[1] for row in cur.fetchall()}

        # Remove UNIQUE constraint rows
        real_columns = [
            col for col in table_columns
            if not col["name"].upper().startswith("UNIQUE(")
        ]

        # Verify required columns exist in the table (metadata columns must also exist)
        core_required = real_columns
        for column in core_required:
            if column["name"] not in existing_columns:
                sys.exit(f"Missing column: {column['name']} ({column['type']}) in '{table_name}'.\nExisting columns: {list(existing_columns.keys())}")

        table_vector_size = get_vector_size(cur, table_name)
        if sql_cfg.get("store_embeddings") == "yes":
            expected_size = getattr(model, 'embedding_vector_size', None)
            if expected_size is not None and table_vector_size != expected_size:
                sys.exit(f"Error: SQL configuration mismatch!\nExpected vector size {expected_size} per model, found {table_vector_size} in database for table '{table_name}'.")
            msg_size = expected_size if expected_size is not None else table_vector_size
            print(f"\t\t\t--> Table '{table_name}' already exists with {len(existing_columns)} columns. Embedding vector size: {msg_size}")
        else:
            print(f"\t\t\t--> Table '{table_name}' already exists with {len(existing_columns)} columns. Embeddings not stored.")



    columns = get_table_columns_and_types(conn, table_name)
    data_base.add_table(table_name, columns)

    # Cleanup
    cur.close()
    db_pool.putconn(conn)
    return table_name, store_embeddings_flag, metadata

def is_string_type(sql_type):
    """Check if the given SQL column type is a string or equivalent."""
    string_types = {"CHAR", "VARCHAR", "TEXT", "CLOB", "NVARCHAR", "NCHAR"}
    return sql_type.upper() in string_types

def store_one_embedding_in_db(data_base, embedding, table_name, bucket_name, s3_key, saved_s3_object=None, checksum=None, content = None, start = None, end = None, md5 = None, model_name=None, metadata = None, silent = False):
    """
    Stores embeddings into the PostgreSQL table safely. Updates if `s3_object` exists.

    Args:
        db_pool: Connection pool object
        embedding: Vector embedding (list of floats)
        table_name: Name of the table to store embeddings
        bucket_name: Name of the S3 bucket
        s3_key: S3 object key
    """
    date = datetime.today().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')

    db_pool = data_base.db_pool
    conn = db_pool.getconn()
    cur = None
    try:
        cur = conn.cursor()

        # Ensure embedding is a list before inserting
        embedding = embedding.tolist() if hasattr(embedding, "tolist") else embedding

        s3_object = f"{bucket_name}/{s3_key}"  # Full S3 path

        # INSERT and UPDATE if s3_object already exists
        # Start building the INSERT query dynamically for the given table
        query = f"""
        INSERT INTO {table_name} (
            s3_object, embedding, date, time, checksum, model
        """
        # Initialize the list of values corresponding to the columns
        values = [s3_object, embedding, date, time, checksum, model_name]

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
        query += """
            embedding = EXCLUDED.embedding,
            date = EXCLUDED.date,
            time = EXCLUDED.time,
            checksum = EXCLUDED.checksum,
            model = EXCLUDED.model
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
                        raise ValueError(f"Metadata '{key}' must be INT but value '{value}' is not.")
                elif not is_string_type(sql_type):
                    # For non-string/non-int types, rely on DB to validate
                    pass

                # Execute the UPDATE query to store metadata in the row with the matching ID
                upd = sql.SQL("UPDATE {} SET {} = %s WHERE id = %s;").format(
                    sql.Identifier(table_name), sql.Identifier(key))
                cur.execute(upd, (value, inserted_id))

            if unknown_keys:
                raise ValueError(f"Unknown metadata key(s) for table '{table_name}': {', '.join(unknown_keys)}")


        conn.commit()

        if not silent :
            print(f"\t--> Stored/Updated embedding in '{table_name}' for '{s3_object}'")
            if metadata:
                print(f"\t--> metadata have been added/updated'")
            if saved_s3_object:
                  print(f"\t--> Output saved in {saved_s3_object}'")
          


    except Exception as e:
        try:
            if conn:
                conn.rollback()
        except Exception as rollback_exc:
            print(f"\t--> Error during rollback: {rollback_exc}")
        print(f"\t--> Error storing embedding for query: {e}")

    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                # Consider logging the exception instead of silent pass
                print(f"Warning: Failed to close cursor: {e}")
        db_pool.putconn(conn)

def store_embeddings_in_db(data_base, embeddings, table_name, bucket_name, s3_keys, saved_s3_objects, modality_name, checksums, model_name, metadata_list=None):
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

    if modality_name in ("image", "tomography"):
        # One embedding per image file
        for embedding, s3_key, saved_s3_object, metadata, checksum in zip(embeddings, s3_keys, saved_s3_objects, metadata_list, checksums):
            # print(f"storing embedding for {s3_key}... ")
            store_one_embedding_in_db(
                data_base,
                embedding=embedding,
                table_name=table_name,
                bucket_name=bucket_name,
                s3_key=s3_key,
                saved_s3_object= saved_s3_object,
                content=None,
                metadata=metadata,
                model_name = model_name,
                checksum=checksum,
                silent = False
            )

    elif modality_name in ("text", "text file"):
        # Multiple embeddings per file (one per text chunk)
        for chunked_embeddings, s3_key, saved_s3_object, metadata, checksum in zip(embeddings, s3_keys, saved_s3_objects, metadata_list, checksums):
            sorted_chunks = sorted(chunked_embeddings, key=lambda chunk: chunk["start"])
            for chunk in sorted_chunks:
                store_one_embedding_in_db(
                    data_base,
                    embedding=chunk["embedding"],
                    table_name=table_name,
                    bucket_name=bucket_name,
                    s3_key=s3_key,
                    saved_s3_object= saved_s3_object,
                    content=chunk["text"],
                    start=chunk["start"],
                    end=chunk["end"],
                    md5=chunk["md5"],
                    metadata=metadata,
                    model_name = model_name,
                    checksum=checksum
                )

    else:
        raise ValueError(f"Unsupported modality '{modality_name}' for embedding storage.")

def check_status_db(bucket_name, s3_key, checksum, data_base, table_name):
    db_pool = data_base.db_pool
    conn = db_pool.getconn()
    cur = None
    try:
        cur = conn.cursor()
        s3_object = f"{bucket_name}/{s3_key}"  # Full S3 path

        # Query to get the existing checksum for this S3 object
        query = sql.SQL("SELECT checksum FROM {} WHERE s3_object = %s;").format(sql.Identifier(table_name))
        cur.execute(query, (s3_object,))
        results = cur.fetchall()

        # If no rows exist yet for this object, report as not_found (not mismatch)
        if not results:
            return {
                "status": "not_found",
                "message": f"No entry found in table '{table_name}' for key: {s3_object}"
            }


        # Extract checksums into a list (flatten from rows)
        db_checksums = [row[0] for row in results]

        # print(f"{checksum=} <=> {db_checksums=}")

        if checksum in db_checksums:
            return {
                "status": "match",
                "message": f"Checksum match for '{s3_object}'."
            }
        else:
            return {
                "status": "mismatch",
                "message": f"Checksum mismatch for '{s3_object}': expected '{checksum}', found {db_checksums}"
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception occurred during DB check: {e}"
        }
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception as e:
                print(f"Warning: failed to close cursor: {e}")
        db_pool.putconn(conn)

def check_S3_object_status_in_db(path_to_bucket, s3_objects, checksums, buckets_cfg: BucketsCfg):
    
    selected_fields = ["data_base", "table_name"]
    data_base, table_name  = buckets_cfg.get(path_to_bucket, filter=selected_fields)
    status = []
    for i, s3_object in enumerate(s3_objects):
        _, s3_key = s3_object.replace("s3://", "").split("/", 1)
        status.append(check_status_db(path_to_bucket, s3_key, checksums[i], data_base, table_name))
    return status
