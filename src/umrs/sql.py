#!/usr/bin/python3
import os
import sys
import psycopg2
from psycopg2 import sql, pool
from datetime import datetime
import psycopg2


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

    if db_user_exists(conn, username):
        print(f"==> User '{username}' exist in DB")
        conn.close()
        return

    password = username

    cur = conn.cursor()
    cur.execute(
        sql.SQL("CREATE USER {} WITH PASSWORD %s;").format(sql.Identifier(username)),
        (password,)
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
        cur.close()
        conn.close()
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
        # Return a new connection to the created database
        DB_USER_PASSWORD = os.environ.get("DB_USER_PASSWORD", "ejeannot")
        db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=5,  # Adjust based on your needs
            dbname=db_name,
            user=sql_username,
            password=DB_USER_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return DataBase(db_pool, db_name, sql_username), db_name

    # Clean up
    cur.close()
    conn.close()
    # Return a new connection to the created database
    DB_USER_PASSWORD = os.environ.get("DB_USER_PASSWORD", "ejeannot")
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1, maxconn=5,  # Adjust based on your needs
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

def configure_table(data_base, sql_cfg,  modality_name, model):

    db_pool = data_base.db_pool;
    table_name = modality_name.replace(" ","_").lower()
    print(f"\t\t\t--> Configuring table '{table_name}'...")
    table_columns = [];
    column = {"name": "id", "type": "SERIAL PRIMARY KEY"}
    table_columns.append(column)
    column = {"name": "s3_object", "type": "TEXT"}
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
        column = {"name": "model", "type": f"CHAR(1024)"}
        table_columns.append(column)

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

    else:
        # Validate existing table schema
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;", (table_name,))
        existing_columns = {row[0]: row[1] for row in cur.fetchall()}

        # Remove UNIQUE constraint rows
        real_columns = [
            col for col in table_columns
            if not col["name"].upper().startswith("UNIQUE(")
        ]

        # Vérification des colonnes dans la table existante
        for column in real_columns:
            if column["name"] not in existing_columns:
                sys.exit(f"Missing column: {column['name']} ({column['type']}) in '{table_name}'.\nExisting columns: {list(existing_columns.keys())}")

        table_vector_size = get_vector_size(cur, table_name)
        if table_vector_size != vector_size:
            sys.exit(f"Error: SQL configuration mispatch!\nThe vector size to store embeddings is {vector_size} in the json config file and {table_vector_size} in the sql database.")
        print(f"\t\t\t--> Table '{table_name}' already exists with {len(existing_columns)} columns. Embedding vector size: {vector_size}")



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

def store_one_embedding_in_db(data_base, embedding, table_name, bucket_name, s3_key,  content = None, start = None, end = None, md5 = None, metadata = {}, silent = False):
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
    try:
        cur = conn.cursor()

        # Ensure embedding is a list before inserting
        embedding = embedding.tolist() if hasattr(embedding, "tolist") else embedding

        s3_object = f"{bucket_name}/{s3_key}"  # Full S3 path

        # INSERT and UPDATE if s3_object already exists
        if data_base.has_column(table_name, "_content"):
            query = f"""
            INSERT INTO {table_name} (
            s3_object, _content, embedding, date, time,
            chunk_start, chunk_end, chunk_md5
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (s3_object, chunk_start) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            date = EXCLUDED.date,
            time = EXCLUDED.time,
            _content = EXCLUDED._content,
            chunk_start = EXCLUDED.chunk_start,
            chunk_end = EXCLUDED.chunk_end,
            chunk_md5 = EXCLUDED.chunk_md5
            RETURNING id;
            """
            cur.execute(query, (s3_object, content, embedding, date, time, start, end, md5))
        else:
            query = f"""
            INSERT INTO {table_name} (s3_object, embedding, date, time)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (s3_object) DO UPDATE
            SET embedding = EXCLUDED.embedding, date = EXCLUDED.date, time = EXCLUDED.time
            RETURNING id;
            """
            cur.execute(query, (s3_object, embedding, date, time))

         # Fetch the inserted/updated ID
        inserted_id = cur.fetchone()[0]  # Fetch the first column of the result

         # Now manage metadata if any
        if metadata:
            for key, value in metadata.items():
                sql_type = data_base.get_column_type(table_name, key)
                if sql_type is None:
                    print(f"Error: Field '{key}' unknown in table {table_name}. Skipping...")
                    continue

                if sql_type.upper() in ["INT", "INTEGER"]:
                    if not value.isdigit():
                        print(f"Key '{key}' must be of type INT but value '{value}' is not. Skipping...")
                        continue
                elif not is_string_type(sql_type):
                    print(f"Checking field of type '{sql_type}' is not implemented. This might raise an error...")

                # Execute the UPDATE query to store metadata in the row with the matching ID
                try:
                    query = f"""
                    UPDATE {table_name}
                    SET {key} = %s
                    WHERE id = %s;
                    """
                    cur.execute(query, (value, inserted_id))
                except Exception as e:
                    print(f"Error updating metadata '{key}' in {table_name}: {e}")


        conn.commit()

        if not silent :
            print(f"\t--> Stored/Updated embedding in '{table_name}' for '{s3_object}'")
            if metadata:
                print(f"\t--> metadata have been added/updated'")


    except Exception as e:
        print(f"\t--> Error storing embedding: {e}")

    finally:
        cur.close()
        db_pool.putconn(conn)

def store_embeddings_in_db(data_base, embeddings, table_name, bucket_name, s3_keys, modality_name, metadata_list=[]):
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

    if modality_name == "image":
        # One embedding per image file
        for embedding, s3_key, metadata in zip(embeddings, s3_keys, metadata_list):
            store_one_embedding_in_db(
                data_base,
                embedding=embedding,
                table_name=table_name,
                bucket_name=bucket_name,
                s3_key=s3_key,
                content=None,
                metadata=metadata
            )

    elif modality_name in ("text", "text file"):
        # Multiple embeddings per file (one per text chunk)
        for chunked_embeddings, s3_key, metadata in zip(embeddings, s3_keys, metadata_list):
            for chunk in chunked_embeddings:
                store_one_embedding_in_db(
                    data_base,
                    embedding=chunk["embedding"],
                    table_name=table_name,
                    bucket_name=bucket_name,
                    s3_key=s3_key,
                    content=chunk["text"],
                    start=chunk["start"],
                    end=chunk["end"],
                    md5=chunk["md5"],
                    metadata=metadata
                )

    else:
        raise ValueError(f"Unsupported modality '{modality_name}' for embedding storage.")


def get_table_name_from_bucket_name(bucket_name):
    """
    Generate a PostgreSQL table name from a bucket name.

    Args:
        bucket_name (str): The S3 bucket name (e.g., 'user1.images-and-texts.bucket-1')

    Returns:
        str: A valid PostgreSQL table name
    """
    # Replace dots and hyphens with underscores and make lowercase
    table_name = bucket_name.replace('.', '_').replace('-', '_').lower()

    # Ensure it starts with a letter or underscore (PostgreSQL requirement)
    if table_name[0].isdigit():
        table_name = 'table_' + table_name

    return table_name





