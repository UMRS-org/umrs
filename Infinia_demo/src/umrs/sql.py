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

        
def create_db(sql_username, xp_name):
    db_name = xp_name+"_"+sql_username
    db_name = db_name.replace(" ","_").lower()
    print(f"\t-->  Configuring database '{db_name}'...")
    # Database credentials
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_SUPERUSER = "postgres"  # Use a superuser or admin account
    DB_NAME = db_name  # Replace with the target database
    DB_USER = sql_username  # Replace with the target user

    # Connect to PostgreSQL default database (`postgres`)
    conn = psycopg2.connect(dbname="postgres", user=DB_SUPERUSER, host=DB_HOST, port=DB_PORT)
    conn.autocommit = True  # Enable auto-commit mode
    cur = conn.cursor()

    # Check if the database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    exists = cur.fetchone()

    if exists:
        print(f"\t-->  Database '{db_name}' already exists.")
    else:
        # Create the database if it does not exist
        cur.execute(sql.SQL("CREATE DATABASE {} OWNER {};").format(
            sql.Identifier(DB_NAME), sql.Identifier(DB_USER)
        ))
        print(f"\t-->  Database '{db_name}' created successfully.")

        # Clean up
    cur.close()
    conn.close()
    # Return a new connection to the created database
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1, maxconn=5,  # Adjust based on your needs
        dbname=db_name,
        user=sql_username,
        host=DB_HOST,
        port=DB_PORT
    )
    return DataBase(db_pool, db_name, sql_username)


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
        
def configure_table(data_base, sql_cfg,  modality_name, model):
    db_pool = data_base.db_pool;
    table_name = modality_name.replace(" ","_").lower()
    print(f"\t\t\t--> Configuring table '{table_name}'...")
    table_columns = [];
    column = {"name": "id", "type": "SERIAL PRIMARY KEY"}
    table_columns.append(column)
    column = {"name": "s3_object", "type": "TEXT UNIQUE"}
    model_name = None
    vector_size = None
    table_columns.append(column)
    store_embeddings_flag = False
    if sql_cfg["store_embeddings"] == "yes":
        store_embeddings_flag = True
        vector_size = model.embedding_vector_size
        column = {"name": "embedding", "type": f"VECTOR({vector_size})"}
        table_columns.append(column)
        column = {"name": "model", "type": f"CHAR(1024)"}
        table_columns.append(column)

    for column in sql_cfg["meta_data"]:
        table_columns.append(column)

        
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

        for column in table_columns:
            if column["name"] not in existing_columns:
                sys.exit(f" Missing column: {column['name']} ({column['type']}) in '{table_name}'.")


        table_vector_size = get_vector_size(cur, table_name)
        if table_vector_size != vector_size:
            sys.exit(f"Error: SQL configuration mispatch!\nThe vector size to store embeddings is {vector_size} in the json config file and {table_vector_size} in the sql database.")
        print(f"\t\t\t--> Table '{table_name}' already exists with {len(existing_columns)} columns. Embedding vector size: {vector_size}")

    

    columns = get_table_columns_and_types(conn, table_name)
    data_base.add_table(table_name, columns)
    
    # Cleanup
    cur.close()
    db_pool.putconn(conn)
    return table_name, store_embeddings_flag



def store_one_embedding_in_db(data_base, embedding, table_name, bucket_name, s3_key, text = None, silent = False):
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

        s3_object = f"{bucket_name}/{s3_key}"  # ✅ Full S3 path

        # INSERT and UPDATE if s3_object already exists
        if data_base.has_column(table_name, "text"):
            query = f"""
            INSERT INTO {table_name} (s3_object, text, embedding, date, time)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (s3_object) DO UPDATE
            SET text = EXCLUDED.text, embedding = EXCLUDED.embedding, date = EXCLUDED.date, time = EXCLUDED.time;
            """
            cur.execute(query, (s3_object, text, embedding, date, time))
        else:
            query = f"""
            INSERT INTO {table_name} (s3_object, embedding, date, time)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (s3_object) DO UPDATE
            SET embedding = EXCLUDED.embedding, date = EXCLUDED.date, time = EXCLUDED.time;
            """
            cur.execute(query, (s3_object, embedding, date, time))
        conn.commit()

        if not silent : 
            print(f"\t--> Stored/Updated embedding in '{table_name}' for '{s3_object}'")

    except Exception as e:
        print(f"\t--> Error storing embedding: {e}")

    finally:
        cur.close()
        db_pool.putconn(conn)

def store_embeddings_in_db(data_base, embeddings, table_name, bucket_name, s3_keys, texts=None):
    for embedding, s3_key, text in zip(embeddings, s3_keys, texts or [None] * len(embeddings)): #"or" retuns the first Truthy value (None being not a Truthy value)
        store_one_embedding_in_db(data_base, embedding, table_name, bucket_name, s3_key, text)
        


        
