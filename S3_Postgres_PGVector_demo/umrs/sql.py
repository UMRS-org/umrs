#!/usr/bin/python3
import os
import sys
import psycopg2
from psycopg2 import sql, pool
from datetime import datetime


def create_db_and_get_pool(sql_username, xp_name):
    db_name = xp_name+"_"+sql_username
    db_name = db_name.replace(" ","_").lower()
    print(f"\t-->  Configuring database '{db_name}'...")
    # Database credentials
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_SUPERUSER = "ejeannot"  # Use a superuser or admin account
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
    return db_pool


import psycopg2
from datetime import datetime

def store_embeddings_in_db(db_pool, embedding, table_name, bucket_name, s3_key, text = None):
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
    
    conn = db_pool.getconn()
    try:
        cur = conn.cursor()

        # Ensure embedding is a list before inserting
        embedding = embedding.tolist() if hasattr(embedding, "tolist") else embedding

        s3_object = f"{bucket_name}/{s3_key}"  # ✅ Full S3 path

        # INSERT and UPDATE if s3_object already exists
        if text:
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

        print(f"\t--> Stored/Updated embedding in '{table_name}' for '{s3_object}'")

    except Exception as e:
        print(f"\t--> Error storing embedding: {e}")

    finally:
        cur.close()
        db_pool.putconn(conn)

        
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
        
def configure_sql(db_pool, sql_cfg,  modality_name):
    table_name = modality_name.replace(" ","_").lower()
    print(f"\t\t\t--> Configuring table '{table_name}'...")
    table_fields = [];
    field = {"name": "id", "type": "SERIAL PRIMARY KEY"}
    table_fields.append(field)
    field = {"name": "s3_object", "type": "TEXT UNIQUE"}
    model_name = None
    vector_size = None
    table_fields.append(field)
    if sql_cfg["embeddings"]["store_embeddings"] == "yes":
        model_name = sql_cfg["embeddings"]["model_name"]
        vector_size = int(sql_cfg["embeddings"]["vector_size"])
        field = {"name": "embedding", "type": f"VECTOR({vector_size})"}
        table_fields.append(field)
        field = {"name": "model", "type": f"CHAR(1024)"}
        table_fields.append(field)

    for field in sql_cfg["meta_data"]:
        table_fields.append(field)

        
    conn = db_pool.getconn()
    ensure_pgvector_enabled(conn)
    cur = conn.cursor()

    # Check if table exists
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);", (table_name,))
    table_exists = cur.fetchone()[0]

    if not table_exists:
        # Create table if it doesn't exist
        field_defs = ", ".join([f"{field['name']} {field['type']}" for field in table_fields])
        create_query = sql.SQL("CREATE TABLE {} ({});").format(
            sql.Identifier(table_name), sql.SQL(field_defs)
        )
        cur.execute(create_query)
        conn.commit()
        print(f"\t\t\t--> Table '{table_name}' created successfully.")

    else:
        # Validate existing table schema
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;", (table_name,))
        existing_fields = {row[0]: row[1] for row in cur.fetchall()}

        for field in table_fields:
            if field["name"] not in existing_fields:
                sys.exit(f" Missing field: {field['name']} ({field['type']}) in '{table_name}'.")


        table_vector_size = get_vector_size(cur, table_name)
        if table_vector_size != vector_size:
            sys.exit(f"Error: SQL configuration mispatch!\nThe vector size to store embeddings is {vector_size} in the json config file and {table_vector_size} in the sql database.")
        print(f"\t\t\t--> Table '{table_name}' already exists with {len(existing_fields)} fields. Embedding vector size: {vector_size}")

    # 🧹 Cleanup
    cur.close()
    db_pool.putconn(conn)
    return table_name, model_name
