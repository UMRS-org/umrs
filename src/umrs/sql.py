#!/usr/bin/python3
import os
import sys
import psycopg2
from psycopg2 import sql, pool
from datetime import datetime
from umrs.db_store import query_db_model


class DataBase:
    def __init__(self, db_pool, db_name, username):
        self.db_pool = db_pool
        self.db_name = db_name
        self.username = username
        self.tables = dict()

    def add_table(self, table_name, columns):
        self.tables[table_name] = columns

    def has_column(self, table_name, column_name):
        if table_name not in self.tables:
            return False
        return column_name in self.tables[table_name]

    def get_column_type(self, table_name, column_name):
        return self.tables.get(table_name, {}).get(column_name)


def create_db_name(sql_username, xp_name):
    db_name = f"{xp_name}_{sql_username}".replace(" ", "_").lower()
    return db_name


def db_user_exists(conn, username):
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s;", (username,))
        return cur.fetchone() is not None


def create_db_user(username):
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_SUPERUSER = os.environ.get("DB_SUPERUSER", "postgres")
    DB_SUPERUSER_PASSWORD = os.environ.get("DB_SUPERUSER_PASSWORD", "postgres")

    conn = psycopg2.connect(dbname="postgres", user=DB_SUPERUSER, password=DB_SUPERUSER_PASSWORD, host=DB_HOST, port=DB_PORT)
    conn.autocommit = True

    if db_user_exists(conn, username):
        print(f"==> User '{username}' exists in DB")
        conn.close()
        return

    password = username
    cur = conn.cursor()
    cur.execute(sql.SQL("CREATE USER {} WITH PASSWORD %s;").format(sql.Identifier(username)), (password,))
    print(f"==> User '{username}' created in DB")
    cur.close()
    conn.close()


def create_db(sql_username, xp_name):
    db_name = create_db_name(sql_username, xp_name)
    print(f"\t\t-->  Configuring database '{db_name}'...")

    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_SUPERUSER = os.environ.get("DB_SUPERUSER", "postgres")
    DB_SUPERUSER_PASSWORD = os.environ.get("DB_SUPERUSER_PASSWORD", "postgres")

    conn = psycopg2.connect(dbname="postgres", user=DB_SUPERUSER, password=DB_SUPERUSER_PASSWORD, host=DB_HOST, port=DB_PORT)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
    exists = cur.fetchone()

    if exists:
        print(f"\t\t-->  Database '{db_name}' already exists.")
    else:
        cur.execute(sql.SQL("CREATE DATABASE {} OWNER {};").format(sql.Identifier(db_name), sql.Identifier(sql_username)))
        print(f"\t\t-->  Database '{db_name}' created successfully.")

        cur.close()
        conn.close()
        conn_new = psycopg2.connect(dbname=db_name, user=DB_SUPERUSER, password=DB_SUPERUSER_PASSWORD, host=DB_HOST, port=DB_PORT)
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

    # Clean up admin connection
    try:
        cur.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass

    # Use admin credentials for the connection pool by default
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1, maxconn=5,
        dbname=db_name,
        user=DB_SUPERUSER,
        password=DB_SUPERUSER_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return DataBase(db_pool, db_name, sql_username), db_name


def get_table_columns_and_types(conn, table_name, schema='public'):
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
            column_info = {column: data_type for column, data_type in rows}
    except Exception as e:
        print(f"Error fetching column info: {e}")
    return column_info


def ensure_pgvector_enabled(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()


def get_vector_size(cursor, table_name):
    query = f"""
    SELECT atttypmod AS vector_size
    FROM pg_attribute
    WHERE attrelid = %s::regclass AND attname = 'embedding';
    """
    cursor.execute(query, (table_name,))
    result = cursor.fetchone()
    return result[0] if result else None


def metadata_dict(sql_cfg):
    metadata = {}
    for i, column in enumerate(sql_cfg.get("meta_data", [])):
        metadata[f"key{i}"] = column
    return metadata


def configure_table(data_base, sql_cfg, modality_name, model):
    db_pool = data_base.db_pool
    table_name = modality_name.replace(" ", "_").lower()
    print(f"\t\t\t--> Configuring table '{table_name}'...")
    table_columns = []
    table_columns.append({"name": "id", "type": "SERIAL PRIMARY KEY"})
    table_columns.append({"name": "s3_object", "type": "TEXT"})
    table_columns.append({"name": "checksum", "type": "TEXT"})
    store_embeddings_flag = False
    table_columns.append({"name": "date", "type": "DATE"})
    table_columns.append({"name": "time", "type": "TIME"})

    vector_size = None
    if sql_cfg.get("store_embeddings") == "yes":
        store_embeddings_flag = True
        vector_size = getattr(model, "embedding_vector_size", None)
        table_columns.append({"name": "embedding", "type": f"VECTOR({vector_size})"})
        table_columns.append({"name": "model", "type": "TEXT"})

    for column in sql_cfg.get("meta_data", []):
        table_columns.append(column)

    if modality_name in ("text", "text file"):
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
    try:
        ensure_pgvector_enabled(conn)
        cur = conn.cursor()

        cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);", (table_name,))
        table_exists = cur.fetchone()[0]

        if not table_exists:
            column_defs = ", ".join([f"{column['name']} {column['type']}" for column in table_columns])
            create_query = sql.SQL("CREATE TABLE {} ({});").format(sql.Identifier(table_name), sql.SQL(column_defs))
            cur.execute(create_query)
            conn.commit()
            print(f"\t\t\t--> Table '{table_name}' created successfully.")
        else:
            cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;", (table_name,))
            existing_columns = {row[0]: row[1] for row in cur.fetchall()}

            real_columns = [col for col in table_columns if not col["name"].upper().startswith("UNIQUE(")]
            for column in real_columns:
                if column["name"] not in existing_columns:
                    sys.exit(f"Missing column: {column['name']} ({column['type']}) in '{table_name}'.\nExisting columns: {list(existing_columns.keys())}")

            if store_embeddings_flag and vector_size is not None:
                table_vector_size = get_vector_size(cur, table_name)
                if table_vector_size != vector_size:
                    sys.exit(f"Error: SQL configuration mismatch! The vector size to store embeddings is {vector_size} in the json config file and {table_vector_size} in the sql database.")
                print(f"\t\t\t--> Table '{table_name}' already exists with {len(existing_columns)} columns. Embedding vector size: {vector_size}")

        columns = get_table_columns_and_types(conn, table_name)
        data_base.add_table(table_name, columns)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        db_pool.putconn(conn)

    return table_name, store_embeddings_flag, metadata


def is_string_type(sql_type):
    string_types = {"CHAR", "VARCHAR", "TEXT", "CLOB", "NVARCHAR", "NCHAR"}
    return sql_type.upper() in string_types


def store_one_embedding_in_db(data_base, embedding, table_name, bucket_name, s3_key, checksum=None, content=None, start=None, end=None, md5=None, model_name=None, metadata=None, silent=False):
    metadata = metadata or {}
    date_str = datetime.today().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M:%S')

    db_pool = data_base.db_pool
    conn = db_pool.getconn()
    cur = conn.cursor()
    query = None
    try:
        embedding = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        s3_object = f"{bucket_name}/{s3_key}"

        if data_base.has_column(table_name, "_content"):
            if start == 0:
                query = f"DELETE FROM {table_name} WHERE s3_object = %s;"
                cur.execute(query, (s3_object,))
            query = f"""
            INSERT INTO {table_name} (
            s3_object, _content, embedding, date, time,
            chunk_start, chunk_end, chunk_md5, checksum, model
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (s3_object, chunk_start) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            date = EXCLUDED.date,
            time = EXCLUDED.time,
            _content = EXCLUDED._content,
            chunk_start = EXCLUDED.chunk_start,
            chunk_end = EXCLUDED.chunk_end,
            chunk_md5 = EXCLUDED.chunk_md5,
            checksum = EXCLUDED.checksum,
            model = EXCLUDED.model
            RETURNING id;
            """
            cur.execute(query, (s3_object, content, embedding, date_str, time_str, start, end, md5, checksum, model_name))
        else:
            query = f"""
            INSERT INTO {table_name} (s3_object, embedding, date, time, checksum, model)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (s3_object) DO UPDATE
            SET embedding = EXCLUDED.embedding, date = EXCLUDED.date, time = EXCLUDED.time, checksum = EXCLUDED.checksum, model = EXCLUDED.model
            RETURNING id;
            """
            cur.execute(query, (s3_object, embedding, date_str, time_str, checksum, model_name))

        inserted_id = cur.fetchone()[0]

        if metadata:
            for key, value in metadata.items():
                sql_type = data_base.get_column_type(table_name, key)
                if sql_type is None:
                    print(f"Error: Field '{key}' unknown in table {table_name}. Skipping...")
                    continue
                if sql_type.upper() in ["INT", "INTEGER"]:
                    if not str(value).isdigit():
                        print(f"Key '{key}' must be of type INT but value '{value}' is not. Skipping...")
                        continue
                elif not is_string_type(sql_type):
                    print(f"Checking field of type '{sql_type}' is not implemented. This might raise an error...")

                try:
                    update_query = f"UPDATE {table_name} SET {key} = %s WHERE id = %s;"
                    cur.execute(update_query, (value, inserted_id))
                except Exception as e:
                    print(f"Error updating metadata '{key}' in {table_name}: {e}")

        conn.commit()
        if not silent:
            print(f"\t--> Stored/Updated embedding in '{table_name}' for '{s3_object}'")
            if metadata:
                print(f"\t--> metadata have been added/updated')")

    except Exception as e:
        print(f"\t--> Error storing embedding for query {query}: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        db_pool.putconn(conn)


def store_embeddings_in_db(data_base, embeddings, table_name, bucket_name, s3_keys, modality_name, checksums, model_name, metadata_list=None):
    metadata_list = metadata_list or []
    if modality_name == "image":
        for embedding, s3_key, metadata, checksum in zip(embeddings, s3_keys, metadata_list, checksums):
            store_one_embedding_in_db(data_base, embedding=embedding, table_name=table_name, bucket_name=bucket_name, s3_key=s3_key, content=None, metadata=metadata, model_name=model_name, checksum=checksum)
    elif modality_name in ("text", "text file"):
        for chunked_embeddings, s3_key, metadata, checksum in zip(embeddings, s3_keys, metadata_list, checksums):
            sorted_chunks = sorted(chunked_embeddings, key=lambda chunk: chunk["start"])
            for chunk in sorted_chunks:
                store_one_embedding_in_db(data_base, embedding=chunk["embedding"], table_name=table_name, bucket_name=bucket_name, s3_key=s3_key, content=chunk.get("text"), start=chunk.get("start"), end=chunk.get("end"), md5=chunk.get("md5"), metadata=metadata, model_name=model_name, checksum=checksum)
    else:
        raise ValueError(f"Unsupported modality '{modality_name}' for embedding storage.")


def check_status_db(bucket_name, s3_key, checksum, data_base, table_name):
    db_pool = data_base.db_pool
    conn = db_pool.getconn()
    cur = conn.cursor()
    query = None
    try:
        s3_object = f"{bucket_name}/{s3_key}"
        query = f"SELECT checksum FROM {table_name} WHERE s3_object = %s;"
        cur.execute(query, (s3_object,))
        results = cur.fetchall()
        if not results:
            return {"status": "not_found", "message": f"No entry found in table '{table_name}' for key: {s3_object}"}
        db_checksums = [row[0] for row in results]
        if checksum in db_checksums:
            return {"status": "match", "message": f"Checksum match for '{s3_object}'."}
        else:
            db_first = db_checksums[0] if db_checksums else None
            return {"status": "mismatch", "message": f"Checksum mismatch for '{s3_object}': DB has '{db_first}', expected '{checksum}'"}
    except Exception as e:
        return {"status": "error", "message": f"Exception occurred during DB check for query {query}: {e}"}
    finally:
        try:
            cur.close()
        except Exception:
            pass
        db_pool.putconn(conn)


def check_S3_object_status_in_db(path_to_bucket, s3_objects, checksums, db_model_store):
    username, data_base, table_name, model, strore_embeddings_flag, modality_name, s3_connection = query_db_model(db_model_store, path_to_bucket)
    status = []
    for i, s3_object in enumerate(s3_objects):
        _, s3_key = s3_object.replace("s3://", "").split("/", 1)
        status.append(check_status_db(path_to_bucket, s3_key, checksums[i], data_base, table_name))
    return status
