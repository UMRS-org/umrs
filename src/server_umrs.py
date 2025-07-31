
# =============================
# FastAPI Imports, App, Models
# =============================
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

import os
import sys
import json
import subprocess
import tempfile
import psycopg2
import argparse

from umrs.sql import *
from umrs.s3 import *
from umrs.models import load_model
from umrs.perf_test import run_perf_test
from  umrs.MessageQueue import MessageQueue
from umrs.pretty_print import pretty_print_cfg
from collections import defaultdict
import torch

import uvicorn


server_status = {"running": False, "last_event": None}
ADMIN_API_TOKEN = os.environ.get("UMRS_ADMIN_TOKEN", "test_admin_token")




app = FastAPI()

# -----------------------------
# Pydantic Models
# -----------------------------
class CreateBucketRequest(BaseModel):
    username: str
    bucket_alias: str
    xp_name: str
    modality: str
    model: str
    store_embeddings: bool
    metadata: Optional[List[str]] = []

class CreateUserRequest(BaseModel):
    username: str
    password: str
    s3_secret: str

# -----------------------------
# Utility/Auth Functions
# -----------------------------
def get_user_from_token(token: str):
    # In production, validate token and return user info
    return token  # For now, just return the token

# =============================
# FastAPI Endpoints
# =============================

# List buckets endpoint
@app.get("/api/user/list_buckets")
async def list_buckets(request: Request, username: str):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)
    if user != username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not match username")
    # Use mc (MinIO client) to list buckets for the user
    import subprocess
    import os
    import json
    minio_alias = os.environ.get("MINIO_ALIAS", "localminio")
    try:
        result = subprocess.run([
            "mc", "ls", f"{minio_alias}", "--json"
        ], capture_output=True, text=True, check=True)
        buckets = []
        for line in result.stdout.splitlines():
            try:
                entry = json.loads(line)
                if entry.get("type") == "folder" and entry.get("key", "").startswith(f"{username}."):
                    buckets.append(entry["key"])
            except Exception:
                continue
        return {"username": username, "buckets": buckets}
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to list buckets: {str(e)}"}, status_code=500)

# Show user info endpoint
@app.get("/api/user/info")
async def user_info(request: Request, username: str):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)
    # For now, just echo username and token; in production, fetch from DB
    return {"username": username, "token": token, "info": f"User info for {username} (stub)"}

# Create bucket endpoint
@app.post("/api/user/create_bucket")
async def create_bucket_api(request: Request, payload: CreateBucketRequest):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)
    # Only allow user to create buckets for themselves (or admin logic)
    if user != payload.username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not match username")
    try:
        # MinIO: create bucket if not exists
        s3_connection = connect_to_s3(f"/root/.s3cfg")  # Use default config or user config as needed
        # Sanitize xp_name for S3 bucket naming: replace spaces with hyphens and lowercase
        safe_xp_name = payload.xp_name.replace(' ', '-').lower()
        bucket_name = f"{payload.username}.{safe_xp_name}.{payload.bucket_alias}"
        try:
            s3_connection.create_bucket(Bucket=bucket_name)
        except Exception as e:
            if "BucketAlreadyOwnedByYou" not in str(e):
                return JSONResponse({"status": "error", "message": f"MinIO bucket creation failed: {str(e)}"}, status_code=500)
        # DB: create entry for bucket (simulate, real logic may differ)
        # Here you would update db_model_store or DB as needed
        # For now, just return success
        return JSONResponse({"status": "ok", "message": f"Bucket '{bucket_name}' created for user '{payload.username}'."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Bucket creation failed: {str(e)}"}, status_code=500)

"""
UMRS Server: FastAPI app and CLI for multi-user, multi-bucket, multi-modality data management.
"""
def create_minio_user_mc(username, secret):
    """Create a MinIO user and attach a policy."""
    minio_alias = os.environ.get("MINIO_ALIAS", "localminio")
    minio_url = os.environ.get("MINIO_URL", "http://minio:9000")
    minio_root_user = os.environ.get("MINIO_ROOT_USER", "minioadmin")
    minio_root_password = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
    subprocess.run(["mc", "alias", "set", minio_alias, minio_url, minio_root_user, minio_root_password], check=False)
    subprocess.run(["mc", "admin", "user", "add", minio_alias, username, secret], check=True)
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:ListAllMyBuckets"], "Resource": ["arn:aws:s3:::*"]},
            {"Effect": "Allow", "Action": ["s3:*"], "Resource": [f"arn:aws:s3:::{username}*", f"arn:aws:s3:::{username}*/*"]}
        ]
    }
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(policy, f)
        policy_path = f.name
    subprocess.run(["mc", "admin", "policy", "create", minio_alias, f"{username}-policy", policy_path], check=True)
    subprocess.run(["mc", "admin", "policy", "attach", minio_alias, f"{username}-policy", "--user", username], check=True)

def create_postgres_user(username, password):
    """Create a PostgreSQL user if not exists."""
    pg_host = os.environ.get("PGHOST", "postgres")
    pg_port = int(os.environ.get("PGPORT", "5432"))
    pg_admin = os.environ.get("PGADMINUSER", "postgres")
    pg_admin_pass = os.environ.get("PGADMINPASS", "postgres")
    conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_admin, password=pg_admin_pass, dbname="postgres")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (username,))
    if not cur.fetchone():
        cur.execute(f"CREATE USER {username} WITH PASSWORD %s", (password,))
    cur.close()
    conn.close()

@app.post("/api/admin/create_user")
async def create_user_api(request: Request, payload: CreateUserRequest):
    """Admin endpoint to create a new user in MinIO and PostgreSQL."""
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1]
    if token != ADMIN_API_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin token")
    try:
        create_minio_user_mc(payload.username, payload.s3_secret)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"MinIO user creation failed: {str(e)}"}, status_code=500)
    try:
        create_postgres_user(payload.username, payload.password)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"PostgreSQL user creation failed: {str(e)}"}, status_code=500)
    return JSONResponse({"status": "ok", "message": f"User '{payload.username}' created in MinIO and PostgreSQL."})

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/status")
def server_status_info():
    """Server status endpoint."""
    return server_status



# --- FastAPI web server ---

# server_status is already defined at the top, do not redefine


# The /health and /status routes are already defined at the top, do not redefine

def store_db_model(db_model_store, path_to_bucket, bucket_alias, username, db_pool, table_name, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name):
    """Stores data base and model for a given (user, bucket_alias)."""
    # if user not in db_model_store:
    #     storage[user] = {}  # Initialize user entry

    db_model_store[path_to_bucket] = {"bucket_alias": bucket_alias,
                                      "username": username,
                                      "data_base": db_pool,
                                      "table_name": table_name,
                                      "model": model,
                                      "store_embeddings_flag": store_embeddings_flag,
                                      "modality_name": modality_name,
                                      "s3_connection": s3_connection,
                                      "metadata": metadata,
                                      "xp_name": xp_name}


def query_db_model(db_model_store, path_to_bucket):
    """Retrieves database and model for a given (user, bucket_alias)."""
    entry = db_model_store.get(path_to_bucket, None)
    if not entry:
        return None, None, None, None, None, None
    else:
        return entry["username"], entry["data_base"], entry["table_name"], entry["model"], entry["store_embeddings_flag"], entry["modality_name"], entry["s3_connection"]


def configure_user(user_cfg, db_model_store):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nConfiguring {fullname} ({username})\n######################")

    user_s3_configuration_file = user_cfg["s3_configuration_file"]

    s3_connection = connect_to_s3(user_s3_configuration_file)
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username  = xp["sql_username"]
        data_base = create_db(sql_username, xp_name)
        if data_base is None:
            print(f"\t\t-->  Error: Failed to create database for user '{sql_username}' and experience '{xp_name}'. Skipping this experience.")
            continue
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            print(f"\t\tModality: {modality_name}")
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = configure_bucket(s3_connection, bucket_alias, xp_name, username)
            model = load_model(modality['model'], modality_name)
            table_name, store_embeddings_flag, metadata = configure_table(data_base, modality["sql"], modality_name, model)
            store_db_model(db_model_store, path_to_bucket, bucket_alias, username, data_base, table_name, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name)
            print(f"\t\t\t==>Bucket alias: {bucket_alias}")


def compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection):
    """
    Loads data from S3, computes embeddings **in batch**, and stores them in the database.

    Args:
        bucket_alias (str): The name of the S3 bucket.
        s3_keys (list): List of S3 object keys.
        db_pool: Database connection pool.
        table_name (str): Target table for storing embeddings.
        model (SentenceTransformer): Model used for embedding computation.
        device (str): "cpu" or "cuda".
        modality_name (str): "text" or "image".
        s3_connection: Boto3 S3 client.
    """
    objects, metadata = load_data_from_s3(s3_connection, path_to_bucket, s3_keys, modality_name)

    embeddings = model.compute_embeddings(objects)

    # Store **all** embeddings in batch
    store_embeddings_in_db(data_base, embeddings, table_name, path_to_bucket, s3_keys, modality_name, metadata_list=metadata)


def initialize_umrs_system(configure_file, queue_type='zeromq'):
    """
    Initialize the UMRS system: configure users, databases, and buckets.

    Args:
        configure_file (str): Path to the configuration file
        queue_type (str): Type of message queue (for compatibility)

    Returns:
        tuple: (db_model_store, device) if successful, (None, None) if failed
    """
    # Check if the file exists
    if not os.path.exists(configure_file):
        print(f"Configuration file '{configure_file}' not found.")
        return None, None

    # Load JSON content from the file
    with open(configure_file, "r", encoding="utf-8") as file:
        content = json.load(file)

    print("Configuration loaded successfully.")

    db_model_store = defaultdict(dict)
    for user in content["users"]:
        configure_user(user, db_model_store)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-->  Models will be run on device: {device}")

    print(f"\n######################\nServer configuration done!\n######################\n")
    pretty_print_cfg(db_model_store)

    return db_model_store, device


def run_server_loop(db_model_store, device, queue_type):
    """
    Run the main server loop that waits for S3 events and processes them.

    Args:
        db_model_store: Dictionary containing database and model configurations
        device (str): Device to run models on ("cpu" or "cuda")
        queue_type (str): Type of message queue to use
    """
    # Create the message queue
    mq = MessageQueue(queue_type=queue_type)
    print(f"Subscribed to message queue '{queue_type}'")

    try:
        # loop for new objects to be created
        while True:
            print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")

            s3_objects = mq.wait_for_s3_events()

            keys_per_bucket = defaultdict(list)  # Create a dictionary with lists
            print(f"\t-->Detected {len(s3_objects)} new file(s):")

            for s3_object in s3_objects:
                print(f"\t\t-->{s3_object}")
                path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)

                # Append s3_key to the list associated with the bucket
                if s3_key not in keys_per_bucket[path_to_bucket]:  # Ensure no duplicates
                    keys_per_bucket[path_to_bucket].append(s3_key)

            for path_to_bucket in keys_per_bucket.keys():
                username, data_base, table_name, model, strore_embeddings_flag, modality_name, s3_connection = query_db_model(db_model_store, path_to_bucket)
                s3_keys = keys_per_bucket[path_to_bucket]
                if strore_embeddings_flag:
                    print(f"\t--> Storing to data base in {table_name}")
                    compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection)
                else:
                    print(f"We do not compute embeddings for this bucket \"{path_to_bucket}\" (modality = \"{modality_name}\")")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
        mq.close()
        sys.exit(0)  # Ensures a clean exit


parser = argparse.ArgumentParser(description="UMRS server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue 'zeromq' or 'posix_ipc', or 'sysv_ipc'")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument('--init-only', action='store_true', help='Initialize system only (users, databases, buckets) without starting the server loop')
parser.add_argument('--web', action='store_true', help='Run FastAPI web server instead of message queue loop')


args = parser.parse_args()

# Initialize the system
db_model_store, device = initialize_umrs_system(args.conf, args.queue_type)

if db_model_store is None:
    sys.exit(1)

# If init-only mode, exit after initialization
if args.init_only:
    print("Initialization completed. Exiting (--init-only mode).")
    sys.exit(0)


# If --web is specified, run FastAPI server
if args.web:
    print("Starting FastAPI web server on port 8000 with background message queue loop...")
    from threading import Thread
    import time
    def fastapi_server_loop():
        server_status["running"] = True
        mq = MessageQueue(queue_type=args.queue_type)
        print(f"Subscribed to message queue '{args.queue_type}' (FastAPI background task)")
        try:
            while True:
                print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")
                s3_objects = mq.wait_for_s3_events()
                server_status["last_event"] = time.time()
                keys_per_bucket = defaultdict(list)
                print(f"\t-->Detected {len(s3_objects)} new file(s):")
                for s3_object in s3_objects:
                    print(f"\t\t-->{s3_object}")
                    path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)
                    if s3_key not in keys_per_bucket[path_to_bucket]:
                        keys_per_bucket[path_to_bucket].append(s3_key)
                for path_to_bucket in keys_per_bucket.keys():
                    username, data_base, table_name, model, strore_embeddings_flag, modality_name, s3_connection = query_db_model(db_model_store, path_to_bucket)
                    s3_keys = keys_per_bucket[path_to_bucket]
                    if strore_embeddings_flag:
                        print(f"\t--> Storing to data base in {table_name}")
                        compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection)
                    else:
                        print(f"We do not compute embeddings for this bucket '{path_to_bucket}' (modality = '{modality_name}')")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting gracefully...")
            mq.close()
            server_status["running"] = False
            sys.exit(0)
    thread = Thread(target=fastapi_server_loop, daemon=True)
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # Otherwise, run the full server loop
    run_server_loop(db_model_store, device, args.queue_type)
