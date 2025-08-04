#!/usr/bin/python3

# =============================
# FastAPI Imports, App, Models
# =============================
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

import argparse
import json
import os
import sys
import subprocess
import tempfile
import psycopg2
from umrs.sql import *
from umrs.s3 import *
from umrs.models import load_model
from umrs.perf_test import run_perf_test
from umrs.MessageQueue import MessageQueue
from umrs.pretty_print import pretty_print_cfg
from umrs.modality import *
from umrs.json_utils import load_server_cfg_file
from collections import defaultdict
import signal
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
        # Use the new umrs functions for S3 connection
        s3_connection, bucket_location = connect_to_s3(f"/root/.s3cfg")  # Use default config or user config as needed
        # Sanitize xp_name for S3 bucket naming: replace spaces with hyphens and lowercase
        safe_xp_name = payload.xp_name.replace(' ', '-').lower()
        bucket_name = f"{payload.username}.{safe_xp_name}.{payload.bucket_alias}"
        try:
            s3_connection.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': bucket_location
                }
            )
        except Exception as e:
            if "BucketAlreadyOwnedByYou" not in str(e):
                return JSONResponse({"status": "error", "message": f"MinIO bucket creation failed: {str(e)}"}, status_code=500)
        # DB: create entry for bucket (simulate, real logic may differ)
        # Here you would update db_model_store or DB as needed
        # For now, just return success
        return JSONResponse({"status": "ok", "message": f"Bucket '{bucket_name}' created for user '{payload.username}'."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Bucket creation failed: {str(e)}"}, status_code=500)

def create_minio_user_mc(username, secret):
    """Create a MinIO user and attach a policy using the new umrs minio functions."""
    from umrs.minio import minio_s3_create_user, minio_grant_user_access

    # Use the admin S3 configuration
    admin_s3_config = os.environ.get("ADMIN_S3_CONFIG", "/root/.s3cfg")
    user_s3_config = f"/tmp/{username}.s3cfg"

    # Create user using the new minio functions
    alias = minio_s3_create_user(username, None, user_s3_config, admin_s3_config)
    return alias

def create_postgres_user(username, password):
    """Create a PostgreSQL user if not exists using the new umrs sql functions."""
    create_db_user(username)

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

# =============================
# Main Branch Functions (preserved)
# =============================

def store_db_model(db_model_store, path_to_bucket, bucket_alias, s3_username, db_pool, db_name, table_name, sql_username, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name):
    """Stores data base and model for a given (user, bucket_alias)."""
    # if user not in db_model_store:
    #     storage[user] = {}  # Initialize user entry

    db_model_store[path_to_bucket] = {"bucket_alias": bucket_alias,
                                      "s3_username": s3_username,
                                      "sql_username": sql_username,
                                      "data_base": db_pool,
                                      "db_name": db_name,
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
        return entry["s3_username"], entry["data_base"], entry["table_name"], entry["model"], entry["store_embeddings_flag"], entry["modality_name"], entry["s3_connection"]


def create_s3_sql_users(user_cfg, admin_s3_configuration_file):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nCreating {fullname} ({username})\n######################")

    bucket_list = []
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        sql_username  = xp["sql_username"]
        create_db_user(sql_username)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            # print(f"{bucket_alias} -> {path_to_bucket}")
            bucket_list.append(path_to_bucket)

    user_s3_configuration_file = user_cfg["s3_configuration_file"]
    s3_connection, _ = connect_to_s3(admin_s3_configuration_file)
    s3_create_users_and_buckets(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file, bucket_list)
    s3_connection.close()

def configure_user(user_cfg, db_model_store):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nConfiguring {fullname} ({username})\n######################")

    user_s3_configuration_file = user_cfg["s3_configuration_file"]

    s3_connection, bucket_location = connect_to_s3(user_s3_configuration_file)
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username  = xp["sql_username"]
        data_base, db_name = create_db(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            print(f"\t\tModality: {modality_name}")
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = configure_bucket(s3_connection, bucket_alias, xp_name, username, bucket_location)
            model = load_model(modality['model'], modality_name)
            table_name, store_embeddings_flag, metadata = configure_table(data_base, modality["sql"], modality_name, model)
            store_db_model(db_model_store, path_to_bucket, bucket_alias, username, data_base, db_name, table_name, sql_username, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name)
            print(f"\t\t\t==>Bucket alias: {bucket_alias}")

def load_json_conf_file(configure_file):
# Load JSON content from the file
    users = []
    with open(configure_file, "r", encoding="utf-8") as file:
        content = json.load(file)
        admin_s3_configuration_file = content["admin_s3_configuration_file"]
        for user_conf_file in content["users"]:
            with open(user_conf_file, "r", encoding="utf-8") as file:
                user_content = json.load(file)
                users.append(user_content)

    return admin_s3_configuration_file, users

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


parser = argparse.ArgumentParser(description="UMRS server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue 'zeromq' or 'posix_ipc', or 'sysv_ipc'")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument('--web', action='store_true', help='Run FastAPI web server instead of message queue loop')
parser.add_argument('--port', type=int, default=8000, help='Port for web server (default: 8000)')

args = parser.parse_args()

configure_file = args.conf

# Check if the file exists
if not os.path.exists(configure_file):
    sys.exit(f"Configuration file '{configure_file}' not found.")

admin_s3_configuration_file, users = load_server_cfg_file(configure_file)
# Print loaded content (optional)
print("Configuration loaded successfully.")

# If --web is specified, run FastAPI server
if args.web:
    print(f"Starting FastAPI web server on port {args.port}...")
    print("API endpoints available:")
    print("  GET  /health - Health check")
    print("  GET  /status - Server status")
    print("  GET  /api/user/list_buckets?username=<user> - List user buckets")
    print("  GET  /api/user/info?username=<user> - User information")
    print("  POST /api/user/create_bucket - Create a bucket")
    print("  POST /api/admin/create_user - Create a user (admin only)")

    # Update server status
    server_status["running"] = True

    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    sys.exit(0)

# Otherwise run the traditional message queue server

# create the message queue:
mq = MessageQueue(queue_type = args.queue_type);
response_mq = MessageQueue(queue_type = args.queue_type, role='client', queue_name="/tmp/response")
print (f"Subscribed to message queue '{args.queue_type}'")


# create S3/SQL  users if necessary
for user in users:
    create_s3_sql_users(user, admin_s3_configuration_file)


#configure server for each users
db_model_store = defaultdict(dict)
for user in users:
    configure_user(user, db_model_store)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"-->  Models will be run on device: {device}")

print(f"\n######################\nServer configuration done!\n######################\n")


pretty_print_cfg(db_model_store)

# run_perf_test()

try:
    # loop for new new object to be created
    is_handshake = False
    while True:
        if not is_handshake: #Only print if we did not have handshake in the last round
            print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")

        s3_objects, is_handshake = mq.wait_for_s3_events()
        if is_handshake: # The client ask me if I am ready.
            print("Received handshake request. Answering 'READY!'")
            response_mq.handshake("READY!")
            continue;

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
