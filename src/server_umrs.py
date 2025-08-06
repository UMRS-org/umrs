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

    # Use admin's S3 configuration to list all buckets, then filter by user
    admin_s3_config = f"/root/.s3cfg"
    try:
        s3_connection, _ = connect_to_s3(admin_s3_config)
        # List all buckets accessible to admin
        response = s3_connection.list_buckets()
        buckets = []
        for bucket in response.get('Buckets', []):
            bucket_name = bucket['Name']
            # Only include buckets that belong to this user (start with username.)
            if bucket_name.startswith(f"{username}."):
                buckets.append(bucket_name)
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
        # Step 1: Create bucket using admin credentials (only admin can create buckets)
        admin_s3_config = f"/root/.s3cfg"
        try:
            admin_s3_connection, bucket_location = connect_to_s3(admin_s3_config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect with admin S3 config: {str(e)}")

        # Sanitize xp_name for S3 bucket naming: replace spaces with hyphens and lowercase
        safe_xp_name = payload.xp_name.replace(' ', '-').lower()
        bucket_name = f"{payload.username}.{safe_xp_name}.{payload.bucket_alias}"

        # Step 2: Create the bucket using admin credentials
        try:
            admin_s3_connection.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': bucket_location
                }
            )
        except Exception as e:
            if "BucketAlreadyOwnedByYou" not in str(e):
                return JSONResponse({"status": "error", "message": f"MinIO bucket creation failed: {str(e)}"}, status_code=500)

        # Step 3: Grant user access to the bucket using MinIO admin tools
        from umrs.minio import minio_grant_user_access
        user_s3_config = f"/tmp/{payload.username}.s3cfg"
        try:
            # Read user config to get the alias
            import configparser
            config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
            config.read(user_s3_config)
            alias = config.get("default", "alias", fallback="myminio")

            # Grant user access to the bucket
            minio_grant_user_access(payload.username, bucket_name, alias)
        except Exception as e:
            # If granting access fails, log but don't fail the bucket creation
            print(f"Warning: Failed to grant user access to bucket: {str(e)}")

        # DB: create entry for bucket (simulate, real logic may differ)
        # Here you would update db_model_store or DB as needed
        # For now, just return success
        return JSONResponse({"status": "ok", "message": f"Bucket '{bucket_name}' created for user '{payload.username}'."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Bucket creation failed: {str(e)}"}, status_code=500)

# Delete bucket endpoint
@app.delete("/api/user/delete_bucket")
async def delete_bucket_api(request: Request, payload: dict):
    """Delete a bucket and all its contents, including PostgreSQL records."""
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)

    # Extract parameters from payload
    username = payload.get("username")
    bucket_name = payload.get("bucket_name")
    confirmation = payload.get("confirmation", "").lower()

    if not username or not bucket_name:
        raise HTTPException(status_code=400, detail="Missing username or bucket_name")

    # Only allow user to delete their own buckets
    if user != username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not match username")

    # Verify bucket belongs to user (must start with username.)
    if not bucket_name.startswith(f"{username}."):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bucket does not belong to this user")

    # Require confirmation
    if confirmation != "delete":
        raise HTTPException(status_code=400, detail="Confirmation required: set 'confirmation' to 'delete' to proceed")

    try:
        # Step 1: Connect using admin credentials (needed for bucket deletion)
        admin_s3_config = f"/root/.s3cfg"
        try:
            admin_s3_connection, _ = connect_to_s3(admin_s3_config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect with admin S3 config: {str(e)}")

        # Step 2: First, check if bucket exists and belongs to user
        try:
            response = admin_s3_connection.list_buckets()
            bucket_exists = False
            for bucket in response.get('Buckets', []):
                if bucket['Name'] == bucket_name:
                    bucket_exists = True
                    break

            if not bucket_exists:
                return JSONResponse({"status": "error", "message": f"Bucket '{bucket_name}' not found"}, status_code=404)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to check bucket existence: {str(e)}")

        # Step 3: Delete all objects in the bucket first
        try:
            # List all objects in the bucket
            objects_response = admin_s3_connection.list_objects_v2(Bucket=bucket_name)

            if objects_response.get('Contents'):
                # Delete all objects
                objects_to_delete = [{'Key': obj['Key']} for obj in objects_response['Contents']]
                admin_s3_connection.delete_objects(
                    Bucket=bucket_name,
                    Delete={'Objects': objects_to_delete}
                )
                print(f"Deleted {len(objects_to_delete)} objects from bucket '{bucket_name}'")
        except Exception as e:
            # If no objects exist, that's fine
            print(f"No objects to delete in bucket '{bucket_name}' or error listing: {str(e)}")

        # Step 4: Delete the bucket itself
        try:
            admin_s3_connection.delete_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' deleted successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete bucket: {str(e)}")

        # Step 5: Clean up PostgreSQL records
        try:
            from umrs.sql import connect_to_db, get_table_name_from_bucket_name
            db_conn = connect_to_db()

            # Get table name from bucket name
            table_name = get_table_name_from_bucket_name(bucket_name)

            # Drop the table if it exists
            with db_conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                db_conn.commit()
                print(f"PostgreSQL table '{table_name}' dropped successfully")

        except Exception as e:
            # Log the error but don't fail the entire operation
            print(f"Warning: Failed to clean up PostgreSQL records: {str(e)}")

        # Step 6: Remove user policies for this bucket (MinIO cleanup)
        try:
            from umrs.minio import minio_remove_user_bucket_access
            user_s3_config = f"/tmp/{username}.s3cfg"

            # Read user config to get the alias
            import configparser
            config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
            config.read(user_s3_config)
            alias = config.get("default", "alias", fallback="myminio")

            # Remove user access policies for this bucket
            minio_remove_user_bucket_access(username, bucket_name, alias)
        except Exception as e:
            print(f"Warning: Failed to remove user bucket access policies: {str(e)}")

        return JSONResponse({"status": "ok", "message": f"Bucket '{bucket_name}' and all associated data deleted successfully."})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Bucket deletion failed: {str(e)}"}, status_code=500)

# =============================
# Hybrid Upload Endpoints (API -> MessageQueue)
# =============================

# Global message queue instance for hybrid mode
hybrid_mq = None

def init_hybrid_message_queue(queue_type='zeromq'):
    """Initialize message queue for hybrid mode."""
    global hybrid_mq
    if hybrid_mq is None:
        hybrid_mq = MessageQueue(queue_type=queue_type, role='client')
    return hybrid_mq

@app.post("/api/user/upload_file")
async def upload_file_hybrid(request: Request):
    """Hybrid upload endpoint: receives file via HTTP API and forwards to message queue."""
    from fastapi import File, UploadFile, Form

    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ", 1)[1]

    # Parse multipart form data
    form = await request.form()
    username = form.get("username")
    bucket_alias = form.get("bucket_alias")
    xp_name = form.get("xp_name")
    modality = form.get("modality")
    file = form.get("file")

    if not all([username, bucket_alias, xp_name, modality, file]):
        raise HTTPException(status_code=400, detail="Missing required fields: username, bucket_alias, xp_name, modality, file")

    user = get_user_from_token(token)
    if user != username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not match username")

    try:
        # Save uploaded file to temporary location with proper extension
        original_filename = file.filename if hasattr(file, 'filename') and file.filename else "uploaded_file"
        file_extension = os.path.splitext(original_filename)[1] or ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Create S3 bucket path
        safe_xp_name = xp_name.replace(' ', '-').lower()
        bucket_name = f"{username}.{safe_xp_name}.{bucket_alias}"

        # Connect to S3 using user's configuration
        user_s3_config = f"/tmp/{username}.s3cfg"
        try:
            s3_connection, _ = connect_to_s3(user_s3_config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Store file to S3 and notify message queue
        if modality == "image":
            from umrs.s3 import s3_store_image
            mq = init_hybrid_message_queue()
            s3_key = s3_store_image(s3_connection, bucket_name, temp_path, mq, {})
        elif modality == "text file":
            from umrs.s3 import s3_store_text_file
            mq = init_hybrid_message_queue()
            s3_key = s3_store_text_file(s3_connection, bucket_name, temp_path, mq, {})
        else:
            os.unlink(temp_path)  # Clean up temp file
            raise HTTPException(status_code=400, detail=f"Unsupported modality: {modality}")

        # Clean up temp file
        os.unlink(temp_path)

        if s3_key:
            return JSONResponse({
                "status": "ok",
                "message": f"File uploaded successfully to bucket '{bucket_name}'",
                "s3_key": s3_key,
                "bucket": bucket_name
            })
        else:
            raise HTTPException(status_code=500, detail="File upload failed")

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return JSONResponse({"status": "error", "message": f"Upload failed: {str(e)}"}, status_code=500)

@app.post("/api/user/upload_batch")
async def upload_batch_hybrid(request: Request):
    """Hybrid batch upload endpoint: receives multiple files via HTTP API and forwards to message queue."""
    from fastapi import File, UploadFile

    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ", 1)[1]

    # Parse multipart form data
    form = await request.form()
    username = form.get("username")
    bucket_alias = form.get("bucket_alias")
    xp_name = form.get("xp_name")
    modality = form.get("modality")

    # Get all files from form
    files = []
    for key, value in form.items():
        if key.startswith("file"):
            files.append(value)

    if not all([username, bucket_alias, xp_name, modality]) or not files:
        raise HTTPException(status_code=400, detail="Missing required fields or no files provided")

    user = get_user_from_token(token)
    if user != username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not match username")

    uploaded_files = []
    temp_paths = []

    try:
        # Create S3 bucket path
        safe_xp_name = xp_name.replace(' ', '-').lower()
        bucket_name = f"{username}.{safe_xp_name}.{bucket_alias}"

        # Connect to S3 using user's configuration
        user_s3_config = f"/tmp/{username}.s3cfg"
        s3_connection, _ = connect_to_s3(user_s3_config)

        # Initialize message queue
        mq = init_hybrid_message_queue()

        # Process each file
        for file in files:
            # Save uploaded file to temporary location with proper extension
            original_filename = file.filename if hasattr(file, 'filename') and file.filename else "uploaded_file"
            file_extension = os.path.splitext(original_filename)[1] or ""

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name
                temp_paths.append(temp_path)

            # Store file to S3 and notify message queue
            s3_key = None
            if modality == "image":
                from umrs.s3 import s3_store_image
                s3_key = s3_store_image(s3_connection, bucket_name, temp_path, mq, {})
            elif modality == "text file":
                from umrs.s3 import s3_store_text_file
                s3_key = s3_store_text_file(s3_connection, bucket_name, temp_path, mq, {})

            if s3_key:
                uploaded_files.append({
                    "filename": file.filename,
                    "s3_key": s3_key,
                    "status": "success"
                })
            else:
                uploaded_files.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"Unsupported modality or upload error: {modality}"
                })

        # Clean up temp files
        for temp_path in temp_paths:
            try:
                os.unlink(temp_path)
            except:
                pass

        return JSONResponse({
            "status": "ok",
            "message": f"Batch upload completed. {len([f for f in uploaded_files if f['status'] == 'success'])} files uploaded successfully.",
            "bucket": bucket_name,
            "files": uploaded_files
        })

    except Exception as e:
        # Clean up temp files
        for temp_path in temp_paths:
            try:
                os.unlink(temp_path)
            except:
                pass
        return JSONResponse({"status": "error", "message": f"Batch upload failed: {str(e)}"}, status_code=500)

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


parser = argparse.ArgumentParser(description="UMRS server - Hybrid API + MessageQueue mode", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue 'zeromq' or 'posix_ipc', or 'sysv_ipc'")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument('--port', type=int, default=8000, help='Port for web server (default: 8000)')

args = parser.parse_args()

configure_file = args.conf

# Check if the file exists
if not os.path.exists(configure_file):
    sys.exit(f"Configuration file '{configure_file}' not found.")

admin_s3_configuration_file, users = load_server_cfg_file(configure_file)
# Print loaded content (optional)
print("Configuration loaded successfully.")

# Initialize MinIO alias for admin operations
print("Initializing MinIO client alias...")
try:
    from umrs.minio import minio_check_create_alias
    import configparser

    # Read admin S3 config to get connection details
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(admin_s3_configuration_file)
    s3_host = config.get("default", "host_base")
    use_https = config.getboolean("default", "use_https", fallback=False)
    access_key = config.get("default", "access_key")
    secret_key = config.get("default", "secret_key")
    alias = config.get("default", "alias", fallback="myminio")

    # Create endpoint URL
    protocol = "https" if use_https else "http"
    endpoint_url = f"{protocol}://{s3_host}"

    # Set up the MinIO alias
    minio_check_create_alias(alias, endpoint_url, access_key, secret_key, insecure=True)
    print(f"MinIO alias '{alias}' initialized successfully.")
except Exception as e:
    print(f"Warning: Failed to initialize MinIO alias: {e}")
    print("MinIO user operations may fail.")

# Always run in hybrid mode (API + MessageQueue)
print(f"Starting UMRS server in hybrid mode on port {args.port}...")
print("API endpoints available:")
print("  GET  /health - Health check")
print("  GET  /status - Server status")
print("  GET  /api/user/list_buckets?username=<user> - List user buckets")
print("  GET  /api/user/info?username=<user> - User information")
print("  POST /api/user/create_bucket - Create a bucket")
print("  POST /api/admin/create_user - Create a user (admin only)")
print("  POST /api/user/upload_file - Upload single file")
print("  POST /api/user/upload_batch - Upload multiple files")
print(f"Message queue backend: {args.queue_type}")

# Initialize hybrid message queue backend
init_hybrid_message_queue(args.queue_type)

# Set up background message queue processing
import threading

def run_message_queue_processor():
    """Background thread to process message queue events in hybrid mode."""
    try:
        # Initialize message queue for processing
        mq = MessageQueue(queue_type=args.queue_type, role='server')
        response_mq = MessageQueue(queue_type=args.queue_type, role='client', queue_name="/tmp/response")

        # Initialize database and model store for hybrid mode
        db_model_store = defaultdict(dict)

        # Create S3/SQL users if necessary
        for user in users:
            create_s3_sql_users(user, admin_s3_configuration_file)

        # Configure server for each user
        for user in users:
            configure_user(user, db_model_store)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Hybrid] Message queue processor running on device: {device}")

        # Process message queue events
        is_handshake = False
        while True:
            if not is_handshake:
                print("[Hybrid] Waiting for message queue events...")

            s3_objects, is_handshake = mq.wait_for_s3_events()
            if is_handshake:
                print("[Hybrid] Received handshake request. Answering 'READY!'")
                response_mq.handshake("READY!")
                continue

            if s3_objects:
                keys_per_bucket = defaultdict(list)
                print(f"[Hybrid] Processing {len(s3_objects)} new file(s) from API uploads:")

                for s3_object in s3_objects:
                    print(f"[Hybrid] --> {s3_object}")
                    path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)

                    if s3_key not in keys_per_bucket[path_to_bucket]:
                        keys_per_bucket[path_to_bucket].append(s3_key)

                for path_to_bucket in keys_per_bucket.keys():
                    username, data_base, table_name, model, store_embeddings_flag, modality_name, s3_connection = query_db_model(db_model_store, path_to_bucket)
                    s3_keys = keys_per_bucket[path_to_bucket]
                    if store_embeddings_flag:
                        print(f"[Hybrid] --> Storing embeddings to database in {table_name}")
                        compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection)
                    else:
                        print(f"[Hybrid] No embeddings computed for bucket \"{path_to_bucket}\" (modality = \"{modality_name}\")")

    except KeyboardInterrupt:
        print("[Hybrid] Message queue processor interrupted.")
    except Exception as e:
        print(f"[Hybrid] Message queue processor error: {e}")

# Start background message queue processor
mq_thread = threading.Thread(target=run_message_queue_processor, daemon=True)
mq_thread.start()
print("[Hybrid] Background message queue processor started.")

# Update server status
server_status["running"] = True

# Run the FastAPI server
uvicorn.run(app, host="0.0.0.0", port=args.port)
