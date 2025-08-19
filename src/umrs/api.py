#!/usr/bin/env python3
"""
FastAPI implementation for UMRS server.
This module provides HTTP API endpoints for the UMRS service.
"""

import os
import json
import time
import tempfile
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from collections import defaultdict

# Import UMRS modules - these should be available from the current implementation

from .sql import *
from .s3 import *
from .models import load_model
from .modality import *
from .db_store import *
from .utils import zero_out_embeddings_for_empty_objects
from .json_utils import load_server_cfg_file
from .pretty_print import pretty_print_cfg, pretty_print_recompute, pretty_print_check
from .MessageQueue import MessageQueue


# We need to import the functions from server_umrs.py at runtime to avoid circular imports
def get_server_functions():
    """Dynamically import server functions to avoid circular imports."""
    # Add the parent directory to the path to import server_umrs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    try:
        import server_umrs
        return server_umrs
    except ImportError as e:
        print(f"Warning: Could not import server_umrs: {e}")
        return None


class UMRSAPIServer:
    """Main UMRS API Server class that wraps the existing functionality."""

    def __init__(self, queue_type='zeromq'):
        self.app = FastAPI(
            title="UMRS API Server",
            description="Universal Multi-modal Retrieval System API",
            version="1.0.0"
        )
        self.db_model_store = defaultdict(dict)
        self.admin_s3_configuration_file = None
        self.users = []
        self.device = "cpu"
        self.initialized = False
        # Initialize MessageQueue in client mode for API
        self.queue_type = queue_type
        self.mq = MessageQueue(queue_type=self.queue_type, role='client')

        self._setup_routes()

    def _setup_routes(self):
        """Setup all API routes."""

        from pydantic import BaseModel, ValidationError as PydanticValidationError
        from fastapi import Body
        import jsonschema
        import traceback
        import shutil
        from fastapi import status
        USERS_DIR = "/app/users"
        os.makedirs(USERS_DIR, exist_ok=True)

        class UserConfigModel(BaseModel):
            username: str
            fullname: str
            s3_configuration_file: str
            experiences: list

        @self.app.post("/admin/add_user_hot_reload", status_code=status.HTTP_201_CREATED)
        async def add_user_hot_reload(user_config: dict = Body(...)):
            """Add a user, store JSON, update config, and hot-reload user (no restart)."""
            from .json_utils import client_schema
            try:
                jsonschema.validate(instance=user_config, schema=client_schema)
            except jsonschema.ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Invalid user config: {e.message}")
            username = user_config.get("username")
            if not username:
                raise HTTPException(status_code=400, detail="Missing username in user config")
            user_json_path = os.path.join(USERS_DIR, f"{username}.json")
            # Store user JSON atomically
            try:
                tmp_path = user_json_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(user_config, f, indent=2)
                os.replace(tmp_path, user_json_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to store user JSON: {e}")
            # Atomically update conf_umrs_init.json
            conf_path = "/app/conf_umrs_init.json"
            try:
                with open(conf_path, "r", encoding="utf-8") as f:
                    conf = json.load(f)
                users_list = conf.get("users", [])
                if user_json_path not in users_list:
                    users_list.append(user_json_path)
                    conf["users"] = users_list
                    tmp_conf_path = conf_path + ".tmp"
                    with open(tmp_conf_path, "w", encoding="utf-8") as f:
                        json.dump(conf, f, indent=2)
                    os.replace(tmp_conf_path, conf_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to update conf_umrs_init.json: {e}")
            # Hot-reload: initialize user in-memory
            try:
                server_umrs = get_server_functions()
                if not server_umrs:
                    raise Exception("Could not import server functions")
                # Create S3/SQL users and buckets
                server_umrs.create_s3_sql_users(user_config, self.admin_s3_configuration_file)
                server_umrs.configure_user(user_config, self.db_model_store, self.admin_s3_configuration_file)
                self.users.append(user_config)
            except Exception as e:
                traceback_str = traceback.format_exc()
                raise HTTPException(status_code=500, detail=f"Failed to hot-reload user: {e}\n{traceback_str}")
            return {"success": True, "message": f"User '{username}' added, config updated, and hot-reloaded."}

        @self.app.post("/admin/create_user")
        async def create_user(user_config: dict = Body(...)):
            """Create and initialize a user (admin/server side)."""
            # Validate using client_schema
            from .json_utils import client_schema
            try:
                jsonschema.validate(instance=user_config, schema=client_schema)
            except jsonschema.ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Invalid user config: {e.message}")
            try:
                server_umrs = get_server_functions()
                if not server_umrs:
                    raise Exception("Could not import server functions")
                # Create S3/SQL users and buckets
                server_umrs.create_s3_sql_users(user_config, self.admin_s3_configuration_file)
                # Optionally configure user (models/tables)
                server_umrs.configure_user(user_config, self.db_model_store, self.admin_s3_configuration_file)
                return {"success": True, "message": f"User '{user_config['username']}' created and initialized."}
            except Exception as e:
                traceback_str = traceback.format_exc()
                raise HTTPException(status_code=500, detail=f"Failed to create user: {e}\n{traceback_str}")

        @self.app.post("/admin/create_buckets")
        async def create_buckets(user_config: dict = Body(...)):
            """Create all buckets for a user from a client JSON file (admin/server side)."""
            from .json_utils import client_schema
            try:
                jsonschema.validate(instance=user_config, schema=client_schema)
            except jsonschema.ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Invalid user config: {e.message}")
            try:
                server_umrs = get_server_functions()
                if not server_umrs:
                    raise Exception("Could not import server functions")
                # Only create S3/SQL users and buckets (no model/table config)
                server_umrs.create_s3_sql_users(user_config, self.admin_s3_configuration_file)
                return {"success": True, "message": f"Buckets for user '{user_config['username']}' created."}
            except Exception as e:
                traceback_str = traceback.format_exc()
                raise HTTPException(status_code=500, detail=f"Failed to create buckets: {e}\n{traceback_str}")

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for Docker healthcheck."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "initialized": self.initialized
            }

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "message": "UMRS API Server",
                "status": "running",
                "initialized": self.initialized
            }

        @self.app.get("/status")
        async def get_status():
            """Get server status and configuration."""
            if not self.initialized:
                raise HTTPException(status_code=503, detail="Server not initialized")

            status = {
                "initialized": self.initialized,
                "device": self.device,
                "users_count": len(self.users),
                "buckets_count": len(self.db_model_store),
                "buckets": {}
            }

            # Add bucket information
            for path_to_bucket, entry in self.db_model_store.items():
                status["buckets"][path_to_bucket] = {
                    "bucket_alias": entry.get("bucket_alias", ""),
                    "modality": entry.get("modality_name", ""),
                    "username": entry.get("s3_username", ""),
                    "model": str(entry.get("model", ""))
                }

            return status

        @self.app.post("/upload/text")
        async def upload_text(
            text: str = Form(...),
            bucket_alias: str = Form(...),
            username: str = Form(...),
            metadata: Optional[str] = Form(None)
        ):
            """Upload text data to specified bucket."""
            if not self.initialized:
                raise HTTPException(status_code=503, detail="Server not initialized")

            try:
                # Parse metadata if provided
                metadata_args = {"Metadata": {}}
                if metadata:
                    try:
                        metadata_dict = json.loads(metadata)
                        metadata_args = {"Metadata": metadata_dict}
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

                # Find the bucket configuration
                bucket_info = self._find_bucket_by_alias_and_user(bucket_alias, username)
                if not bucket_info:
                    raise HTTPException(status_code=404, detail=f"Bucket '{bucket_alias}' not found for user '{username}'")

                path_to_bucket, entry = bucket_info

                if entry["modality_name"] != "text":
                    raise HTTPException(status_code=400, detail=f"Bucket '{bucket_alias}' is not configured for text modality")

                # Store the text
                # Validate that text is a string
                if not isinstance(text, str):
                    raise HTTPException(status_code=400, detail="The 'text' field must be a string.")
                try:
                    s3_key = s3_store_txt(entry["s3_connection"], path_to_bucket, text, None, metadata_args)
                except Exception as s3_exc:
                    raise HTTPException(status_code=500, detail=f"Failed to store text in S3: {s3_exc}")


                # Enqueue for embedding computation if needed
                if entry.get("store_embeddings_flag", False):
                    try:
                        s3_object = f"s3://{path_to_bucket}/{s3_key}"
                        self.mq.notify_s3(s3_object)
                    except Exception as mq_exc:
                        raise HTTPException(status_code=500, detail=f"Failed to enqueue embedding computation: {mq_exc}")

                return {
                    "success": True,
                    "s3_key": s3_key,
                    "bucket": path_to_bucket,
                    "message": f"Text stored successfully"
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error storing text: {str(e)}")

        @self.app.post("/upload/image")
        async def upload_image(
            file: UploadFile = File(...),
            bucket_alias: str = Form(...),
            username: str = Form(...),
            metadata: Optional[str] = Form(None)
        ):
            """Upload image file to specified bucket."""
            if not self.initialized:
                raise HTTPException(status_code=503, detail="Server not initialized")

            try:
                # Parse metadata if provided
                metadata_args = {"Metadata": {}}
                if metadata:
                    try:
                        metadata_dict = json.loads(metadata)
                        metadata_args = {"Metadata": metadata_dict}
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

                # Find the bucket configuration
                bucket_info = self._find_bucket_by_alias_and_user(bucket_alias, username)
                if not bucket_info:
                    raise HTTPException(status_code=404, detail=f"Bucket '{bucket_alias}' not found for user '{username}'")

                path_to_bucket, entry = bucket_info

                if entry["modality_name"] != "image":
                    raise HTTPException(status_code=400, detail=f"Bucket '{bucket_alias}' is not configured for image modality")

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                try:
                    # Store the image
                    s3_key = s3_store_image(entry["s3_connection"], path_to_bucket, temp_file_path, None, metadata_args)

                    if not s3_key:
                        raise HTTPException(status_code=400, detail="Failed to store image")


                    # Enqueue for embedding computation if needed
                    if entry.get("store_embeddings_flag", False):
                        try:
                            s3_object = f"s3://{path_to_bucket}/{s3_key}"
                            self.mq.notify_s3(s3_object)
                        except Exception as mq_exc:
                            raise HTTPException(status_code=500, detail=f"Failed to enqueue embedding computation: {mq_exc}")

                    return {
                        "success": True,
                        "s3_key": s3_key,
                        "bucket": path_to_bucket,
                        "filename": file.filename,
                        "message": f"Image stored successfully"
                    }

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error storing image: {str(e)}")

        @self.app.post("/upload/textfile")
        async def upload_text_file(
            file: UploadFile = File(...),
            bucket_alias: str = Form(...),
            username: str = Form(...),
            metadata: Optional[str] = Form(None)
        ):
            """Upload text file to specified bucket."""
            if not self.initialized:
                raise HTTPException(status_code=503, detail="Server not initialized")

            try:
                # Parse metadata if provided
                metadata_args = {"Metadata": {}}
                if metadata:
                    try:
                        metadata_dict = json.loads(metadata)
                        metadata_args = {"Metadata": metadata_dict}
                    except json.JSONDecodeError:
                        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

                # Find the bucket configuration
                bucket_info = self._find_bucket_by_alias_and_user(bucket_alias, username)
                if not bucket_info:
                    raise HTTPException(status_code=404, detail=f"Bucket '{bucket_alias}' not found for user '{username}'")

                path_to_bucket, entry = bucket_info

                if entry["modality_name"] != "text file":
                    raise HTTPException(status_code=400, detail=f"Bucket '{bucket_alias}' is not configured for text file modality")

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                try:
                    # Store the text file
                    s3_key = s3_store_text_file(entry["s3_connection"], path_to_bucket, temp_file_path, None, metadata_args)

                    if not s3_key:
                        raise HTTPException(status_code=400, detail="Failed to store text file")


                    # Enqueue for embedding computation if needed
                    if entry.get("store_embeddings_flag", False):
                        try:
                            s3_object = f"s3://{path_to_bucket}/{s3_key}"
                            self.mq.notify_s3(s3_object)
                        except Exception as mq_exc:
                            raise HTTPException(status_code=500, detail=f"Failed to enqueue embedding computation: {mq_exc}")

                    return {
                        "success": True,
                        "s3_key": s3_key,
                        "bucket": path_to_bucket,
                        "filename": file.filename,
                        "message": f"Text file stored successfully"
                    }

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error storing text file: {str(e)}")

        @self.app.get("/buckets")
        async def list_buckets(username: Optional[str] = Query(None)):
            """List available buckets for a user or all buckets."""
            if not self.initialized:
                raise HTTPException(status_code=503, detail="Server not initialized")

            buckets = []
            for path_to_bucket, entry in self.db_model_store.items():
                if username is None or entry.get("s3_username") == username:
                    buckets.append({
                        "bucket_alias": entry.get("bucket_alias", ""),
                        "path_to_bucket": path_to_bucket,
                        "modality": entry.get("modality_name", ""),
                        "username": entry.get("s3_username", ""),
                        "experience": entry.get("xp_name", ""),
                        "store_embeddings": entry.get("store_embeddings_flag", False)
                    })

            return {"buckets": buckets}

        @self.app.post("/recompute")
        async def recompute_embeddings():
            """Recompute all embeddings."""
            if not self.initialized:
                raise HTTPException(status_code=503, detail="Server not initialized")

            try:
                server_umrs = get_server_functions()
                if server_umrs:
                    server_umrs.recompute_all_objects(self.db_model_store, self.device)
                else:
                    raise Exception("Could not import server functions")
                return {"success": True, "message": "Recomputation completed"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during recomputation: {str(e)}")

        @self.app.post("/update")
        async def update_embeddings():
            """Update missing/mismatched embeddings."""
            if not self.initialized:
                raise HTTPException(status_code=503, detail="Server not initialized")

            try:
                server_umrs = get_server_functions()
                if server_umrs:
                    server_umrs.update_objects(self.db_model_store, self.device)
                else:
                    raise Exception("Could not import server functions")
                return {"success": True, "message": "Update completed"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during update: {str(e)}")

        @self.app.get("/check")
        async def check_consistency():
            """Check consistency between S3 and database."""
            if not self.initialized:
                raise HTTPException(status_code=503, detail="Server not initialized")

            try:
                server_umrs = get_server_functions()
                if server_umrs:
                    server_umrs.check_objects(self.db_model_store, verbose=False)
                else:
                    raise Exception("Could not import server functions")
                return {"success": True, "message": "Consistency check completed"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during consistency check: {str(e)}")

    def _find_bucket_by_alias_and_user(self, bucket_alias: str, username: str):
        """Find bucket configuration by alias and username."""
        for path_to_bucket, entry in self.db_model_store.items():
            if (entry.get("bucket_alias") == bucket_alias and
                entry.get("s3_username") == username):
                return path_to_bucket, entry
        return None

    def _process_single_object(self, s3_object: str):
        """Process a single S3 object for embedding computation."""
        try:
            server_umrs = get_server_functions()
            if server_umrs:
                server_umrs.process_s3_objects([s3_object], self.db_model_store, self.device)
        except Exception as e:
            print(f"Warning: Failed to process object {s3_object}: {e}")

    def initialize_server(self, configure_file: str):
        """Initialize the server with configuration."""
        try:
            # Load configuration
            self.admin_s3_configuration_file, self.users = load_server_cfg_file(configure_file)

            # Get server functions
            server_umrs = get_server_functions()
            if not server_umrs:
                raise Exception("Could not import server functions")

            # Create S3/SQL users if necessary
            for user in self.users:
                server_umrs.create_s3_sql_users(user, self.admin_s3_configuration_file)

            # Configure server for each user
            for user in self.users:
                server_umrs.configure_user(user, self.db_model_store, self.admin_s3_configuration_file)

            # Set device
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.initialized = True
            print(f"UMRS API Server initialized successfully on device: {self.device}")
            print(f"Configured {len(self.users)} users with {len(self.db_model_store)} buckets")

        except Exception as e:
            print(f"Failed to initialize UMRS API Server: {e}")
            raise

    def run(self, host: str = "0.0.0.0", port: int = 8000, ssl_keyfile: str = None, ssl_certfile: str = None):
        """Run the FastAPI server."""
        if ssl_keyfile and ssl_certfile:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile
            )
        else:
            uvicorn.run(self.app, host=host, port=port)


def create_api_server():
    """Factory function to create UMRS API server."""
    return UMRSAPIServer()
