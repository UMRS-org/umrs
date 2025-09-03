#!/usr/bin/env python3
"""
ZeroMQ RPC implementation for UMRS server.

This module exposes a simple ZeroMQ REP/REQ JSON RPC interface that
mirrors the previous FastAPI endpoints. Clients send JSON objects with
an "action" key and parameters; the server performs the action and
returns a JSON response.
"""

import os
import sys
import json
import time
import tempfile
import base64
import traceback
from collections import defaultdict
from typing import Optional

try:
    import zmq
except Exception:
    zmq = None

from .sql import *
from .s3 import *
from .models import load_model
from .modality import *
from .db_store import *
from .utils import zero_out_embeddings_for_empty_objects
from .json_utils import load_server_cfg_file
from .pretty_print import pretty_print_cfg, pretty_print_recompute, pretty_print_check
from .MessageQueue import MessageQueue


def get_server_functions():
    """Dynamically import server functions to avoid circular imports."""
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
    """ZeroMQ-based RPC server offering UMRS functionality."""

    def __init__(self, queue_type='zeromq'):
        if zmq is None:
            raise ImportError("zmq package is required for ZeroMQ API server")

        self.db_model_store = defaultdict(dict)
        self.admin_s3_configuration_file = None
        self.users = []
        self.device = "cpu"
        self.initialized = False
        self.queue_type = queue_type
        # MessageQueue client used to notify background workers about S3 objects
        self.mq = MessageQueue(queue_type=self.queue_type, role='client')

    def _find_bucket_by_alias_and_user(self, bucket_alias: str, username: str):
        for path_to_bucket, entry in self.db_model_store.items():
            if (entry.get("bucket_alias") == bucket_alias and entry.get("s3_username") == username):
                return path_to_bucket, entry
        return None

    def initialize_server(self, configure_file: str):
        try:
            self.admin_s3_configuration_file, self.users = load_server_cfg_file(configure_file)
            server_umrs = get_server_functions()
            if not server_umrs:
                raise Exception("Could not import server functions")

            for user in self.users:
                server_umrs.create_s3_sql_users(user, self.admin_s3_configuration_file)

            for user in self.users:
                server_umrs.configure_user(user, self.db_model_store, self.admin_s3_configuration_file)

            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.initialized = True
            print(f"UMRS ZeroMQ API Server initialized on device: {self.device}")
        except Exception as e:
            print(f"Failed to initialize UMRS API Server: {e}")
            raise
    def _handle_upload_text(self, params):
        text = params.get('text')
        bucket_alias = params.get('bucket_alias')
        username = params.get('username')
        metadata = params.get('metadata')

        if not self.initialized:
            return {"error": "Server not initialized"}
        if not isinstance(text, str):
            return {"error": "The 'text' field must be a string"}

        bucket_info = self._find_bucket_by_alias_and_user(bucket_alias, username)
        if not bucket_info:
            return {"error": f"Bucket '{bucket_alias}' not found for user '{username}'"}
        path_to_bucket, entry = bucket_info
        if entry['modality_name'] != 'text':
            return {"error": f"Bucket '{bucket_alias}' is not configured for text modality"}

        metadata_args = {"Metadata": metadata or {}}
        try:
            s3_key = s3_store_txt(entry['s3_connection'], path_to_bucket, text, None, metadata_args)
            if entry.get('store_embeddings_flag', False):
                self.mq.notify_s3(f"s3://{path_to_bucket}/{s3_key}")
            return {"success": True, "s3_key": s3_key, "bucket": path_to_bucket}
        except Exception as e:
            return {"error": str(e)}

    def _handle_upload_file(self, params, expected_modality):
        filename = params.get('filename')
        file_b64 = params.get('file_bytes')
        bucket_alias = params.get('bucket_alias')
        username = params.get('username')
        metadata = params.get('metadata')

        if not self.initialized:
            return {"error": "Server not initialized"}
        if not filename or not file_b64:
            return {"error": "Missing filename or file_bytes"}

        bucket_info = self._find_bucket_by_alias_and_user(bucket_alias, username)
        if not bucket_info:
            return {"error": f"Bucket '{bucket_alias}' not found for user '{username}'"}
        path_to_bucket, entry = bucket_info
        if entry['modality_name'] != expected_modality:
            return {"error": f"Bucket '{bucket_alias}' is not configured for {expected_modality} modality"}

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}")
        try:
            data = base64.b64decode(file_b64)
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
            metadata_args = {"Metadata": metadata or {}}
            if expected_modality == 'image':
                s3_key = s3_store_image(entry['s3_connection'], path_to_bucket, tmp_path, None, metadata_args)
            else:
                s3_key = s3_store_text_file(entry['s3_connection'], path_to_bucket, tmp_path, None, metadata_args)

            if entry.get('store_embeddings_flag', False):
                self.mq.notify_s3(f"s3://{path_to_bucket}/{s3_key}")

            return {"success": True, "s3_key": s3_key, "bucket": path_to_bucket, "filename": filename}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}
        finally:
            try:
                tmp.close()
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
            except Exception:
                pass

    def _handle_list_buckets(self, params):
        username = params.get('username')
        buckets = []
        for path_to_bucket, entry in self.db_model_store.items():
            if username is None or entry.get('s3_username') == username:
                buckets.append({
                    "bucket_alias": entry.get("bucket_alias", ""),
                    "path_to_bucket": path_to_bucket,
                    "modality": entry.get("modality_name", ""),
                    "username": entry.get("s3_username", ""),
                    "experience": entry.get("xp_name", ""),
                    "store_embeddings": entry.get("store_embeddings_flag", False)
                })
        return {"buckets": buckets}

    def _handle_admin_actions(self, action, params):
        server_umrs = get_server_functions()
        if server_umrs is None:
            return {"error": "Server administrative functions are unavailable (could not import server_umrs)."}

        try:
            if action == 'add_user_hot_reload':
                server_umrs.create_s3_sql_users(params, self.admin_s3_configuration_file)
                server_umrs.configure_user(params, self.db_model_store, self.admin_s3_configuration_file)
                self.users.append(params)
                return {"success": True}
            if action == 'create_user':
                server_umrs.create_s3_sql_users(params, self.admin_s3_configuration_file)
                server_umrs.configure_user(params, self.db_model_store, self.admin_s3_configuration_file)
                return {"success": True}
            if action == 'create_buckets':
                server_umrs.create_s3_sql_users(params, self.admin_s3_configuration_file)
                return {"success": True}
            if action == 'recompute':
                server_umrs.recompute_all_objects(self.db_model_store, self.device)
                return {"success": True}
            if action == 'update':
                server_umrs.update_objects(self.db_model_store, self.device)
                return {"success": True}
            if action == 'check':
                server_umrs.check_objects(self.db_model_store, verbose=False)
                return {"success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def run(self, host: str = '127.0.0.1', port: int = 8000, ssl_keyfile: Optional[str] = None, ssl_certfile: Optional[str] = None):
        """Run ZeroMQ REP server (blocks)."""
        if zmq is None:
            raise ImportError("zmq package is required to run ZeroMQ API server")

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        bind_addr = f"tcp://{host}:{port}"
        socket.bind(bind_addr)
        print(f"ZeroMQ API server listening on {bind_addr}")

        try:
            while True:
                msg = socket.recv()
                try:
                    request = json.loads(msg.decode('utf-8'))
                except Exception:
                    socket.send_json({"error": "Invalid JSON request"})
                    continue

                action = request.get('action')
                params = request.get('params', {})

                if action == 'health':
                    resp = {"status": "healthy", "timestamp": time.time(), "initialized": self.initialized}
                elif action == 'status':
                    if not self.initialized:
                        resp = {"error": "Server not initialized"}
                    else:
                        resp = {
                            "initialized": self.initialized,
                            "device": self.device,
                            "users_count": len(self.users),
                            "buckets_count": len(self.db_model_store)
                        }
                elif action == 'upload_text':
                    resp = self._handle_upload_text(params)
                elif action == 'upload_image':
                    resp = self._handle_upload_file(params, 'image')
                elif action == 'upload_textfile':
                    resp = self._handle_upload_file(params, 'text file')
                elif action == 'buckets':
                    resp = self._handle_list_buckets(params)
                elif action in ('add_user_hot_reload', 'create_user', 'create_buckets', 'recompute', 'update', 'check'):
                    resp = self._handle_admin_actions(action, params)
                else:
                    resp = {"error": f"Unknown action: {action}"}

                socket.send_json(resp)

        except KeyboardInterrupt:
            print("ZeroMQ API server interrupted, shutting down")
        finally:
            socket.close()
            context.term()


def create_api_server():
    return UMRSAPIServer()
