#!/usr/bin/python3
"""
UMRS Client: Multi-user, multi-bucket, multi-modality data uploader and admin tool.
Supports both local message queue mode and remote API mode.
"""

import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
from umrs.MessageQueue import MessageQueue
from umrs.metadata import check_and_build_metadata
from umrs.modality import *
from umrs.json_utils import load_client_cfg_file 
from umrs.pretty_print import pretty_print_user_cfg
from glob import glob
import argparse
from collections import defaultdict

# API client imports
try:
    import requests
except ImportError:
    requests = None


def is_server_ready(mq, response_mq, timeout_ms=5000):
    """Check if the UMRS server is ready for local message queue communication."""
    print("Checking for server availability...")
    if timeout_ms < 1500:
        timeout_ms = 1500

    while timeout_ms > 100:
        mq.handshake("READY?")
        answer = response_mq.busy_wait(100) 
        if answer == "READY!":
            return True
    
    mq.handshake("READY?")
    answer = response_mq.busy_wait(timeout_ms - 1000) 
    return answer == "READY!"


def store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user, s3_configuration_file, metadata, sql_username, db_name):
    """Store bucket configuration in bucket store."""
    if bucket_store[bucket_alias]:
        sys.exit(f"Error: duplicate bucket_alias {bucket_alias}")
    bucket_store[bucket_alias] = {
        "path_to_bucket": path_to_bucket,
        "username": username,
        "xp_name": xp_name,
        "modality_name": modality_name,
        "s3_configuration_file": s3_configuration_file,
        "metadata": metadata,
        "sql_username": sql_username,
        "db_name": db_name
    }


def configure_bucket(user_cfg, bucket_store):
    """Configure buckets from user configuration."""
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    s3_configuration_file = user_cfg["s3_configuration_file"]
    for xp in user_cfg["experiences"]:
        sql_username = xp["sql_username"]
        xp_name = xp["xp_name"]
        db_name = create_db_name(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            metadata = metadata_dict(modality["sql"])
            store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user_cfg, s3_configuration_file, metadata, sql_username, db_name)


# =============================
# API Client Functions
# =============================

def handle_init_user(args):
    """Handle the init-user subcommand."""
    if not requests:
        print("Error: 'requests' module is required for API commands. Install with: pip install requests")
        sys.exit(1)
        
    admin_token = args.admin_token or os.environ.get("UMRS_ADMIN_TOKEN")
    if not admin_token:
        print("Error: Admin token must be provided via --admin-token or UMRS_ADMIN_TOKEN env var.")
        sys.exit(1)
    payload = {
        "username": args.username,
        "password": args.password,
        "s3_secret": args.s3_secret
    }
    headers = {"Authorization": f"Bearer {admin_token}"}
    url = args.server_url.rstrip("/") + "/api/admin/create_user"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"[UMRS admin] POST {url} -> {resp.status_code}")
        print(resp.json())
        if resp.status_code == 200:
            print(f"User '{args.username}' created successfully.")
        else:
            print(f"Error: {resp.json().get('message')}")
    except Exception as e:
        print(f"Request failed: {e}")
    sys.exit(0)


def handle_create_bucket(args):
    """Handle the create-bucket subcommand."""
    if not requests:
        print("Error: 'requests' module is required for API commands. Install with: pip install requests")
        sys.exit(1)
        
    token = args.token or os.environ.get("UMRS_USER_TOKEN")
    if not token:
        print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
        sys.exit(1)
    # Sanitize xp_name for S3 bucket naming: replace spaces with hyphens and lowercase
    safe_xp_name = args.xp_name.replace(' ', '-').lower()
    payload = {
        "username": args.username,
        "bucket_alias": args.bucket_alias,
        "xp_name": safe_xp_name,
        "modality": args.modality,
        "model": args.model,
        "store_embeddings": args.store_embeddings,
        "metadata": args.metadata or []
    }
    headers = {"Authorization": f"Bearer {token}"}
    url = args.server_url.rstrip("/") + "/api/user/create_bucket"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"[UMRS user] POST {url} -> {resp.status_code}")
        print(resp.json())
        if resp.status_code == 200:
            print(f"Bucket '{args.bucket_alias}' created successfully.")
        else:
            print(f"Error: {resp.json().get('message')}")
    except Exception as e:
        print(f"Request failed: {e}")
    sys.exit(0)


def handle_list_buckets(args):
    """Handle the list-buckets subcommand."""
    if not requests:
        print("Error: 'requests' module is required for API commands. Install with: pip install requests")
        sys.exit(1)
        
    token = args.token or os.environ.get("UMRS_USER_TOKEN")
    if not token:
        print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
        sys.exit(1)
    
    headers = {"Authorization": f"Bearer {token}"}
    url = args.server_url.rstrip("/") + f"/api/user/list_buckets?username={args.username}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"[UMRS user] GET {url} -> {resp.status_code}")
        print(resp.json())
    except Exception as e:
        print(f"Request failed: {e}")
    sys.exit(0)


def handle_show_user(args):
    """Handle the show-user subcommand."""
    if not requests:
        print("Error: 'requests' module is required for API commands. Install with: pip install requests")
        sys.exit(1)
        
    token = args.token or os.environ.get("UMRS_USER_TOKEN")
    if not token:
        print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
        sys.exit(1)
    
    headers = {"Authorization": f"Bearer {token}"}
    url = args.server_url.rstrip("/") + f"/api/user/info?username={args.username}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"[UMRS user] GET {url} -> {resp.status_code}")
        print(resp.json())
    except Exception as e:
        print(f"Request failed: {e}")
    sys.exit(0)


def handle_create_buckets_from_config(args):
    """Handle the create-buckets-from-config subcommand."""
    if not requests:
        print("Error: 'requests' module is required for API commands. Install with: pip install requests")
        sys.exit(1)
        
    token = args.token or os.environ.get("UMRS_USER_TOKEN")
    if not token:
        print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
        sys.exit(1)
    
    # Load config file
    try:
        user_cfg = load_client_cfg_file(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)
    
    headers = {"Authorization": f"Bearer {token}"}
    base_url = args.server_url.rstrip("/")
    
    # Create buckets for each experience
    for xp in user_cfg.get("experiences", []):
        for modality in xp.get("modalities", []):
            # Sanitize xp_name for S3 bucket naming
            safe_xp_name = xp["xp_name"].replace(' ', '-').lower()
            
            # Extract bucket info from modality config
            bucket_alias = modality["s3"]["bucket_alias"]
            modality_name = modality["modality_name"]
            
            payload = {
                "username": user_cfg["username"],
                "bucket_alias": bucket_alias,
                "xp_name": safe_xp_name,
                "modality": modality_name,
                "model": modality.get("model", ""),
                "store_embeddings": modality.get("store_embeddings", False),
                "metadata": list(modality.get("sql", {}).keys()) if modality.get("sql") else []
            }
            
            url = base_url + "/api/user/create_bucket"
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                print(f"[UMRS user] POST {url} -> {resp.status_code}")
                result = resp.json()
                print(result)
                if resp.status_code == 200:
                    print(f"Bucket '{bucket_alias}' created successfully.")
                else:
                    print(f"Error creating bucket '{bucket_alias}': {result.get('message')}")
            except Exception as e:
                print(f"Request failed for bucket '{bucket_alias}': {e}")
    
    sys.exit(0)


def handle_upload(args):
    """Handle the traditional upload command using local message queues."""
    user_cfg = load_client_cfg_file(args.conf)
    bucket_store = defaultdict(lambda: False)
    configure_bucket(user_cfg, bucket_store)
    
    selected_bucket_alias = args.bucket
    if not args.bucket:
        print("Available buckets:")
        for bucket_alias in bucket_store:
            print(f"  {bucket_alias}")
        selected_bucket_alias = input("Please select a bucket: ")
    
    selected_bucket_obj = bucket_store[selected_bucket_alias]
    if not selected_bucket_obj:
        print(f"Error: bucket '{selected_bucket_alias}' not in configuration.")
        sys.exit(1)
    
    # Initialize message queues
    mq = MessageQueue(queue_type=args.queue_type, role="client")
    response_mq = MessageQueue(queue_type=args.queue_type, role="server", queue_name="/tmp/response")
    
    if not is_server_ready(mq, response_mq, timeout_ms=5000):
        sys.exit("Server connection timeout...\nCannot continue!")
    
    print(f"Server ready.\nConnected to message queue of type '{args.queue_type}'")
    
    # Process metadata
    metadata_args = {}
    if args.k:
        for _data in args.k:
            if ":" in _data:
                split_data = _data.split(":")
                metadata_args[split_data[0]] = split_data[1]
    
    # Extract bucket configuration
    bucket_alias = selected_bucket_alias
    username = selected_bucket_obj["username"]
    xp_name = selected_bucket_obj["xp_name"]
    modality_name = selected_bucket_obj["modality_name"]
    path_to_bucket = selected_bucket_obj["path_to_bucket"]
    s3_configuration_file = selected_bucket_obj["s3_configuration_file"]
    
    # Connect to S3
    s3 = connect_to_s3(s3_configuration_file)
    if not s3:
        sys.exit(1)
    
    full_bucket_name = create_path_to_bucket(path_to_bucket, xp_name)
    modality = modality_name  # Use modality name directly
    
    if args.bucket:
        pretty_print_user_cfg(user_cfg, bucket_alias)
        check_bucket_name(s3, full_bucket_name)
    
    if not args.data:
        data_path = input("Please specify a data path: ")
    else:
        data_path = args.data
    
    print(f"Using bucket: {full_bucket_name}")
    print(f"Modality: {modality}")
    
    if modality == "image":
        if os.path.isfile(data_path):
            s3_key = s3_store_image(s3, full_bucket_name, data_path, mq, metadata_args)
            if s3_key:
                print(f"'{s3_key}' saved on: {full_bucket_name}")
        elif os.path.isdir(data_path):
            for file in glob(os.path.join(data_path, "*")):
                if os.path.isfile(file):
                    s3_key = s3_store_image(s3, full_bucket_name, file, mq, metadata_args)
                    if s3_key:
                        print(f"'{s3_key}' saved on: {full_bucket_name}")
        else:
            print(f"Error: do not know what to do with '{data_path}'...")
            print("Provide a path to an image or to a directory of images")
    
    elif modality == "text file":
        if os.path.isfile(data_path):
            s3_key = s3_store_text_file(s3, full_bucket_name, data_path, mq, metadata_args)
            if s3_key:
                print(f"'{s3_key}' saved on: {full_bucket_name}")
        elif os.path.isdir(data_path):
            for file in glob(os.path.join(data_path, "*")):
                if os.path.isfile(file):
                    s3_key = s3_store_text_file(s3, full_bucket_name, file, mq, metadata_args)
                    if s3_key:
                        print(f"'{s3_key}' saved on: {full_bucket_name}")
        else:
            print(f"Error: do not know what to do with '{data_path}'...")
            print("Provide a path to a text file or to a directory of text_files")
    
    else:
        print(f"Error: Unknown modality '{modality}'. Nothing has been sent to the server")


# =============================
# Argument Parsing
# =============================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Traditional upload command (local message queue mode)
    upload_parser = subparsers.add_parser("upload", help="Upload data file to a bucket (local mode)")
    upload_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    upload_parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue between 'zeromq', 'posix_ipc', or 'sysv_ipc'")
    upload_parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
    upload_parser.add_argument("-k", action="append", help="Metadata key:value", required=False)
    upload_parser.add_argument("data", type=str, nargs="?", help="input data", default=None)
    upload_parser.add_argument("bucket", type=str, nargs="?", help="target bucket", default=None)

    # API-based commands (remote server mode)
    # Subcommand: list-buckets
    list_buckets_parser = subparsers.add_parser("list-buckets", help="List all buckets for a user")
    list_buckets_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    list_buckets_parser.add_argument("--token", required=False, help="User API token (or set UMRS_USER_TOKEN env var)")
    list_buckets_parser.add_argument("--username", required=True, help="Username")

    # Subcommand: init-user
    init_user_parser = subparsers.add_parser("init-user", help="Initialize a new user (admin only)")
    init_user_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    init_user_parser.add_argument("--admin-token", required=False, help="Admin API token (or set UMRS_ADMIN_TOKEN env var)")
    init_user_parser.add_argument("--username", required=True, help="Username to create")
    init_user_parser.add_argument("--password", required=True, help="Password for PostgreSQL user")
    init_user_parser.add_argument("--s3-secret", required=True, help="Secret for S3 user")

    # Subcommand: create-bucket
    create_bucket_parser = subparsers.add_parser("create-bucket", help="Create a new bucket for a user")
    create_bucket_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    create_bucket_parser.add_argument("--token", required=False, help="User API token (or set UMRS_USER_TOKEN env var)")
    create_bucket_parser.add_argument("--username", required=True, help="Username")
    create_bucket_parser.add_argument("--bucket-alias", required=True, help="Bucket alias")
    create_bucket_parser.add_argument("--xp-name", required=True, help="Experiment name")
    create_bucket_parser.add_argument("--modality", required=True, help="Modality (e.g., 'image', 'text file')")
    create_bucket_parser.add_argument("--model", required=False, help="Model name", default="")
    create_bucket_parser.add_argument("--store-embeddings", action="store_true", help="Store embeddings")
    create_bucket_parser.add_argument("--metadata", action="append", help="Metadata fields", default=[])

    # Subcommand: create-buckets-from-config
    create_buckets_from_config_parser = subparsers.add_parser("create-buckets-from-config", help="Create all buckets for the user from the config file")
    create_buckets_from_config_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    create_buckets_from_config_parser.add_argument("--token", required=False, help="User API token (or set UMRS_USER_TOKEN env var)")
    create_buckets_from_config_parser.add_argument("--config", required=True, help="User configuration file")

    # Subcommand: show-user
    show_user_parser = subparsers.add_parser("show-user", help="Show user information")
    show_user_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    show_user_parser.add_argument("--token", required=False, help="User API token (or set UMRS_USER_TOKEN env var)")
    show_user_parser.add_argument("--username", required=True, help="Username")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Handle API-based commands
    if args.command == "list-buckets":
        handle_list_buckets(args)
    elif args.command == "init-user":
        handle_init_user(args)
    elif args.command == "create-bucket":
        handle_create_bucket(args)
    elif args.command == "create-buckets-from-config":
        handle_create_buckets_from_config(args)
    elif args.command == "show-user":
        handle_show_user(args)
    elif args.command == "upload":
        # Handle traditional upload command (existing code logic)
        handle_upload(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
