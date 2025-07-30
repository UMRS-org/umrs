

#!/usr/bin/env python3
"""
UMRS Client: Multi-user, multi-bucket, multi-modality data uploader and admin tool.
"""
# =============================
# Imports
# =============================
import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob
import requests


# Only import these for upload/data commands, not for CLI-only utilities
try:
    from umrs.MessageQueue import MessageQueue
    from umrs.metadata import check_and_build_metadata
    from umrs.s3 import create_path_to_bucket, connect_to_s3, check_bucket_name, s3_store_image, s3_store_text_file
    from umrs.sql import metadata_dict
except ImportError:
    pass  # Not needed for simple CLI utilities like list-buckets

# =============================
# Argparse/Subcommand Setup
# =============================
def parse_args():
    parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: list-buckets
    list_buckets_parser = subparsers.add_parser("list-buckets", help="List all buckets for a user")
    list_buckets_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    list_buckets_parser.add_argument("--token", required=True, help="User API token (or set UMRS_USER_TOKEN env var)")
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
    create_bucket_parser.add_argument("--token", required=True, help="User API token (or set UMRS_USER_TOKEN env var)")
    create_bucket_parser.add_argument("--username", required=True, help="Username")
    create_bucket_parser.add_argument("--bucket-alias", required=True, help="Bucket alias (short name)")
    create_bucket_parser.add_argument("--xp-name", required=True, help="Experience name")
    create_bucket_parser.add_argument("--modality", required=True, help="Modality name (e.g. image, text, text_file)")
    create_bucket_parser.add_argument("--model", required=True, help="Model name for embeddings")
    create_bucket_parser.add_argument("--store-embeddings", action="store_true", help="Whether to store embeddings for this bucket")
    create_bucket_parser.add_argument("--metadata", nargs="*", help="Metadata fields (e.g. key:type)")

    # Subcommand: create-buckets-from-config
    create_buckets_from_config_parser = subparsers.add_parser("create-buckets-from-config", help="Create all buckets for the user from the config file")
    create_buckets_from_config_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    create_buckets_from_config_parser.add_argument("--token", required=True, help="User API token (or set UMRS_USER_TOKEN env var)")
    create_buckets_from_config_parser.add_argument("--username", required=True, help="Username")
    create_buckets_from_config_parser.add_argument("-c", "--conf", default="./conf_umrs.json", help="Config file path")

    # Subcommand: show-user
    show_user_parser = subparsers.add_parser("show-user", help="Show user information")
    show_user_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    show_user_parser.add_argument("--token", required=True, help="User API token (or set UMRS_USER_TOKEN env var)")
    show_user_parser.add_argument("--username", required=True, help="Username")

    # Subcommand: upload
    upload_parser = subparsers.add_parser("upload", help="Upload a data file to a bucket")
    upload_parser.add_argument("-c", "--conf", required=True, help="Config file path")
    upload_parser.add_argument("-k", action="append", help="Metadata key:value", required=False)
    upload_parser.add_argument("data", help="Path to data file to upload")
    upload_parser.add_argument("bucket", help="Bucket alias (short name)")

    # Common options
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue: 'zeromq', 'posix_ipc', or 'sysv_ipc'")
    #parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
    #parser.add_argument("-k", action="append", help="Metadata key:value", required=False)
    args = parser.parse_args()
    return args

# =============================
# Argparse/Subcommand Setup
# =============================

# =============================
# Subcommand Handler Functions
# =============================
def handle_init_user(args):
    """Handle the init-user subcommand."""
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

# ...existing code for other handlers (create-buckets-from-config, show-user, upload, etc.)...

# =============================
# Main Logic
# =============================
def main():
    args = parse_args()

    if args.command == "init-user":
        handle_init_user(args)
        return
    elif args.command == "create-bucket":
        handle_create_bucket(args)
        return
    elif args.command == "list-buckets":
        token = args.token or os.environ.get("UMRS_USER_TOKEN")
        if not token:
            print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
            sys.exit(1)
        url = args.server_url.rstrip("/") + f"/api/user/list_buckets?username={args.username}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            print(f"[UMRS user] GET {url} -> {resp.status_code}")
            print(resp.json())
        except Exception as e:
            print(f"Request failed: {e}")
        sys.exit(0)

if __name__ == "__main__":
    main()

def store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user, s3_configuration_file, metadata):
    """Store bucket configuration, ensuring no duplicates."""
    if bucket_store[bucket_alias]:
        sys.exit(f"Error: duplicate bucket_alias {bucket_alias}")
    bucket_store[bucket_alias] = {
        "path_to_bucket": path_to_bucket,
        "username": username,
        "xp_name": xp_name,
        "modality_name": modality_name,
        "s3_configuration_file": s3_configuration_file,
        "metadata": metadata
    }

def configure_bucket(user_cfg, bucket_store):
    """Configure all buckets for a user from config."""
    username = user_cfg["username"]
    s3_configuration_file = user_cfg["s3_configuration_file"]
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            metadata = metadata_dict(modality["sql"])
            store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user_cfg, s3_configuration_file, metadata)



def handle_init_user(args):
    """Handle the init-user subcommand."""
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
    token = args.token or os.environ.get("UMRS_USER_TOKEN")
    if not token:
        print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
        sys.exit(1)
    payload = {
        "username": args.username,
        "bucket_alias": args.bucket_alias,
        "xp_name": args.xp_name,
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

def is_likely_path(data_str):
    """Heuristic to determine if a string is likely a file/directory path."""
    if ('/' in data_str or '\\' in data_str or
        data_str.startswith('.') or data_str.startswith('~') or
        data_str.startswith('/') or
        (len(data_str.split()) == 1 and '.' in data_str)):
        return True
    if ' ' in data_str and len(data_str.split()) > 2:
        return False
    return False

def main():
    args = parse_args()
    if args.command == "show-user":
        token = args.token or os.environ.get("UMRS_USER_TOKEN")
        if not token:
            print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
            sys.exit(1)
        url = args.server_url.rstrip("/") + f"/api/user/info?username={args.username}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            print(f"[UMRS user] GET {url} -> {resp.status_code}")
            print(resp.json())
        except Exception as e:
            print(f"Request failed: {e}")
        sys.exit(0)
    if args.command == "create-buckets-from-config":
        # Read config and create all buckets for the user
        configure_file = args.conf
        if not os.path.exists(configure_file):
            sys.exit(f"Configuration file '{configure_file}' not found.")
        with open(configure_file, "r", encoding="utf-8") as file:
            content = json.load(file)
        user_entry = None
        for user in content["users"]:
            if user["username"] == args.username:
                user_entry = user
                break
        if not user_entry:
            sys.exit(f"User '{args.username}' not found in config.")
        for xp in user_entry["experiences"]:
            xp_name = xp["xp_name"]
            for modality in xp["modalities"]:
                modality_name = modality["modality_name"]
                bucket_alias = modality["s3"]["bucket_alias"]
                model_name = modality["model"]["model_name"] if "model" in modality and "model_name" in modality["model"] else ""
                store_embeddings = modality["sql"].get("store_embeddings", "no") == "yes"
                metadata = [f'{m["name"]}:{m["type"]}' for m in modality["sql"].get("meta_data", [])]
                # Sanitize xp_name for S3 bucket naming: replace spaces with hyphens and lowercase
                safe_xp_name = xp_name.replace(' ', '-').lower()
                payload = {
                    "username": args.username,
                    "bucket_alias": bucket_alias,
                    "xp_name": safe_xp_name,
                    "modality": modality_name,
                    "model": model_name,
                    "store_embeddings": store_embeddings,
                    "metadata": metadata
                }
                headers = {"Authorization": f"Bearer {args.token}"}
                url = args.server_url.rstrip("/") + "/api/user/create_bucket"
                try:
                    resp = requests.post(url, json=payload, headers=headers, timeout=10)
                    print(f"[UMRS user] POST {url} -> {resp.status_code}")
                    print(resp.json())
                    if resp.status_code == 200:
                        print(f"Bucket '{bucket_alias}' created successfully.")
                    else:
                        print(f"Error: {resp.json().get('message')}")
                except Exception as e:
                    print(f"Request failed: {e}")
        sys.exit(0)


    if args.command == "init-user":
        handle_init_user(args)
        return
    elif args.command == "create-bucket":
        handle_create_bucket(args)
        return
    elif args.command == "upload":
        # --- Upload logic ---
        mq = MessageQueue(queue_type=args.queue_type, role="client")
        print(f"Connected to message queue of type '{args.queue_type}'")

        configure_file = args.conf
        if not os.path.exists(configure_file):
            sys.exit(f"Configuration file '{configure_file}' not found.")

        with open(configure_file, "r", encoding="utf-8") as file:
            content = json.load(file)

        bucket_store = defaultdict(dict)
        for user in content["users"]:
            configure_bucket(user, bucket_store)
        if args.verbose:
            for bucket_alias, entry in bucket_store.items():
                print(f"{bucket_alias}: {entry['path_to_bucket']}, {entry['username']}, {entry['xp_name']}, {entry['modality_name']}")
        print("Configuration loaded successfully.")

        bucket_alias = args.bucket
        if not bucket_store[bucket_alias]:
            sys.exit(f"Error: unknown bucket name '{bucket_alias}' in configuration")

        entry = bucket_store[bucket_alias]
        path_to_bucket = entry['path_to_bucket']
        username = entry['username']
        xp_name = entry['xp_name']
        modality = entry['modality_name']
        s3_configuration_file = entry['s3_configuration_file']
        metadata = entry['metadata']

        s3 = connect_to_s3(s3_configuration_file)
        full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username)
        if not success or full_bucket_name != path_to_bucket:
            sys.exit(f"No bucket {bucket_alias} for modality {modality} and for user {username}")
        print(f"Bucket '{bucket_alias}' for modality '{modality}' and for user '{username}': OK!")

        metadata_args = check_and_build_metadata(metadata, args.k)
        print(f"Storing data for:\n\tUser: '{username}'\n\tXP: '{xp_name}'\n\tModality: '{modality}'")

        # --- Data upload logic ---
        if modality == "text":
            if os.path.exists(args.data):
                if os.path.isfile(args.data):
                    try:
                        with open(args.data, "r", encoding="utf-8") as file:
                            for line in file.readlines():
                                # Example: upload each line as a text object (replace with actual upload logic)
                                print(f"Uploading line: {line.strip()}")
                    except UnicodeDecodeError as e:
                        print(f"Unicode decoding error: {e}")
                    except IOError as e:
                        print(f"Error reading file {args.data}: {e}")
                else:
                    if os.path.isdir(args.data):
                        print(f"Error: '{args.data}' is a directory. For text modality, provide a file path or text content.")
                    else:
                        print(f"Error: {args.data} is not a file.")
            else:
                print(f"Error: File {args.data} does not exist.")
        else:
            print(f"Upload for modality '{modality}' is not implemented yet.")
        print("Upload complete.")
        sys.exit(0)

    elif modality == "image":
        if os.path.exists(args.data):
            if os.path.isfile(args.data):
                try:
                    s3_key = s3_store_image(s3, full_bucket_name, args.data, mq, metadata_args)
                    if s3_key:
                        print(f"'{s3_key}' saved on: {full_bucket_name}")
                    else:
                        print(f"Error: Failed to store image '{args.data}'")
                        sys.exit(1)
                except Exception as e:
                    print(f"Error storing image {args.data}: {e}")
                    sys.exit(1)
            elif os.path.isdir(args.data):
                image_files = [file for file in glob(os.path.join(args.data, "*")) if os.path.isfile(file)]
                if not image_files:
                    print(f"Error: No files found in directory '{args.data}'")
                    sys.exit(1)
                for file in image_files:
                    try:
                        s3_key = s3_store_image(s3, full_bucket_name, file, mq, metadata_args)
                        if s3_key:
                            print(f"'{s3_key}' saved on: {full_bucket_name}")
                        else:
                            print(f"Warning: Failed to store image '{file}' - skipping")
                    except Exception as e:
                        print(f"Error storing image {file}: {e} - skipping")
        else:
            print(f"Error: Image file or directory '{args.data}' does not exist.")
            print("Provide a path to an existing image file or directory containing images.")
            sys.exit(1)

    elif modality == "text_file":
        if os.path.exists(args.data):
            if os.path.isfile(args.data):
                try:
                    s3_key = s3_store_text_file(s3, full_bucket_name, args.data, mq, metadata_args)
                    if s3_key:
                        print(f"'{s3_key}' saved on: {full_bucket_name}")
                    else:
                        print(f"Error: Failed to store text file '{args.data}'")
                        sys.exit(1)
                except Exception as e:
                    print(f"Error storing text file {args.data}: {e}")
                    sys.exit(1)
            elif os.path.isdir(args.data):
                text_files = [file for file in glob(os.path.join(args.data, "*")) if os.path.isfile(file)]
                if not text_files:
                    print(f"Error: No files found in directory '{args.data}'")
                    sys.exit(1)
                for file in text_files:
                    try:
                        s3_key = s3_store_text_file(s3, full_bucket_name, file, mq, metadata_args)
                        if s3_key:
                            print(f"'{s3_key}' saved on: {full_bucket_name}")
                        else:
                            print(f"Warning: Failed to store text file '{file}' - skipping")
                    except Exception as e:
                        print(f"Error storing text file {file}: {e} - skipping")
        else:
            print(f"Error: Text file or directory '{args.data}' does not exist.")
            print("Provide a path to an existing text file or directory containing text files.")
            sys.exit(1)

    else:
        print(f"Error: Unknown modality '{modality}'. Nothing has been sent to the server")


if __name__ == "__main__":
    main()