#!/usr/bin/python3
"""
UMRS Client: Multi-user, multi-bucket, multi-modality data uploader and admin tool.
API-only mode - connects to hybrid UMRS server.
"""

import json
import os
import sys
import argparse
from glob import glob
from umrs.json_utils import load_client_cfg_file

# API client imports
try:
    import requests
except ImportError:
    requests = None


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
                "model": modality.get("model", {}).get("model_name", "") if isinstance(modality.get("model"), dict) else modality.get("model", ""),
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


def handle_delete_bucket(args):
    """Handle the delete-bucket subcommand with confirmation."""
    if not requests:
        print("Error: 'requests' module is required for API commands. Install with: pip install requests")
        sys.exit(1)

    token = args.token or os.environ.get("UMRS_USER_TOKEN")
    if not token:
        print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
        sys.exit(1)

    # Display warning and require confirmation
    print(f"⚠️  WARNING: You are about to DELETE bucket '{args.bucket_name}' and ALL its contents!")
    print(f"   - This will delete all files in the bucket")
    print(f"   - This will delete all PostgreSQL records")
    print(f"   - This action CANNOT be undone")
    print()

    if not args.confirm:
        # Interactive confirmation
        confirmation = input(f"Type 'DELETE {args.bucket_name}' to confirm: ")
        expected_confirmation = f"DELETE {args.bucket_name}"

        if confirmation != expected_confirmation:
            print("❌ Deletion cancelled - confirmation did not match.")
            sys.exit(0)

        # Double confirmation
        final_confirm = input("Are you absolutely sure? Type 'YES' to proceed: ")
        if final_confirm != "YES":
            print("❌ Deletion cancelled.")
            sys.exit(0)

    payload = {
        "username": args.username,
        "bucket_name": args.bucket_name,
        "confirmation": "delete"
    }
    headers = {"Authorization": f"Bearer {token}"}
    url = args.server_url.rstrip("/") + "/api/user/delete_bucket"

    try:
        print(f"🗑️  Deleting bucket '{args.bucket_name}'...")
        resp = requests.delete(url, json=payload, headers=headers, timeout=30)
        print(f"[UMRS delete] DELETE {url} -> {resp.status_code}")

        result = resp.json()
        print(result)

        if resp.status_code == 200:
            print(f"✅ Bucket '{args.bucket_name}' deleted successfully.")
        else:
            print(f"❌ Error deleting bucket: {result.get('message')}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Delete request failed: {e}")
        sys.exit(1)


def handle_upload(args):
    """Handle upload command via API (server is always hybrid)."""
    if not requests:
        print("Error: 'requests' module is required for API upload. Install with: pip install requests")
        sys.exit(1)

    token = args.token or os.environ.get("UMRS_USER_TOKEN")
    if not token:
        print("Error: User token must be provided via --token or UMRS_USER_TOKEN env var.")
        sys.exit(1)

    # Get data path from --file parameter or positional argument
    if hasattr(args, 'file') and args.file:
        data_path = args.file
    elif args.data:
        data_path = args.data
    else:
        data_path = input("Please specify a data path: ")

    # Get bucket alias from --bucket-alias parameter or positional argument
    if hasattr(args, 'bucket_alias') and args.bucket_alias:
        bucket_alias = args.bucket_alias
    elif args.bucket:
        bucket_alias = args.bucket
    else:
        print("Error: Bucket alias must be specified for upload.")
        sys.exit(1)

    # For API mode, we need additional parameters
    username = getattr(args, 'username', None) or input("Please specify username: ")
    xp_name = getattr(args, 'xp_name', None) or input("Please specify experiment name: ")
    modality = getattr(args, 'modality', None) or input("Please specify modality (image/text file): ")

    headers = {"Authorization": f"Bearer {token}"}
    base_url = args.server_url.rstrip("/")

    if os.path.isfile(data_path):
        # Single file upload
        upload_single_file_api(base_url, headers, username, bucket_alias, xp_name, modality, data_path)
    elif os.path.isdir(data_path):
        # Batch upload for directory
        files = glob(os.path.join(data_path, "*"))
        files = [f for f in files if os.path.isfile(f)]
        upload_batch_files_api(base_url, headers, username, bucket_alias, xp_name, modality, files)
    else:
        print(f"Error: '{data_path}' is not a valid file or directory.")
        sys.exit(1)

def upload_single_file_api(base_url, headers, username, bucket_alias, xp_name, modality, file_path):
    """Upload a single file via API."""
    url = base_url + "/api/user/upload_file"

    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            data = {
                'username': username,
                'bucket_alias': bucket_alias,
                'xp_name': xp_name,
                'modality': modality
            }

            resp = requests.post(url, files=files, data=data, headers=headers, timeout=30)
            print(f"[UMRS upload] POST {url} -> {resp.status_code}")
            result = resp.json()
            print(result)

            if resp.status_code == 200:
                print(f"✓ File '{os.path.basename(file_path)}' uploaded successfully to bucket '{result.get('bucket')}'")
            else:
                print(f"✗ Upload failed: {result.get('message')}")

    except Exception as e:
        print(f"✗ Upload failed for '{file_path}': {e}")

def upload_batch_files_api(base_url, headers, username, bucket_alias, xp_name, modality, file_paths):
    """Upload multiple files via API batch endpoint."""
    url = base_url + "/api/user/upload_batch"

    try:
        files = []
        for i, file_path in enumerate(file_paths):
            files.append((f'file{i}', (os.path.basename(file_path), open(file_path, 'rb'))))

        data = {
            'username': username,
            'bucket_alias': bucket_alias,
            'xp_name': xp_name,
            'modality': modality
        }

        resp = requests.post(url, files=files, data=data, headers=headers, timeout=60)

        # Close file handles
        for _, (_, file_handle) in files:
            file_handle.close()

        print(f"[UMRS batch upload] POST {url} -> {resp.status_code}")
        result = resp.json()
        print(result)

        if resp.status_code == 200:
            successful_uploads = [f for f in result.get('files', []) if f.get('status') == 'success']
            print(f"✓ Batch upload completed: {len(successful_uploads)}/{len(file_paths)} files uploaded successfully")

            # Show individual file results
            for file_result in result.get('files', []):
                status_icon = "✓" if file_result.get('status') == 'success' else "✗"
                print(f"  {status_icon} {file_result.get('filename')}: {file_result.get('status')}")
        else:
            print(f"✗ Batch upload failed: {result.get('message')}")

    except Exception as e:
        print(f"✗ Batch upload failed: {e}")

# =============================
# Argument Parsing
# =============================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload command (API mode only - server is always hybrid)
    upload_parser = subparsers.add_parser("upload", help="Upload data file to a bucket via API")
    upload_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    upload_parser.add_argument("data", type=str, nargs="?", help="input data path", default=None)
    upload_parser.add_argument("bucket", type=str, nargs="?", help="target bucket alias", default=None)

    # API mode parameters (required)
    upload_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    upload_parser.add_argument("--token", required=False, help="User API token (or set UMRS_USER_TOKEN env var)")
    upload_parser.add_argument("--username", required=True, help="Username")
    upload_parser.add_argument("--xp-name", required=True, help="Experiment name")
    upload_parser.add_argument("--modality", required=True, help="Modality (e.g., 'image', 'text file')")
    upload_parser.add_argument("--bucket-alias", required=False, help="Bucket alias (alternative to positional bucket argument)")
    upload_parser.add_argument("--file", required=False, help="File path (alternative to positional data argument)")

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

    # Subcommand: delete-bucket
    delete_bucket_parser = subparsers.add_parser("delete-bucket", help="Delete a bucket and all its contents (DANGEROUS)")
    delete_bucket_parser.add_argument("--server-url", required=True, help="UMRS server base URL (e.g. http://localhost:8000)")
    delete_bucket_parser.add_argument("--token", required=False, help="User API token (or set UMRS_USER_TOKEN env var)")
    delete_bucket_parser.add_argument("--username", required=True, help="Username")
    delete_bucket_parser.add_argument("--bucket-name", required=True, help="Full bucket name to delete")
    delete_bucket_parser.add_argument("--confirm", action="store_true", help="Skip interactive confirmation (DANGEROUS)")

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
    elif args.command == "delete-bucket":
        handle_delete_bucket(args)
    elif args.command == "upload":
        # Handle traditional upload command (existing code logic)
        handle_upload(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
