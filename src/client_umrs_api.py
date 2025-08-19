"""
UMRS API Client: Functions to interact with UMRS FastAPI endpoints.
"""

import requests
import json
from typing import Optional, Dict, Any
import argparse
import sys


class UMRSApiClientError(Exception):
    """Custom exception for UMRSApiClient errors."""
    pass

class UMRSApiClient:

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def add_user_hot_reload(self, user_config: dict) -> Dict[str, Any]:
        """Add a user, store JSON, update config, and hot-reload user (no restart)."""
        try:
            headers = {"Content-Type": "application/json"}
            resp = requests.post(f"{self.base_url}/admin/add_user_hot_reload", json=user_config, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Add user hot-reload failed: {e}")
    def create_user(self, user_config: dict) -> Dict[str, Any]:
        """Create and initialize a user (admin/server side)."""
        try:
            headers = {"Content-Type": "application/json"}
            resp = requests.post(f"{self.base_url}/admin/create_user", json=user_config, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Create user failed: {e}")

    def create_buckets(self, user_config: dict) -> Dict[str, Any]:
        """Create all buckets for a user from a client JSON file (admin/server side)."""
        try:
            headers = {"Content-Type": "application/json"}
            resp = requests.post(f"{self.base_url}/admin/create_buckets", json=user_config, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Create buckets failed: {e}")

    def health(self) -> Dict[str, Any]:
        import os, traceback
        try:
            resp = requests.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.SSLError as ssl_err:
            print("[SSL ERROR]", ssl_err)
            print("REQUESTS_CA_BUNDLE:", os.environ.get("REQUESTS_CA_BUNDLE"))
            print("Retrying with verify=False (NOT SECURE, for debug only)...")
            try:
                resp = requests.get(f"{self.base_url}/health", verify=False)
                resp.raise_for_status()
                print("[WARNING] SSL verification was skipped!")
                return resp.json()
            except Exception as e2:
                print("[SSL fallback failed]")
                traceback.print_exc()
                raise UMRSApiClientError(f"Health check failed (SSL fallback): {e2}")
        except Exception as e:
            raise UMRSApiClientError(f"Health check failed: {e}")

    def status(self) -> Dict[str, Any]:
        import os, traceback
        try:
            resp = requests.get(f"{self.base_url}/status")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.SSLError as ssl_err:
            print("[SSL ERROR]", ssl_err)
            print("REQUESTS_CA_BUNDLE:", os.environ.get("REQUESTS_CA_BUNDLE"))
            print("Retrying with verify=False (NOT SECURE, for debug only)...")
            try:
                resp = requests.get(f"{self.base_url}/status", verify=False)
                resp.raise_for_status()
                print("[WARNING] SSL verification was skipped!")
                return resp.json()
            except Exception as e2:
                print("[SSL fallback failed]")
                traceback.print_exc()
                raise UMRSApiClientError(f"Status check failed (SSL fallback): {e2}")
        except Exception as e:
            raise UMRSApiClientError(f"Status check failed: {e}")

    def list_buckets(self, username: Optional[str] = None) -> Dict[str, Any]:
        import os, traceback
        try:
            params = {"username": username} if username else {}
            resp = requests.get(f"{self.base_url}/buckets", params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.SSLError as ssl_err:
            print("[SSL ERROR]", ssl_err)
            print("REQUESTS_CA_BUNDLE:", os.environ.get("REQUESTS_CA_BUNDLE"))
            print("Retrying with verify=False (NOT SECURE, for debug only)...")
            try:
                resp = requests.get(f"{self.base_url}/buckets", params=params, verify=False)
                resp.raise_for_status()
                print("[WARNING] SSL verification was skipped!")
                return resp.json()
            except Exception as e2:
                print("[SSL fallback failed]")
                traceback.print_exc()
                raise UMRSApiClientError(f"List buckets failed (SSL fallback): {e2}")
        except Exception as e:
            raise UMRSApiClientError(f"List buckets failed: {e}")

    def upload_text(self, text: str, bucket_alias: str, username: str, metadata: Optional[dict] = None) -> Dict[str, Any]:
        try:
            data = {
                "text": text,
                "bucket_alias": bucket_alias,
                "username": username,
                "metadata": None if metadata is None else json.dumps(metadata)
            }
            resp = requests.post(f"{self.base_url}/upload/text", data=data)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Upload text failed: {e}")

    def upload_image(self, file_path: str, bucket_alias: str, username: str, metadata: Optional[dict] = None) -> Dict[str, Any]:
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {
                    "bucket_alias": bucket_alias,
                    "username": username,
                    "metadata": None if metadata is None else json.dumps(metadata)
                }
                resp = requests.post(f"{self.base_url}/upload/image", data=data, files=files)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Upload image failed: {e}")

    def upload_textfile(self, file_path: str, bucket_alias: str, username: str, metadata: Optional[dict] = None) -> Dict[str, Any]:
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {
                    "bucket_alias": bucket_alias,
                    "username": username,
                    "metadata": None if metadata is None else json.dumps(metadata)
                }
                resp = requests.post(f"{self.base_url}/upload/textfile", data=data, files=files)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Upload textfile failed: {e}")

    def recompute_embeddings(self) -> Dict[str, Any]:
        try:
            resp = requests.post(f"{self.base_url}/recompute")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Recompute embeddings failed: {e}")

    def update_embeddings(self) -> Dict[str, Any]:
        try:
            resp = requests.post(f"{self.base_url}/update")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Update embeddings failed: {e}")

    def check_consistency(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/check")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise UMRSApiClientError(f"Check consistency failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="UMRS API Client CLI")
    parser.add_argument("--base_url", required=True, help="Base URL of the UMRS API (e.g., https://localhost:8000)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # health
    subparsers.add_parser("health", help="Check API health")
    # status
    subparsers.add_parser("status", help="Check API status")
    # list_buckets
    parser_list_buckets = subparsers.add_parser("list_buckets", help="List buckets for a username")
    parser_list_buckets.add_argument("username", help="Username to filter buckets", nargs="?")
    # upload_text
    parser_upload_text = subparsers.add_parser("upload_text", help="Upload text to a bucket")
    parser_upload_text.add_argument("text")
    parser_upload_text.add_argument("bucket_alias")
    parser_upload_text.add_argument("username")
    parser_upload_text.add_argument("metadata", nargs="?")
    # upload_image
    parser_upload_image = subparsers.add_parser("upload_image", help="Upload image to a bucket")
    parser_upload_image.add_argument("file_path")
    parser_upload_image.add_argument("bucket_alias")
    parser_upload_image.add_argument("username")
    parser_upload_image.add_argument("metadata", nargs="?")
    # upload_textfile
    parser_upload_textfile = subparsers.add_parser("upload_textfile", help="Upload text file to a bucket")
    parser_upload_textfile.add_argument("file_path")
    parser_upload_textfile.add_argument("bucket_alias")
    parser_upload_textfile.add_argument("username")
    parser_upload_textfile.add_argument("metadata", nargs="?")
    # recompute_embeddings
    subparsers.add_parser("recompute_embeddings", help="Recompute all embeddings")
    # update_embeddings
    subparsers.add_parser("update_embeddings", help="Update embeddings")
    # check_consistency
    subparsers.add_parser("check_consistency", help="Check consistency")
    # add_user_hot_reload
    parser_add_user = subparsers.add_parser("add_user_hot_reload", help="Add user with hot reload (admin)")
    parser_add_user.add_argument("user_config")
    # create_user
    parser_create_user = subparsers.add_parser("create_user", help="Create and initialize user (admin)")
    parser_create_user.add_argument("user_config")
    # create_buckets
    parser_create_buckets = subparsers.add_parser("create_buckets", help="Create all buckets for a user (admin)")
    parser_create_buckets.add_argument("user_config")

    args = parser.parse_args()
    client = UMRSApiClient(args.base_url)

    try:
        if args.command == "health":
            print(client.health())
        elif args.command == "status":
            print(client.status())
        elif args.command == "list_buckets":
            print(client.list_buckets(username=args.username))
        elif args.command == "upload_text":
            metadata = json.loads(args.metadata) if args.metadata else None
            print(client.upload_text(args.text, args.bucket_alias, args.username, metadata=metadata))
        elif args.command == "upload_image":
            metadata = json.loads(args.metadata) if args.metadata else None
            print(client.upload_image(args.file_path, args.bucket_alias, args.username, metadata=metadata))
        elif args.command == "upload_textfile":
            metadata = json.loads(args.metadata) if args.metadata else None
            print(client.upload_textfile(args.file_path, args.bucket_alias, args.username, metadata=metadata))
        elif args.command == "recompute_embeddings":
            print(client.recompute_embeddings())
        elif args.command == "update_embeddings":
            print(client.update_embeddings())
        elif args.command == "check_consistency":
            print(client.check_consistency())
        elif args.command == "add_user_hot_reload":
            with open(args.user_config, "r") as f:
                user_config = json.load(f)
            print(client.add_user_hot_reload(user_config))
        elif args.command == "create_user":
            with open(args.user_config, "r") as f:
                user_config = json.load(f)
            print(client.create_user(user_config))
        elif args.command == "create_buckets":
            with open(args.user_config, "r") as f:
                user_config = json.load(f)
            print(client.create_buckets(user_config))
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)
    except UMRSApiClientError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()

