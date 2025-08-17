"""
UMRS API Client: Functions to interact with UMRS FastAPI endpoints.
"""
import requests
from typing import Optional, Dict, Any


class UMRSApiClientError(Exception):
    """Custom exception for UMRSApiClient errors."""
    pass

class UMRSApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

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
                "metadata": None if metadata is None else requests.utils.json.dumps(metadata)
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
                    "metadata": None if metadata is None else requests.utils.json.dumps(metadata)
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
                    "metadata": None if metadata is None else requests.utils.json.dumps(metadata)
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
