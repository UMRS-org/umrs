"""
UMRS Python Client.

Provides a clean Python interface for UMRS REST API.
Standalone-capable: works without Gateway or aisci SDK.

Note: Context/memory clients excluded from selective merge.

Usage:
    from umrs.client import UMRSClient

    client = UMRSClient(base_url="http://umrs:8000")
"""

from .cli import main
from .client import UMRSClient

__all__ = [
    "UMRSClient",
    "main",
]
