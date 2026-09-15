"""Shared UMRS primitives safe to import from both client and server installs."""

from umrs.shared.buckets import BucketsCfg, ClientBucketEntry, ServerBucketEntry

__all__ = [
    "BucketsCfg",
    "ClientBucketEntry",
    "ServerBucketEntry",
]
