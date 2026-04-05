"""Compatibility re-export for bucket models during the client/server split."""

from umrs.shared.buckets import BucketsCfg, ClientBucketEntry, ServerBucketEntry

__all__ = ["BucketsCfg", "ClientBucketEntry", "ServerBucketEntry"]

