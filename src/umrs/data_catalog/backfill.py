"""One-time backfill of legacy S3 .meta.json sidecars into Postgres."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import boto3
from botocore.config import Config as BotoConfig
from psycopg2 import pool

from .legacy_s3_store import list_legacy_meta_keys, read_meta
from .paths import decode_s3_key
from .pg_store import CatalogMetaStore, derive_project_id

logger = logging.getLogger(__name__)


def _build_s3_client() -> object:
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", ""),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", ""),
        region_name=os.getenv("S3_REGION", "us-east-1"),
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
        use_ssl=os.getenv("S3_USE_SSL", "false").lower() == "true",
    )


def backfill_catalog(
    *,
    db_url: str,
    bucket: str,
    prefix: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Backfill legacy sidecars into ``catalog_objects``."""
    s3_client = _build_s3_client()
    from umrs.shared.pg_settings import pg_connect_kwargs

    db_pool = pool.SimpleConnectionPool(
        1, 4, db_url, **pg_connect_kwargs("catalog-backfill")
    )
    store = CatalogMetaStore(db_pool)

    stats = {"seen": 0, "upserted": 0, "skipped": 0, "errors": 0}
    meta_keys = list_legacy_meta_keys(s3_client, bucket, prefix=prefix)

    for meta_key in meta_keys:
        stats["seen"] += 1
        encoded = meta_key.removeprefix("_data_catalog/objects/").removesuffix(
            ".meta.json"
        )
        s3_key = decode_s3_key(encoded)
        try:
            meta = read_meta(s3_client, bucket, s3_key)
            if meta is None:
                stats["skipped"] += 1
                continue
            project_id = derive_project_id(meta)
            if not project_id:
                logger.warning("Skipping %s — cannot derive project_id", s3_key)
                stats["skipped"] += 1
                continue
            if dry_run:
                stats["upserted"] += 1
                continue
            store.upsert(meta, project_id)
            stats["upserted"] += 1
        except Exception:
            logger.exception("Failed to backfill %s", s3_key)
            stats["errors"] += 1

    db_pool.closeall()
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill catalog metadata to Postgres"
    )
    parser.add_argument("--db-url", default=os.getenv("UMRS_JOBS_DB_URL", ""))
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument(
        "--prefix", default=None, help="Optional object-key prefix filter"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if not args.db_url:
        logger.error("--db-url or UMRS_JOBS_DB_URL is required")
        return 1
    if not args.bucket:
        logger.error("--bucket or S3_BUCKET is required")
        return 1

    stats = backfill_catalog(
        db_url=args.db_url,
        bucket=args.bucket,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )
    logger.info("Backfill complete: %s", stats)
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
