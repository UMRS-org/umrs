"""Catalog reconciliation CLI — remove phantom rows for deleted S3 objects."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import boto3
import psycopg2
from botocore.config import Config as BotoConfig
from psycopg2 import pool

from .pg_store import CatalogMetaStore, reconcile_project

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile catalog rows against live S3 objects",
    )
    parser.add_argument("--db-url", default=os.getenv("UMRS_JOBS_DB_URL", ""))
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", ""))
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if not args.db_url:
        logger.error("--db-url or UMRS_JOBS_DB_URL is required")
        return 1
    if not args.bucket:
        logger.error("--bucket or S3_BUCKET is required")
        return 1

    from umrs.shared.pg_settings import pg_connect_kwargs

    db_pool = pool.SimpleConnectionPool(
        1, 2, args.db_url, **pg_connect_kwargs("catalog-reconcile")
    )
    store = CatalogMetaStore(db_pool)
    s3_client = _build_s3_client()

    result = reconcile_project(
        store,
        s3_client,
        args.bucket,
        args.project_id,
        dry_run=args.dry_run,
    )
    db_pool.closeall()

    logger.info(
        "Reconcile complete: checked=%d deleted=%d missing=%d dry_run=%s",
        result.checked,
        result.deleted,
        len(result.missing_keys or []),
        args.dry_run,
    )
    if result.missing_keys:
        for key in result.missing_keys[:20]:
            logger.info("  missing: %s", key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
