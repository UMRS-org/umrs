"""
UMRS Migrations CLI entry point.

Usage:
    python -m umrs.migrations --db-url postgresql://...
    python -m umrs.migrations --status
"""

import argparse
import logging
import os
import sys


def main():
    """Run migrations CLI."""
    # Import here to avoid circular imports
    try:
        import psycopg2
    except ImportError:
        print("Error: psycopg2 is required. Install with: pip install psycopg2-binary")
        sys.exit(1)

    from umrs.migrations import (
        apply_migrations,
        get_current_version,
        get_pending_migrations,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="UMRS Database Migrations")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status instead of applying",
    )
    parser.add_argument("--target", help="Target version to migrate to")

    args = parser.parse_args()

    if not args.db_url:
        print("Error: --db-url or DATABASE_URL required")
        sys.exit(1)

    from umrs.shared.pg_settings import pg_connect_kwargs

    try:
        conn = psycopg2.connect(args.db_url, **pg_connect_kwargs("migrations"))
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    try:
        if args.status:
            current = get_current_version(conn)
            pending = get_pending_migrations(conn)
            print(f"Current version: {current or '(none)'}")
            print(f"Pending migrations: {len(pending)}")
            for v, p in pending:
                print(f"  - {v}: {p.name}")
        else:
            count = apply_migrations(conn, target_version=args.target)
            print(f"Applied {count} migration(s)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
