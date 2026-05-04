"""
UMRS Database Migrations.

Provides versioned, idempotent SQL migrations for UMRS schema evolution.
Migrations are tracked in the `schema_migrations` table.

Usage:
    from umrs.migrations import apply_migrations

    # Apply all pending migrations
    apply_migrations(db_pool)

    # Or from CLI
    python -m umrs.migrations --db-url postgresql://...
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory containing migration SQL files
MIGRATIONS_DIR = Path(__file__).parent


def get_migration_files() -> list[tuple[str, Path]]:
    """
    Get all migration files sorted by version.

    Returns:
        List of (version, path) tuples sorted by version.
    """
    migrations = []
    for f in MIGRATIONS_DIR.glob("*.sql"):
        # Extract version from filename: 0001_context_tables.sql -> 0001
        version = f.stem.split("_")[0]
        migrations.append((version, f))

    return sorted(migrations, key=lambda x: x[0])


def ensure_migrations_table(cursor) -> None:
    """
    Create the schema_migrations table if it doesn't exist.

    Args:
        cursor: Database cursor.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def get_applied_migrations(cursor) -> set:
    """
    Get set of already applied migration versions.

    Args:
        cursor: Database cursor.

    Returns:
        Set of version strings.
    """
    cursor.execute("SELECT version FROM schema_migrations")
    return {row[0] for row in cursor.fetchall()}


def apply_migration(cursor, version: str, sql_path: Path) -> bool:
    """
    Apply a single migration.

    Args:
        cursor: Database cursor.
        version: Migration version string.
        sql_path: Path to SQL file.

    Returns:
        True if applied, False if skipped (already applied).
    """
    sql_content = sql_path.read_text()

    # Execute migration SQL
    cursor.execute(sql_content)

    # Record migration
    cursor.execute(
        "INSERT INTO schema_migrations (version) VALUES (%s)",
        (version,)
    )

    logger.info(f"Applied migration: {version} ({sql_path.name})")
    return True


def apply_migrations(
    db_pool,
    target_version: str | None = None,
) -> int:
    """
    Apply all pending migrations up to target_version.

    Migrations are applied transactionally - if any migration fails,
    all changes are rolled back.

    Args:
        db_pool: psycopg2 connection pool or connection.
        target_version: Optional version to stop at. If None, apply all.

    Returns:
        Number of migrations applied.
    """
    # Get connection from pool or use directly
    if hasattr(db_pool, "getconn"):
        conn = db_pool.getconn()
        return_to_pool = True
    else:
        conn = db_pool
        return_to_pool = False

    try:
        with conn.cursor() as cursor:
            # Ensure migrations table exists
            ensure_migrations_table(cursor)
            conn.commit()

            # Get applied migrations
            applied = get_applied_migrations(cursor)

            # Get pending migrations
            all_migrations = get_migration_files()
            pending = [
                (v, p) for v, p in all_migrations
                if v not in applied and (target_version is None or v <= target_version)
            ]

            if not pending:
                logger.info("No pending migrations")
                return 0

            # Apply each migration in a transaction
            applied_count = 0
            for version, sql_path in pending:
                try:
                    apply_migration(cursor, version, sql_path)
                    conn.commit()
                    applied_count += 1
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Migration {version} failed: {e}")
                    raise

            logger.info(f"Applied {applied_count} migration(s)")
            return applied_count

    finally:
        if return_to_pool:
            db_pool.putconn(conn)


def get_current_version(db_pool) -> str | None:
    """
    Get the current schema version.

    Args:
        db_pool: psycopg2 connection pool or connection.

    Returns:
        Latest applied version or None if no migrations applied.
    """
    if hasattr(db_pool, "getconn"):
        conn = db_pool.getconn()
        return_to_pool = True
    else:
        conn = db_pool
        return_to_pool = False

    try:
        with conn.cursor() as cursor:
            try:
                cursor.execute(
                    "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
                )
                row = cursor.fetchone()
                return row[0] if row else None
            except Exception:
                # Table might not exist
                return None
    finally:
        if return_to_pool:
            db_pool.putconn(conn)


def get_pending_migrations(db_pool) -> list[tuple[str, Path]]:
    """
    Get list of pending migrations.

    Args:
        db_pool: psycopg2 connection pool or connection.

    Returns:
        List of (version, path) tuples for pending migrations.
    """
    if hasattr(db_pool, "getconn"):
        conn = db_pool.getconn()
        return_to_pool = True
    else:
        conn = db_pool
        return_to_pool = False

    try:
        with conn.cursor() as cursor:
            try:
                ensure_migrations_table(cursor)
                conn.commit()
                applied = get_applied_migrations(cursor)
            except Exception:
                applied = set()

        all_migrations = get_migration_files()
        return [(v, p) for v, p in all_migrations if v not in applied]
    finally:
        if return_to_pool:
            db_pool.putconn(conn)


# CLI entry point
if __name__ == "__main__":
    import argparse

    import psycopg2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="UMRS Database Migrations")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status instead of applying"
    )
    parser.add_argument(
        "--target",
        help="Target version to migrate to"
    )

    args = parser.parse_args()

    if not args.db_url:
        print("Error: --db-url or DATABASE_URL required")
        exit(1)

    conn = psycopg2.connect(args.db_url)

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

    conn.close()
