"""Resource governance — ownership tracking and quota enforcement.

Provides durable Postgres tables for:
- resource_ownership : who owns which resource (endpoint, model, MCP server, etc.)
- resource_quotas    : per-project limits on resource types

This is the foundation layer for multi-tenant governance. Higher-level
policy (who can create what, approval flows) will build on top.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

GOVERNANCE_SCHEMA_VERSION = 4

CREATE_RESOURCE_OWNERSHIP_TABLE = """
CREATE TABLE IF NOT EXISTS resource_ownership (
    resource_type VARCHAR(64) NOT NULL,
    resource_id VARCHAR(128) NOT NULL,
    owner_id VARCHAR(128) NOT NULL,
    project_id VARCHAR(128) NOT NULL DEFAULT '',
    tier VARCHAR(32) NOT NULL DEFAULT 'user',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (resource_type, resource_id)
);
"""

CREATE_RESOURCE_QUOTAS_TABLE = """
CREATE TABLE IF NOT EXISTS resource_quotas (
    project_id VARCHAR(128) NOT NULL,
    resource_type VARCHAR(64) NOT NULL,
    max_count INTEGER NOT NULL DEFAULT 0,
    current_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (project_id, resource_type)
);
"""

CREATE_GOVERNANCE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_ownership_owner
    ON resource_ownership(owner_id);
CREATE INDEX IF NOT EXISTS idx_ownership_project
    ON resource_ownership(project_id) WHERE project_id != '';
CREATE INDEX IF NOT EXISTS idx_ownership_type
    ON resource_ownership(resource_type);
CREATE INDEX IF NOT EXISTS idx_quotas_project
    ON resource_quotas(project_id);
"""

# Resource sharing — explicit grants beyond scope-based visibility
CREATE_RESOURCE_SHARES_TABLE = """
CREATE TABLE IF NOT EXISTS resource_shares (
    share_id SERIAL PRIMARY KEY,
    resource_type VARCHAR(64) NOT NULL,
    resource_id VARCHAR(128) NOT NULL,
    granted_to VARCHAR(128) NOT NULL,
    granted_to_type VARCHAR(16) NOT NULL DEFAULT 'user',
    access_level VARCHAR(16) NOT NULL DEFAULT 'read',
    granted_by VARCHAR(128) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NULL,
    UNIQUE (resource_type, resource_id, granted_to, granted_to_type)
);
"""

CREATE_SHARES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_shares_grantee
    ON resource_shares(granted_to, granted_to_type);
CREATE INDEX IF NOT EXISTS idx_shares_resource
    ON resource_shares(resource_type, resource_id);
"""

CREATE_GOVERNANCE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS governance_schema_version (
    id INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CHECK (id = 1)
);
"""


def ensure_governance_schema(conn) -> None:
    """Create governance tables and indexes if they don't exist.

    Performs forward-only schema migration with version checking.
    Raises RuntimeError if stored version is newer than code.
    """
    cur = conn.cursor()
    try:
        # Bootstrap tables
        cur.execute(CREATE_RESOURCE_OWNERSHIP_TABLE)
        cur.execute(CREATE_RESOURCE_QUOTAS_TABLE)
        cur.execute(CREATE_GOVERNANCE_INDEXES)
        cur.execute(CREATE_RESOURCE_SHARES_TABLE)
        cur.execute(CREATE_SHARES_INDEXES)
        cur.execute(CREATE_GOVERNANCE_SCHEMA_VERSION_TABLE)
        conn.commit()

        # Check existing version
        cur.execute("SELECT version FROM governance_schema_version WHERE id = 1")
        row = cur.fetchone()
        stored_version = row[0] if row else 0

        if stored_version > GOVERNANCE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Governance schema version {stored_version} is newer than code version "
                f"{GOVERNANCE_SCHEMA_VERSION}. Cannot downgrade. "
                "Deploy the newer code version or restore from backup."
            )

        if stored_version < GOVERNANCE_SCHEMA_VERSION:
            _apply_governance_migrations(
                conn, cur, stored_version, GOVERNANCE_SCHEMA_VERSION
            )

        # Upsert version
        cur.execute(
            "INSERT INTO governance_schema_version (id, version) VALUES (1, %s) "
            "ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version, "
            "updated_at = NOW()",
            (GOVERNANCE_SCHEMA_VERSION,),
        )
        conn.commit()
        logger.info(
            "Governance schema ensured (version %d, migrated from %d)",
            GOVERNANCE_SCHEMA_VERSION,
            stored_version,
        )
    except RuntimeError:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _apply_governance_migrations(conn, cur, from_version: int, to_version: int) -> None:
    """Apply sequential governance schema migrations.

    Version 2: Added ownership lifecycle columns and orphan detection support.
    Version 3: Added warn_count to resource_quotas for soft quota warnings.
    Version 4: Added resource_shares table for explicit sharing, group_id to ownership.
    """
    migrations: dict[int, list[str]] = {
        # Version 1: initial schema (tables created above)
        # Version 2: ownership lifecycle support — no DDL needed, only
        # new store methods (transfer_owner, find_orphans, cleanup_orphans)
        3: [
            "ALTER TABLE resource_quotas ADD COLUMN IF NOT EXISTS warn_count INTEGER DEFAULT NULL"
        ],
        4: [
            "ALTER TABLE resource_ownership ADD COLUMN IF NOT EXISTS group_id VARCHAR(128) NOT NULL DEFAULT ''",
            CREATE_RESOURCE_SHARES_TABLE,
            CREATE_SHARES_INDEXES,
        ],
    }

    for version in range(from_version + 1, to_version + 1):
        sqls = migrations.get(version)
        if sqls:
            logger.info("Applying governance schema migration to version %d", version)
            for sql in sqls:
                cur.execute(sql)
            conn.commit()


class GovernanceStore:
    """Postgres-backed resource ownership and quota store."""

    def __init__(self, conn, pool=None) -> None:
        self.conn = conn
        self._pool = pool

    def _reconnect(self) -> None:
        """Swap a dead connection for a fresh pool connection (bounded).

        Original failure class 2026-09-08: umrs-server-bdrt booted while the
        postgres container cycled during converge; the boot-issued conn then
        died once ("server closed the connection"), and every later query
        from this long-lived object raised psycopg2.InterfaceError
        "connection already closed" — provisions then 500'd at check_quota
        until the server was restarted. When a pool is available, exchange
        instead of 500-ing.
        """
        if self._pool is None:
            return
        try:
            self._pool.putconn(self.conn)
        except Exception:
            try:
                self._pool.putconn(self.conn, close=True)
            except Exception:
                pass
        self.conn = self._pool.getconn()

    @staticmethod
    def _is_conn_dead(exc: Exception) -> bool:
        try:
            import psycopg2

            return isinstance(
                exc, (psycopg2.InterfaceError, psycopg2.OperationalError)
            )
        except Exception:
            return False

    # ---- Ownership ----

    def register_owner(
        self,
        resource_type: str,
        resource_id: str,
        owner_id: str,
        *,
        project_id: str = "",
        tier: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register or update ownership of a resource."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "INSERT INTO resource_ownership "
                "(resource_type, resource_id, owner_id, project_id, tier, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (resource_type, resource_id) DO UPDATE SET "
                "owner_id = EXCLUDED.owner_id, project_id = EXCLUDED.project_id, "
                "tier = EXCLUDED.tier, metadata = EXCLUDED.metadata",
                (
                    resource_type,
                    resource_id,
                    owner_id,
                    project_id,
                    tier,
                    __import__("json").dumps(metadata or {}),
                ),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def remove_owner(self, resource_type: str, resource_id: str) -> bool:
        """Remove ownership record. Returns True if a row was deleted."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "DELETE FROM resource_ownership WHERE resource_type = %s AND resource_id = %s",
                (resource_type, resource_id),
            )
            deleted = cur.rowcount > 0
            self.conn.commit()
            return deleted
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def get_owner(self, resource_type: str, resource_id: str) -> dict[str, Any] | None:
        """Look up the owner of a resource."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT resource_type, resource_id, owner_id, project_id, tier, "
                "created_at, metadata FROM resource_ownership "
                "WHERE resource_type = %s AND resource_id = %s",
                (resource_type, resource_id),
            )
            row = cur.fetchone()
        except Exception:
            self.conn.rollback()
            return None
        finally:
            cur.close()

        if not row:
            return None
        return {
            "resource_type": row[0],
            "resource_id": row[1],
            "owner_id": row[2],
            "project_id": row[3],
            "tier": row[4],
            "created_at": str(row[5]),
            "metadata": row[6] if isinstance(row[6], dict) else {},
        }

    def list_owned(
        self,
        owner_id: str,
        *,
        resource_type: str = "",
        project_id: str = "",
    ) -> list[dict[str, Any]]:
        """List resources owned by a user, optionally filtered by type or project."""
        conditions = ["owner_id = %s"]
        params: list[Any] = [owner_id]
        if resource_type:
            conditions.append("resource_type = %s")
            params.append(resource_type)
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)

        where = " AND ".join(conditions)
        cur = self.conn.cursor()
        try:
            cur.execute(
                f"SELECT resource_type, resource_id, owner_id, project_id, tier, "
                f"created_at, metadata FROM resource_ownership WHERE {where} "
                f"ORDER BY created_at DESC",
                params,
            )
            rows = cur.fetchall()
        except Exception:
            self.conn.rollback()
            return []
        finally:
            cur.close()

        return [
            {
                "resource_type": r[0],
                "resource_id": r[1],
                "owner_id": r[2],
                "project_id": r[3],
                "tier": r[4],
                "created_at": str(r[5]),
                "metadata": r[6] if isinstance(r[6], dict) else {},
            }
            for r in rows
        ]

    # ---- Quotas ----

    def set_quota(
        self,
        project_id: str,
        resource_type: str,
        max_count: int,
        *,
        warn_count: int | None = None,
    ) -> None:
        """Set or update the quota for a project+resource_type.

        Parameters
        ----------
        warn_count : int | None
            Optional soft warning threshold. When current_count reaches
            warn_count, a governance.quota_warning audit event is emitted
            but the operation is still allowed. Set to None to disable.
        """
        cur = self.conn.cursor()
        try:
            cur.execute(
                "INSERT INTO resource_quotas (project_id, resource_type, max_count, warn_count) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (project_id, resource_type) DO UPDATE SET "
                "max_count = EXCLUDED.max_count, warn_count = EXCLUDED.warn_count, "
                "updated_at = NOW()",
                (project_id, resource_type, max_count, warn_count),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def get_quota(
        self,
        project_id: str,
        resource_type: str,
    ) -> dict[str, Any] | None:
        """Get quota for a project+resource_type."""
        for attempt in (1, 2):
            cur = self.conn.cursor()
            try:
                cur.execute(
                    "SELECT project_id, resource_type, max_count, current_count, "
                    "updated_at, warn_count "
                    "FROM resource_quotas WHERE project_id = %s AND resource_type = %s",
                    (project_id, resource_type),
                )
                row = cur.fetchone()
            except Exception as exc:
                last_exc = exc
                if attempt == 1 and self._is_conn_dead(exc):
                    try:
                        self._reconnect()
                        continue
                    except Exception as recon_exc:
                        logger.warning(
                            "Governance store reconnect failed: %s", recon_exc
                        )
                        return None
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                return None
            finally:
                cur.close()

        if not row:
            return None
        return {
            "project_id": row[0],
            "resource_type": row[1],
            "max_count": row[2],
            "current_count": row[3],
            "updated_at": str(row[4]),
            "warn_count": row[5],
        }

    def increment_usage(self, project_id: str, resource_type: str) -> bool:
        """Increment current_count if under quota. Returns True if allowed."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "UPDATE resource_quotas SET current_count = current_count + 1, "
                "updated_at = NOW() "
                "WHERE project_id = %s AND resource_type = %s "
                "AND current_count < max_count",
                (project_id, resource_type),
            )
            ok = cur.rowcount > 0
            self.conn.commit()
            return ok
        except Exception:
            self.conn.rollback()
            return False
        finally:
            cur.close()

    def decrement_usage(self, project_id: str, resource_type: str) -> None:
        """Decrement current_count (min 0)."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "UPDATE resource_quotas SET "
                "current_count = GREATEST(current_count - 1, 0), "
                "updated_at = NOW() "
                "WHERE project_id = %s AND resource_type = %s",
                (project_id, resource_type),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
        finally:
            cur.close()

    def list_quotas(self, project_id: str) -> list[dict[str, Any]]:
        """List all quotas for a project."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT project_id, resource_type, max_count, current_count, "
                "updated_at, warn_count "
                "FROM resource_quotas WHERE project_id = %s ORDER BY resource_type",
                (project_id,),
            )
            rows = cur.fetchall()
        except Exception:
            self.conn.rollback()
            return []
        finally:
            cur.close()

        return [
            {
                "project_id": r[0],
                "resource_type": r[1],
                "max_count": r[2],
                "current_count": r[3],
                "updated_at": str(r[4]),
                "warn_count": r[5],
            }
            for r in rows
        ]

    # ---- Ownership lifecycle ----

    def transfer_owner(
        self,
        resource_type: str,
        resource_id: str,
        new_owner_id: str,
        *,
        new_project_id: str | None = None,
    ) -> bool:
        """Transfer ownership of a resource to a new owner.

        Optionally moves the resource to a new project. Returns True if
        a record was updated.  Uses SELECT ... FOR UPDATE to serialize
        concurrent transfers of the same resource.
        """
        cur = self.conn.cursor()
        try:
            # Lock the row to prevent concurrent transfer races
            cur.execute(
                "SELECT resource_id FROM resource_ownership "
                "WHERE resource_type = %s AND resource_id = %s FOR UPDATE",
                (resource_type, resource_id),
            )
            if cur.fetchone() is None:
                self.conn.rollback()
                return False

            if new_project_id is not None:
                cur.execute(
                    "UPDATE resource_ownership SET owner_id = %s, project_id = %s "
                    "WHERE resource_type = %s AND resource_id = %s",
                    (new_owner_id, new_project_id, resource_type, resource_id),
                )
            else:
                cur.execute(
                    "UPDATE resource_ownership SET owner_id = %s "
                    "WHERE resource_type = %s AND resource_id = %s",
                    (new_owner_id, resource_type, resource_id),
                )
            updated = cur.rowcount > 0
            self.conn.commit()
            return updated
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def list_by_project(
        self,
        project_id: str,
        *,
        resource_type: str = "",
    ) -> list[dict[str, Any]]:
        """List all resources belonging to a project."""
        conditions = ["project_id = %s"]
        params: list[Any] = [project_id]
        if resource_type:
            conditions.append("resource_type = %s")
            params.append(resource_type)

        where = " AND ".join(conditions)
        cur = self.conn.cursor()
        try:
            cur.execute(
                f"SELECT resource_type, resource_id, owner_id, project_id, tier, "
                f"created_at, metadata FROM resource_ownership WHERE {where} "
                f"ORDER BY created_at DESC",
                params,
            )
            rows = cur.fetchall()
        except Exception:
            self.conn.rollback()
            return []
        finally:
            cur.close()

        return [
            {
                "resource_type": r[0],
                "resource_id": r[1],
                "owner_id": r[2],
                "project_id": r[3],
                "tier": r[4],
                "created_at": str(r[5]),
                "metadata": r[6] if isinstance(r[6], dict) else {},
            }
            for r in rows
        ]

    def find_orphans(
        self,
        resource_type: str,
        known_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Find ownership records whose resource_id is not in the known set.

        This is the primary orphan detection mechanism. Compare ownership
        records against the live registry (endpoints, MCP servers, etc.)
        to find stale ownership entries.
        """
        if not known_ids:
            # If no known IDs, all records of this type are orphans
            return self.list_by_project("", resource_type=resource_type)

        cur = self.conn.cursor()
        try:
            placeholders = ",".join(["%s"] * len(known_ids))
            cur.execute(
                f"SELECT resource_type, resource_id, owner_id, project_id, tier, "
                f"created_at, metadata FROM resource_ownership "
                f"WHERE resource_type = %s AND resource_id NOT IN ({placeholders})",
                [resource_type, *known_ids],
            )
            rows = cur.fetchall()
        except Exception:
            self.conn.rollback()
            return []
        finally:
            cur.close()

        return [
            {
                "resource_type": r[0],
                "resource_id": r[1],
                "owner_id": r[2],
                "project_id": r[3],
                "tier": r[4],
                "created_at": str(r[5]),
                "metadata": r[6] if isinstance(r[6], dict) else {},
            }
            for r in rows
        ]

    def cleanup_orphans(
        self,
        resource_type: str,
        known_ids: list[str],
    ) -> int:
        """Remove ownership records not in the known set. Returns count deleted."""
        if not known_ids:
            cur = self.conn.cursor()
            try:
                cur.execute(
                    "DELETE FROM resource_ownership WHERE resource_type = %s",
                    (resource_type,),
                )
                count = cur.rowcount
                self.conn.commit()
                return count
            except Exception:
                self.conn.rollback()
                return 0
            finally:
                cur.close()

        cur = self.conn.cursor()
        try:
            placeholders = ",".join(["%s"] * len(known_ids))
            cur.execute(
                f"DELETE FROM resource_ownership "
                f"WHERE resource_type = %s AND resource_id NOT IN ({placeholders})",
                [resource_type, *known_ids],
            )
            count = cur.rowcount
            self.conn.commit()
            return count
        except Exception:
            self.conn.rollback()
            return 0
        finally:
            cur.close()

    # ---- Resource Sharing ----

    def create_share(
        self,
        resource_type: str,
        resource_id: str,
        granted_to: str,
        *,
        granted_to_type: str = "user",
        access_level: str = "read",
        granted_by: str = "",
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Create or update an explicit resource share.

        Parameters
        ----------
        granted_to_type : str
            "user" or "group" — the kind of entity receiving access.
        access_level : str
            "read" or "write".
        expires_at : str | None
            ISO 8601 timestamp for share expiry, or None for permanent.
        """
        cur = self.conn.cursor()
        try:
            cur.execute(
                "INSERT INTO resource_shares "
                "(resource_type, resource_id, granted_to, granted_to_type, "
                "access_level, granted_by, expires_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (resource_type, resource_id, granted_to, granted_to_type) "
                "DO UPDATE SET access_level = EXCLUDED.access_level, "
                "granted_by = EXCLUDED.granted_by, expires_at = EXCLUDED.expires_at "
                "RETURNING share_id, created_at",
                (
                    resource_type,
                    resource_id,
                    granted_to,
                    granted_to_type,
                    access_level,
                    granted_by,
                    expires_at,
                ),
            )
            row = cur.fetchone()
            self.conn.commit()
            return {
                "share_id": row[0],
                "resource_type": resource_type,
                "resource_id": resource_id,
                "granted_to": granted_to,
                "granted_to_type": granted_to_type,
                "access_level": access_level,
                "granted_by": granted_by,
                "created_at": str(row[1]),
                "expires_at": expires_at,
            }
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def revoke_share(
        self,
        resource_type: str,
        resource_id: str,
        granted_to: str,
        *,
        granted_to_type: str = "user",
    ) -> bool:
        """Revoke a specific share. Returns True if deleted."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "DELETE FROM resource_shares "
                "WHERE resource_type = %s AND resource_id = %s "
                "AND granted_to = %s AND granted_to_type = %s",
                (resource_type, resource_id, granted_to, granted_to_type),
            )
            deleted = cur.rowcount > 0
            self.conn.commit()
            return deleted
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def revoke_all_shares(
        self,
        resource_type: str,
        resource_id: str,
    ) -> int:
        """Revoke all shares for a resource. Returns count deleted."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "DELETE FROM resource_shares "
                "WHERE resource_type = %s AND resource_id = %s",
                (resource_type, resource_id),
            )
            count = cur.rowcount
            self.conn.commit()
            return count
        except Exception:
            self.conn.rollback()
            return 0
        finally:
            cur.close()

    def list_shares_for_resource(
        self,
        resource_type: str,
        resource_id: str,
    ) -> list[dict[str, Any]]:
        """List all active shares for a resource."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT share_id, resource_type, resource_id, granted_to, "
                "granted_to_type, access_level, granted_by, created_at, expires_at "
                "FROM resource_shares "
                "WHERE resource_type = %s AND resource_id = %s "
                "AND (expires_at IS NULL OR expires_at > NOW()) "
                "ORDER BY created_at",
                (resource_type, resource_id),
            )
            rows = cur.fetchall()
        except Exception:
            self.conn.rollback()
            return []
        finally:
            cur.close()

        return [
            {
                "share_id": r[0],
                "resource_type": r[1],
                "resource_id": r[2],
                "granted_to": r[3],
                "granted_to_type": r[4],
                "access_level": r[5],
                "granted_by": r[6],
                "created_at": str(r[7]),
                "expires_at": str(r[8]) if r[8] else None,
            }
            for r in rows
        ]

    def list_shares_for_user(
        self,
        user_id: str,
        *,
        user_groups: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List all resources shared with a user (directly or via groups).

        Includes both direct user shares and group-inherited shares.
        Filters out expired shares.
        """
        cur = self.conn.cursor()
        try:
            # Build conditions: direct user shares + group shares
            conditions = [
                "(granted_to = %s AND granted_to_type = 'user')",
            ]
            params: list[Any] = [user_id]

            if user_groups:
                placeholders = ",".join(["%s"] * len(user_groups))
                conditions.append(
                    f"(granted_to IN ({placeholders}) AND granted_to_type = 'group')"
                )
                params.extend(user_groups)

            where = " OR ".join(conditions)
            cur.execute(
                f"SELECT share_id, resource_type, resource_id, granted_to, "
                f"granted_to_type, access_level, granted_by, created_at, expires_at "
                f"FROM resource_shares "
                f"WHERE ({where}) AND (expires_at IS NULL OR expires_at > NOW()) "
                f"ORDER BY created_at DESC",
                params,
            )
            rows = cur.fetchall()
        except Exception:
            self.conn.rollback()
            return []
        finally:
            cur.close()

        return [
            {
                "share_id": r[0],
                "resource_type": r[1],
                "resource_id": r[2],
                "granted_to": r[3],
                "granted_to_type": r[4],
                "access_level": r[5],
                "granted_by": r[6],
                "created_at": str(r[7]),
                "expires_at": str(r[8]) if r[8] else None,
            }
            for r in rows
        ]

    def has_share_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        *,
        user_groups: list[str] | None = None,
        required_level: str = "read",
    ) -> bool:
        """Check if a user has explicit share access to a resource.

        Parameters
        ----------
        required_level : str
            "read" — any access level grants read.
            "write" — only "write" access grants this.
        """
        cur = self.conn.cursor()
        try:
            # Build grantee conditions
            conditions = [
                "(granted_to = %s AND granted_to_type = 'user')",
            ]
            params: list[Any] = [user_id]

            if user_groups:
                placeholders = ",".join(["%s"] * len(user_groups))
                conditions.append(
                    f"(granted_to IN ({placeholders}) AND granted_to_type = 'group')"
                )
                params.extend(user_groups)

            grantee_where = " OR ".join(conditions)

            # Access level filter
            if required_level == "write":
                level_filter = "AND access_level = 'write'"
            else:
                level_filter = ""  # Any access level grants read

            cur.execute(
                f"SELECT 1 FROM resource_shares "
                f"WHERE resource_type = %s AND resource_id = %s "
                f"AND ({grantee_where}) "
                f"AND (expires_at IS NULL OR expires_at > NOW()) "
                f"{level_filter} "
                f"LIMIT 1",
                [resource_type, resource_id, *params],
            )
            return cur.fetchone() is not None
        except Exception:
            self.conn.rollback()
            return False
        finally:
            cur.close()


# ---------------------------------------------------------------------------
# Module-level accessor (same pattern as JobStore, MCPServerStore)
# ---------------------------------------------------------------------------

_governance_store: GovernanceStore | None = None


def set_governance_store(store: GovernanceStore) -> None:
    """Register the global GovernanceStore instance (called during sidecar init)."""
    global _governance_store
    _governance_store = store


def get_governance_store() -> GovernanceStore | None:
    """Return the governance store, or None if not initialized."""
    return _governance_store


# ---------------------------------------------------------------------------
# Enforcement helpers — called from resource creation/deletion paths
# ---------------------------------------------------------------------------


def guard_ingest(
    project_id: str,
    source_kind: str,
    *,
    user_id: str,
    details: dict[str, Any],
) -> None:
    """Unified ingest governance (R6.4): quota check + audit trail.

    Single entry point for every 'put knowledge into the platform' path
    (rag upload, durable KNOWLEDGE_INGEST jobs). Quota is keyed on the
    shared 'rag_documents' family (existing resource_type — relabeling
    would lose accumulated quota state).
    """
    import logging as _log

    check_quota(project_id, "rag_documents")
    from .audit_events import emit_audit

    emit_audit(
        f"ingest.{source_kind}",
        user_id=user_id,
        project_id=project_id,
        details=details,
    )
    _log.getLogger(__name__).debug(
        "guard_ingest: kind=%s project=%s user=%s", source_kind, project_id, user_id
    )


def check_quota(project_id: str, resource_type: str) -> None:
    """Check if the project is within quota for resource_type.

    Raises HTTPException(409) if quota exceeded, or does nothing if no
    governance store or no quota is configured. Emits a governance audit
    event on quota violation.
    """
    store = _governance_store
    if store is None or not project_id:
        return  # governance not active — allow

    quota = store.get_quota(project_id, resource_type)
    if quota is None:
        return  # no quota defined — allow

    # Soft quota warning (warn_count)
    warn_count = quota.get("warn_count")
    if (
        warn_count is not None
        and quota["current_count"] >= warn_count
        and quota["current_count"] < quota["max_count"]
    ):
        from .audit_events import emit_audit

        emit_audit(
            "governance.quota_warning",
            project_id=project_id,
            outcome="success",
            details={
                "resource_type": resource_type,
                "current": quota["current_count"],
                "warn_at": warn_count,
                "max": quota["max_count"],
            },
        )

    if quota["current_count"] >= quota["max_count"]:
        from .audit_events import emit_audit

        emit_audit(
            "governance.quota_exceeded",
            project_id=project_id,
            details={
                "resource_type": resource_type,
                "current": quota["current_count"],
                "max": quota["max_count"],
            },
        )
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Quota exceeded: project '{project_id}' has "
                f"{quota['current_count']}/{quota['max_count']} {resource_type}. "
                "Contact admin to increase quota."
            ),
        )


def record_resource_created(
    resource_type: str,
    resource_id: str,
    owner_id: str,
    *,
    project_id: str = "",
    tier: str = "user",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record ownership and increment quota usage for a newly created resource.

    Best-effort — never raises on governance failures.
    Emits governance.resource_created audit event.
    """
    store = _governance_store
    if store is None:
        return

    try:
        store.register_owner(
            resource_type,
            resource_id,
            owner_id,
            project_id=project_id,
            tier=tier,
            metadata=metadata,
        )
    except Exception:
        logger.debug(
            "Governance: failed to record ownership for %s/%s",
            resource_type,
            resource_id,
            exc_info=True,
        )

    if project_id:
        try:
            store.increment_usage(project_id, resource_type)
        except Exception:
            logger.debug(
                "Governance: failed to increment quota for %s/%s",
                project_id,
                resource_type,
                exc_info=True,
            )

    try:
        from .audit_events import emit_audit

        emit_audit(
            "governance.resource_created",
            actor=owner_id,
            project_id=project_id,
            scope=tier,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )
    except Exception:
        pass


def record_resource_deleted(
    resource_type: str,
    resource_id: str,
    project_id: str = "",
) -> None:
    """Remove ownership and decrement quota usage for a deleted resource.

    Best-effort — never raises on governance failures.
    Emits governance.resource_deleted audit event.
    """
    store = _governance_store
    if store is None:
        return

    try:
        store.remove_owner(resource_type, resource_id)
    except Exception:
        logger.debug(
            "Governance: failed to remove ownership for %s/%s",
            resource_type,
            resource_id,
            exc_info=True,
        )

    if project_id:
        try:
            store.decrement_usage(project_id, resource_type)
        except Exception:
            logger.debug(
                "Governance: failed to decrement quota for %s/%s",
                project_id,
                resource_type,
                exc_info=True,
            )

    try:
        from .audit_events import emit_audit

        emit_audit(
            "governance.resource_deleted",
            project_id=project_id,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )
    except Exception:
        pass
