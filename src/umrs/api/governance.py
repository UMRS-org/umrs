"""Resource governance — ownership tracking and quota enforcement.

Provides durable Postgres tables for:
- resource_ownership : who owns which resource (endpoint, model, MCP server, etc.)
- resource_quotas    : per-project limits on resource types

This is the foundation layer for multi-tenant governance. Higher-level
policy (who can create what, approval flows) will build on top.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

GOVERNANCE_SCHEMA_VERSION = 1

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

CREATE_GOVERNANCE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS governance_schema_version (
    id INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CHECK (id = 1)
);
"""


def ensure_governance_schema(conn) -> None:
    """Create governance tables and indexes if they don't exist."""
    cur = conn.cursor()
    try:
        cur.execute(CREATE_RESOURCE_OWNERSHIP_TABLE)
        cur.execute(CREATE_RESOURCE_QUOTAS_TABLE)
        cur.execute(CREATE_GOVERNANCE_INDEXES)
        cur.execute(CREATE_GOVERNANCE_SCHEMA_VERSION_TABLE)
        cur.execute(
            "INSERT INTO governance_schema_version (id, version) VALUES (1, %s) "
            "ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version, "
            "updated_at = NOW()",
            (GOVERNANCE_SCHEMA_VERSION,),
        )
        conn.commit()
        logger.info("Governance schema ensured (version %d)", GOVERNANCE_SCHEMA_VERSION)
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


class GovernanceStore:
    """Postgres-backed resource ownership and quota store."""

    def __init__(self, conn) -> None:
        self.conn = conn

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
    ) -> None:
        """Set or update the quota for a project+resource_type."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "INSERT INTO resource_quotas (project_id, resource_type, max_count) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (project_id, resource_type) DO UPDATE SET "
                "max_count = EXCLUDED.max_count, updated_at = NOW()",
                (project_id, resource_type, max_count),
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
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT project_id, resource_type, max_count, current_count, updated_at "
                "FROM resource_quotas WHERE project_id = %s AND resource_type = %s",
                (project_id, resource_type),
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
            "project_id": row[0],
            "resource_type": row[1],
            "max_count": row[2],
            "current_count": row[3],
            "updated_at": str(row[4]),
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
                "SELECT project_id, resource_type, max_count, current_count, updated_at "
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
            }
            for r in rows
        ]
