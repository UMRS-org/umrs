"""Postgres-backed tenant registry (durable K1 implementation).

Implements the same interface as the in-memory :class:`TenantManager`
against the ``sat_tenants`` table (migration 0060). The in-memory manager
remains the API contract reference.

Per-tenant schema DDL is opt-in: pass ``execute_schema_ddl=True`` to run
SCHEMA_CREATE_SQL / SCHEMA_DROP_SQL at provision/deprovision time. The
default only maintains the registry — tenant data-plane provisioning is a
separate concern (X1/X3 track).
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import (
    SCHEMA_CREATE_SQL,
    SCHEMA_DROP_SQL,
    _DEFAULT_LIMITS,
    TenantManager,
    TenantSpec,
    TenantStatus,
)

logger = logging.getLogger(__name__)

_COLUMNS = (
    "tenant_id, project_id, project_name, schema_name, owner_id, status, "
    "created_at, metadata, resource_limits"
)


def _row_to_spec(row: dict[str, Any]) -> TenantSpec:
    metadata = row.get("metadata") or {}
    resource_limits = row.get("resource_limits") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if isinstance(resource_limits, str):
        resource_limits = json.loads(resource_limits)
    return TenantSpec(
        tenant_id=str(row["tenant_id"]),
        project_id=str(row["project_id"]),
        project_name=str(row["project_name"]),
        schema_name=str(row["schema_name"]),
        owner_id=str(row["owner_id"]),
        status=TenantStatus(str(row["status"])),
        created_at=row.get("created_at"),
        metadata=dict(metadata),
        resource_limits=dict(resource_limits),
    )


class PgTenantManager(TenantManager):
    """Durable tenant registry backed by Postgres."""

    def __init__(self, db_pool: Any, *, execute_schema_ddl: bool = False) -> None:
        # Skip the in-memory dict initialization — state lives in Postgres.
        self._pool = db_pool
        self._execute_schema_ddl = execute_schema_ddl
        self._tenants: Dict[str, TenantSpec] = {}

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _cursor(self):
        from psycopg2.extras import RealDictCursor

        with self._conn() as conn:
            yield conn.cursor(cursor_factory=RealDictCursor)

    def provision(
        self,
        project_id: str,
        project_name: str,
        owner_id: str,
        tier: str = "standard",
    ) -> TenantSpec:
        """Register a tenant (and optionally execute the schema DDL)."""
        spec = TenantSpec(
            project_id=project_id,
            project_name=project_name,
            owner_id=owner_id,
            created_at=datetime.utcnow(),
            resource_limits=_DEFAULT_LIMITS.get(tier, _DEFAULT_LIMITS["standard"]),
        )
        spec.status = TenantStatus.ACTIVE
        logger.info(
            "Provisioning tenant schema: %s for project %s",
            spec.schema_name,
            project_id,
        )
        with self._cursor() as cur:
            if self._execute_schema_ddl:
                self._execute_schema_sql(cur, spec, SCHEMA_CREATE_SQL)
            cur.execute(
                f"INSERT INTO sat_tenants ({_COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb) "
                "ON CONFLICT (tenant_id) DO UPDATE SET "
                "project_name = excluded.project_name, "
                "owner_id = excluded.owner_id, "
                "status = excluded.status, "
                "resource_limits = excluded.resource_limits",
                (
                    spec.tenant_id,
                    spec.project_id,
                    spec.project_name,
                    spec.schema_name,
                    spec.owner_id,
                    spec.status.value,
                    spec.created_at,
                    json.dumps(spec.metadata),
                    json.dumps(spec.resource_limits),
                ),
            )
        return spec

    def get(self, tenant_id: str) -> Optional[TenantSpec]:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM sat_tenants WHERE tenant_id = %s",
                (tenant_id,),
            )
            row = cur.fetchone()
        return _row_to_spec(row) if row else None

    def get_by_project(self, project_id: str) -> Optional[TenantSpec]:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM sat_tenants WHERE project_id = %s",
                (project_id,),
            )
            row = cur.fetchone()
        return _row_to_spec(row) if row else None

    def list_tenants(self, owner_id: Optional[str] = None) -> List[TenantSpec]:
        with self._cursor() as cur:
            if owner_id:
                cur.execute(
                    f"SELECT {_COLUMNS} FROM sat_tenants WHERE owner_id = %s",
                    (owner_id,),
                )
            else:
                cur.execute(f"SELECT {_COLUMNS} FROM sat_tenants")
            return [_row_to_spec(row) for row in cur.fetchall()]

    def suspend(self, tenant_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE sat_tenants SET status = %s WHERE tenant_id = %s",
                (TenantStatus.SUSPENDED.value, tenant_id),
            )
            return cur.rowcount > 0

    def deprovision(self, tenant_id: str) -> bool:
        """Remove the tenant (and optionally drop its schema)."""
        spec = self.get(tenant_id)
        if not spec:
            return False
        logger.info("Deprovisioning tenant schema: %s", spec.schema_name)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE sat_tenants SET status = %s WHERE tenant_id = %s",
                (TenantStatus.DEPROVISIONING.value, tenant_id),
            )
            if self._execute_schema_ddl:
                self._execute_schema_sql(cur, spec, SCHEMA_DROP_SQL)
            cur.execute("DELETE FROM sat_tenants WHERE tenant_id = %s", (tenant_id,))
            return True

    def check_limits(self, tenant_id: str, resource: str, current: int) -> bool:
        """Check if a tenant is within resource limits (parity with in-memory)."""
        spec = self.get(tenant_id)
        if spec is None:
            return False
        limit = spec.resource_limits.get(resource, -1)
        if limit == -1:  # unlimited
            return True
        return current < limit

    def get_schema_sql(self, tenant_id: str) -> Optional[str]:
        """Generate the CREATE SCHEMA SQL for a tenant (parity with in-memory)."""
        spec = self.get(tenant_id)
        if spec is None:
            return None
        from . import SCHEMA_CREATE_SQL

        safe = re.sub(r"[^a-z0-9_]", "_", spec.schema_name)
        return SCHEMA_CREATE_SQL.format(
            schema_name=spec.schema_name,
            schema_name_safe=safe,
        )

    def _execute_schema_sql(cur, spec: TenantSpec, template: str) -> None:
        safe = re.sub(r"[^a-z0-9_]", "_", spec.schema_name)
        cur.execute(
            template.format(schema_name=spec.schema_name, schema_name_safe=safe)
        )
