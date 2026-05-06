"""UMRS Multi-tenant schema isolation.

Provides per-project schema isolation using Postgres schemas.
Each project gets its own schema with the standard UMRS tables,
while a shared ``public`` schema holds cross-project metadata.

Designed for Alembic migration compatibility. Each tenant schema
runs the same migration set independently.

No dependency on r4.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PROVISIONING = "provisioning"
    DEPROVISIONING = "deprovisioning"


@dataclass
class TenantSpec:
    """Specification for a tenant (project)."""

    tenant_id: str = ""
    project_id: str = ""
    project_name: str = ""
    schema_name: str = ""
    owner_id: str = ""
    status: TenantStatus = TenantStatus.PROVISIONING
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tenant_id and self.project_id:
            self.tenant_id = hashlib.sha256(
                self.project_id.encode()
            ).hexdigest()[:16]
        if not self.schema_name and self.project_id:
            # Sanitize project_id for use as schema name
            safe = re.sub(r"[^a-z0-9_]", "_", self.project_id.lower())
            self.schema_name = f"tenant_{safe}"


# Default resource limits per tier
_DEFAULT_LIMITS = {
    "free": {
        "max_documents": 1000,
        "max_storage_mb": 100,
        "max_connectors": 2,
        "max_scheduled_jobs": 1,
    },
    "standard": {
        "max_documents": 50000,
        "max_storage_mb": 5000,
        "max_connectors": 10,
        "max_scheduled_jobs": 10,
    },
    "enterprise": {
        "max_documents": -1,  # unlimited
        "max_storage_mb": -1,
        "max_connectors": -1,
        "max_scheduled_jobs": -1,
    },
}


# ── Schema SQL templates ────────────────────────────────────────────────────

SCHEMA_CREATE_SQL = """
CREATE SCHEMA IF NOT EXISTS {schema_name};

-- Documents table
CREATE TABLE IF NOT EXISTS {schema_name}.documents (
    doc_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT,
    content TEXT,
    content_type TEXT DEFAULT 'text/plain',
    metadata JSONB DEFAULT '{{}}',
    url TEXT,
    last_modified TIMESTAMPTZ,
    size_bytes BIGINT DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table (for RAG)
CREATE TABLE IF NOT EXISTS {schema_name}.chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES {schema_name}.documents(doc_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INT NOT NULL,
    metadata JSONB DEFAULT '{{}}',
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Entities table (knowledge graph)
CREATE TABLE IF NOT EXISTS {schema_name}.entities (
    entity_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases TEXT[] DEFAULT ARRAY[]::TEXT[],
    mention_count INT DEFAULT 1,
    metadata JSONB DEFAULT '{{}}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Relations table (knowledge graph)
CREATE TABLE IF NOT EXISTS {schema_name}.relations (
    id SERIAL PRIMARY KEY,
    source_entity_id TEXT REFERENCES {schema_name}.entities(entity_id),
    target_entity_id TEXT REFERENCES {schema_name}.entities(entity_id),
    relation_type TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    source_doc_ids TEXT[] DEFAULT ARRAY[]::TEXT[],
    metadata JSONB DEFAULT '{{}}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Connector sync state
CREATE TABLE IF NOT EXISTS {schema_name}.connector_sync (
    connector_name TEXT PRIMARY KEY,
    last_sync TIMESTAMPTZ,
    cursor TEXT,
    docs_synced INT DEFAULT 0,
    errors INT DEFAULT 0,
    metadata JSONB DEFAULT '{{}}'
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_{schema_name_safe}_docs_source
    ON {schema_name}.documents(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_{schema_name_safe}_chunks_doc
    ON {schema_name}.chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_{schema_name_safe}_entities_type
    ON {schema_name}.entities(entity_type);
"""

SCHEMA_DROP_SQL = """
DROP SCHEMA IF EXISTS {schema_name} CASCADE;
"""


# ── Tenant manager ──────────────────────────────────────────────────────────


class TenantManager:
    """Manages tenant lifecycle and schema isolation.

    In production, this executes SQL against Postgres. This
    implementation provides the in-memory API contract.
    """

    def __init__(self) -> None:
        self._tenants: Dict[str, TenantSpec] = {}

    def provision(
        self,
        project_id: str,
        project_name: str,
        owner_id: str,
        tier: str = "standard",
    ) -> TenantSpec:
        """Provision a new tenant schema."""
        spec = TenantSpec(
            project_id=project_id,
            project_name=project_name,
            owner_id=owner_id,
            created_at=datetime.utcnow(),
            resource_limits=_DEFAULT_LIMITS.get(tier, _DEFAULT_LIMITS["standard"]),
        )

        # In production: execute SCHEMA_CREATE_SQL
        logger.info(
            "Provisioning tenant schema: %s for project %s",
            spec.schema_name,
            project_id,
        )

        spec.status = TenantStatus.ACTIVE
        self._tenants[spec.tenant_id] = spec
        return spec

    def get(self, tenant_id: str) -> Optional[TenantSpec]:
        return self._tenants.get(tenant_id)

    def get_by_project(self, project_id: str) -> Optional[TenantSpec]:
        for t in self._tenants.values():
            if t.project_id == project_id:
                return t
        return None

    def list_tenants(
        self, owner_id: Optional[str] = None
    ) -> List[TenantSpec]:
        results: List[TenantSpec] = []
        for t in self._tenants.values():
            if owner_id and t.owner_id != owner_id:
                continue
            results.append(t)
        return results

    def suspend(self, tenant_id: str) -> bool:
        t = self._tenants.get(tenant_id)
        if t:
            t.status = TenantStatus.SUSPENDED
            return True
        return False

    def deprovision(self, tenant_id: str) -> bool:
        """Remove a tenant schema."""
        t = self._tenants.get(tenant_id)
        if not t:
            return False
        t.status = TenantStatus.DEPROVISIONING
        # In production: execute SCHEMA_DROP_SQL
        logger.info("Deprovisioning tenant schema: %s", t.schema_name)
        del self._tenants[tenant_id]
        return True

    def get_schema_sql(self, tenant_id: str) -> Optional[str]:
        """Generate the CREATE SCHEMA SQL for a tenant."""
        t = self._tenants.get(tenant_id)
        if not t:
            return None
        safe = re.sub(r"[^a-z0-9_]", "_", t.schema_name)
        return SCHEMA_CREATE_SQL.format(
            schema_name=t.schema_name,
            schema_name_safe=safe,
        )

    def check_limits(self, tenant_id: str, resource: str, current: int) -> bool:
        """Check if a tenant is within resource limits."""
        t = self._tenants.get(tenant_id)
        if not t:
            return False
        limit = t.resource_limits.get(resource, -1)
        if limit == -1:  # unlimited
            return True
        return current < limit
