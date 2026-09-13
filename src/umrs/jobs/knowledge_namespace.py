"""Knowledge namespace isolation for project- and experiment-scoped partitions (MEM-C4).

Extends the ``namespace`` column added by migration ``0035_rag_namespace.sql``
with structured project-scoped and experiment-scoped knowledge partitions.
Workflows can declare which namespaces are visible via visibility policies.

This module lives in ``umrs.jobs`` because it reads from the same Postgres
database and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Namespace scope constants
SCOPE_DEFAULT = "default"
SCOPE_PROJECT = "project"
SCOPE_EXPERIMENT = "experiment"
SCOPE_USER = "user"
VALID_SCOPES = {SCOPE_DEFAULT, SCOPE_PROJECT, SCOPE_EXPERIMENT, SCOPE_USER}


@dataclass
class KnowledgeNamespace:
    """A knowledge namespace partition."""

    namespace_id: str
    name: str
    scope: str  # default | project | experiment | user
    project_id: str | None = None
    experiment_id: str | None = None
    owner_id: str | None = None
    description: str = ""
    is_active: bool = True
    item_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace_id": self.namespace_id,
            "name": self.name,
            "scope": self.scope,
            "project_id": self.project_id,
            "experiment_id": self.experiment_id,
            "owner_id": self.owner_id,
            "description": self.description,
            "is_active": self.is_active,
            "item_count": self.item_count,
            "created_at": self.created_at.isoformat()
            if isinstance(self.created_at, datetime)
            else str(self.created_at),
        }


@dataclass
class NamespaceVisibility:
    """Declares which namespaces a workflow or job can access."""

    policy_id: str
    job_id: str | None = None
    workflow_name: str | None = None
    visible_namespaces: list[str] = field(default_factory=list)
    include_default: bool = True
    include_project: bool = True
    include_experiment: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "job_id": self.job_id,
            "workflow_name": self.workflow_name,
            "visible_namespaces": self.visible_namespaces,
            "include_default": self.include_default,
            "include_project": self.include_project,
            "include_experiment": self.include_experiment,
            "created_at": self.created_at.isoformat()
            if isinstance(self.created_at, datetime)
            else str(self.created_at),
        }

    def resolve_namespaces(
        self, project_id: str | None = None, experiment_id: str | None = None
    ) -> list[str]:
        """Return the list of namespace names this policy allows access to."""
        ns: list[str] = list(self.visible_namespaces)
        if self.include_default and SCOPE_DEFAULT not in ns:
            ns.append(SCOPE_DEFAULT)
        if self.include_project and project_id:
            project_ns = f"project:{project_id}"
            if project_ns not in ns:
                ns.append(project_ns)
        if self.include_experiment and experiment_id:
            exp_ns = f"experiment:{experiment_id}"
            if exp_ns not in ns:
                ns.append(exp_ns)
        return ns


class KnowledgeNamespaceStore:
    """Store for knowledge namespace partitions.

    Postgres-backed with in-memory fallback for dev/test.
    """

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url = db_url
        self._psycopg2 = None
        if db_url:
            try:
                import psycopg2

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning("psycopg2 unavailable; using in-memory store")
        # In-memory fallback
        self._namespaces: dict[str, KnowledgeNamespace] = {}
        self._policies: dict[str, NamespaceVisibility] = {}

    @property
    def db_enabled(self) -> bool:
        return self._psycopg2 is not None and self._db_url is not None

    def _get_conn(self):
        from umrs.shared.pg_settings import pg_connect_kwargs

        return self._psycopg2.connect(self._db_url, **pg_connect_kwargs("knowledge-namespace"))

    # =========================================================================
    # Namespace CRUD
    # =========================================================================

    def create_namespace(
        self,
        name: str,
        scope: str,
        project_id: str | None = None,
        experiment_id: str | None = None,
        owner_id: str | None = None,
        description: str = "",
    ) -> KnowledgeNamespace:
        """Create a new knowledge namespace."""
        if scope not in VALID_SCOPES:
            raise ValueError(f"Invalid scope: {scope}. Must be one of {VALID_SCOPES}")

        ns = KnowledgeNamespace(
            namespace_id=f"ns-{uuid.uuid4().hex[:12]}",
            name=name,
            scope=scope,
            project_id=project_id,
            experiment_id=experiment_id,
            owner_id=owner_id,
            description=description,
        )

        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO knowledge_namespaces
                            (namespace_id, name, scope, project_id,
                             experiment_id, owner_id, description,
                             is_active, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            ns.namespace_id,
                            ns.name,
                            ns.scope,
                            ns.project_id,
                            ns.experiment_id,
                            ns.owner_id,
                            ns.description,
                            ns.is_active,
                            ns.created_at,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        else:
            self._namespaces[ns.namespace_id] = ns

        return ns

    def get_namespace(self, namespace_id: str) -> KnowledgeNamespace | None:
        """Get a namespace by ID."""
        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM knowledge_namespaces WHERE namespace_id = %s",
                        (namespace_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return self._row_to_namespace(cur, row)
            finally:
                conn.close()
        return self._namespaces.get(namespace_id)

    def list_namespaces(
        self,
        project_id: str | None = None,
        scope: str | None = None,
        active_only: bool = True,
    ) -> list[KnowledgeNamespace]:
        """List namespaces with optional filters."""
        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    conditions = []
                    params: list[Any] = []
                    if project_id:
                        conditions.append("project_id = %s")
                        params.append(project_id)
                    if scope:
                        conditions.append("scope = %s")
                        params.append(scope)
                    if active_only:
                        conditions.append("is_active = TRUE")

                    where = ""
                    if conditions:
                        where = "WHERE " + " AND ".join(conditions)

                    cur.execute(
                        f"SELECT * FROM knowledge_namespaces {where} ORDER BY created_at",
                        params,
                    )
                    return [self._row_to_namespace(cur, r) for r in cur.fetchall()]
            finally:
                conn.close()

        results = list(self._namespaces.values())
        if project_id:
            results = [ns for ns in results if ns.project_id == project_id]
        if scope:
            results = [ns for ns in results if ns.scope == scope]
        if active_only:
            results = [ns for ns in results if ns.is_active]
        return sorted(results, key=lambda ns: str(ns.created_at))

    def deactivate_namespace(self, namespace_id: str) -> bool:
        """Soft-delete a namespace."""
        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE knowledge_namespaces SET is_active = FALSE WHERE namespace_id = %s",
                        (namespace_id,),
                    )
                    updated = cur.rowcount > 0
                conn.commit()
                return updated
            finally:
                conn.close()

        ns = self._namespaces.get(namespace_id)
        if ns:
            ns.is_active = False
            return True
        return False

    def get_namespace_item_count(self, namespace_name: str) -> int:
        """Count memory_items in a given namespace."""
        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM memory_items WHERE namespace = %s",
                        (namespace_name,),
                    )
                    return cur.fetchone()[0]
            finally:
                conn.close()
        return 0

    # =========================================================================
    # Visibility policies
    # =========================================================================

    def create_visibility_policy(
        self,
        job_id: str | None = None,
        workflow_name: str | None = None,
        visible_namespaces: list[str] | None = None,
        include_default: bool = True,
        include_project: bool = True,
        include_experiment: bool = False,
    ) -> NamespaceVisibility:
        """Create a visibility policy for a job or workflow."""
        policy = NamespaceVisibility(
            policy_id=f"pol-{uuid.uuid4().hex[:12]}",
            job_id=job_id,
            workflow_name=workflow_name,
            visible_namespaces=visible_namespaces or [],
            include_default=include_default,
            include_project=include_project,
            include_experiment=include_experiment,
        )

        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO namespace_visibility_policies
                            (policy_id, job_id, workflow_name,
                             visible_namespaces, include_default,
                             include_project, include_experiment, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            policy.policy_id,
                            policy.job_id,
                            policy.workflow_name,
                            ",".join(policy.visible_namespaces),
                            policy.include_default,
                            policy.include_project,
                            policy.include_experiment,
                            policy.created_at,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        else:
            self._policies[policy.policy_id] = policy

        return policy

    def get_visibility_policy(
        self,
        job_id: str | None = None,
        workflow_name: str | None = None,
    ) -> NamespaceVisibility | None:
        """Get the visibility policy for a job or workflow."""
        if self.db_enabled:
            return self._db_get_visibility_policy(job_id, workflow_name)
        for policy in self._policies.values():
            if job_id and policy.job_id == job_id:
                return policy
            if workflow_name and policy.workflow_name == workflow_name:
                return policy
        return None

    def _db_get_visibility_policy(
        self, job_id: str | None, workflow_name: str | None
    ) -> NamespaceVisibility | None:
        """Latest policy row keyed by job (preferred) or workflow name."""
        if job_id:
            column, key = "job_id", job_id
        elif workflow_name:
            column, key = "workflow_name", workflow_name
        else:
            return None
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM namespace_visibility_policies WHERE {column} = %s "
                    "ORDER BY created_at DESC LIMIT 1",
                    (key,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_policy(cur, row)
        finally:
            conn.close()

    def resolve_visible_namespaces(
        self,
        job_id: str | None = None,
        workflow_name: str | None = None,
        project_id: str | None = None,
        experiment_id: str | None = None,
    ) -> list[str]:
        """Resolve the set of namespaces visible to a job or workflow.

        Falls back to default visibility (default + project) if no policy exists.
        """
        policy = self.get_visibility_policy(job_id=job_id, workflow_name=workflow_name)
        if policy:
            return policy.resolve_namespaces(project_id, experiment_id)

        # Default: include default and project namespaces
        default_policy = NamespaceVisibility(
            policy_id="__default__",
            include_default=True,
            include_project=True,
            include_experiment=False,
        )
        return default_policy.resolve_namespaces(project_id, experiment_id)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _row_to_namespace(cur, row) -> KnowledgeNamespace:
        cols = [desc[0] for desc in cur.description]
        d = dict(zip(cols, row))
        return KnowledgeNamespace(
            namespace_id=d["namespace_id"],
            name=d["name"],
            scope=d["scope"],
            project_id=d.get("project_id"),
            experiment_id=d.get("experiment_id"),
            owner_id=d.get("owner_id"),
            description=d.get("description", ""),
            is_active=d.get("is_active", True),
            item_count=d.get("item_count", 0),
            created_at=d.get("created_at", datetime.utcnow()),
        )

    @staticmethod
    def _row_to_policy(cur, row) -> NamespaceVisibility:
        cols = [desc[0] for desc in cur.description]
        d = dict(zip(cols, row))
        vis_ns_raw = d.get("visible_namespaces", "")
        vis_ns = (
            [n.strip() for n in vis_ns_raw.split(",") if n.strip()]
            if vis_ns_raw
            else []
        )
        return NamespaceVisibility(
            policy_id=d["policy_id"],
            job_id=d.get("job_id"),
            workflow_name=d.get("workflow_name"),
            visible_namespaces=vis_ns,
            include_default=d.get("include_default", True),
            include_project=d.get("include_project", True),
            include_experiment=d.get("include_experiment", False),
            created_at=d.get("created_at", datetime.utcnow()),
        )
