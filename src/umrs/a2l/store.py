"""Durable store for the A2L conservation domain (ADR-034).

Asyncpg pool against the UMRS Postgres. Stores only conserved *additions*:
artifact identities, operational lineage, agent-execution annotations,
decisions — keyed by ``job_id`` / ``attempt_id``. Query helpers here expose
read-joins against the authoritative spine (jobs/attempts) without copying
any of its state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from r4_contracts.a2l import a2l_artifact_transition_ok

from .models import (
    AgentExecution,
    AgentExecutionCreate,
    AgentExecutionUpdate,
    ArtifactDerive,
    ArtifactRecord,
    DecisionCreate,
    DecisionRecord,
)

logger = logging.getLogger(__name__)

_pool = None
_instance: ConservationStore | None = None


def _add_project_scope(
    clauses: list[str], params: list[Any], readable_projects: list[str] | None
) -> None:
    """Restrict a list query to the caller's readable project namespaces
    (``None`` = unrestricted). Rows with no project are global, as in
    knowledge.access_scope.project_readable."""
    if readable_projects is None:
        return
    params.append(readable_projects)
    clauses.append(f"(project_id IS NULL OR project_id = '' OR project_id = ANY(${len(params)}))")


class ConservationStoreUnavailable(RuntimeError):
    """Raised when no pool has been built for the conservation store."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_artifact_id(
    content_hash: str, operation: str, type_: str, hash_version: int = 2
) -> str:
    """Deterministic content+context identity (SHA-256)."""
    material = f"v{hash_version}:{content_hash}:{operation}:{type_}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ConservationStore:
    """Asyncpg-backed store for a2l conservation tables (migration 0077)."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # ── Artifacts ─────────────────────────────────────────────────────────

    async def create_artifact(self, record: ArtifactRecord) -> ArtifactRecord:
        """Insert a parent-less (ingested) artifact. Idempotent by id."""
        await self._pool.execute(
            """
            INSERT INTO a2l_artifacts (
                id, type, type_schema, uri, metadata, status, user_id,
                project_id, hash_version, artifact_class, job_id, attempt_id,
                created_at
            ) VALUES ($1,$2,$3::jsonb,$4,$5::jsonb,$6,$7,$8,$9,$10,$11,$12,$13::text::timestamptz)
            ON CONFLICT (id) DO NOTHING
            """,
            record.id,
            record.type,
            json.dumps(record.type_schema),
            record.uri,
            json.dumps(record.metadata),
            record.status,
            record.user_id,
            record.project_id,
            record.hash_version,
            record.artifact_class,
            record.job_id,
            record.attempt_id,
            record.created_at or _utcnow(),
        )
        return record

    async def derive_artifact(self, req: ArtifactDerive) -> ArtifactRecord:
        """Derive a child artifact from parents; lineage edges atomically.

        Idempotent: re-deriving the same (content, operation, type) returns
        the existing record and adds no duplicate edges.
        """
        if not req.parents:
            raise ValueError("derive requires at least one parent")
        aid = compute_artifact_id(req.input_data_hash, req.operation, req.type)
        record = ArtifactRecord(
            id=aid,
            type=req.type,
            uri=req.uri,
            user_id=req.user_id,
            type_schema=req.type_schema,
            metadata=req.metadata,
            project_id=req.project_id,
            artifact_class=req.artifact_class,
            job_id=req.job_id,
            attempt_id=req.attempt_id,
            created_at=_utcnow(),
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO a2l_artifacts (
                        id, type, type_schema, uri, metadata, status, user_id,
                        project_id, hash_version, artifact_class, job_id, attempt_id,
                        created_at
                    ) VALUES ($1,$2,$3::jsonb,$4,$5::jsonb,$6,$7,$8,$9,$10,$11,$12,$13::text::timestamptz)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    record.id,
                    record.type,
                    json.dumps(record.type_schema),
                    record.uri,
                    json.dumps(record.metadata),
                    record.status,
                    record.user_id,
                    record.project_id,
                    record.hash_version,
                    record.artifact_class,
                    record.job_id,
                    record.attempt_id,
                    record.created_at,
                )
                for parent in req.parents:
                    await conn.execute(
                        """
                        INSERT INTO a2l_artifact_edges (parent_id, child_id, relationship_type)
                        VALUES ($1, $2, 'derived_from')
                        ON CONFLICT (parent_id, child_id) DO NOTHING
                        """,
                        parent,
                        record.id,
                    )
        return record

    async def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM a2l_artifacts WHERE id = $1 AND deleted_at IS NULL",
            artifact_id,
        )
        return self._to_artifact(row) if row else None

    async def list_artifacts(
        self,
        *,
        job_id: str | None = None,
        user_id: str | None = None,
        artifact_type: str | None = None,
        status: str | None = None,
        readable_projects: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ArtifactRecord]:
        """``readable_projects`` None = every project (admin); otherwise rows
        in those namespaces or with no project (global)."""
        clauses, params = ["deleted_at IS NULL"], []
        for col, val in (
            ("job_id", job_id),
            ("user_id", user_id),
            ("type", artifact_type),
            ("status", status),
        ):
            if val:
                params.append(val)
                clauses.append(f"{col} = ${len(params)}")
        _add_project_scope(clauses, params, readable_projects)
        params.extend([limit, offset])
        rows = await self._pool.fetch(
            f"SELECT * FROM a2l_artifacts WHERE {' AND '.join(clauses)}"
            " ORDER BY created_at DESC"
            f" LIMIT ${len(params) - 1} OFFSET ${len(params)}",
            *params,
        )
        return [self._to_artifact(r) for r in rows]

    async def lineage(
        self, artifact_id: str, *, direction: str = "both", max_depth: int = 3
    ) -> dict[str, Any]:
        """Breadth-first lineage around ``artifact_id`` over the derive edges:
        ``{"root_id", "nodes": [artifact dicts], "edges": [edge dicts]}``."""
        seen, frontier, edges = {artifact_id}, [artifact_id], {}
        for _ in range(max(0, min(max_depth, 10))):
            rows = await self._pool.fetch(
                "SELECT parent_id, child_id, relationship_type FROM a2l_artifact_edges"
                " WHERE ($2 AND parent_id = ANY($1)) OR ($3 AND child_id = ANY($1))",
                frontier,
                direction in ("down", "both"),
                direction in ("up", "both"),
            )
            frontier = []
            for row in rows:
                edges[(row["parent_id"], row["child_id"])] = row["relationship_type"]
                for node in (row["parent_id"], row["child_id"]):
                    if node not in seen:
                        seen.add(node)
                        frontier.append(node)
            if not frontier:
                break
        nodes = await self._pool.fetch("SELECT * FROM a2l_artifacts WHERE id = ANY($1)", list(seen))
        return {
            "root_id": artifact_id,
            "nodes": [self._to_artifact(r).to_dict() for r in nodes],
            "edges": [
                {"parent_id": p, "child_id": c, "relationship_type": t}
                for (p, c), t in edges.items()
            ],
        }

    async def update_artifact_status(
        self, artifact_id: str, status: str, error: str = ""
    ) -> ArtifactRecord:
        """FSM-guarded status patch. Raises ValueError on illegal transition."""
        current = await self.get_artifact(artifact_id)
        if current is None:
            raise ValueError(f"artifact not found: {artifact_id}")
        if not a2l_artifact_transition_ok(current.status, status):
            raise ValueError(f"illegal transition {current.status} -> {status}")
        completed = _utcnow() if status in ("completed", "failed") else None
        await self._pool.execute(
            """
            UPDATE a2l_artifacts
            SET status = $2::text::a2l_artifact_status,
                error = $3,
                started_at = COALESCE(started_at, CASE WHEN $2::text = 'running' THEN now() END),
                completed_at = COALESCE($4::text::timestamptz, completed_at)
            WHERE id = $1
            """,
            artifact_id,
            status,
            error or None,
            completed,
        )
        refreshed = await self.get_artifact(artifact_id)
        if refreshed is None:  # pragma: no cover — defensive
            raise ValueError(f"artifact vanished after update: {artifact_id}")
        return refreshed

    # ── Agent executions ──────────────────────────────────────────────────

    async def create_execution(self, req: AgentExecutionCreate) -> AgentExecution:
        eid = uuid.uuid4().hex
        rec = AgentExecution(
            id=eid,
            agent_name=req.agent_name,
            user_id=req.user_id,
            agent_version=req.agent_version,
            model_ref=req.model_ref,
            prompt_hash=req.prompt_hash,
            params=req.params,
            project_id=req.project_id,
            job_id=req.job_id,
            attempt_id=req.attempt_id,
            created_at=_utcnow(),
        )
        await self._pool.execute(
            """
            INSERT INTO a2l_agent_executions (
                id, agent_name, agent_version, model_ref, prompt_hash, params,
                status, user_id, project_id, job_id, attempt_id, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10,$11,$12::text::timestamptz)
            """,
            rec.id,
            rec.agent_name,
            rec.agent_version,
            rec.model_ref,
            rec.prompt_hash,
            json.dumps(rec.params),
            rec.status,
            rec.user_id,
            rec.project_id,
            rec.job_id,
            rec.attempt_id,
            rec.created_at,
        )
        return rec

    async def get_execution(self, execution_id: str) -> AgentExecution | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM a2l_agent_executions WHERE id = $1", execution_id
        )
        return self._to_execution(row) if row else None

    async def list_executions(
        self,
        *,
        job_id: str | None = None,
        attempt_id: str | None = None,
        agent_name: str | None = None,
        user_id: str | None = None,
        readable_projects: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentExecution]:
        clauses, params = ["TRUE"], []
        for col, val in (
            ("job_id", job_id),
            ("attempt_id", attempt_id),
            ("agent_name", agent_name),
            ("user_id", user_id),
        ):
            if val:
                params.append(val)
                clauses.append(f"{col} = ${len(params)}")
        _add_project_scope(clauses, params, readable_projects)
        params.extend([limit, offset])
        rows = await self._pool.fetch(
            f"SELECT * FROM a2l_agent_executions WHERE {' AND '.join(clauses)}"
            " ORDER BY created_at DESC"
            f" LIMIT ${len(params) - 1} OFFSET ${len(params)}",
            *params,
        )
        return [self._to_execution(r) for r in rows]

    async def update_execution(
        self, execution_id: str, patch: AgentExecutionUpdate
    ) -> AgentExecution:
        existing = await self.get_execution(execution_id)
        if existing is None:
            raise ValueError(f"execution not found: {execution_id}")
        await self._pool.execute(
            """
            UPDATE a2l_agent_executions
            SET status = COALESCE($2::text::a2l_execution_status, status),
                execution_trace = COALESCE($3::jsonb, execution_trace),
                error = COALESCE($4, error),
                started_at = COALESCE(started_at, CASE WHEN $2::text = 'running' THEN now() END),
                completed_at = COALESCE($5::text::timestamptz, completed_at)
            WHERE id = $1
            """,
            execution_id,
            patch.status,
            json.dumps(patch.execution_trace)
            if patch.execution_trace is not None
            else None,
            patch.error,
            patch.completed_at
            if patch.status in ("completed", "failed", "cancelled")
            else None,
        )
        refreshed = await self.get_execution(execution_id)
        if refreshed is None:  # pragma: no cover — defensive
            raise ValueError(f"execution vanished after update: {execution_id}")
        return refreshed

    # ── Decisions ─────────────────────────────────────────────────────��───

    async def create_decision(self, req: DecisionCreate) -> DecisionRecord:
        did = uuid.uuid4().hex
        created = _utcnow()
        await self._pool.execute(
            """
            INSERT INTO a2l_decisions (
                id, job_id, attempt_id, artifact_id, decided_by, decision,
                rationale, payload, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::text::timestamptz)
            """,
            did,
            req.job_id or None,
            req.attempt_id or None,
            req.artifact_id or None,
            req.decided_by,
            req.decision,
            req.rationale,
            json.dumps(req.payload),
            created,
        )
        return DecisionRecord(
            id=did,
            decided_by=req.decided_by,
            decision=req.decision,
            rationale=req.rationale,
            job_id=req.job_id,
            attempt_id=req.attempt_id,
            artifact_id=req.artifact_id,
            payload=req.payload,
            created_at=created,
        )

    async def list_decisions(
        self, *, job_id: str | None = None, limit: int = 50
    ) -> list[DecisionRecord]:
        if job_id:
            rows = await self._pool.fetch(
                "SELECT * FROM a2l_decisions WHERE job_id = $1 ORDER BY created_at DESC LIMIT $2",
                job_id,
                limit,
            )
        else:
            rows = await self._pool.fetch(
                "SELECT * FROM a2l_decisions ORDER BY created_at DESC LIMIT $1", limit
            )
        return [
            DecisionRecord(
                id=r["id"],
                decided_by=r["decided_by"],
                decision=r["decision"],
                rationale=r["rationale"],
                job_id=r["job_id"] or "",
                attempt_id=r["attempt_id"] or "",
                artifact_id=r["artifact_id"] or "",
                payload=r["payload"] or {},
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]

    # ── Spine join read (ADR-034: compositional query, never a copy) ──────

    async def job_conservation_view(self, job_id: str) -> dict[str, Any]:
        """Compositional view: spine-row (if readable) + conserved additions.

        Joins jobs/attempts for the authoritative state and returns it beside
        a2l's conserved rows. If the spine tables are unreadable or the job is
        unknown, returns conserved rows with ``spine`` = None — a reconcile
        finding, not a second truth.
        """
        spine_row = None
        try:
            spine_row = await self._pool.fetchrow(
                """
                SELECT j.id, j.status, j.job_type, j.commit_attempt_id
                FROM jobs j WHERE j.id = $1
                """,
                job_id,
            )
        except Exception:
            logger.debug("a2l conservation view: spine join unavailable", exc_info=True)
        artifacts = [
            {
                "id": a.id,
                "type": a.type,
                "status": a.status,
                "uri": a.uri,
                "attempt_id": a.attempt_id,
            }
            for a in await self.list_artifacts(job_id=job_id)
        ]
        executions = [
            {
                "id": e.id,
                "agent_name": e.agent_name,
                "status": e.status,
                "attempt_id": e.attempt_id,
            }
            for e in await self.list_executions(job_id=job_id)
        ]
        decisions = [vars(d) for d in await self.list_decisions(job_id=job_id)]
        return {
            "job_id": job_id,
            "spine": (
                {
                    "status": spine_row["status"],
                    "job_type": spine_row["job_type"],
                    "commit_attempt_id": spine_row["commit_attempt_id"],
                }
                if spine_row
                else None
            ),
            "artifacts": artifacts,
            "executions": executions,
            "decisions": decisions,
        }

    # ── Row mapping ───────────────────────────────────────────────────────

    @staticmethod
    def _to_artifact(row: Any) -> ArtifactRecord:
        return ArtifactRecord(
            id=row["id"],
            type=row["type"],
            uri=row["uri"],
            user_id=row["user_id"],
            status=str(row["status"]),
            type_schema=row["type_schema"] or {},
            metadata=row["metadata"] or {},
            project_id=row["project_id"],
            hash_version=row["hash_version"],
            artifact_class=row["artifact_class"],
            job_id=row["job_id"] or "",
            attempt_id=row["attempt_id"] or "",
            created_at=str(row["created_at"] or ""),
            completed_at=str(row["completed_at"] or ""),
            error=row["error"] or "",
        )

    @staticmethod
    def _to_execution(row: Any) -> AgentExecution:
        return AgentExecution(
            id=row["id"],
            agent_name=row["agent_name"],
            user_id=row["user_id"],
            status=str(row["status"]),
            agent_version=row["agent_version"],
            model_ref=row["model_ref"],
            prompt_hash=row["prompt_hash"],
            params=row["params"] or {},
            execution_trace=row["execution_trace"] or {},
            project_id=row["project_id"],
            job_id=row["job_id"] or "",
            attempt_id=row["attempt_id"] or "",
            error=row["error"] or "",
            created_at=str(row["created_at"] or ""),
            completed_at=str(row["completed_at"] or ""),
        )


# ── Pool / singleton management ───────────────────────────────────────────


def build_pool(uri: str, **kwargs: Any) -> Any:
    """Create the asyncpg pool synchronously (mirrors knowledge store)."""
    import asyncio

    import asyncpg  # type: ignore[import-untyped]

    global _pool
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # sidecar lifespan is async; caller should use build_pool_async
            raise RuntimeError("use build_pool_async from an async context")
        from umrs.shared.pg_settings import asyncpg_pool_kwargs

        _pool = loop.run_until_complete(
            asyncpg.create_pool(dsn=uri, **asyncpg_pool_kwargs("a2l-store", **kwargs))
        )
    except RuntimeError:
        raise
    return _pool


async def build_pool_async(uri: str, **kwargs: Any) -> Any:
    """Create the asyncpg pool from an async context."""
    import asyncpg  # type: ignore[import-untyped]

    from umrs.shared.pg_settings import asyncpg_pool_kwargs

    global _pool
    _pool = await asyncpg.create_pool(
        dsn=uri, **asyncpg_pool_kwargs("a2l-store", **kwargs)
    )
    return _pool


def get_store() -> ConservationStore:
    """Return the module singleton, building from the current pool."""
    global _instance
    if _pool is None:
        raise ConservationStoreUnavailable("a2l conservation pool not built")
    if _instance is None:
        _instance = ConservationStore(_pool)
    return _instance


def init_store(pool: Any) -> ConservationStore:
    """Inject an externally built pool (sidecar lifespan path)."""
    global _pool, _instance
    _pool = pool
    _instance = ConservationStore(pool)
    return _instance


def reset_store() -> None:
    """Reset the singleton for testing (pool is preserved)."""
    global _instance
    _instance = None
