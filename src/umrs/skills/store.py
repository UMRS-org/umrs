"""Persistent skill registry store.

CRUD operations for the ``skills`` table.  Mirrors the public surface
of the in-memory :class:`~umrs.skills.SkillRegistry` so the REST API
layer can use either backend interchangeably.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2

from . import RegisteredSkill, SkillScope

logger = logging.getLogger(__name__)


class SkillStoreError(Exception):
    """Base error for skill store operations."""


class SkillNotFoundError(SkillStoreError):
    """Raised when a skill is not found."""


class DuplicateSkillError(SkillStoreError):
    """Raised when registering a duplicate skill."""


class SkillStore:
    """Postgres-backed skill registry.

    Provides CRUD operations for skill bundle registrations.
    """

    def __init__(self, conn: psycopg2.extensions.connection):
        self.conn = conn

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register_skill(
        self,
        *,
        name: str,
        version: str = "0.1.0",
        scope: SkillScope = SkillScope.USER,
        owner_id: str = "",
        project_id: str | None = None,
        skill_md: str = "",
        manifest: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> RegisteredSkill:
        """Register or update a skill (upsert).

        Returns:
            The registered ``RegisteredSkill``.
        """
        skill_id = f"skl-{__hash_str(f'{scope.value}:{owner_id}:{name}')}"
        manifest_json = json.dumps(manifest or {})

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO skills (
                    skill_id, name, version, scope, owner_id, project_id,
                    skill_md, manifest, enabled, registered_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, NOW(), NOW()
                )
                ON CONFLICT (scope, owner_id, name) DO UPDATE SET
                    version = EXCLUDED.version,
                    project_id = EXCLUDED.project_id,
                    skill_md = EXCLUDED.skill_md,
                    manifest = EXCLUDED.manifest,
                    enabled = EXCLUDED.enabled,
                    updated_at = NOW()
                RETURNING skill_id, registered_at, updated_at
                """,
                (
                    skill_id,
                    name,
                    version,
                    scope.value,
                    owner_id,
                    project_id,
                    skill_md,
                    manifest_json,
                    enabled,
                ),
            )
            row = cur.fetchone()
            if row is None:  # pragma: no cover - defensive
                raise SkillStoreError("Failed to register skill")
            actual_skill_id = row[0]
            registered_at = row[1]
            updated_at = row[2]

        logger.info("Registered skill: %s (id=%s)", name, actual_skill_id)
        return RegisteredSkill(
            skill_id=actual_skill_id,
            manifest=_manifest_from_row(manifest or {}, name, version),
            scope=scope,
            owner_id=owner_id,
            project_id=project_id,
            skill_md=skill_md,
            registered_at=registered_at,
            enabled=enabled,
        )

    def get_skill(self, skill_id: str) -> RegisteredSkill:
        """Get a skill by ID.

        Raises:
            SkillNotFoundError: If the skill is not found.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT skill_id, name, version, scope, owner_id, project_id,
                       skill_md, manifest, enabled, registered_at, updated_at
                FROM skills
                WHERE skill_id = %s
                """,
                (skill_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise SkillNotFoundError(f"Skill not found: {skill_id}")
            return _row_to_skill(row)

    def get_skill_by_name(
        self,
        name: str,
        scope: SkillScope,
        owner_id: str,
    ) -> RegisteredSkill | None:
        """Get a skill by (scope, owner, name)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT skill_id, name, version, scope, owner_id, project_id,
                       skill_md, manifest, enabled, registered_at, updated_at
                FROM skills
                WHERE name = %s AND scope = %s AND owner_id = %s
                """,
                (name, scope.value, owner_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_skill(row)

    def list_skills(
        self,
        *,
        scope: SkillScope | None = None,
        owner_id: str | None = None,
        project_id: str | None = None,
    ) -> list[RegisteredSkill]:
        """List skills with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []
        if scope is not None:
            conditions.append("scope = %s")
            params.append(scope.value)
        if owner_id is not None:
            conditions.append("owner_id = %s")
            params.append(owner_id)
        if project_id is not None:
            conditions.append("project_id = %s")
            params.append(project_id)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT skill_id, name, version, scope, owner_id, project_id,
                       skill_md, manifest, enabled, registered_at, updated_at
                FROM skills{where}
                ORDER BY name
                """,
                params,
            )
            return [_row_to_skill(row) for row in cur.fetchall()]

    def set_enabled(self, skill_id: str, enabled: bool) -> bool:
        """Enable or disable a skill. Returns True if updated."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE skills SET enabled = %s, updated_at = NOW()
                WHERE skill_id = %s
                RETURNING skill_id
                """,
                (enabled, skill_id),
            )
            row = cur.fetchone()
            self.conn.commit()
            return row is not None

    def unregister_skill(self, skill_id: str) -> bool:
        """Delete a skill by ID. Returns True if deleted."""
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM skills WHERE skill_id = %s RETURNING skill_id",
                (skill_id,),
            )
            row = cur.fetchone()
            self.conn.commit()
            return row is not None

    def search(self, query: str, limit: int = 20) -> list[RegisteredSkill]:
        """Search skills by name/description/tags."""
        q = f"%{query}%"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT skill_id, name, version, scope, owner_id, project_id,
                       skill_md, manifest, enabled, registered_at, updated_at
                FROM skills
                WHERE enabled = TRUE AND (
                    name ILIKE %s OR
                    skill_md ILIKE %s OR
                    manifest::text ILIKE %s
                )
                ORDER BY name
                LIMIT %s
                """,
                (q, q, q, limit),
            )
            return [_row_to_skill(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        """Get registry statistics."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM skills")
            count = cur.fetchone()[0]
            return {"skills": count}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_str(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _row_to_skill(row: tuple) -> RegisteredSkill:
    (
        skill_id,
        name,
        version,
        scope_str,
        owner_id,
        project_id,
        skill_md,
        manifest_raw,
        enabled,
        registered_at,
        updated_at,
    ) = row

    manifest_data: dict[str, Any]
    if isinstance(manifest_raw, str):
        try:
            manifest_data = json.loads(manifest_raw)
        except json.JSONDecodeError:
            manifest_data = {}
    elif isinstance(manifest_raw, dict):
        manifest_data = manifest_raw
    else:
        manifest_data = {}

    from r4_contracts.skills import SkillManifest

    manifest = SkillManifest.from_dict(manifest_data)
    if not manifest.name:
        manifest.name = name
    if not manifest.version:
        manifest.version = version

    try:
        scope = SkillScope(scope_str)
    except ValueError:
        scope = SkillScope.USER

    return RegisteredSkill(
        skill_id=skill_id,
        manifest=manifest,
        scope=scope,
        owner_id=owner_id,
        project_id=project_id,
        skill_md=skill_md,
        registered_at=registered_at,
        enabled=enabled,
    )


def _manifest_from_row(
    manifest: dict[str, Any],
    name: str,
    version: str,
) -> Any:
    from r4_contracts.skills import SkillManifest

    m = SkillManifest.from_dict(manifest)
    if not m.name:
        m.name = name
    if not m.version:
        m.version = version
    return m
