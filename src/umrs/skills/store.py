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

# Column list shared by every skill SELECT (order matters: `_row_to_skill`).
_SKILL_COLUMNS = (
    "skill_id, name, version, scope, owner_id, project_id, "
    "skill_md, manifest, review_state, enabled, registered_at, updated_at"
)


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
        skill_id = f"skl-{_hash_str(f'{scope.value}:{owner_id}:{name}')}"
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
                f"""
                SELECT {_SKILL_COLUMNS}
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
                f"""
                SELECT {_SKILL_COLUMNS}
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
                SELECT {_SKILL_COLUMNS}
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

    def list_pending_reviews(self) -> list[RegisteredSkill]:
        """Loop-proposed skills awaiting human review (D-27 edit plane)."""
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_SKILL_COLUMNS}
                FROM skills
                WHERE enabled = FALSE
                  AND review_state->>'status' = 'pending_review'
                ORDER BY registered_at
                """
            )
            return [_row_to_skill(row) for row in cur.fetchall()]

    def decide_review(
        self,
        skill_id: str,
        decision: str,
        *,
        reviewer: str,
        note: str = "",
        test_receipt: dict[str, Any] | None = None,
    ) -> RegisteredSkill | None:
        """Merge a human review decision into a pending proposal (atomic).

        The UPDATE fires only while the row is still ``pending_review``
        (atomic guard); approve is the merge — the only transition that
        enables a compiled proposal. Returns the updated skill, or ``None``
        when the skill is absent or already decided (caller maps to a
        typed conflict).
        """
        from . import _now_iso

        block: dict[str, Any] = {
            "status": decision,
            "decision": decision,
            "reviewer": reviewer,
            "decided_at": _now_iso(),
        }
        if note:
            block["note"] = note
        if test_receipt is not None:
            block["test_receipt"] = test_receipt

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE skills
                SET review_state = COALESCE(review_state, '{{}}'::jsonb) || %s::jsonb,
                    enabled = CASE WHEN %s = 'approved' THEN TRUE ELSE enabled END,
                    updated_at = NOW()
                WHERE skill_id = %s
                  AND review_state->>'status' = 'pending_review'
                RETURNING {_SKILL_COLUMNS}
                """,
                (json.dumps(block), decision, skill_id),
            )
            row = cur.fetchone()
            self.conn.commit()
            if row is None:
                return None
            return _row_to_skill(row)

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


def _jsonb_field(raw: Any) -> dict[str, Any] | None:
    """Parse a JSONB-ish column value (dict passthrough, JSON str, junk → None)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _manifest_data(raw: Any) -> dict[str, Any]:
    """Manifest dict from a JSONB column (empty on junk — never raises)."""
    return _jsonb_field(raw) or {}


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
        review_state_raw,
        enabled,
        registered_at,
        updated_at,
    ) = row

    manifest_data = _manifest_data(manifest_raw)
    review_state = _jsonb_field(review_state_raw)

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
        review_state=review_state,
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
