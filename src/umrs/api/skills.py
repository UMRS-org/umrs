"""UMRS Skill Registry REST API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from umrs.skills import RegisteredSkill, SkillManifest, SkillRegistry, SkillScope
from umrs.skills import SkillValidationError

router = APIRouter(prefix="/umrs/skills", tags=["skills"])

_registry: SkillRegistry | None = SkillRegistry()
_skill_store = None  # type: ignore


def set_skill_registry(registry: SkillRegistry | None) -> None:
    """Set the active skill registry, primarily for tests and sidecar init."""
    global _registry
    _registry = registry


def set_skill_store(conn: Any) -> None:
    """Wire a Postgres-backed SkillStore into the REST layer.

    Call once at server startup when a database connection is available.
    Passing *None* clears the store and falls back to the in-memory registry.
    """
    global _skill_store
    if conn is None:
        _skill_store = None
        return
    from umrs.skills.store import SkillStore

    _skill_store = _SkillStoreAdapter(SkillStore(conn))


class _SkillStoreAdapter(SkillRegistry):
    """Adapts :class:`~umrs.skills.store.SkillStore` to the
    :class:`~umrs.skills.SkillRegistry` interface so the REST layer can
    use either backend interchangeably.
    """

    def __init__(self, store) -> None:  # type: ignore[no-untyped-def]
        self._store = store

    def validate_manifest(self, manifest: SkillManifest) -> list[str]:  # type: ignore[override]
        # Delegate to base implementation
        return super().validate_manifest(manifest)

    def register(
        self,
        manifest: SkillManifest,
        skill_md: str,
        scope: SkillScope = SkillScope.USER,
        owner_id: str = "",
        project_id: str | None = None,
    ) -> RegisteredSkill:
        return self._store.register_skill(
            name=manifest.name,
            version=manifest.version,
            scope=scope,
            owner_id=owner_id,
            project_id=project_id,
            skill_md=skill_md,
            manifest=manifest.to_dict(),
            enabled=True,
        )

    def get(self, skill_id: str) -> RegisteredSkill | None:  # type: ignore[override]
        try:
            return self._store.get_skill(skill_id)
        except Exception:
            return None

    def list_skills(
        self,
        scope: SkillScope | None = None,
        owner_id: str | None = None,
        tag: str | None = None,
    ) -> list[RegisteredSkill]:
        skills = self._store.list_skills(
            scope=scope,
            owner_id=owner_id,
        )
        if tag:
            skills = [s for s in skills if tag in s.manifest.tags]
        return skills

    def unregister(self, skill_id: str) -> bool:  # type: ignore[override]
        return self._store.unregister_skill(skill_id)

    def search(self, query: str, limit: int = 20) -> list[RegisteredSkill]:  # type: ignore[override]
        return self._store.search(query, limit=limit)


def _get_registry() -> SkillRegistry:
    if _skill_store is not None:
        return _skill_store
    if _registry is None:
        raise_typed(
            PlatformError.skills(
                "unavailable",
                "Skill registry unavailable"
            ),
        )
    return _registry


class SkillRegisterRequest(BaseModel):
    """Request body for registering a skill bundle."""

    manifest: dict[str, Any] = Field(..., description="Skill manifest/config.json")
    skill_md: str = Field(default="", description="SKILL.md content")
    scope: str = Field(default="user", description="system, project, or user")
    project_id: str | None = Field(
        default=None, description="Owning project for project skills"
    )


class SkillResponse(BaseModel):
    """Skill registry response."""

    skill_id: str
    manifest: dict[str, Any]
    scope: str
    owner_id: str
    project_id: str | None = None
    skill_md: str = ""
    registered_at: str | None = None
    enabled: bool = True


class _CallerIdentity(BaseModel):
    user_id: str
    project_id: str
    is_admin: bool = False


def _caller_identity(
    request: Request,
    x_user_id: str = Header(default=""),
    x_project_id: str = Header(default=""),
    x_verified_project_id: str = Header(default=""),
    x_verified_is_admin: str = Header(default="false"),
) -> _CallerIdentity:
    """Canonical tier chain (R1) → skills identity dataclass."""
    from ._identity import request_identity

    identity = request_identity(request, require_project=True)
    return _CallerIdentity(
        user_id=identity.user_id,
        project_id=identity.project_id,
        is_admin=identity.is_admin,
    )


def _is_visible(skill: RegisteredSkill, caller: _CallerIdentity) -> bool:
    if caller.is_admin:
        return True
    if skill.scope == SkillScope.SYSTEM:
        return True
    if skill.scope == SkillScope.PROJECT:
        return skill.project_id == caller.project_id
    if skill.scope == SkillScope.USER:
        return bool(caller.user_id) and skill.owner_id == caller.user_id
    return False


def _skill_response(skill: RegisteredSkill) -> SkillResponse:
    return SkillResponse(
        skill_id=skill.skill_id,
        manifest=skill.manifest.to_dict(),
        scope=skill.scope.value,
        owner_id=skill.owner_id,
        project_id=skill.project_id,
        skill_md=skill.skill_md,
        registered_at=skill.registered_at.isoformat() if skill.registered_at else None,
        enabled=skill.enabled,
    )


def _resolve_scope(value: str) -> SkillScope:
    try:
        return SkillScope(value)
    except ValueError:
        raise_typed(
            PlatformError.skills(
                "bad_request",
                "scope must be one of: system, project, user"
            ),
            cause=None,
        )


def _visible_skills(caller: _CallerIdentity) -> list[RegisteredSkill]:
    registry = _get_registry()
    return [skill for skill in registry.list_skills() if _is_visible(skill, caller)]


@router.get("", response_model=list[SkillResponse])
@router.get("/", response_model=list[SkillResponse])
async def list_skills(
    scope: str | None = Query(default=None),
    tag: str | None = Query(default=None),
    caller: _CallerIdentity = Depends(_caller_identity),
) -> list[SkillResponse]:
    """List skills visible to the authenticated caller."""
    scope_filter = _resolve_scope(scope) if scope else None
    skills = _visible_skills(caller)
    if scope_filter:
        skills = [skill for skill in skills if skill.scope == scope_filter]
    if tag:
        skills = [skill for skill in skills if tag in skill.manifest.tags]
    return [_skill_response(skill) for skill in skills]


@router.get("/search", response_model=list[SkillResponse])
async def search_skills(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
    caller: _CallerIdentity = Depends(_caller_identity),
) -> list[SkillResponse]:
    """Search skills visible to the authenticated caller."""
    registry = _get_registry()
    matches = [
        skill for skill in registry.search(q, limit=limit) if _is_visible(skill, caller)
    ]
    return [_skill_response(skill) for skill in matches[:limit]]


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SkillResponse)
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=SkillResponse)
async def register_skill(
    request: SkillRegisterRequest,
    caller: _CallerIdentity = Depends(_caller_identity),
) -> SkillResponse:
    """Register a skill bundle in the UMRS registry."""
    scope = _resolve_scope(request.scope)
    if scope == SkillScope.SYSTEM and not caller.is_admin:
        raise_typed(
            PlatformError.skills(
                "forbidden",
                "admin required"
            ),
        )
    if scope == SkillScope.PROJECT:
        project_id = request.project_id or caller.project_id
    elif scope == SkillScope.SYSTEM:
        project_id = request.project_id
    else:
        project_id = None

    manifest = SkillManifest.from_dict(request.manifest)
    try:
        skill = _get_registry().register(
            manifest=manifest,
            skill_md=request.skill_md,
            scope=scope,
            owner_id=caller.user_id,
            project_id=project_id,
        )
    except SkillValidationError as exc:
        raise_typed(
            PlatformError.skills(
                "bad_request",
                str(exc)
            ),
            cause=exc,
        )
    return _skill_response(skill)


@router.get("/{skill_id_or_name}", response_model=SkillResponse)
async def get_skill(
    skill_id_or_name: str,
    caller: _CallerIdentity = Depends(_caller_identity),
) -> SkillResponse:
    """Get a visible skill by ID or manifest name."""
    for skill in _visible_skills(caller):
        if (
            skill.skill_id == skill_id_or_name
            or skill.manifest.name == skill_id_or_name
        ):
            return _skill_response(skill)
    raise_typed(
        PlatformError.skills(
            "not_found",
            f"Skill not found: {skill_id_or_name}"
        ),
    )


@router.delete("/{skill_id_or_name}")
async def unregister_skill(
    skill_id_or_name: str,
    caller: _CallerIdentity = Depends(_caller_identity),
) -> dict[str, str]:
    """Unregister a skill by ID or manifest name."""
    target: RegisteredSkill | None = None
    for skill in _visible_skills(caller):
        if (
            skill.skill_id == skill_id_or_name
            or skill.manifest.name == skill_id_or_name
        ):
            target = skill
            break
    if target is None:
        raise_typed(
            PlatformError.skills(
                "not_found",
                f"Skill not found: {skill_id_or_name}"
            ),
        )
    if not caller.is_admin and target.owner_id != caller.user_id:
        raise_typed(
            PlatformError.skills(
                "forbidden",
                "owner or admin required"
            ),
        )
    _get_registry().unregister(target.skill_id)
    return {"status": "unregistered", "skill_id": target.skill_id}
