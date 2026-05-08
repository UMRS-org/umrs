"""UMRS Skill Registry REST API."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from umrs.skills import RegisteredSkill, SkillManifest, SkillRegistry, SkillScope
from umrs.skills import SkillValidationError

router = APIRouter(prefix="/umrs/skills", tags=["skills"])

_registry: SkillRegistry | None = SkillRegistry()


def set_skill_registry(registry: SkillRegistry | None) -> None:
    """Set the active skill registry, primarily for tests and sidecar init."""
    global _registry
    _registry = registry


def _get_registry() -> SkillRegistry:
    if _registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Skill registry unavailable",
        )
    return _registry


class SkillRegisterRequest(BaseModel):
    """Request body for registering a skill bundle."""

    manifest: dict[str, Any] = Field(..., description="Skill manifest/config.json")
    skill_md: str = Field(default="", description="SKILL.md content")
    scope: str = Field(default="user", description="system, project, or user")
    project_id: str | None = Field(default=None, description="Owning project for project skills")


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
    x_user_id: str = Header(default=""),
    x_project_id: str = Header(default=""),
    x_verified_project_id: str = Header(default=""),
    x_verified_is_admin: str = Header(default="false"),
) -> _CallerIdentity:
    project_id = (x_verified_project_id or x_project_id or "").strip()
    strict = os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict"
    if strict and not x_verified_project_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Verified-Project-ID header (Gateway must set this)",
        )
    if not project_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Project-ID header (auth sidecar must set this)",
        )
    return _CallerIdentity(
        user_id=x_user_id.strip(),
        project_id=project_id,
        is_admin=x_verified_is_admin.strip().lower() in ("1", "true", "yes"),
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scope must be one of: system, project, user",
        ) from None


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
    matches = [skill for skill in registry.search(q, limit=limit) if _is_visible(skill, caller)]
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
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin required")
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _skill_response(skill)


@router.get("/{skill_id_or_name}", response_model=SkillResponse)
async def get_skill(
    skill_id_or_name: str,
    caller: _CallerIdentity = Depends(_caller_identity),
) -> SkillResponse:
    """Get a visible skill by ID or manifest name."""
    for skill in _visible_skills(caller):
        if skill.skill_id == skill_id_or_name or skill.manifest.name == skill_id_or_name:
            return _skill_response(skill)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Skill not found: {skill_id_or_name}",
    )


@router.delete("/{skill_id_or_name}")
async def unregister_skill(
    skill_id_or_name: str,
    caller: _CallerIdentity = Depends(_caller_identity),
) -> dict[str, str]:
    """Unregister a skill by ID or manifest name."""
    target: RegisteredSkill | None = None
    for skill in _visible_skills(caller):
        if skill.skill_id == skill_id_or_name or skill.manifest.name == skill_id_or_name:
            target = skill
            break
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill not found: {skill_id_or_name}",
        )
    if not caller.is_admin and target.owner_id != caller.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="owner or admin required")
    _get_registry().unregister(target.skill_id)
    return {"status": "unregistered", "skill_id": target.skill_id}
