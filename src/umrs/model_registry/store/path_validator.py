"""Path validation for model storage (U-804)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from ..models import ModelScope
from .errors import PathNotAllowedError

# Default allowed path patterns for model storage
# These can be overridden via environment variable MODEL_ALLOWED_PATHS.
# Scope-aware placeholders let administrators opt in extra experimental roots
# without giving every user access to every matching directory.
DEFAULT_ALLOWED_PATHS = [
    "/shared/models/system/*",                 # System models
    "/shared/projects/{project_id}/models/*", # Project models
    "/shared/users/{owner_id}/models/*",      # User models
    "/models/*",                              # Development/Docker paths
    "/home/{owner_id}/models/*",              # Home directory models (dev)
]


def _append_unique(patterns: List[str], candidate: Optional[str]) -> None:
    if not candidate:
        return
    normalized = os.path.normpath(candidate)
    if normalized not in patterns:
        patterns.append(normalized)


def _site_scope_patterns(scope: Optional[ModelScope] = None) -> List[str]:
    try:
        from umrs.orchestrator.site_config import get_site_config

        storage = get_site_config().storage
    except Exception:
        return []

    patterns: List[str] = []

    if scope in (None, ModelScope.SYSTEM):
        model_cache_dir = getattr(storage, "model_cache_dir", "")
        if model_cache_dir:
            system_root = os.path.dirname(os.path.normpath(model_cache_dir))
            _append_unique(patterns, os.path.join(system_root, "*"))

        for attr_name in ("sam3_model_path", "hmae_model_path"):
            _append_unique(patterns, getattr(storage, attr_name, ""))

    if scope in (None, ModelScope.USER):
        user_model_root = getattr(storage, "user_model_path", "")
        if user_model_root:
            _append_unique(patterns, os.path.join(user_model_root, "*"))

    return patterns


@dataclass
class PathValidator:
    """
    Validates that storage paths are within allowed directories.

    This prevents users from serving arbitrary filesystem paths.
    """

    allowed_patterns: List[str]

    def __init__(self, allowed_patterns: Optional[List[str]] = None):
        if allowed_patterns is not None:
            self.allowed_patterns = allowed_patterns
        else:
            env_paths = os.environ.get("MODEL_ALLOWED_PATHS")
            if env_paths:
                self.allowed_patterns = [p.strip() for p in env_paths.split(":")]
            else:
                self.allowed_patterns = DEFAULT_ALLOWED_PATHS.copy()

        for pattern in _site_scope_patterns():
            _append_unique(self.allowed_patterns, pattern)

    def is_allowed(self, path: str) -> bool:
        """Check if a path matches any allowed pattern."""
        path = os.path.normpath(path)
        for pattern in self.allowed_patterns:
            if self._match_pattern(path, pattern):
                return True
        return False

    def validate(self, path: str) -> None:
        """Validate a path, raising PathNotAllowedError if not allowed."""
        if not self.is_allowed(path):
            raise PathNotAllowedError(path, self.allowed_patterns)

    def _compile_pattern(
        self,
        pattern: str,
        *,
        owner_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        regex_pattern = re.escape(pattern)
        regex_pattern = regex_pattern.replace(
            re.escape("{owner_id}"),
            re.escape(owner_id) if owner_id else r"[^/]+",
        )
        regex_pattern = regex_pattern.replace(
            re.escape("{project_id}"),
            re.escape(project_id) if project_id else r"[^/]+",
        )
        regex_pattern = regex_pattern.replace(r"\*", r"[^/]+")
        return f"^{regex_pattern}$"

    def _match_pattern(
        self,
        path: str,
        pattern: str,
        *,
        owner_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bool:
        """Match path against a glob-like pattern."""
        regex_pattern = self._compile_pattern(
            pattern,
            owner_id=owner_id,
            project_id=project_id,
        )
        return bool(re.match(regex_pattern, path))

    def _scope_patterns(self, scope: ModelScope) -> List[str]:
        if scope == ModelScope.SYSTEM:
            patterns = self._system_patterns()
            for pattern in _site_scope_patterns(ModelScope.SYSTEM):
                _append_unique(patterns, pattern)
            return patterns

        token = None
        if scope == ModelScope.PROJECT:
            token = "{project_id}"
        elif scope == ModelScope.USER:
            token = "{owner_id}"

        if token is None:
            return []
        patterns = [pattern for pattern in self.allowed_patterns if token in pattern]
        for pattern in _site_scope_patterns(scope):
            _append_unique(patterns, pattern)
        return patterns

    def _system_patterns(self) -> List[str]:
        return [
            pattern for pattern in self.allowed_patterns
            if "{owner_id}" not in pattern and "{project_id}" not in pattern
        ]

    def validate_scope_path(
        self,
        path: str,
        scope: ModelScope,
        owner_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """
        Validate that path matches the claimed scope.

        For example, a user model should be in /shared/users/<owner_id>/models/,
        not in /shared/models/system/.
        """
        self.validate(path)
        normalized = os.path.normpath(path)

        if scope == ModelScope.PROJECT and not project_id:
            raise ValueError("project_id required for project scope")
        if scope == ModelScope.USER and not owner_id:
            raise ValueError("owner_id required for user scope")

        scope_patterns = self._scope_patterns(scope)
        if any(
            self._match_pattern(
                normalized,
                pattern,
                owner_id=owner_id,
                project_id=project_id,
            )
            for pattern in scope_patterns
        ):
            return

        raise PathNotAllowedError(path, scope_patterns or self.allowed_patterns)
