"""Path validation for model storage (U-804)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from ..models import ModelScope
from .errors import PathNotAllowedError

# Default allowed path patterns for model storage
# These can be overridden via environment variable MODEL_ALLOWED_PATHS
DEFAULT_ALLOWED_PATHS = [
    "/shared/models/system/*",       # System models
    "/shared/projects/*/models/*",   # Project models
    "/shared/users/*/models/*",      # User models
    "/models/*",                     # Development/Docker paths
    "/home/*/models/*",              # Home directory models (dev)
]


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

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Match path against a glob-like pattern."""
        regex_pattern = re.escape(pattern)
        regex_pattern = regex_pattern.replace(r"\*", r"[^/]+")
        regex_pattern = f"^{regex_pattern}$"
        return bool(re.match(regex_pattern, path))

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

        if scope == ModelScope.SYSTEM:
            valid_prefixes = ["/shared/models/system", "/models"]
            if not any(normalized.startswith(p) for p in valid_prefixes):
                raise PathNotAllowedError(
                    path,
                    [f"{p}/*" for p in valid_prefixes],
                )

        elif scope == ModelScope.PROJECT:
            if not project_id:
                raise ValueError("project_id required for project scope")
            expected_prefix = f"/shared/projects/{project_id}/models"
            if not normalized.startswith(expected_prefix):
                if not normalized.startswith("/shared/projects/"):
                    raise PathNotAllowedError(path, [f"{expected_prefix}/*"])

        elif scope == ModelScope.USER:
            if not owner_id:
                raise ValueError("owner_id required for user scope")
            valid_prefixes = [
                f"/shared/users/{owner_id}/models",
                f"/home/{owner_id}/models",
            ]
            if not any(normalized.startswith(p) for p in valid_prefixes):
                raise PathNotAllowedError(
                    path,
                    [f"{p}/*" for p in valid_prefixes],
                )
