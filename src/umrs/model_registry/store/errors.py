"""Model Registry store exceptions."""

from __future__ import annotations

from ..models import ModelScope


class ModelNotFoundError(Exception):
    """Raised when a model is not found."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class DuplicateModelError(Exception):
    """Raised when a model with the same name already exists in scope."""

    def __init__(self, name: str, scope: ModelScope, project_id: str | None = None):
        self.name = name
        self.scope = scope
        self.project_id = project_id
        if scope == ModelScope.PROJECT:
            super().__init__(f"Model '{name}' already exists in project '{project_id}'")
        else:
            super().__init__(f"Model '{name}' already exists in {scope.value} scope")


class PermissionDeniedError(Exception):
    """Raised when user lacks permission to access/modify a model."""

    def __init__(
        self,
        user_id: str,
        model_id: str,
        action: str = "access",
        reason: str | None = None,
    ):
        self.user_id = user_id
        self.model_id = model_id
        self.action = action
        msg = f"User '{user_id}' cannot {action} model '{model_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class HiddenModelError(ModelNotFoundError, PermissionDeniedError):
    """A model the caller may not even read: it reads as not found.

    Both a ``ModelNotFoundError`` (so every surface answers exactly as for an
    unknown id: same status, same message) and a ``PermissionDeniedError``
    (so code that asks "was this refused?" still sees a refusal). H-1 /
    ADR-086 existence hiding (2026-10-21: a foreign id answered 403, an
    unknown id 404, an existence oracle).
    """

    def __init__(self, lookup_key: str, user_id: str = ""):
        # Exception.__init__ directly: the cooperative chain would hand the
        # message to PermissionDeniedError's signature. str() stays exactly
        # ModelNotFoundError's.
        Exception.__init__(self, f"Model not found: {lookup_key}")
        self.model_id = lookup_key
        self.user_id = user_id
        self.action = "read"


class PathNotAllowedError(Exception):
    """Raised when a storage path is not in the allowed directories."""

    def __init__(self, path: str, allowed_patterns: list[str]):
        self.path = path
        self.allowed_patterns = allowed_patterns
        super().__init__(
            f"Path '{path}' is not allowed. "
            f"Allowed patterns: {', '.join(allowed_patterns)}"
        )
