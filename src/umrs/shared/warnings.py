"""Unified warning and notification framework for UMRS.

Provides consistent, structured feedback for all user-facing operations.
Replaces silent failures with explicit warnings and errors.

Categories:
    - OPERATION_INVALID: User performed an invalid operation
    - OPERATION_UNSUPPORTED: Requested feature not supported in current mode
    - OPERATION_IGNORED: Action was silently ignored (now reports)
    - RESOURCE_MISSING: Required resource not found or unavailable
    - RECOVERABLE_ERROR: Transient failure that can be retried
    - BLOCKING: Operation blocked by policy or state (e.g., task not ready)

Usage:
    # In API endpoints:
    from umrs.shared.warnings import UMRSWarning, WarningCategory

    raise UMRSWarning(
        category=WarningCategory.BLOCKING,
        message="Task is still running — download not available yet",
        details={"task_status": "running", "job_id": "job-abc123"}
    )

    # In CLI:
    from umrs.shared.warnings import print_warning

    print_warning("Task is still running — download not available yet")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Color codes for terminal output (stderr)
_COLOR_YELLOW = "\033[93m"
_COLOR_RED = "\033[91m"
_COLOR_CYAN = "\033[96m"
_COLOR_RESET = "\033[0m"

# Warning prefix labels by category value string
_WARNING_PREFIX: dict[str, str] = {
    "operation_invalid": "INVALID",
    "operation_unsupported": "UNSUPPORTED",
    "operation_ignored": "IGNORED",
    "resource_missing": "MISSING",
    "recoverable_error": "RECOVERABLE",
    "blocking": "BLOCKING",
}


class WarningCategory(str, Enum):
    """Categories of UMRS warnings/notifications."""

    OPERATION_INVALID = "operation_invalid"
    OPERATION_UNSUPPORTED = "operation_unsupported"
    OPERATION_IGNORED = "operation_ignored"
    RESOURCE_MISSING = "resource_missing"
    RECOVERABLE_ERROR = "recoverable_error"
    BLOCKING = "blocking"


@dataclass
class UMRSWarning:
    """Structured warning/notification for UMRS operations.

    Attributes:
        category: Type of warning
        message: Human-readable message
        details: Optional structured data
        suggestion: Optional remediation advice
        retry_after: Optional seconds to wait before retry (for BLOCKING)
    """

    category: WarningCategory
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str | None = None
    retry_after: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result: dict[str, Any] = {
            "category": self.category.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.retry_after is not None:
            result["retry_after_seconds"] = self.retry_after
        return result


class UMRSWarningError(Exception):
    """Exception raised for blocking warnings.

    This can be caught by API layers and converted to HTTP responses,
    or by CLI layers and displayed to the user.

    Attributes:
        warning: The structured warning
    """

    def __init__(
        self,
        warning: UMRSWarning,
        http_status: int = 409,
    ):
        self.warning = warning
        self.http_status = http_status
        super().__init__(f"[{warning.category.value}] {warning.message}")


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------


def print_warning(message: str, *, category: WarningCategory | None = None) -> None:
    """Print a warning to stderr with color coding.

    Args:
        message: Warning message text
        category: Optional category for prefix
    """
    prefix = ""
    color = _COLOR_YELLOW
    if category:
        label = _WARNING_PREFIX.get(category, "WARN")
        prefix = f" {color}[{label}]{_COLOR_RESET} "
        if category == WarningCategory.BLOCKING:
            color = _COLOR_RED
        elif category == WarningCategory.OPERATION_INVALID:
            color = _COLOR_RED

    print(f"{color}{prefix}⚠ {message}{_COLOR_RESET}", file=sys.stderr)


def print_warning_structured(warning: UMRSWarning) -> None:
    """Print a structured warning to stderr.

    Args:
        warning: Structured warning to display
    """
    print_warning(
        warning.message,
        category=warning.category,
    )
    if warning.suggestion:
        print(f"  → {warning.suggestion}", file=sys.stderr)
    if warning.details:
        for key, value in warning.details.items():
            print(f"  {key}: {value}", file=sys.stderr)
    if warning.retry_after:
        print(f"  Retry after {warning.retry_after} seconds", file=sys.stderr)


# ---------------------------------------------------------------------------
# API response helpers
# ---------------------------------------------------------------------------


def warning_to_http_response(warning: UMRSWarning) -> dict[str, Any]:
    """Convert a warning to an HTTP response body.

    Returns a dict suitable for FastAPI JSONResponse or as
    a 'warnings' key in a 2xx response body.

    Args:
        warning: The warning to serialize

    Returns:
        Dict with status, warnings, and optional details
    """
    body: dict[str, Any] = {
        "status": "warning",
        "warnings": [warning.to_dict()],
    }
    if warning.retry_after:
        body["retry_after"] = warning.retry_after
    return body


def blocking_warning_response(warning: UMRSWarning) -> dict[str, Any]:
    """Convert a blocking warning to an error HTTP response body.

    Use for HTTP 4xx/5xx error responses that need structured warning data.

    Args:
        warning: The blocking warning

    Returns:
        Error response body dict
    """
    body: dict[str, Any] = {
        "error_id": f"warning.{warning.category.value}",
        "message": warning.message,
        "details": warning.details,
    }
    if warning.suggestion:
        body["suggestion"] = warning.suggestion
    if warning.retry_after:
        body["retry_after_seconds"] = warning.retry_after
    return body


# ---------------------------------------------------------------------------
# Download-specific blocking helpers
# ---------------------------------------------------------------------------


def blocking_download_warning(
    job_id: str,
    task_status: str,
    retry_after: int | None = None,
) -> UMRSWarning:
    """Create a blocking warning for premature download attempts.

    Args:
        job_id: The job ID being downloaded from
        task_status: Current task status (e.g., "running", "queued")
        retry_after: Optional seconds to wait before retry

    Returns:
        Structured blocking warning
    """
    return UMRSWarning(
        category=WarningCategory.BLOCKING,
        message=(
            f"Task is still {task_status} — download not available yet for job {job_id}"
        ),
        details={
            "job_id": job_id,
            "task_status": task_status,
        },
        suggestion="Use ?blocking=true to wait, or retry later",
        retry_after=retry_after,
    )


def job_not_found_warning(job_id: str) -> UMRSWarning:
    """Create a warning for missing job resource."""
    return UMRSWarning(
        category=WarningCategory.RESOURCE_MISSING,
        message=f"Job {job_id} not found",
        details={"job_id": job_id},
        suggestion="Check the job ID or list available jobs",
    )


def operation_invalid_warning(operation: str, reason: str) -> UMRSWarning:
    """Create a warning for invalid user operations."""
    return UMRSWarning(
        category=WarningCategory.OPERATION_INVALID,
        message=f"Operation '{operation}' is invalid: {reason}",
        suggestion="Review the operation parameters and try again",
    )


def operation_unsupported_warning(feature: str, reason: str) -> UMRSWarning:
    """Create a warning for unsupported operations."""
    return UMRSWarning(
        category=WarningCategory.OPERATION_UNSUPPORTED,
        message=f"Feature '{feature}' is not supported: {reason}",
        suggestion="Check documentation for available features",
    )


def resource_missing_warning(resource_type: str, resource_id: str) -> UMRSWarning:
    """Create a warning for missing resources."""
    return UMRSWarning(
        category=WarningCategory.RESOURCE_MISSING,
        message=f"{resource_type} {resource_id} not found",
        details={"resource_type": resource_type, "resource_id": resource_id},
        suggestion=f"List available {resource_type}s or create a new one",
    )


def recoverable_error_warning(message: str, retry_after: int = 30) -> UMRSWarning:
    """Create a warning for recoverable/transient errors."""
    return UMRSWarning(
        category=WarningCategory.RECOVERABLE_ERROR,
        message=message,
        suggestion=f"Retry after {retry_after} seconds",
        retry_after=retry_after,
    )


def operation_ignored_warning(action: str, reason: str) -> UMRSWarning:
    """Create a warning for silently ignored actions (now reported)."""
    return UMRSWarning(
        category=WarningCategory.OPERATION_IGNORED,
        message=f"Action '{action}' was ignored: {reason}",
        suggestion="Review the action parameters and try again",
    )


# ---------------------------------------------------------------------------
# Blocking mode for download operations
# ---------------------------------------------------------------------------


def check_download_ready(
    job_id: str,
    attempt_status: str,
    blocking: bool = False,
    timeout: int = 300,
) -> dict[str, Any]:
    """Check if download is ready for a job attempt, optionally blocking.

    This is a utility function used by download endpoints. In production,
    the actual status check should query the JobStore.

    Args:
        job_id: Job ID to check
        attempt_status: Current attempt status string
        blocking: If True, poll until ready or timeout
        timeout: Maximum seconds to block (default 300)

    Returns:
        Dict with ready (bool), status, and optional error
    """
    if attempt_status in ("succeeded", "completed"):
        return {"ready": True, "status": attempt_status, "job_id": job_id}

    if blocking:
        return {
            "ready": False,
            "status": attempt_status,
            "job_id": job_id,
            "blocking": True,
            "message": "Task is still in progress — use retry or wait",
        }

    warning = blocking_download_warning(job_id, attempt_status)
    return {
        "ready": False,
        "status": attempt_status,
        "job_id": job_id,
        "error": warning.to_dict(),
    }
