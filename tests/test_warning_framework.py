"""Tests for the unified warning/notification framework.

Validates:
- All WarningCategory enum values
- UMRSWarning serialization
- CLI warning rendering (stderr capture)
- HTTP response conversion
- Blocking mode for downloads
- Helper factory functions
"""

from io import StringIO
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_warning_exception_handler_returns_structured_409():
    from unittest.mock import MagicMock

    from umrs.api.error_responses import structured_warning_exception_handler
    from umrs.shared.warnings import (
        UMRSWarningError,
        blocking_download_warning,
    )

    response = await structured_warning_exception_handler(
        MagicMock(),
        UMRSWarningError(blocking_download_warning("job-1", "running", retry_after=30)),
    )
    assert response.status_code == 409
    assert b'"task_status":"running"' in response.body
    assert response.headers.get("retry-after") == "30"


class TestWarningCategory:
    """Verify WarningCategory enum values."""

    def test_all_categories_exist(self):
        from umrs.shared.warnings import WarningCategory

        cats = WarningCategory
        assert hasattr(cats, "OPERATION_INVALID")
        assert hasattr(cats, "OPERATION_UNSUPPORTED")
        assert hasattr(cats, "OPERATION_IGNORED")
        assert hasattr(cats, "RESOURCE_MISSING")
        assert hasattr(cats, "RECOVERABLE_ERROR")
        assert hasattr(cats, "BLOCKING")

    def test_category_values(self):
        from umrs.shared.warnings import WarningCategory

        assert WarningCategory.BLOCKING.value == "blocking"
        assert WarningCategory.RESOURCE_MISSING.value == "resource_missing"
        assert WarningCategory.OPERATION_INVALID.value == "operation_invalid"


class TestUMRSWarning:
    """Verify UMRSWarning dataclass."""

    def test_basic_warning(self):
        from umrs.shared.warnings import UMRSWarning, WarningCategory

        w = UMRSWarning(
            category=WarningCategory.BLOCKING,
            message="Task is still running",
        )
        assert w.category == WarningCategory.BLOCKING
        assert w.message == "Task is still running"
        assert w.details == {}
        assert w.suggestion is None
        assert w.retry_after is None

    def test_full_warning(self):
        from umrs.shared.warnings import UMRSWarning, WarningCategory

        w = UMRSWarning(
            category=WarningCategory.OPERATION_INVALID,
            message="Invalid operation",
            details={"op": "foo"},
            suggestion="Check parameters",
            retry_after=30,
        )
        d = w.to_dict()
        assert d["category"] == "operation_invalid"
        assert d["message"] == "Invalid operation"
        assert d["details"] == {"op": "foo"}
        assert d["suggestion"] == "Check parameters"
        assert d["retry_after_seconds"] == 30


class TestUMRSWarningError:
    """Verify UMRSWarningError exception."""

    def test_creates_exception(self):
        from umrs.shared.warnings import UMRSWarning, UMRSWarningError, WarningCategory

        w = UMRSWarning(
            category=WarningCategory.BLOCKING,
            message="Blocked",
        )
        exc = UMRSWarningError(w, http_status=409)
        assert exc.warning is w
        assert exc.http_status == 409
        assert "blocking" in str(exc)  # category value is lowercase

    def test_default_status(self):
        from umrs.shared.warnings import UMRSWarning, UMRSWarningError, WarningCategory

        w = UMRSWarning(
            category=WarningCategory.RESOURCE_MISSING,
            message="Not found",
        )
        exc = UMRSWarningError(w)
        assert exc.http_status == 409  # default


class TestCLIRendering:
    """Verify CLI warning printing to stderr."""

    def test_print_warning_basic(self):
        from umrs.shared.warnings import print_warning

        captured = StringIO()
        with patch("sys.stderr", captured):
            print_warning("Test warning message")
        output = captured.getvalue()
        assert "Test warning message" in output
        assert "⚠" in output

    def test_print_warning_with_category(self):
        from umrs.shared.warnings import WarningCategory, print_warning

        captured = StringIO()
        with patch("sys.stderr", captured):
            print_warning("Blocking error", category=WarningCategory.BLOCKING)
        output = captured.getvalue()
        assert "[BLOCKING]" in output
        assert "Blocking error" in output


class TestHTTPResponseConversion:
    """Verify warning → HTTP response conversion."""

    def test_warning_to_http_response(self):
        from umrs.shared.warnings import UMRSWarning, WarningCategory, warning_to_http_response

        w = UMRSWarning(
            category=WarningCategory.BLOCKING,
            message="Download not ready",
            retry_after=60,
        )
        body = warning_to_http_response(w)
        assert body["status"] == "warning"
        assert len(body["warnings"]) == 1
        assert body["warnings"][0]["message"] == "Download not ready"
        assert body["retry_after"] == 60

    def test_blocking_warning_response(self):
        from umrs.shared.warnings import (
            UMRSWarning,
            WarningCategory,
            blocking_warning_response,
        )

        w = UMRSWarning(
            category=WarningCategory.BLOCKING,
            message="Blocked",
            details={"job_id": "job-123"},
            suggestion="Retry later",
            retry_after=30,
        )
        body = blocking_warning_response(w)
        assert body["error_id"] == "warning.blocking"
        assert body["message"] == "Blocked"
        assert body["details"]["job_id"] == "job-123"
        assert body["suggestion"] == "Retry later"
        assert body["retry_after_seconds"] == 30


class TestDownloadHelpers:
    """Verify download-specific warning helpers."""

    def test_blocking_download_warning(self):
        from umrs.shared.warnings import blocking_download_warning

        w = blocking_download_warning("job-abc123", "running", retry_after=10)
        assert w.category.value == "blocking"
        assert "job-abc123" in w.message
        assert "running" in w.message
        assert w.details["job_id"] == "job-abc123"
        assert w.details["task_status"] == "running"
        assert "Use ?blocking=true" in w.suggestion
        assert w.retry_after == 10

    def test_check_download_ready_not_ready(self):
        from umrs.shared.warnings import check_download_ready

        result = check_download_ready("job-123", "running", blocking=False)
        assert result["ready"] is False
        assert result["status"] == "running"
        assert "error" in result

    def test_check_download_ready_success(self):
        from umrs.shared.warnings import check_download_ready

        result = check_download_ready("job-123", "succeeded")
        assert result["ready"] is True
        assert result["status"] == "succeeded"

    def test_check_download_ready_completed(self):
        from umrs.shared.warnings import check_download_ready

        result = check_download_ready("job-123", "completed")
        assert result["ready"] is True


class TestFactoryFunctions:
    """Verify all warning factory functions."""

    def test_job_not_found_warning(self):
        from umrs.shared.warnings import job_not_found_warning

        w = job_not_found_warning("job-xyz")
        assert w.category.value == "resource_missing"
        assert "job-xyz" in w.message
        assert w.details["job_id"] == "job-xyz"

    def test_operation_invalid_warning(self):
        from umrs.shared.warnings import operation_invalid_warning

        w = operation_invalid_warning("foo", "bad param")
        assert w.category.value == "operation_invalid"
        assert "foo" in w.message

    def test_operation_unsupported_warning(self):
        from umrs.shared.warnings import operation_unsupported_warning

        w = operation_unsupported_warning("bar", "not available")
        assert w.category.value == "operation_unsupported"

    def test_resource_missing_warning(self):
        from umrs.shared.warnings import resource_missing_warning

        w = resource_missing_warning("model", "m-123")
        assert w.category.value == "resource_missing"
        assert w.details["resource_type"] == "model"

    def test_recoverable_error_warning(self):
        from umrs.shared.warnings import recoverable_error_warning

        w = recoverable_error_warning("Network error", retry_after=60)
        assert w.category.value == "recoverable_error"
        assert w.retry_after == 60

    def test_operation_ignored_warning(self):
        from umrs.shared.warnings import operation_ignored_warning

        w = operation_ignored_warning("export", "no data")
        assert w.category.value == "operation_ignored"


class TestNoSilentFailures:
    """Ensure all factory functions produce visible output."""

    def test_all_helpers_create_warning(self):
        from umrs.shared.warnings import (
            blocking_download_warning,
            job_not_found_warning,
            operation_ignored_warning,
            operation_invalid_warning,
            operation_unsupported_warning,
            recoverable_error_warning,
            resource_missing_warning,
        )

        helpers = [
            (blocking_download_warning, ("job-1", "running")),
            (job_not_found_warning, ("job-2",)),
            (operation_invalid_warning, ("foo", "bar")),
            (operation_ignored_warning, ("export", "no data")),
            (operation_unsupported_warning, ("feature", "reason")),
            (recoverable_error_warning, ("error",)),
            (resource_missing_warning, ("type", "id")),
        ]

        for fn, args in helpers:
            w = fn(*args)
            assert w is not None
            assert w.message  # Must have a message
            assert len(w.message) > 0  # Message cannot be empty
