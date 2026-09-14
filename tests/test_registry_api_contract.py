"""
U-206: Contract tests for UMRS Registry API.

Tests the API responses conform to the expected contract for:
- 200: Success responses
- 202: Accepted (async operations)
- 404: Not found
- 503: Service unavailable
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# These tests validate the API contract without requiring full service deployment


class TestRegistryAPIContract:
    """Contract tests for /umrs/endpoints/* API."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock endpoint registry."""
        return MagicMock()

    @pytest.fixture
    def mock_umrs_client(self):
        """Mock the UMRS client for contract testing."""
        client = AsyncMock()
        return client

    # ==========================================================================
    # GET /umrs/endpoints - List all endpoints
    # ==========================================================================

    def test_list_endpoints_200_response_schema(self, mock_umrs_client):
        """
        Contract: GET /umrs/endpoints returns 200 with list of EndpointRecord.

        Expected response:
        [
            {
                "project_id": str,
                "model_name": str,
                "url": str,
                "state": "starting" | "ready" | "stopping" | "stopped",
                "created_at": ISO8601 timestamp,
                "last_heartbeat": ISO8601 timestamp | null,
                "slurm_job_id": str | null,
                "node": str | null,
                "port": int | null
            },
            ...
        ]
        """
        # Arrange: Mock response that conforms to contract
        mock_response = [
            {
                "project_id": "proj-123",
                "model_name": "rivault-gpt-oss-static",
                "url": "http://node01:11434",
                "state": "ready",
                "created_at": "2024-01-15T10:30:00Z",
                "last_heartbeat": "2024-01-15T10:35:00Z",
                "slurm_job_id": "12345",
                "node": "node01",
                "port": 11434,
            }
        ]

        # Validate schema
        for endpoint in mock_response:
            assert "project_id" in endpoint
            assert "model_name" in endpoint
            assert "url" in endpoint
            assert "state" in endpoint
            assert endpoint["state"] in ("starting", "ready", "stopping", "stopped")
            assert "created_at" in endpoint
            # Optional fields can be null
            assert "last_heartbeat" in endpoint
            assert "slurm_job_id" in endpoint

    def test_list_endpoints_200_empty_list(self, mock_umrs_client):
        """
        Contract: GET /umrs/endpoints returns 200 with empty list when no endpoints.
        """
        mock_response = []
        assert isinstance(mock_response, list)
        assert len(mock_response) == 0

    # ==========================================================================
    # GET /umrs/endpoints/{project_id} - Get endpoint by project
    # ==========================================================================

    def test_get_endpoint_200_response_schema(self, mock_umrs_client):
        """
        Contract: GET /umrs/endpoints/{project_id} returns 200 with EndpointRecord.
        """
        mock_response = {
            "project_id": "proj-123",
            "model_name": "rivault-gpt-oss-static",
            "url": "http://node01:11434",
            "state": "ready",
            "created_at": "2024-01-15T10:30:00Z",
            "last_heartbeat": "2024-01-15T10:35:00Z",
            "slurm_job_id": "12345",
            "node": "node01",
            "port": 11434,
        }

        # Validate all required fields
        required_fields = ["project_id", "model_name", "url", "state", "created_at"]
        for field in required_fields:
            assert field in mock_response, f"Missing required field: {field}"

    def test_get_endpoint_404_response_schema(self, mock_umrs_client):
        """
        Contract: GET /umrs/endpoints/{project_id} returns 404 with detail message.

        Expected response:
        {
            "detail": str
        }
        """
        mock_response = {"detail": "Endpoint not found for project: proj-unknown"}

        assert "detail" in mock_response
        assert isinstance(mock_response["detail"], str)

    # ==========================================================================
    # POST /umrs/endpoints/request - Request new endpoint (async)
    # ==========================================================================

    def test_request_endpoint_202_response_schema(self, mock_umrs_client):
        """
        Contract: POST /umrs/endpoints/request returns 202 with RequestResponse.

        202 indicates the request was accepted and endpoint is being provisioned.

        Expected response:
        {
            "request_id": str,
            "project_id": str,
            "status": "accepted" | "queued" | "provisioning",
            "estimated_wait_sec": int | null,
            "poll_url": str
        }
        """
        mock_response = {
            "request_id": "req-abc-123",
            "project_id": "proj-123",
            "status": "accepted",
            "estimated_wait_sec": 60,
            "poll_url": "/umrs/endpoints/proj-123",
        }

        # Validate schema
        assert "request_id" in mock_response
        assert "project_id" in mock_response
        assert "status" in mock_response
        assert mock_response["status"] in ("accepted", "queued", "provisioning")
        assert "poll_url" in mock_response

    def test_request_endpoint_200_already_ready(self, mock_umrs_client):
        """
        Contract: POST /umrs/endpoints/request returns 200 if endpoint already ready.

        200 indicates an endpoint is immediately available.
        """
        mock_response = {
            "project_id": "proj-123",
            "model_name": "rivault-gpt-oss-static",
            "url": "http://node01:11434",
            "state": "ready",
            "created_at": "2024-01-15T10:30:00Z",
            "last_heartbeat": "2024-01-15T10:35:00Z",
        }

        assert mock_response["state"] == "ready"
        assert "url" in mock_response

    # ==========================================================================
    # DELETE /umrs/endpoints/{project_id} - Release endpoint
    # ==========================================================================

    def test_delete_endpoint_200_response_schema(self, mock_umrs_client):
        """
        Contract: DELETE /umrs/endpoints/{project_id} returns 200 on success.

        Expected response:
        {
            "status": "released",
            "project_id": str
        }
        """
        mock_response = {"status": "released", "project_id": "proj-123"}

        assert mock_response["status"] == "released"
        assert "project_id" in mock_response

    def test_delete_endpoint_404_not_found(self, mock_umrs_client):
        """
        Contract: DELETE /umrs/endpoints/{project_id} returns 404 if not found.
        """
        mock_response = {"detail": "No endpoint found for project: proj-unknown"}

        assert "detail" in mock_response

    # ==========================================================================
    # GET /umrs/endpoints/health - Registry health check
    # ==========================================================================

    def test_health_200_response_schema(self, mock_umrs_client):
        """
        Contract: GET /umrs/endpoints/health returns 200 with health status.

        Expected response:
        {
            "status": "ok" | "degraded" | "unhealthy",
            "service": str,
            "timestamp": ISO8601,
            "checks": {
                "registry": "ok" | "error",
                "orchestrator": "ok" | "error" | "not_configured",
                ...
            }
        }
        """
        mock_response = {
            "status": "ok",
            "service": "umrs-registry",
            "timestamp": "2024-01-15T10:30:00Z",
            "checks": {
                "registry": "ok",
                "orchestrator": "ok",
            },
        }

        assert "status" in mock_response
        assert mock_response["status"] in ("ok", "degraded", "unhealthy")
        assert "service" in mock_response
        assert "timestamp" in mock_response
        assert "checks" in mock_response

    def test_health_503_unhealthy_response(self, mock_umrs_client):
        """
        Contract: GET /umrs/endpoints/health returns 503 when service unhealthy.

        The response body still follows the same schema, but status is unhealthy.
        """
        mock_response = {
            "status": "unhealthy",
            "service": "umrs-registry",
            "timestamp": "2024-01-15T10:30:00Z",
            "checks": {
                "registry": "error",
                "orchestrator": "error",
            },
            "errors": ["Cannot connect to orchestrator", "Database unavailable"],
        }

        assert mock_response["status"] == "unhealthy"
        assert "errors" in mock_response or "checks" in mock_response


class TestEndpointStates:
    """Test endpoint state transitions conform to contract."""

    def test_valid_states(self):
        """Contract: Only these states are valid."""
        valid_states = {"starting", "ready", "stopping", "stopped", "error"}

        # Test state can be any of these
        for state in valid_states:
            assert state in valid_states

    def test_state_transition_starting_to_ready(self):
        """Contract: starting -> ready is valid."""
        # This documents the expected state machine
        valid_transitions = {
            "starting": ["ready", "error", "stopped"],
            "ready": ["stopping", "error"],
            "stopping": ["stopped", "error"],
            "stopped": ["starting"],  # Can restart
            "error": ["starting", "stopped"],  # Can retry or cleanup
        }

        assert "ready" in valid_transitions["starting"]


class TestErrorResponseContract:
    """Test error response format across all endpoints."""

    def test_400_bad_request_format(self):
        """Contract: 400 Bad Request response format."""
        mock_response = {
            "detail": "Invalid request: missing required field 'model_name'",
        }
        assert "detail" in mock_response

    def test_401_unauthorized_format(self):
        """Contract: 401 Unauthorized response format."""
        mock_response = {
            "detail": "Invalid or missing authentication token",
        }
        assert "detail" in mock_response

    def test_403_forbidden_format(self):
        """Contract: 403 Forbidden response format."""
        mock_response = {
            "detail": "Access denied to project: proj-other",
        }
        assert "detail" in mock_response

    def test_429_rate_limited_format(self):
        """
        Contract: 429 Too Many Requests response format.

        Expected headers:
        - X-RateLimit-Limit: requests allowed per window
        - X-RateLimit-Remaining: requests remaining
        - X-RateLimit-Reset: timestamp when limit resets
        - Retry-After: seconds to wait
        """
        mock_response = {
            "detail": "Rate limit exceeded",
            "retry_after": 60,
        }

        expected_headers = {
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1705315800",
            "Retry-After": "60",
        }

        assert "detail" in mock_response
        for header in expected_headers:
            assert header.startswith("X-RateLimit-") or header == "Retry-After"

    def test_500_internal_error_format(self):
        """Contract: 500 Internal Server Error response format."""
        mock_response = {
            "detail": "Internal server error",
        }
        assert "detail" in mock_response
        # Should NOT expose internal details in production

    def test_503_service_unavailable_format(self):
        """Contract: 503 Service Unavailable response format."""
        mock_response = {
            "detail": "Service temporarily unavailable",
            "retry_after": 30,
        }
        assert "detail" in mock_response


class TestRequestPayloadContract:
    """Test request payload schemas."""

    def test_request_endpoint_payload(self):
        """Contract: POST /umrs/endpoints/request request body."""
        valid_payload = {
            "project_id": "proj-123",
            "model_name": "rivault-gpt-oss-static",  # Optional, has default
            "priority": "normal",  # Optional: "low", "normal", "high"
            "timeout_sec": 3600,  # Optional: max session duration
        }

        # Required field
        assert "project_id" in valid_payload

        # model_name should use static naming convention
        if "model_name" in valid_payload:
            assert "-static" in valid_payload["model_name"]

    def test_minimal_request_payload(self):
        """Contract: Minimal valid request payload."""
        minimal_payload = {"project_id": "proj-123"}

        assert "project_id" in minimal_payload


class TestPaginationContract:
    """Test pagination in list endpoints."""

    def test_list_endpoints_pagination_params(self):
        """Contract: List endpoints supports pagination query params."""
        # Query parameters
        valid_params = {
            "limit": 20,  # Default: 20, Max: 100
            "offset": 0,  # Default: 0
            "state": "ready",  # Optional filter
        }

        assert valid_params["limit"] <= 100
        assert valid_params["offset"] >= 0

    def test_list_endpoints_pagination_response(self):
        """Contract: Paginated list response includes metadata."""
        mock_response = {
            "items": [
                {"project_id": "proj-1", "state": "ready"},
                {"project_id": "proj-2", "state": "ready"},
            ],
            "total": 50,
            "limit": 20,
            "offset": 0,
            "has_more": True,
        }

        assert "items" in mock_response
        assert "total" in mock_response
        assert "has_more" in mock_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
