"""
UMRS Client core.

Provides HTTP client with authentication and error handling.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin

# Note: Context/memory API imports excluded from selective merge

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    import urllib.error
    import urllib.request

logger = logging.getLogger(__name__)


class UMRSError(Exception):
    """Base exception for UMRS client errors."""

    def __init__(
        self, message: str, status_code: int | None = None, detail: str | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class UMRSNotFoundError(UMRSError):
    """Resource not found."""


class UMRSAccessDeniedError(UMRSError):
    """Access denied to resource."""


class UMRSRateLimitError(UMRSError):
    """Rate limit exceeded."""


class UMRSClient:
    """
    UMRS Python client.

    Provides HTTP client with authentication and error handling.

    Example:
        client = UMRSClient(
            base_url="http://umrs:8000",
            user_id="user123",
            project_id="myproject"
        )
        response = client._request("GET", "/umrs/endpoints/active")

    Note: Context/memory APIs excluded from selective merge.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        user_id: str | None = None,
        project_id: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize UMRS client.

        Args:
            base_url: UMRS server URL.
            user_id: Default user ID for requests.
            project_id: Default project ID for requests.
            api_key: API key for authentication (optional).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.project_id = project_id
        self.api_key = api_key
        self.timeout = timeout

        # Initialize httpx client if available
        if HTTPX_AVAILABLE:
            self._client = httpx.Client(timeout=timeout)
        else:
            self._client = None

        # Note: Context/memory API namespaces excluded from selective merge

    def _get_headers(
        self,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Use provided or default user/project
        uid = user_id or self.user_id
        pid = project_id or self.project_id

        if uid:
            headers["X-User-ID"] = uid
        if pid:
            headers["X-Project-ID"] = pid
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Make HTTP request to UMRS API.

        Args:
            method: HTTP method.
            path: API path (relative to base_url).
            json_data: JSON body data.
            params: Query parameters.
            user_id: Override user ID for this request.
            project_id: Override project ID for this request.

        Returns:
            Response JSON as dict.

        Raises:
            UMRSError: On API errors.
        """
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        headers = self._get_headers(user_id, project_id)

        # Filter None params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        if HTTPX_AVAILABLE:
            return self._request_httpx(method, url, headers, json_data, params)
        else:
            return self._request_urllib(method, url, headers, json_data, params)

    def _request_httpx(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json_data: dict[str, Any] | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Make request using httpx."""
        try:
            response = self._client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                params=params,
            )

            if response.status_code == 404:
                raise UMRSNotFoundError(
                    "Resource not found",
                    status_code=404,
                    detail=response.text,
                )
            elif response.status_code == 403:
                raise UMRSAccessDeniedError(
                    "Access denied",
                    status_code=403,
                    detail=response.text,
                )
            elif response.status_code == 429:
                raise UMRSRateLimitError(
                    "Rate limit exceeded",
                    status_code=429,
                    detail=response.text,
                )
            elif response.status_code >= 400:
                raise UMRSError(
                    f"API error: {response.status_code}",
                    status_code=response.status_code,
                    detail=response.text,
                )

            if response.status_code == 204:
                return {}

            return response.json()

        except httpx.RequestError as e:
            raise UMRSError(f"Request failed: {e}") from e

    def _request_urllib(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json_data: dict[str, Any] | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Make request using urllib (fallback)."""
        import json as json_module

        # Build URL with params
        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"

        # Prepare request
        data = None
        if json_data:
            data = json_module.dumps(json_data).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                if not body:
                    return {}
                try:
                    return json_module.loads(body)
                except (json_module.JSONDecodeError, ValueError) as e:
                    raise UMRSError(f"Invalid JSON response from server: {e}") from e

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""

            if e.code == 404:
                raise UMRSNotFoundError(
                    "Resource not found", status_code=404, detail=body
                ) from e
            elif e.code == 403:
                raise UMRSAccessDeniedError(
                    "Access denied", status_code=403, detail=body
                ) from e
            elif e.code == 429:
                raise UMRSRateLimitError(
                    "Rate limit exceeded", status_code=429, detail=body
                ) from e
            else:
                raise UMRSError(
                    f"API error: {e.code}", status_code=e.code, detail=body
                ) from e

        except urllib.error.URLError as e:
            raise UMRSError(f"Request failed: {e.reason}") from e

    def close(self) -> None:
        """Close the client and release resources."""
        if self._client and HTTPX_AVAILABLE:
            self._client.close()

    def __enter__(self) -> UMRSClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # Note: API namespaces (conversations, messages, context, memory)
    # excluded from selective merge. Use _request() directly for HTTP calls.
