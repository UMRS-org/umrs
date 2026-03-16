"""BentoML async task queue client for UMRS vision services.

Wrapps the BentoML task queue REST API (POST /submit, GET /status, GET /get,
POST /cancel, PUT /retry) for use by UMRS components.

Can be used with the BentoML Python client (bentoml.SyncHTTPClient) or with
raw HTTP requests for environments where bentoml is not installed.
"""

import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class BentoMLTaskClient:
    """Client for interacting with BentoML task queue endpoints.

    Args:
        base_url: Base URL of the BentoML service (e.g. "http://localhost:3000").
        timeout: Default HTTP request timeout in seconds.
    """

    def __init__(self, base_url, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def submit(self, endpoint_name, **kwargs):
        """Submit a task to a @bentoml.task endpoint.

        Args:
            endpoint_name: Name of the task endpoint (e.g. "sam3_segment").
            **kwargs: Arguments to pass to the task function.

        Returns:
            str: Unique task identifier.
        """
        url = f"{self.base_url}/{endpoint_name}/submit"
        resp = self._session.post(url, json=kwargs, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("task_id") or data.get("id")
        logger.info("Task submitted: %s -> %s", endpoint_name, task_id)
        return task_id

    def get_status(self, endpoint_name, task_id):
        """Get the status of a submitted task.

        Returns:
            str: One of "pending", "in_progress", "success", "failure", "cancelled".
        """
        url = f"{self.base_url}/{endpoint_name}/status"
        resp = self._session.get(url, params={"task_id": task_id}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("status", data.get("value", "unknown"))

    def get_result(self, endpoint_name, task_id):
        """Get the result of a completed task.

        Returns:
            dict: Task result payload.
        """
        url = f"{self.base_url}/{endpoint_name}/get"
        resp = self._session.get(url, params={"task_id": task_id}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def cancel(self, endpoint_name, task_id):
        """Attempt to cancel a pending task.

        Returns:
            bool: True if cancel request was accepted.
        """
        url = f"{self.base_url}/{endpoint_name}/cancel"
        resp = self._session.post(url, json={"task_id": task_id}, timeout=self.timeout)
        resp.raise_for_status()
        return True

    def retry(self, endpoint_name, task_id):
        """Retry a failed task.

        Returns:
            str: New task identifier.
        """
        url = f"{self.base_url}/{endpoint_name}/retry"
        resp = self._session.put(url, json={"task_id": task_id}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("task_id") or data.get("id")

    def wait_for_result(self, endpoint_name, task_id, poll_interval=2.0,
                        timeout=300.0):
        """Poll until a task completes, then return the result.

        Args:
            endpoint_name: Task endpoint name.
            task_id: Task identifier from submit().
            poll_interval: Seconds between status polls.
            timeout: Maximum seconds to wait.

        Returns:
            dict: Task result.

        Raises:
            TimeoutError: If the task doesn't complete within timeout.
            RuntimeError: If the task fails.
        """
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Task {task_id} did not complete within {timeout}s"
                )

            status = self.get_status(endpoint_name, task_id)
            logger.debug("Task %s status: %s (%.1fs elapsed)", task_id, status, elapsed)

            if status in ("success", "completed"):
                return self.get_result(endpoint_name, task_id)
            elif status == "failure":
                raise RuntimeError(f"Task {task_id} failed")
            elif status == "cancelled":
                raise RuntimeError(f"Task {task_id} was cancelled")

            time.sleep(poll_interval)

    def is_healthy(self):
        """Check if the BentoML service is healthy."""
        try:
            resp = self._session.get(f"{self.base_url}/healthz", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
