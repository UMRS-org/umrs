"""A staged step's HTTP client to its routed endpoint (ADR-109 D-5).

A step on the connected-HPC lane runs with ``--network=none``. When its
routing decision names an endpoint (``requires`` / ``model_type`` /
``model``), the job runs a forwarder and gives the step one Unix socket,
named by ``UMRS_STEP_ENDPOINT_SOCKET``. The forwarder adds the endpoint's
credential and allows only that endpoint kind's paths (``/v1/*`` for a vLLM
endpoint; ``/infer/*``, ``/vision/*`` and ``/readyz`` for the BentoML vision
service); the step never holds a key.

    http = ctx.endpoint_http()
    reply = http.post("/v1/chat/completions", json={"model": ..., "messages": ...})
    text = reply.json()["choices"][0]["message"]["content"]

    vision = ctx.endpoint_vision()
    masks = vision.submit_and_wait("sam3", "s3://bucket/frame.png", {"prompt": "cell"})

Standard library only (``http.client`` over ``AF_UNIX``): the platform step
image ships no httpx. :func:`endpoint_httpx` gives an ``httpx.Client`` over
the same socket for an image that has httpx.
"""

from __future__ import annotations

import http.client
import json as _json
import os
import pickle
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

ENDPOINT_SOCKET_ENV = "UMRS_STEP_ENDPOINT_SOCKET"
#: The host name requests carry; the forwarder sends them to its one upstream.
ENDPOINT_HOST = "endpoint"
DEFAULT_TIMEOUT = 600.0


class NoRoutedEndpoint(RuntimeError):
    """The step has no routed endpoint, so no socket to reach one."""


class EndpointHTTPError(RuntimeError):
    """The endpoint (or the forwarder, by name) answered with an error."""

    def __init__(self, response: EndpointResponse) -> None:
        super().__init__(
            f"endpoint answered {response.status} {response.reason}: "
            f"{response.text()[:300]}"
        )
        self.response = response


def endpoint_socket_path(path: str | None = None) -> str:
    """The step's endpoint socket; :class:`NoRoutedEndpoint` without one."""
    found = path or os.environ.get(ENDPOINT_SOCKET_ENV, "")
    if not found:
        raise NoRoutedEndpoint(
            "this step has no routed endpoint (no UMRS_STEP_ENDPOINT_SOCKET): "
            "declare requires / model_type / model in workflow.json so the "
            "platform routes it to one"
        )
    return found


class _UnixConnection(http.client.HTTPConnection):
    """``http.client`` over a Unix socket."""

    def __init__(self, path: str, timeout: float) -> None:
        super().__init__(ENDPOINT_HOST, timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self._path)
        except OSError:
            sock.close()
            raise
        self.sock = sock


@dataclass(frozen=True)
class EndpointResponse:
    status: int
    reason: str
    headers: dict[str, str]
    body: bytes

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        return _json.loads(self.body)

    def raise_for_status(self) -> EndpointResponse:
        if not self.ok:
            raise EndpointHTTPError(self)
        return self


class EndpointHTTP:
    """Requests to the step's routed endpoint, one connection each."""

    def __init__(self, socket_path: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.socket_path = socket_path
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> EndpointResponse:
        body, sent = _body(json, content, headers)
        target = path + ("?" + urlencode(params) if params else "")
        conn = _UnixConnection(self.socket_path, timeout or self.timeout)
        try:
            conn.request(method, target, body=body, headers=sent)
            reply = conn.getresponse()
            data = reply.read()
            return EndpointResponse(
                reply.status,
                reply.reason,
                {k.lower(): v for k, v in reply.getheaders()},
                data,
            )
        finally:
            conn.close()

    def get(self, path: str, **kw: Any) -> EndpointResponse:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> EndpointResponse:
        return self.request("POST", path, **kw)


def _body(
    json: Any, content: bytes | None, headers: dict[str, str] | None
) -> tuple[bytes | None, dict[str, str]]:
    sent = dict(headers or {})
    if json is not None:
        sent.setdefault("Content-Type", "application/json")
        return _json.dumps(json).encode("utf-8"), sent
    return content, sent


def endpoint_http(
    socket_path: str | None = None, timeout: float = DEFAULT_TIMEOUT
) -> EndpointHTTP:
    """A client to the step's routed endpoint (base URL ``http://endpoint``)."""
    return EndpointHTTP(endpoint_socket_path(socket_path), timeout)


def endpoint_httpx(
    socket_path: str | None = None, timeout: float = DEFAULT_TIMEOUT
) -> Any:
    """An ``httpx.Client`` over the same socket, for an image with httpx."""
    import httpx

    path = endpoint_socket_path(socket_path)
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=path),
        base_url=f"http://{ENDPOINT_HOST}",
        timeout=timeout,
    )


# ── the BentoML vision service (``svc-vision2``, SAM3) ──────────────────────

_DONE = frozenset({"success", "completed"})
_FAILED = frozenset({"failure", "failed"})
_CANCELLED = frozenset({"cancelled", "canceled"})


class VisionClient:
    """The vision service's task API through the socket: the request shape
    of ``vision_service.bentoml_client.submit_vision_and_wait``."""

    def __init__(
        self, http: EndpointHTTP, *, poll_interval: float = 2.0, timeout: float = 300.0
    ) -> None:
        self.http = http
        self.poll_interval = poll_interval
        self.timeout = timeout

    def submit(
        self,
        model_type: str,
        image_url: str,
        inference_config: dict[str, Any] | None = None,
        **extra: Any,
    ) -> str:
        """``POST /infer/submit``; the task id."""
        payload = {
            "model_type": model_type,
            "inference_config": dict(inference_config or {}),
            "image_url": image_url,
            **extra,
        }
        answer = self.http.post("/infer/submit", json=payload).raise_for_status()
        task_id = answer.json().get("task_id")
        if not task_id:
            raise RuntimeError(
                f"vision submit returned no task_id: {answer.text()[:200]}"
            )
        return str(task_id)

    def status(self, task_id: str) -> str:
        answer = self.http.get("/infer/status", params={"task_id": task_id})
        return str(answer.raise_for_status().json().get("status") or "")

    def result(self, task_id: str) -> Any:
        """The task's result: JSON, or the service's pickled dict."""
        answer = self.http.get("/infer/get", params={"task_id": task_id})
        answer.raise_for_status()
        if answer.headers.get("content-type", "").startswith("application/json"):
            return answer.json()
        # The platform's own service pickles its result dict.
        return pickle.loads(answer.body)

    def submit_and_wait(
        self,
        model_type: str,
        image_url: str,
        inference_config: dict[str, Any] | None = None,
        **extra: Any,
    ) -> Any:
        """Submit, poll until done (bounded by ``timeout``), return the result."""
        task_id = self.submit(model_type, image_url, inference_config, **extra)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            state = self.status(task_id)
            if state in _DONE:
                return self.result(task_id)
            if state in _FAILED or state in _CANCELLED:
                raise RuntimeError(f"vision task {task_id} ended {state}")
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"vision task {task_id} did not finish within {self.timeout}s"
        )
