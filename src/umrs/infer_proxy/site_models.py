"""``umrs-infer``'s ``site`` source: EndpointRegistry rows (ADR-092 slice 8b, D-63).

A request on the reserved site source (path prefix ``site/``) names a model
in its OpenAI-compatible body; it is served by a READY/DEGRADED registry row
of the lease's OWN project only (no shared rows, no other project, no other
site's registry), its model rewritten to the row's served name (ADR-068),
forwarded to the row's internal listener with the site's service key
injected here (the sandbox never holds it). The internal listener is
reachable only because the registry named it: every other upstream keeps
the public-host pin of the pushed-source path.

ADR-062: a row that is a public passthrough (``metadata.locality ==
"public"`` or a public provider kind) is refused for a sensitive project;
self-deployed rows are the sensitive project's own route.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from umrs.shared.project_data_class import PUBLIC_PROVIDER_KINDS

MAX_SITE_BODY_BYTES = 32 * 1024 * 1024
#: OpenAI-compatible verbs a site model serves to a sandbox.
SITE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")
SITE_MODELS_PATH = "/v1/models"
_VARIANT = re.compile(r"^\d+[km]?$|^tep(?:-\d+[km]?)?$|^canary$")


class SiteSourceRefused(Exception):
    """A named refusal: ``reason`` and its HTTP status."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def service_key() -> str:
    """The site's LiteLLM service key (as the ``/umrs/llm`` lane injects it)."""
    return os.environ.get("GATEWAY_LITELLM_SERVICE_KEY") or os.environ.get("LITELLM_MASTER_KEY", "")


def default_registry() -> Any:
    from umrs.orchestrator.endpoint_registry import get_registry

    return get_registry()


def _ready(registry: Any, project: str) -> list[Any]:
    from umrs.orchestrator.endpoint_registry import EndpointStatus

    return registry.list_all(
        project_id=project, status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED]
    )


def served_name(endpoint: Any) -> str:
    metadata = getattr(endpoint, "metadata", None) or {}
    return str(metadata.get("served_model_name") or endpoint.model)


def _names(endpoint: Any) -> set[str]:
    names = {endpoint.model, served_name(endpoint)}
    if endpoint.variant:
        names |= {f"{n}-{endpoint.variant}" for n in list(names)}
    return names


def resolve(registry: Any, project: str, model: str) -> Any:
    """The project's own row serving ``model``; refused by name otherwise."""
    if not project:
        raise SiteSourceRefused(403, "site_scope_missing")
    base, _, suffix = model.rpartition("-")
    variant = suffix if base and _VARIANT.match(suffix) else None
    endpoint = registry.select_endpoint(project, base if variant else model, variant=variant)
    if endpoint is None:
        endpoint = next((ep for ep in _ready(registry, project) if model in _names(ep)), None)
    if endpoint is None or getattr(endpoint, "project_id", None) != project:
        raise SiteSourceRefused(404, "site_model_not_found")
    return endpoint


def is_public_row(endpoint: Any) -> bool:
    metadata = getattr(endpoint, "metadata", None) or {}
    return (
        metadata.get("locality") == "public"
        or metadata.get("provider_kind") in PUBLIC_PROVIDER_KINDS
    )


def request_model(body: bytes) -> tuple[dict[str, Any], str]:
    """The request's JSON object and its ``model``; refused when absent."""
    if len(body) > MAX_SITE_BODY_BYTES:
        raise SiteSourceRefused(413, "body_too_large")
    try:
        payload = json.loads(body or b"{}")
    except ValueError as exc:
        raise SiteSourceRefused(400, "body_not_json") from exc
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, str) or not model:
        raise SiteSourceRefused(400, "model_required")
    return payload, model


def models_listing(registry: Any, project: str) -> dict[str, Any]:
    """``GET /v1/models`` on the site source: the project's served names."""
    if not project:
        raise SiteSourceRefused(403, "site_scope_missing")
    names = sorted({served_name(ep) for ep in _ready(registry, project)})
    return {"object": "list", "data": [{"id": n, "object": "model"} for n in names]}


def upstream_url(endpoint: Any, path: str) -> str:
    return f"http://{endpoint.host}:{int(endpoint.port)}{path}"


__all__ = [
    "MAX_SITE_BODY_BYTES",
    "SITE_PATHS",
    "SiteSourceRefused",
    "is_public_row",
    "models_listing",
    "request_model",
    "resolve",
    "served_name",
    "service_key",
    "upstream_url",
]
