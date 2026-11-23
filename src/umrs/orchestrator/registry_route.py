"""One gated registry resolver for every server consumer (DEC-063, gap D-63).

Before this module each server consumer picked endpoint-registry rows on its
own (chat and llm discovery, knowledge extraction, the agentic executor,
perception), and only umrs-infer's site source applied the ADR-062 class
gate: a sensitive project could reach a public-passthrough row through every
other consumer. Every consumer now passes its rows through here:

* :func:`gate_rows` drops the public rows a sensitive project may not use
  (the class is read only when a public row is present, so the usual path
  costs no lookup);
* :func:`resolve_registry_endpoint` / :func:`resolve_candidates` pick the
  rows serving ``model`` (a model name, a served name or an ADR-106 role
  alias) in a role (``chat`` or ``any``), and refuse BY NAME
  (``source_class_refused``) when rows matched but the gate admitted none;
* :func:`agent_url_exception` is the one hook point for the deprecated
  ``UMRS_AGENT_LITELLM_URL`` override (slice S-D63-3 classifies and gates
  that URL here; today it is returned unchanged, as before).

The module lives in ``umrs.orchestrator`` (beside the registry and ADR-078
access) rather than ``umrs.inference`` as the 2026-11-24 record named it:
importing ``umrs.inference`` loads torch through its package ``__init__``,
which the llm proxy and umrs-infer must not pay.

Hosted and self-hosted UMRS behave alike: the gate reads only the row and
the project's ``data_class``; a stand-alone project that the site creates
``non_sensitive`` keeps every row it registers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from typing import Any

from umrs.shared import endpoint_locality
from umrs.shared.endpoint_locality import (
    SOURCE_CLASS_REFUSED,
    is_public_row,
    row_admitted,
)

logger = logging.getLogger(__name__)


class RegistryRouteRefused(Exception):
    """Rows matched the request, but the project's class admits none of them."""

    def __init__(self, project_id: str, model: str | None) -> None:
        self.reason = SOURCE_CLASS_REFUSED
        self.project_id = project_id
        self.model = model or ""
        target = f"model {model!r}" if model else "any model"
        super().__init__(
            f"{SOURCE_CLASS_REFUSED}: project {project_id!r} is sensitive and every "
            f"registry row serving {target} is a public passthrough (ADR-062)"
        )


def served_name(endpoint: Any) -> str:
    metadata = getattr(endpoint, "metadata", None) or {}
    return str(metadata.get("served_model_name") or endpoint.model)


def is_chat_row(endpoint: Any) -> bool:
    """A row that serves general chat: ``llm``/``vlm``, no service kind naming a
    vision or embedding service (rerankers, OCR and embedders are not chat
    models, 2026-11-02)."""
    from umrs.model_registry.models import CHAT_MODEL_TYPES, normalized_model_type

    meta = getattr(endpoint, "metadata", None) or {}
    if normalized_model_type(meta.get("model_type")) not in CHAT_MODEL_TYPES:
        return False
    kind = str(meta.get("service_kind") or "").lower()
    return not any(word in kind for word in ("vision", "embed"))


_ROLES: dict[str, Callable[[Any], bool]] = {
    "chat": is_chat_row,
    "any": lambda _endpoint: True,
}


def gate_rows(rows: Iterable[Any], project_id: str, *, pool: Any = None) -> list[Any]:
    """``rows`` without the public rows the project's class refuses."""
    rows = list(rows)
    if not any(is_public_row(row) for row in rows):
        return rows
    data_class = endpoint_locality.project_data_class(project_id, pool)
    return [row for row in rows if row_admitted(row, data_class)]


def endpoint_admitted(endpoint: Any, project_id: str, *, pool: Any = None) -> bool:
    return bool(gate_rows([endpoint], project_id, pool=pool))


def wanted_names(model: str) -> set[str]:
    """``model`` plus, for an ADR-106 role alias (``svc-fast``), the models
    whose roster entries serve it."""
    names = {model}
    try:
        from umrs.orchestrator.provision_roster import is_role_alias, role_alias_models

        if is_role_alias(model):
            names.update(role_alias_models(model))
    except Exception as exc:  # noqa: BLE001 — an unreadable roster matches by name only
        logger.warning("role alias %s unresolved: %s", model, exc)
    return names


def _is_system_shared(endpoint: Any) -> bool:
    """Shared with everyone (ADR-078 system scope)."""
    from umrs.orchestrator.endpoint_access import endpoint_scopes

    return any(scope.kind == "system" for scope in endpoint_scopes(endpoint))


def registry_rows(project_id: str, role: str = "chat", registry: Any = None) -> list[Any]:
    """READY/DEGRADED rows in ``role`` the project may use: its own, then
    system-shared (UNGATED; :func:`resolve_candidates` gates them).

    A strict subset of ADR-078 routing (user/group/circle grants need a full
    caller identity a background job does not have). An empty project has
    no rows: the registry filter would otherwise count every project's
    private row as its own.
    """
    if not project_id:
        return []
    try:
        from umrs.orchestrator.endpoint_registry import EndpointStatus, get_registry

        ready = [EndpointStatus.READY, EndpointStatus.DEGRADED]
        registry = registry if registry is not None else get_registry()
        own = registry.list_all(project_id=project_id, status_filter=ready)
        every = registry.list_all(status_filter=ready)
    except Exception as exc:  # noqa: BLE001 — consumers degrade to "no endpoint"
        logger.warning("registry lookup failed: %s", exc)
        return []
    shared = [ep for ep in every if ep.project_id != project_id and _is_system_shared(ep)]
    return [ep for ep in (*own, *shared) if _ROLES[role](ep)]


def _serving(rows: list[Any], model: str | None, role: str) -> list[Any]:
    rows = [row for row in rows if _ROLES[role](row)]
    if not model:
        return rows
    names = wanted_names(model)
    return [row for row in rows if names & {row.model, served_name(row)}]


def resolve_candidates(
    project_id: str,
    model: str | None = None,
    *,
    role: str = "chat",
    rows: Iterable[Any] | None = None,
    pool: Any = None,
) -> list[Any]:
    """The admitted rows serving ``model`` in ``role``, in registry order.

    ``rows`` are the consumer's own candidates (default:
    :func:`registry_rows`). ``[]`` when nothing serves the request;
    :class:`RegistryRouteRefused` when something does but the class gate
    admits none of it.
    """
    pool_rows = registry_rows(project_id, role) if rows is None else list(rows)
    matched = _serving(pool_rows, model, role)
    admitted = gate_rows(matched, project_id, pool=pool)
    if matched and not admitted:
        raise RegistryRouteRefused(project_id, model)
    return admitted


def resolve_registry_endpoint(
    project_id: str,
    model: str | None = None,
    *,
    role: str = "chat",
    rows: Iterable[Any] | None = None,
    pool: Any = None,
) -> Any | None:
    """The first admitted row serving ``model`` in ``role`` (None: nothing
    serves it; :class:`RegistryRouteRefused`: only refused rows serve it)."""
    admitted = resolve_candidates(project_id, model, role=role, rows=rows, pool=pool)
    return admitted[0] if admitted else None


def agent_url_exception(project_id: str) -> str:
    """The deprecated ``UMRS_AGENT_LITELLM_URL`` override, or ``""``.

    DEC-063 keeps it as a single declared exception until an inventory check
    allows its removal. This is its one hook point: slice S-D63-3 classifies
    the URL's host and applies the class gate for ``project_id`` here.
    """
    del project_id  # S-D63-3 gates the URL by the project's class
    return os.environ.get("UMRS_AGENT_LITELLM_URL", "").strip()


__all__ = [
    "RegistryRouteRefused",
    "agent_url_exception",
    "endpoint_admitted",
    "gate_rows",
    "is_chat_row",
    "registry_rows",
    "resolve_candidates",
    "resolve_registry_endpoint",
    "served_name",
    "wanted_names",
]
