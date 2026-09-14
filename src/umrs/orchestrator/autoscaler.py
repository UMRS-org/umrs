"""Pool autoscaler — queue-depth-driven endpoint scaling.

Monitors active request load per (model, variant) across ready endpoints
and provisions or deprovisions endpoints to meet demand. Designed to be
called periodically by the janitor loop.

Scaling decisions:
- **Scale up**: when total active_requests across ready endpoints for a
  (model, variant) exceed ``scale_up_depth * ready_count``, provision a
  new endpoint (up to ``max_replicas``).
- **Scale down**: when an endpoint has been idle longer than
  ``scale_down_idle_sec`` and ``ready_count > min_replicas``, deprovision
  it.

This is a foundation module: stdlib + local orchestrator imports only,
no imports from ``umrs.api`` / ``umrs.server``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .endpoint_registry import EndpointRegistry

logger = logging.getLogger(__name__)


@dataclass
class AutoscalePolicy:
    """Scaling policy for a (model, variant) pair.

    Attributes:
        model: Model name (e.g. "DeepSeek-V4-Pro").
        variant: Variant name (e.g. "128k"), or None for the default.
        min_replicas: Minimum number of ready endpoints to maintain.
        max_replicas: Maximum number of endpoints allowed.
        scale_up_depth: Queue depth per endpoint that triggers scale-up.
            When ``total_active_requests > scale_up_depth * ready_count``,
            a new endpoint is provisioned.
        scale_down_idle_sec: Seconds an endpoint must be idle before
            it is considered for scale-down.
        idle_ttl_sec: TTL for auto-scaled endpoints (shorter than pool
            defaults so they wind down naturally when load drops).
    """

    model: str
    variant: str | None = None
    min_replicas: int = 1
    max_replicas: int = 4
    scale_up_depth: int = 5
    scale_down_idle_sec: int = 600
    idle_ttl_sec: int = 600


def _parse_autoscale_policy(
    entry: dict[str, Any],
) -> AutoscalePolicy | None:
    """Parse an ``autoscale`` block from a roster entry.

    Returns ``None`` if the entry has no ``autoscale`` key.
    """
    autoscale = entry.get("autoscale")
    if not autoscale or not isinstance(autoscale, dict):
        return None
    model = entry.get("model", "")
    variant = entry.get("name")
    if isinstance(variant, str) and variant:
        variant = variant
    else:
        variant = None
    return AutoscalePolicy(
        model=model,
        variant=variant,
        min_replicas=int(autoscale.get("min_replicas", 1)),
        max_replicas=int(autoscale.get("max_replicas", 4)),
        scale_up_depth=int(autoscale.get("scale_up_depth", 5)),
        scale_down_idle_sec=int(autoscale.get("scale_down_idle_sec", 600)),
        idle_ttl_sec=int(autoscale.get("idle_ttl_sec", 600)),
    )


class PoolAutoscaler:
    """Queue-depth-driven autoscaler for pool endpoints.

    Call ``check_and_scale()`` periodically (e.g. from the janitor loop)
    to evaluate scaling decisions against the current registry state.
    """

    def __init__(
        self,
        registry: EndpointRegistry,
        policies: list[AutoscalePolicy] | None = None,
    ):
        self._registry = registry
        self._policies: dict[tuple[str, str | None], AutoscalePolicy] = {}
        if policies:
            for p in policies:
                self._policies[(p.model, p.variant)] = p

    def load_policies_from_roster(self, roster_entries: list[dict[str, Any]]) -> int:
        """Load autoscale policies from roster entries.

        Returns the number of policies loaded.
        """
        count = 0
        for entry in roster_entries:
            policy = _parse_autoscale_policy(entry)
            if policy:
                self._policies[(policy.model, policy.variant)] = policy
                count += 1
        if count:
            logger.info("Loaded %d autoscale policies", count)
        return count

    def get_policy(
        self, model: str, variant: str | None = None
    ) -> AutoscalePolicy | None:
        """Return the autoscale policy for a (model, variant) pair."""
        return self._policies.get((model, variant))

    def check_and_scale(self) -> list[dict[str, Any]]:
        """Evaluate all policies and return scaling decisions.

        Does NOT perform actual provisioning — returns decision records
        that the caller (janitor) can execute. This separation keeps the
        autoscaler testable without side effects.

        Returns:
            List of decision dicts, each with:
            - ``action``: "scale_up" or "scale_down"
            - ``model``: Model name
            - ``variant``: Variant name (or None)
            - ``reason``: Human-readable explanation
            - ``job_id``: (scale_down only) Job ID to deprovision
        """
        from .endpoint_registry import EndpointStatus

        decisions: list[dict[str, Any]] = []
        registry = self._registry

        for (model, variant), policy in self._policies.items():
            # Find all ready/degraded endpoints for this (model, variant)
            candidates = registry._find_candidates(
                project_id="",  # system tier — empty project_id
                model=model,
                variant=variant,
                status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
            )

            # Also check per-project endpoints (user/project tier)
            all_eps = registry.list_all(
                status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
            )
            for ep in all_eps:
                if ep.model == model and ep.variant == variant:
                    if ep not in candidates:
                        candidates.append(ep)

            ready_count = len(candidates)
            total_active = sum(ep.active_requests for ep in candidates)

            # Scale-up check
            if (
                ready_count < policy.max_replicas
                and total_active > policy.scale_up_depth * ready_count
            ):
                decisions.append(
                    {
                        "action": "scale_up",
                        "model": model,
                        "variant": variant,
                        "ready_count": ready_count,
                        "total_active": total_active,
                        "threshold": policy.scale_up_depth * ready_count,
                        "reason": (
                            f"Queue depth {total_active} exceeds threshold "
                            f"{policy.scale_up_depth * ready_count} "
                            f"({ready_count} endpoints × {policy.scale_up_depth})"
                        ),
                    }
                )

            # Scale-down check: find idle endpoints
            import datetime

            now = datetime.datetime.utcnow()
            idle_threshold = now - datetime.timedelta(
                seconds=policy.scale_down_idle_sec
            )
            for ep in candidates:
                if ready_count <= policy.min_replicas:
                    break
                if (
                    ep.last_used_at
                    and ep.last_used_at < idle_threshold
                    and ep.active_requests == 0
                ):
                    decisions.append(
                        {
                            "action": "scale_down",
                            "model": model,
                            "variant": variant,
                            "job_id": ep.job_id,
                            "ready_count": ready_count,
                            "reason": (
                                f"Endpoint {ep.job_id} idle for "
                                f">{policy.scale_down_idle_sec}s "
                                f"(ready_count={ready_count}, min={policy.min_replicas})"
                            ),
                        }
                    )
                    ready_count -= 1  # Track projected count

        return decisions
