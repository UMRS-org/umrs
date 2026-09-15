"""Deployment-scoped model visibility filtering.

Some deployments expose only a curated subset of the shared model registry.
For example, the AI4S frontier cluster should surface only large frontier models
(GLM-5.1, GLM-5.2, Kimi-K2.6, DeepSeek-V4-Pro family), while smaller models live on the
RCCS cluster. Because both clusters can share the same model registry, this
curation is driven by environment so no code or registry change is needed:

- ``UMRS_MODEL_ALLOWLIST``: comma-separated model names. When set, ONLY these
  models are listed. Takes precedence over the denylist.
- ``UMRS_MODEL_DENYLIST``: comma-separated model names. These models are hidden.

Matching is case-insensitive on the model name. Empty/unset means no filtering.

This filter only affects *listing* (model window / ``/v1/models``); it never
blocks provisioning a specific model by name, so operators retain full control.
"""

from __future__ import annotations

import os

_ALLOWLIST_ENV = "UMRS_MODEL_ALLOWLIST"
_DENYLIST_ENV = "UMRS_MODEL_DENYLIST"


def _parse_env_set(var: str) -> set[str]:
    raw = os.environ.get(var, "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def is_model_visible(name: str | None) -> bool:
    """Return True if a model name is visible under the deployment's policy."""
    if not name:
        return True
    key = name.strip().lower()
    allow = _parse_env_set(_ALLOWLIST_ENV)
    if allow:
        return key in allow
    deny = _parse_env_set(_DENYLIST_ENV)
    if deny:
        return key not in deny
    return True
