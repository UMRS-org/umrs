"""Session policy: limit rules, their resolution and storage (ADR-099).

Every remote session carries one resolved policy, computed at start from
the site defaults, the admin and project-owner rules in
``session_policy_rules`` (migration 0125) and the user's web choice:

- :mod:`.model`: the rule document, its validation, :class:`PolicyError`;
- :mod:`.resolve`: the fold (effective limits) and the choice;
- :mod:`.store`: the rule store, the cached lookup for the inference proxy.

A foundation package: no imports from ``umrs.api`` (ADR-053).
"""

from .model import (
    LAYERS,
    SCOPES,
    WEB_MODES,
    PolicyError,
    PolicyRule,
    WebChoice,
    domain_within,
    limit_value,
    validate_rule_doc,
)
from .resolve import (
    check_owner_rule,
    effective_limits,
    legacy_web_default,
    resolve_session_policy,
    site_defaults,
    web_choice_required,
)
from .store import (
    LimitsCache,
    PolicyRuleStore,
    effective_limits_cached,
    effective_limits_for,
)

__all__ = [
    "LAYERS",
    "SCOPES",
    "WEB_MODES",
    "LimitsCache",
    "PolicyError",
    "PolicyRule",
    "PolicyRuleStore",
    "WebChoice",
    "check_owner_rule",
    "domain_within",
    "effective_limits",
    "effective_limits_cached",
    "effective_limits_for",
    "legacy_web_default",
    "limit_value",
    "resolve_session_policy",
    "site_defaults",
    "validate_rule_doc",
    "web_choice_required",
]
