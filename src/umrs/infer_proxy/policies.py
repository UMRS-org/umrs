"""Provider templates for the credential-injecting inference proxy (R1-2).

Path policy per provider kind (design §3.3.3): default-allow of the
inference-class prefixes, default-deny of everything administrative — key
management, organization, billing, dashboards, realtime (R3).

``allow()`` never trusts the raw path: it normalizes first (`.`, `..`-free,
single-slashed, lowercase-compared for the deny prefixes).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPolicy:
    provider_kind: str
    inject_header: str  # header name carrying the provider credential
    inject_format: str  # "bearer" | "plain"
    allowed: tuple[tuple[str, str], ...]  # (method, prefix) default-open
    denied_prefixes: tuple[str, ...]  # default-denied even if matching `allowed`
    opt_in_prefixes: dict[str, tuple[tuple[str, str], ...]]  # allow_paths key → rules
    action_whitelist: tuple[str, ...] = ()  # google-style `<model>:<action>` verbs


def _norm(path: str) -> str:
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if ".." in parts:
        return "__traversal__"
    return "/" + "/".join(parts)


def _deny_norms(denied: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_norm(d) for d in denied)


OPENAI_LIKE = ProviderPolicy(
    provider_kind="openai",
    inject_header="authorization",
    inject_format="bearer",
    allowed=(
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/moderations"),
        ("GET", "/v1/models"),
    ),
    denied_prefixes=(
        "/v1/organization",
        "/v1/api_keys",
        "/v1/dashboard",
        "/v1/billing",
        "/v1/realtime",
        "/v1/fine_tuning",
    ),
    opt_in_prefixes={
        "files": (("POST", "/v1/files"), ("GET", "/v1/files")),
        "batches": (("POST", "/v1/batches"), ("GET", "/v1/batches")),
        "images": (("POST", "/v1/images"),),
        "audio": (("POST", "/v1/audio"),),
        "fine_tuning": (("POST", "/v1/fine_tuning"), ("GET", "/v1/fine_tuning")),
        "vector_stores": (("POST", "/v1/vector_stores"), ("GET", "/v1/vector_stores")),
    },
)

ANTHROPIC = ProviderPolicy(
    provider_kind="anthropic",
    inject_header="x-api-key",
    inject_format="plain",
    allowed=(
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
        ("GET", "/v1/models"),
    ),
    # /v1/messages prefix covers batches — they stay behind the opt-in.
    denied_prefixes=("/v1/organizations", "/v1/messages/batches"),
    opt_in_prefixes={
        "batches": (("POST", "/v1/messages/batches"), ("GET", "/v1/messages/batches")),
        "files": (("POST", "/v1/files"), ("GET", "/v1/files")),
    },
)

AZURE = ProviderPolicy(
    provider_kind="azure_openai",
    inject_header="api-key",
    inject_format="plain",
    allowed=(
        ("POST", "/openai/deployments"),
        ("POST", "/openai/responses"),
        ("GET", "/openai/models"),
    ),
    denied_prefixes=(),  # everything not matched by allowed is denied anyway
    opt_in_prefixes={},
)

GOOGLE = ProviderPolicy(
    provider_kind="google",
    inject_header="x-goog-api-key",
    inject_format="plain",
    allowed=(
        ("POST", "/v1beta/models"),
        ("GET", "/v1beta/models"),
    ),
    denied_prefixes=(),
    opt_in_prefixes={},
    action_whitelist=(
        "generateContent",
        "streamGenerateContent",
        "countTokens",
        "embedContent",
    ),
)


def policy_for(provider_kind: str) -> ProviderPolicy:
    if provider_kind == "openai_compat":
        openai_compat = ProviderPolicy(
            provider_kind="openai_compat",
            inject_header=OPENAI_LIKE.inject_header,
            inject_format=OPENAI_LIKE.inject_format,
            allowed=OPENAI_LIKE.allowed,
            denied_prefixes=OPENAI_LIKE.denied_prefixes,
            opt_in_prefixes=OPENAI_LIKE.opt_in_prefixes,
        )
        return openai_compat
    registry = {
        "openai": OPENAI_LIKE,
        "anthropic": ANTHROPIC,
        "azure_openai": AZURE,
        "google": GOOGLE,
    }
    try:
        return registry[provider_kind]
    except KeyError:
        raise ValueError(f"unknown provider_kind {provider_kind!r}") from None


def _matches(rules: tuple[tuple[str, str], ...], method: str, norm: str) -> bool:
    return any(
        method.upper() == rule_method and norm.startswith(_norm(prefix).lower())
        for rule_method, prefix in rules
    )


def _action_allowed(policy: ProviderPolicy, norm: str) -> bool:
    """google-style verb check: `<model>:<action>` suffix must be whitelisted."""
    if not policy.action_whitelist:
        return True
    if ":" not in norm:
        return True  # path segment without an action (e.g. GET /v1beta/models/x)
    action = norm.rsplit(":", 1)[-1]
    return action in {verb.lower() for verb in policy.action_whitelist}


def allows(
    policy: ProviderPolicy, method: str, path: str, allow_paths: tuple[str, ...] = ()
) -> bool:
    """True when method+path is open for this source's opt-ins.

    Deny first: matching deny prefixes kill the path unless an ENABLED
    opt-in rule also matches (e.g. fine_tuning pushed explicitly). An
    ``action_whitelist`` further gates `<model>:<action>` verbs.
    """
    norm = _norm(path).lower()
    if not _action_allowed(policy, norm):
        return False
    denied_hits = any(norm.startswith(d) for d in _deny_norms(policy.denied_prefixes))
    if denied_hits:
        enabled_opts = tuple(
            rule for key in allow_paths for rule in policy.opt_in_prefixes.get(key, ())
        )
        return _matches(enabled_opts, method, norm)
    if _matches(policy.allowed, method, norm):
        return True
    enabled_opts = tuple(
        rule for key in allow_paths for rule in policy.opt_in_prefixes.get(key, ())
    )
    return _matches(enabled_opts, method, norm)


def strip_credential_headers(headers: dict[str, str]) -> dict[str, str]:
    """Remove every inbound credential shape before injection (design §3.3.1)."""
    drop = {
        "authorization",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "cookie",
        "proxy-authorization",
    }
    return {k: v for k, v in headers.items() if k.lower() not in drop}


def inject_credential(
    policy: ProviderPolicy, headers: dict[str, str], api_key: str
) -> dict[str, str]:
    """Attach the provider credential in the provider's canonical header form."""
    value = f"Bearer {api_key}" if policy.inject_format == "bearer" else api_key
    merged = dict(headers)
    merged[policy.inject_header] = value
    return merged
