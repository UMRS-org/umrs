"""Entry point: ``python -m umrs.infer_proxy`` — uvicorn on UMRS_INFER_PORT (default 8443).

The unseal private key comes from UMRS_UNSEAL_PRIVKEY_PATH (podman secret)
or UMRS_UNSEAL_PRIVKEY_B64 (dev). TLS termination inside the container is
identity-neutral plain HTTP against the internal network until the internal
CA wiring lands (plan R1-3 mounts certs per deployment).

ADR-099 D-5: the quota enforcer reads the session-policy limits (cached
30 s) and the day's usage; ``UMRS_QUOTA_DAY_TZ`` (default UTC) names the
calendar day the daily token caps count.
"""

from __future__ import annotations

import os
from typing import Any

import uvicorn

from .app import create_app


def build_quota_enforcer(pool: Any) -> Any:
    """The proxy's quota enforcer over the session-policy limits (ADR-099)."""
    from umrs.quotas.enforcer import QuotaEnforcer
    from umrs.session_policy.store import effective_limits_cached

    return QuotaEnforcer(
        pool,
        effective_limits_cached(pool),
        day_tz=os.environ.get("UMRS_QUOTA_DAY_TZ", "UTC"),
    )


def main() -> None:
    from umrs.security.seal import recipient_keys
    from umrs.shared.pg import get_default_sync_pool

    pool = get_default_sync_pool()
    if pool is None:
        raise RuntimeError("inference proxy requires the shared umrs_db pool")
    privkey = recipient_keys().unseal_privkey()
    app = create_app(
        umrs_db_pool=pool,
        unseal_privkey=privkey,
        quota_enforcer=build_quota_enforcer(pool),
    )
    uvicorn.run(
        app,
        host=os.environ.get("UMRS_INFER_HOST", "127.0.0.1"),
        port=int(os.environ.get("UMRS_INFER_PORT", "8443")),
        log_level=os.environ.get("UMRS_INFER_LOG_LEVEL", "warning"),
    )


if __name__ == "__main__":
    main()
