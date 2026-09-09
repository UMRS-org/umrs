"""Entry point: ``python -m umrs.infer_proxy`` — uvicorn on UMRS_INFER_PORT (default 8443).

The unseal private key comes from UMRS_UNSEAL_PRIVKEY_PATH (podman secret)
or UMRS_UNSEAL_PRIVKEY_B64 (dev). TLS termination inside the container is
identity-neutral plain HTTP against the internal network until the internal
CA wiring lands (plan R1-3 mounts certs per deployment).
"""

from __future__ import annotations

import os

import uvicorn

from .app import create_app


def main() -> None:
    from umrs.security.seal import recipient_keys
    from umrs.shared.pg import get_default_sync_pool

    pool = get_default_sync_pool()
    if pool is None:
        raise RuntimeError("inference proxy requires the shared umrs_db pool")
    privkey = recipient_keys().unseal_privkey()
    app = create_app(umrs_db_pool=pool, unseal_privkey=privkey)
    uvicorn.run(
        app,
        host=os.environ.get("UMRS_INFER_HOST", "127.0.0.1"),
        port=int(os.environ.get("UMRS_INFER_PORT", "8443")),
        log_level=os.environ.get("UMRS_INFER_LOG_LEVEL", "warning"),
    )


if __name__ == "__main__":
    main()
