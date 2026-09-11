#!/usr/bin/env python3
"""STS renewal loop for the mini-RiVault stack (WP-11 round 3 design leg).

Problem the ADR-043 rollout block names: per-attempt STS sessions cap at
3600s while a serving job runs days; provision-time env-freeze would brick
the serving plane's S3 hooks on the hour. This process is the renewal
counterpart: it polls ``.../credentials/sts`` on a TTL-fraction cadence and
atomically re-renders ``/shared/sts.env`` (loaded via podman's
``env_file:`` by any consumer reading creds per-request).

Rules of play (nom fabrications):
- cadence = min(ttl/2, ttl-180s), floor 60s — the leanest bound that keeps
  a session alive while not storming the STS venue.
- the env file is written ATOMICALLY (tmp + replace); a half-written file
  must never be read under a serving restart.
- nothing secret EVER lands in logs (identity string + expiration only);
  log messages carry bucket/endpoint, never token/access/secret values.
- failures back off (x2, cap 300s) and the LAST-GOOD file stays in place
  until the next successful mint; boot failures halt the service.

Contract called: ``POST {base}/umrs/jobs/{job_id}/attempts/{attempt_id}/
    credentials/sts`` → {credentials:{access_key,secret_key,session_token,
    expiration}, bucket, endpoint_url}.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from urllib import error, request
from urllib.request import urlopen  # noqa: F401 (kept for symptom mapping)

logger = logging.getLogger("sts_renewal")

_ENV_FILE = os.environ.get("STS_ENV_PATH", "/shared/sts.env")
_BASE = (os.environ.get("UMRS_BASE_URL") or "").rstrip("/")
_TOKEN = os.environ.get("UMRS_AUTH_TOKEN") or os.environ.get("R4_API_KEY") or ""
_JOB_ID = os.environ.get("RIVAULT_JOB_ID", "")
_ATTEMPT_ID = os.environ.get("RIVAULT_ATTEMPT_ID", "")
_TTL_HINT = int(os.environ.get("RIVAULT_STS_TTL_SEC", "3000"))

_BACKOFF_BASE = 30.0
_BACKOFF_MAX = 300.0


def renewal_interval(ttl_sec: int) -> float:
    """Cadence below the session in every regime.

    TTL > 240s: min(ttl/2, ttl-180) floored at 60s (the cheapest bound that
    keeps a session alive without storming the venue). TTL ≤ 240s (boot
    probes / fine-grained test lives): ttl/2 with a 5s floor — never
    exceeds the session itself.
    """
    if ttl_sec <= 240:
        return max(ttl_sec / 2.0, 5.0)
    return max(60.0, min(ttl_sec / 2.0, ttl_sec - 180.0))


def mint_sts() -> dict:
    """POST the credentials/sts route; decode its contract payload."""
    url = f"{_BASE}/umrs/jobs/{_JOB_ID}/attempts/{_ATTEMPT_ID}/credentials/sts"
    req = request.Request(url, method="POST", data=b"")
    if _TOKEN:
        req.add_header("Authorization", f"Bearer {_TOKEN}")
    with urlopen(req, timeout=30.0) as resp:  # noqa: S310 — own venue URL
        return json.loads(resp.read().decode("utf-8"))


def render_env_file(payload: dict) -> str:
    """Contract payload → env-file body (no interpolation quirk)."""
    creds = payload.get("credentials") or {}
    lines = [
        f"S3_ENDPOINT_URL={payload.get('endpoint_url', '')}",
        f"S3_BUCKET={payload.get('bucket', '')}",
        f"S3_ACCESS_KEY={creds.get('access_key', '')}",
        f"S3_SECRET_KEY={creds.get('secret_key', '')}",
        f"S3_SESSION_TOKEN={creds.get('session_token', '')}",
    ]
    return "\n".join(lines) + "\n"


def write_env_atomic(body: str, path: str = _ENV_FILE) -> None:
    """Atomic rewrite: tmp-sibling + os.replace (never a torn read)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", opener=lambda *_: os.open(tmp, os.O_WRONLY | os.O_CREAT, 0o600)) as fh:
        fh.write(body)
    os.replace(tmp, path)


_REQUIRED_ENV = ("UMRS_BASE_URL", "RIVAULT_JOB_ID", "RIVAULT_ATTEMPT_ID")


def _check_env() -> list[str]:
    values = {"UMRS_BASE_URL": _BASE, "RIVAULT_JOB_ID": _JOB_ID, "RIVAULT_ATTEMPT_ID": _ATTEMPT_ID}
    return [k for k, v in values.items() if not v]


def run() -> int:
    """Renew-forever loop; only fatal boot config exits nonzero."""
    missing = _check_env()
    if missing:
        logger.error("missing required env: %s", ",".join(missing))
        return 2
    interval = renewal_interval(_TTL_HINT)
    logger.info("sts renewal start (job=%s interval=%ss)", _JOB_ID, interval)
    backoff = _BACKOFF_BASE
    while True:
        try:
            payload = mint_sts()
            write_env_atomic(render_env_file(payload))
            expires = (payload.get("credentials") or {}).get("expiration", "")
            logger.info("sts renewed (expires=%s)", expires)
            backoff = _BACKOFF_BASE
            time.sleep(interval)
        except error.URLError as exc:  # venue fault
            logger.warning("sts mint failed (venue): %s — retry in %ss", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:  # local/write fault
            logger.warning("sts mint failed (local): %s — retry in %ss", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX)


def main() -> int:
    logging.basicConfig(level=os.environ.get("RIVAULT_STS_LOG_LEVEL", "INFO"),
                        format="[sts-renewal] %(levelname)s %(message)s")
    try:
        return run()
    except KeyboardInterrupt:
        logger.info("sts renewal stopped (signal)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
