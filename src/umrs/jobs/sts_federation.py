"""Per-attempt STS federation (WP-11 round 2, G-P5-3).

Round 1 minted a scoped *credential record* — but the signing party stayed
ambient env credentials (named limitation in ADR-043). This module closes
that: for one attempt's recorded key space it asks the S3 venue for **STS
temporary credentials** (`AssumeRole` with an inline prefix-scoped policy),
so anything downstream — mini-RiVault, task workers — can hold an identity
that (a) lives only for the attempt TTL and (b) cannot touch keys outside
the attempt scope even if leaked.

Rules (kept from round 1, now enforced at the boundary too):

- **Fail-closed.** No boto3, no endpoint, any AssumeRole error →
  :class:`STSFederationError`; there is NO silent fallback to ambient
  credentials (the fallback would masquerade as per-attempt identity).
- **Secrets never persist.** The returned AccessKey/SecretKey/SessionToken
  travel to the HTTP caller only. They are never written to ``metrics``
  or logs — the mint record (scope/TTL/expiry, no secret) is the already-
  landed audit trail from round 1.
- **Prefix-true policy.** The inline policy pins
  ``s3:GetObject/PutObject/DeleteObject`` to ``<bucket>/<prefix>*`` (one
  statement per recorded prefix) and ``s3:ListBucket`` conditioned off the
  prefixed; nothing wildcard-host.
"""

from __future__ import annotations

import json
from typing import Any

from .attempt_credentials import AttemptCredentials

_ACTIONS = ("s3:GetObject", "s3:PutObject", "s3:DeleteObject")
_MAX_TTL_SEC = 3600  # MinIO STS hard ceiling for AssumeRole sessions


class STSFederationError(RuntimeError):
    """Raised when per-attempt identity cannot be federated. Fail-closed."""


def attempt_policy_document(
    record: AttemptCredentials, *, bucket: str
) -> dict[str, Any]:
    """Inline policy scoping the role session to the attempt's key space."""
    statements: list[dict[str, Any]] = []
    for idx, prefix in enumerate(record.scope_prefixes):
        statements.append(
            {
                "Sid": f"AttemptScope{idx}",
                "Effect": "Allow",
                "Action": list(_ACTIONS),
                "Resource": [f"arn:aws:s3:::{bucket}/{prefix}*"],
            }
        )
    statements.append(
        {
            "Sid": "AttemptScopeList",
            "Effect": "Allow",
            "Action": ["s3:ListBucket"],
            "Resource": [f"arn:aws:s3:::{bucket}"],
            "Condition": {
                "StringLike": {
                    "s3:prefix": [f"{p}*" for p in record.scope_prefixes]
                }
            },
        }
    )
    return {"Version": "2012-10-17", "Statement": statements}


def assume_role_for_attempt(
    record: AttemptCredentials,
    *,
    bucket: str,
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    ttl_sec: int | None = None,
    sts_client: Any | None = None,
) -> dict[str, str]:
    """Mint per-attempt STS credentials scoped to *record*'s key space.

    ``sts_client`` is injectable for tests; otherwise a boto3 STS client is
    built against the ambient endpoint identity (round-1's ambient creds do
    the signing ONE level up — the venue hands back the scoped identity).
    Returns {access_key, secret_key, session_token, expiration}; raises
    STSFederationError on any failure.
    """
    ttl = min(int(ttl_sec or record.ttl_sec), _MAX_TTL_SEC)
    if ttl <= 0:
        raise STSFederationError("STS ttl must be positive")
    policy = attempt_policy_document(record, bucket=bucket)
    session_name = f"umrs-{record.job_id}-{record.attempt_no}"[:64]
    try:
        client = sts_client
        if client is None:
            import boto3  # imported lazily: venue dep, not module dep

            client = boto3.client(
                "sts",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        resp = client.assume_role(
            RoleArn="arn:aws:iam:::umrs-attempt",
            RoleSessionName=session_name,
            Policy=json.dumps(policy),
            DurationSeconds=ttl,
        )
    except Exception as exc:  # any venue failure must fail closed
        raise STSFederationError(f"AssumeRole failed: {exc}") from exc
    creds = (resp or {}).get("Credentials", {})
    missing = {"AccessKeyId", "SecretAccessKey", "SessionToken"} - set(creds)
    if missing:
        raise STSFederationError(
            f"AssumeRole response missing credential fields: {sorted(missing)}"
        )
    expiration = creds.get("Expiration")
    return {
        "access_key": creds["AccessKeyId"],
        "secret_key": creds["SecretAccessKey"],
        "session_token": creds["SessionToken"],
        "expiration": expiration.isoformat()
        if hasattr(expiration, "isoformat")
        else str(expiration or ""),
    }
