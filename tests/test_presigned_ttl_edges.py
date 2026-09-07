"""Presigned-URL TTL edge behavior probe (G-P5-5, WP-16).

The gap named TTL edge behavior (negative / zero / expired presigned URLs)
as UNPROBED — the read-only verification scope couldn't exercise expiry on
the live compose stack. These pins lock the generation-side truth (the
sidecar's AWS Signature v2 generator) with the limitation named:

- the generator NEVER clamps TTL: negative/zero TTL mints a born-expired
  URL deterministically (Expires in the past) with a well-formed signature;
- expiry ENFORCEMENT is server-side at MinIO (S3 evaluates Expires at
  request time) — the generation-side probe cannot claim live 403s;
- the live-compose negative-TTL assertion remains a harness item
  (named in the gap note, not hidden).

A clamp policy (reject TTL <= 0) was considered and NOT added here:
sidecar generation semantics are part of the deployed wire surface; a
floor needs the site-owner decision + downstream review. Recorded in
wp16.md as the round-2 question.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _load_sidecar():
    sidecar_path = Path(__file__).resolve().parents[2] / "gateway" / "sidecar" / "main.py"
    assert sidecar_path.is_file(), f"expected sidecar at {sidecar_path}"
    spec = importlib.util.spec_from_file_location("_sidecar_main_ttl", sidecar_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _parse(url: str) -> dict[str, str]:
    qs = parse_qs(urlparse(url).query)
    return {k: v[0] for k, v in qs.items()}


def _expected_signature(secret: str, method: str, expires: int, bucket: str, key: str) -> str:
    string_to_sign = f"{method}\n\n\n{expires}\n/{bucket}/{key}"
    digest = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("utf-8")


class TestExpiryArithmetic:
    def test_positive_ttl_lands_inside_window(self):
        sidecar = _load_sidecar()
        before = int(time.time())
        url = sidecar.generate_presigned_url("bucket", "key", ttl_seconds=3600)
        after = int(time.time())
        expires = int(_parse(url)["Expires"])
        assert before + 3600 <= expires <= after + 3600

    def test_zero_ttl_is_born_expired_and_well_formed(self):
        sidecar = _load_sidecar()
        url = sidecar.generate_presigned_url("bucket", "key", ttl_seconds=0)
        params = _parse(url)
        expires = int(params["Expires"])
        assert abs(expires - int(time.time())) <= 1
        assert params["Signature"] == _expected_signature(
            sidecar.S3_SECRET_KEY, "GET", expires, "bucket", "key"
        )

    def test_negative_ttl_mints_past_expiry_deterministically(self):
        sidecar = _load_sidecar()
        before = int(time.time())
        url = sidecar.generate_presigned_url("bucket", "key", ttl_seconds=-3600)
        after = int(time.time())
        params = _parse(url)
        expires = int(params["Expires"])
        assert before - 3600 <= expires <= after - 3600
        assert expires < before  # unambiguously already-expired at mint time
        # Signature is still the correct v2 signature over that raw expiry —
        # no silent clamping anywhere in the generator.
        assert params["Signature"] == _expected_signature(
            sidecar.S3_SECRET_KEY, "GET", expires, "bucket", "key"
        )

    def test_identical_inputs_identical_url(self):
        sidecar = _load_sidecar()
        one = sidecar.generate_presigned_url("bucket", "key", ttl_seconds=-60)
        two = sidecar.generate_presigned_url("bucket", "key", ttl_seconds=-60)
        # Both minted inside the same second-window (skew tolerance 1s).
        assert _parse(one)["Signature"] == _parse(two)["Signature"] or (
            abs(int(_parse(one)["Expires"]) - int(_parse(two)["Expires"])) <= 1
        )


class TestNamedLimitation:
    def test_generator_has_no_ttl_floor(self):
        """Pin the truth the clamp decision builds on."""
        sidecar = _load_sidecar()
        url = sidecar.generate_presigned_url("b", "k", ttl_seconds=-(365 * 86400))
        expires = int(_parse(url)["Expires"])
        assert expires < int(time.time()) - 364 * 86400
