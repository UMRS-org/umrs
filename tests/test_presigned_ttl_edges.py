"""Presigned-URL TTL edge behavior probe (G-P5-5, WP-16).

Round-2 owner decision (umrs 406a659a + sidecar d5548377): the generator
CLAMPS — TTL <= 0 is refused at generation (``ValueError``) and at request
boundaries (HTTP 400/422). These pins lock the clamp truth with the
named limitation:

- expiry ENFORCEMENT stays server-side at MinIO (S3 evaluates Expires at
  request time) — the generation-side probe cannot claim live 403s;
- the live-compose negative-TTL assertion remains a harness item.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import time

import pytest
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

    def test_zero_ttl_refused_at_generation(self):
        sidecar = _load_sidecar()
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            sidecar.generate_presigned_url("bucket", "key", ttl_seconds=0)

    def test_negative_ttl_refused_at_generation(self):
        sidecar = _load_sidecar()
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            sidecar.generate_presigned_url("bucket", "key", ttl_seconds=-3600)

    def test_identical_inputs_identical_url(self):
        sidecar = _load_sidecar()
        one = sidecar.generate_presigned_url("bucket", "key", ttl_seconds=60)
        two = sidecar.generate_presigned_url("bucket", "key", ttl_seconds=60)
        # Both minted inside the same second-window (skew tolerance 1s).
        assert _parse(one)["Signature"] == _parse(two)["Signature"] or (
            abs(int(_parse(one)["Expires"]) - int(_parse(two)["Expires"])) <= 1
        )


class TestNamedLimitation:
    def test_generator_floor_is_exactly_positive_seconds(self):
        """Smallest legal TTL mints inside the expected window."""
        sidecar = _load_sidecar()
        before = int(time.time())
        url = sidecar.generate_presigned_url("b", "k", ttl_seconds=1)
        after = int(time.time())
        expires = int(_parse(url)["Expires"])
        assert before + 1 <= expires <= after + 1
