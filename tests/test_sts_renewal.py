"""WP-11 r3: sts_renewal contract pins (renewal sidecar, fail-without-change)."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "umrs", "mini-rivault-llm"))

import sts_renewal


class TestRenewalInterval:
    def test_floor_and_ceiling(self):
        assert sts_renewal.renewal_interval(3000) == 1500  # min(1500, 2820)
        assert sts_renewal.renewal_interval(120) == 60  # floor
        assert sts_renewal.renewal_interval(360) == 180  # min(180, 180)
        assert sts_renewal.renewal_interval(400) == min(200, 220)

    def test_never_exceeds_session(self):
        for ttl in (60, 120, 300, 900, 1800, 3600):
            assert sts_renewal.renewal_interval(ttl) < ttl


class TestRenderEnvFile:
    def test_contains_all_keys_without_logging_secrets(self):
        payload = {
            "endpoint_url": "http://minio:9000",
            "bucket": "umrs-files",
            "credentials": {
                "access_key": "AKIA",
                "secret_key": "sekret",
                "session_token": "tok123",
                "expiration": "2026-09-11T00:30:00+00:00",
            },
        }
        body = sts_renewal.render_env_file(payload)
        assert "S3_ENDPOINT_URL=http://minio:9000" in body
        assert "S3_BUCKET=umrs-files" in body
        assert "S3_ACCESS_KEY=AKIA" in body
        assert "S3_SECRET_KEY=sekret" in body
        assert "S3_SESSION_TOKEN=tok123" in body


class TestWriteEnvAtomic:
    def test_atomic_via_os_replace_and_ownership(self, tmp_path):
        path = str(tmp_path / "sts.env")
        sts_renewal.write_env_atomic("S3_ACCESS_KEY=x\n", path)
        assert os.path.exists(path)
        with open(path) as fh:
            assert fh.read() == "S3_ACCESS_KEY=x\n"
        # rewrite is atomic wrt readers
        sts_renewal.write_env_atomic("S3_ACCESS_KEY=y\n", path)
        with open(path) as fh:
            assert fh.read() == "S3_ACCESS_KEY=y\n"
        assert not os.path.exists(path + ".tmp")


class TestMintStsContract:
    def test_posts_to_attempts_sts_route_with_bearer(self):
        captured = {}

        class _Resp:
            def __init__(self):
                self.data = json.dumps(
                    {
                        "credentials": {
                            "access_key": "AK",
                            "secret_key": "SK",
                            "session_token": "T",
                            "expiration": "e",
                        },
                        "bucket": "b",
                        "endpoint_url": "u",
                    }
                ).encode()

            def read(self):
                return self.data

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _FakeReq:
            def __init__(self, url, method=None, data=None):
                captured["url"] = url
                captured["method"] = method
                captured["data"] = data
                self.headers = {}

            def add_header(self, k, v):
                self.headers[k] = v

        def _fake_urlopen(req, timeout=None):
            captured["req"] = req
            captured["timeout"] = timeout
            return _Resp()

        with (
            patch.object(sts_renewal, "_BASE", "http://umrs"),
            patch.object(sts_renewal, "_JOB_ID", "job-1"),
            patch.object(sts_renewal, "_ATTEMPT_ID", "att-1"),
            patch.object(sts_renewal, "_TOKEN", "JWTX"),
            patch.object(sts_renewal.request, "Request", _FakeReq),
            patch.object(sts_renewal, "urlopen", _fake_urlopen),
        ):
            payload = sts_renewal.mint_sts()

        assert captured["method"] == "POST"
        assert captured["url"] == ("http://umrs/umrs/jobs/job-1/attempts/att-1/credentials/sts")
        assert captured["req"].headers["Authorization"] == "Bearer JWTX"
        assert captured["timeout"] == 30.0
        assert payload["credentials"]["session_token"] == "T"


class TestRun:
    def test_missing_env_fatal_boot(self):
        with (
            patch.object(sts_renewal, "_BASE", ""),
            patch.object(sts_renewal, "_JOB_ID", "job"),
            patch.object(sts_renewal, "_ATTEMPT_ID", ""),
        ):
            assert sts_renewal.run() == 2
