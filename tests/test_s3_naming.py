"""ADR-055 naming — backend-neutral S3 names win, MinIO-era names still work."""

from __future__ import annotations

import pytest
from umrs.api import ops
from umrs.shared import connection_profiles as cp


class TestConnectInfoServices:
    def _services(self, monkeypatch, **env):
        for k in ("UMRS_S3_URL", "UMRS_MINIO_URL", "MINIO_ENDPOINT", "RAG_URL"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        result: dict = {}
        ops._add_connect_info_extras(result, None)
        return result.get("services", {})

    def test_s3_url_is_canonical_and_minio_url_mirrors_it(self, monkeypatch):
        svc = self._services(monkeypatch, UMRS_S3_URL="https://front.example/s3")
        assert svc["s3_url"] == "https://front.example/s3"
        assert svc["minio_url"] == "https://front.example/s3"

    def test_legacy_env_names_still_populate_both_keys(self, monkeypatch):
        svc = self._services(monkeypatch, UMRS_MINIO_URL="http://legacy:9000")
        assert svc["s3_url"] == svc["minio_url"] == "http://legacy:9000"

    def test_new_name_wins_over_legacy(self, monkeypatch):
        svc = self._services(
            monkeypatch, UMRS_S3_URL="http://new:9000", UMRS_MINIO_URL="http://old:9000"
        )
        assert svc["s3_url"] == "http://new:9000"

    def test_model_declares_both_fields(self):
        fields = ops.ConnectInfoServices.model_fields
        assert {"s3_url", "minio_url", "rag_url"} <= set(fields)


class TestProfileUrlEnvPrecedence:
    @pytest.fixture
    def profile(self):
        prof = cp.ConnectionProfile.__new__(cp.ConnectionProfile)
        object.__setattr__(prof, "minio_url", None)
        object.__setattr__(prof, "server_host", "front.example")
        object.__setattr__(prof, "tunnel_mode", False)
        return prof

    def test_s3_env_first_then_minio_env(self, monkeypatch, profile):
        for k in (
            "UMRS_S3_URL",
            "UMRS_MINIO_URL",
            "UMRS_S3_HOST",
            "UMRS_MINIO_HOST",
            "UMRS_S3_PORT",
            "UMRS_MINIO_PORT",
        ):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("UMRS_MINIO_URL", "http://old:9000")
        assert cp._coerce_minio_url(profile) == "http://old:9000"
        monkeypatch.setenv("UMRS_S3_URL", "http://new:9000")
        assert cp._coerce_minio_url(profile) == "http://new:9000"

    def test_host_port_env_pair(self, monkeypatch, profile):
        for k in ("UMRS_S3_URL", "UMRS_MINIO_URL", "UMRS_MINIO_HOST", "UMRS_MINIO_PORT"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("UMRS_S3_HOST", "s3.example")
        monkeypatch.setenv("UMRS_S3_PORT", "9100")
        assert cp._coerce_minio_url(profile) == "http://s3.example:9100"
