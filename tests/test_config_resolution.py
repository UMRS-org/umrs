"""Regression test for config resolution shadowing bug.

This test ensures that module-level constants in config.py are not
accidentally shadowed by subsequent assignments, which caused
umrs profile resolution to fail silently.

See .github/instructions/umrs-branch-separation.instructions.md §5.
"""

import pytest


class TestConfigResolution:
    """Verify config constants are not shadowed."""

    def test_profile_file_not_none(self):
        """DEFAULT_CURRENT_CONNECTION_PROFILE_FILE must not be None."""
        from umrs.shared.config import DEFAULT_CURRENT_CONNECTION_PROFILE_FILE

        assert DEFAULT_CURRENT_CONNECTION_PROFILE_FILE is not None
        assert DEFAULT_CURRENT_CONNECTION_PROFILE_FILE == "current_connection_profile"

    def test_s3_config_file_not_none(self):
        """DEFAULT_CURRENT_S3_CONFIG_FILE must not be None."""
        from umrs.shared.config import DEFAULT_CURRENT_S3_CONFIG_FILE

        assert DEFAULT_CURRENT_S3_CONFIG_FILE is not None
        assert DEFAULT_CURRENT_S3_CONFIG_FILE == "current_s3_config"

    def test_get_current_connection_profile_file(self):
        """get_current_connection_profile_file() must return a non-None string."""
        from umrs.shared.config import get_current_connection_profile_file

        result = get_current_connection_profile_file()
        assert result is not None
        assert isinstance(result, str)
        assert result == "current_connection_profile"

    def test_get_current_s3_config_file(self):
        """get_current_s3_config_file() must return a non-None string."""
        from umrs.shared.config import get_current_s3_config_file

        result = get_current_s3_config_file()
        assert result is not None
        assert isinstance(result, str)
        assert result == "current_s3_config"


class TestModelStagingRoot:
    """API model_download staging root resolution (fix 2026-09-08).

    Live failure: stage fell back to /etc/umrs/models on umrs-front, the
    registry path validator rejected it, and POST /umrs/models/stage 500ed.
    """

    def test_env_override_wins(self, monkeypatch, tmp_path):
        from umrs.shared.config import get_resolved_model_staging_root

        monkeypatch.setenv("UMRS_MODEL_STAGING_ROOT", str(tmp_path))
        assert get_resolved_model_staging_root() == str(tmp_path)

    def test_site_model_cache_dir_used_when_no_env(self, monkeypatch):
        """site.storage.model_cache_dir is honored when no env override."""
        from types import SimpleNamespace

        import umrs.shared.config as cfg_mod

        monkeypatch.delenv("UMRS_MODEL_STAGING_ROOT", raising=False)

        fake_site = SimpleNamespace(
            storage=SimpleNamespace(model_cache_dir="/lvs0/rccs-dmp/x/umrs_models")
        )

        import umrs.orchestrator.site_config as sc

        monkeypatch.setattr(sc, "get_site_config", lambda: fake_site)
        monkeypatch.setattr(
            cfg_mod, "get_resolved_umrs_subdir", lambda name: f"/should-not-be-used/{name}"
        )

        result = cfg_mod.get_resolved_model_staging_root()
        assert result == "/lvs0/rccs-dmp/x/umrs_models"

    def test_fallback_to_default_subdir_when_no_site_config(self, monkeypatch):
        import umrs.shared.config as cfg_mod

        monkeypatch.delenv("UMRS_MODEL_STAGING_ROOT", raising=False)

        import umrs.orchestrator.site_config as sc

        monkeypatch.setattr(
            sc,
            "get_site_config",
            lambda: (_ for _ in ()).throw(RuntimeError("no site")),
        )
        monkeypatch.setattr(
            cfg_mod, "get_resolved_umrs_subdir", lambda name: f"/etc/umrs/{name}"
        )

        assert cfg_mod.get_resolved_model_staging_root() == "/etc/umrs/models"
