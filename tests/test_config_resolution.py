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
