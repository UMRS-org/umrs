from __future__ import annotations

import json

import pytest

# Every variable resolve_connection() / resolve_profile_name() read, including
# the object-store and RAG overrides of _coerce_minio_url / _coerce_rag_url. A
# value left in the developer's shell would otherwise steer these tests.
_UMRS_CONNECTION_ENV = (
    "UMRS_PROFILE",
    "XDG_CONFIG_HOME",
    "UMRS_SERVER_HOST",
    "UMRS_CONTROL_PORT",
    "UMRS_DATA_PORT",
    "UMRS_DATA_PORT_RANGE",
    "UMRS_DATA_BASE_PORT",
    "UMRS_TUNNEL_MODE",
    "UMRS_HEALTH_CHECK_TIMEOUT_MS",
    "UMRS_ZMQ_AUTH_TYPE",
    "UMRS_S3_URL",
    "UMRS_MINIO_URL",
    "UMRS_S3_HOST",
    "UMRS_MINIO_HOST",
    "UMRS_S3_PORT",
    "UMRS_MINIO_PORT",
    "RAG_URL",
)


@pytest.fixture(autouse=True)
def _isolated_umrs_config(tmp_path, monkeypatch):
    """Hermetic (plan H-3.6, 2026-10-18): the config dir is a temp dir.

    Before this, resolve_connection() read ~/.config/umrs/current_connection_profile
    and the profile it names, so a developer's real profile (e.g. one carrying a
    key the loader did not know yet) turned this file red on that machine only.
    """
    cfg_dir = tmp_path / "umrs-config"
    cfg_dir.mkdir()
    monkeypatch.setenv("UMRS_CONFIG", str(cfg_dir))
    _clear_connection_env(monkeypatch)
    return cfg_dir


def _clear_connection_env(monkeypatch) -> None:
    for name in _UMRS_CONNECTION_ENV:
        monkeypatch.delenv(name, raising=False)


def test_profile_lookup_reads_the_test_config_dir_not_home(_isolated_umrs_config):
    from umrs.shared.connection_profiles import get_connections_dir, resolve_connection

    connection = resolve_connection()

    assert get_connections_dir().is_relative_to(_isolated_umrs_config.resolve())
    assert connection.sources["server_host"] == "default"


def test_shell_object_store_and_rag_overrides_are_cleared(tmp_path, monkeypatch):
    """A developer's UMRS_S3_URL / RAG_URL must not beat the profile (2026-10-18)."""
    from umrs.shared.connection_profiles import (
        _coerce_minio_url,
        _coerce_rag_url,
        load_connection_profile_file,
    )

    monkeypatch.setenv("UMRS_S3_URL", "http://shell-s3:1")
    monkeypatch.setenv("UMRS_S3_HOST", "shell-s3-host")
    monkeypatch.setenv("RAG_URL", "http://shell-rag:2")
    _clear_connection_env(monkeypatch)
    profile_path = tmp_path / "svc.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "svc",
                "mode": "direct",
                "zmq": {"host": "data0.example"},
                "services": {"s3_url": "https://s3.example", "rag_url": "https://rag.example"},
            }
        ),
        encoding="utf-8",
    )
    profile = load_connection_profile_file(profile_path)

    assert _coerce_minio_url(profile) == "https://s3.example"
    assert _coerce_rag_url(profile) == "https://rag.example"


def test_parse_data_port_range_accepts_single_value_string():
    from umrs.shared.connection_profiles import parse_data_port_range

    port_range = parse_data_port_range("5556")

    assert port_range.start == 5556
    assert port_range.end == 5556
    assert port_range.label == "5556"


def test_parse_data_port_range_accepts_range_mapping():
    from umrs.shared.connection_profiles import parse_data_port_range

    port_range = parse_data_port_range({"start": 5556, "end": 5564})

    assert port_range.start == 5556
    assert port_range.end == 5564
    assert port_range.label == "5556-5564"


def test_load_connection_profile_file_with_data_port_range(tmp_path):
    from umrs.shared.connection_profiles import load_connection_profile_file

    profile_path = tmp_path / "range-profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "range-profile",
                "mode": "ssh_tunnel",
                "zmq": {
                    "host": "localhost",
                    "control_port": 5555,
                    "data_port_range": {"start": 5556, "end": 5560},
                    "tunnel_mode": True,
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_connection_profile_file(profile_path)

    assert profile.data_base_port == 5556
    assert profile.data_port_range.start == 5556
    assert profile.data_port_range.end == 5560
    assert profile.to_dict()["data_port_range_label"] == "5556-5560"


def test_load_connection_profile_file_preserves_legacy_data_base_port(tmp_path):
    from umrs.shared.connection_profiles import load_connection_profile_file

    profile_path = tmp_path / "legacy-profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "legacy-profile",
                "mode": "direct",
                "zmq": {
                    "host": "127.0.0.1",
                    "data_base_port": 6001,
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_connection_profile_file(profile_path)

    assert profile.data_base_port == 6001
    assert profile.data_port_range.start == 6001
    assert profile.data_port_range.end == 6001


def test_resolve_connection_prefers_env_data_port_range_override(monkeypatch):
    from umrs.shared.connection_profiles import resolve_connection

    monkeypatch.setenv("UMRS_DATA_PORT_RANGE", "7000-7002")
    monkeypatch.delenv("UMRS_DATA_PORT", raising=False)
    monkeypatch.delenv("UMRS_DATA_BASE_PORT", raising=False)

    connection = resolve_connection()

    assert connection.data_base_port == 7000
    assert connection.data_port_range.start == 7000
    assert connection.data_port_range.end == 7002
    assert connection.to_dict()["data_port_range_label"] == "7000-7002"


# ── OBJ-20260914-001 naming migration: profile key minio_url → s3_url ───


def test_profile_accepts_s3_url_key(tmp_path):
    """ADR-055 names: the object-store key is s3_url; minio_url still accepted."""
    import json

    from umrs.shared.connection_profiles import (
        load_connection_profile_file,
    )

    profile_path = tmp_path / "s3-named.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "s3-named",
                "mode": "direct",
                "zmq": {"host": "data0.example"},
                "services": {
                    "s3_url": "https://data0.example",
                    "rag_url": "http://data0.example:8081",
                },
            }
        ),
        encoding="utf-8",
    )
    profile = load_connection_profile_file(profile_path)
    assert profile.s3_url == "https://data0.example"
    assert profile.minio_url is None  # not invented from the s3 key


def test_s3_url_wins_over_minio_url_when_both(tmp_path):
    import json

    from umrs.shared.connection_profiles import (
        _coerce_minio_url,
        load_connection_profile_file,
    )

    profile_path = tmp_path / "both.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "both",
                "mode": "direct",
                "zmq": {"host": "data0.example"},
                "services": {
                    "s3_url": "https://new.example",
                    "minio_url": "https://old.example",
                },
            }
        ),
        encoding="utf-8",
    )
    profile = load_connection_profile_file(profile_path)
    assert _coerce_minio_url(profile) == "https://new.example"


def test_minio_url_only_profile_still_resolves(tmp_path):
    import json

    from umrs.shared.connection_profiles import (
        _coerce_minio_url,
        load_connection_profile_file,
    )

    profile_path = tmp_path / "legacy.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "legacy",
                "mode": "direct",
                "zmq": {"host": "data0.example"},
                "services": {"minio_url": "https://old.example"},
            }
        ),
        encoding="utf-8",
    )
    profile = load_connection_profile_file(profile_path)
    assert _coerce_minio_url(profile) == "https://old.example"
