from __future__ import annotations

import json


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
