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