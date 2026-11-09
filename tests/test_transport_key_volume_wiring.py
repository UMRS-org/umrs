"""Compose wiring of the transport-key directory (ADR-112 wave B1, plan B1.2).

Each umrs-front server keeps its users' transport keys on its own named
volume at ``UMRS_TRANSPORT_KEY_DIR``. No other container mounts a key volume,
and no server mounts another front's (the private halves of people's keys).
"""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "deploy" / "umrs-front" / "docker-compose.yaml"
KEY_DIR = "/var/lib/umrs/transport-keys"
FRONTS = ("rccs", "ai4s", "bdrt", "octopus")


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def _key_mounts(service: dict) -> list[str]:
    return [v for v in service.get("volumes", []) if isinstance(v, str) and "transport_keys" in v]


def test_every_server_has_its_own_key_volume_and_env() -> None:
    services = _compose()["services"]
    for front in FRONTS:
        server = services[f"umrs-server-{front}"]
        assert server["environment"]["UMRS_TRANSPORT_KEY_DIR"] == KEY_DIR, front
        assert _key_mounts(server) == [f"umrs_transport_keys_{front}:{KEY_DIR}"], front


def test_no_other_container_mounts_a_key_volume() -> None:
    services = _compose()["services"]
    others = {name: s for name, s in services.items() if not name.startswith("umrs-server-")}
    assert not {name: _key_mounts(s) for name, s in others.items() if _key_mounts(s)}


def test_the_key_volumes_are_declared_named_volumes() -> None:
    volumes = _compose()["volumes"]
    for front in FRONTS:
        assert f"umrs_transport_keys_{front}" in volumes
        assert not (volumes[f"umrs_transport_keys_{front}"] or {}).get("external")
