from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from umrs.shared.config import (
    get_connections_default_cfg_subdir,
    get_current_connection_profile_file,
    get_resolved_umrs_cfg_dir,
)

UMRS_PROFILE_ENV_VAR_NAME = "UMRS_PROFILE"
DEFAULT_DATA_PORT = 5556


@dataclass(frozen=True)
class DataPortRange:
    start: int = DEFAULT_DATA_PORT
    end: int = DEFAULT_DATA_PORT

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < 1:
            raise ValueError("Data ports must be positive integers")
        if self.end < self.start:
            raise ValueError(
                "Data port range end must be greater than or equal to start"
            )

    @classmethod
    def single(cls, port: int) -> DataPortRange:
        return cls(start=int(port), end=int(port))

    @property
    def base_port(self) -> int:
        return self.start

    @property
    def is_single(self) -> bool:
        return self.start == self.end

    @property
    def label(self) -> str:
        if self.is_single:
            return str(self.start)
        return f"{self.start}-{self.end}"

    def contains(self, port: int) -> bool:
        return self.start <= int(port) <= self.end


def parse_data_port_range(
    raw: Any,
    *,
    default: DataPortRange | None = None,
) -> DataPortRange | None:
    if raw is None or raw == "":
        return default

    if isinstance(raw, DataPortRange):
        return raw

    if isinstance(raw, int):
        return DataPortRange.single(raw)

    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return default
        if "-" not in value:
            return DataPortRange.single(int(value))
        start_raw, end_raw = value.split("-", 1)
        return DataPortRange(start=int(start_raw.strip()), end=int(end_raw.strip()))

    if isinstance(raw, dict):
        if "start" not in raw:
            raise ValueError("data_port_range requires a start value")
        start = int(raw["start"])
        end = int(raw.get("end", start))
        return DataPortRange(start=start, end=end)

    raise ValueError(f"Unsupported data port range value: {raw!r}")


def _env_flag(name: str, default: bool | None = None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _env_port_range(name: str) -> DataPortRange | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return parse_data_port_range(raw)


def _default_service_url(host: str, port: int, tunnel_mode: bool) -> str:
    service_host = "127.0.0.1" if tunnel_mode else host
    return f"http://{service_host}:{port}"


def _coerce_minio_url(profile: ConnectionProfile) -> str:
    env_minio_url = os.environ.get("UMRS_MINIO_URL")
    if env_minio_url:
        return env_minio_url

    env_minio_host = os.environ.get("UMRS_MINIO_HOST")
    env_minio_port = os.environ.get("UMRS_MINIO_PORT")
    if env_minio_host:
        return _default_service_url(env_minio_host, int(env_minio_port or 9000), False)

    if profile.minio_url:
        return profile.minio_url

    return _default_service_url(profile.server_host, 9000, profile.tunnel_mode)


def _coerce_rag_url(profile: ConnectionProfile) -> str:
    env_rag_url = os.environ.get("RAG_URL")
    if env_rag_url:
        return env_rag_url

    if profile.rag_url:
        return profile.rag_url

    return _default_service_url(profile.server_host, 8081, profile.tunnel_mode)


profile_schema: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["name", "mode", "zmq"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "mode": {"type": "string", "enum": ["ssh_tunnel", "direct"]},
        "zmq": {
            "type": "object",
            "required": ["host"],
            "properties": {
                "host": {"type": "string", "minLength": 1},
                "control_port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "data_base_port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "data_port_range": {
                    "type": "object",
                    "required": ["start"],
                    "properties": {
                        "start": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "end": {"type": "integer", "minimum": 1, "maximum": 65535},
                    },
                    "additionalProperties": False,
                },
                "tunnel_mode": {"type": "boolean"},
                "health_check_timeout_ms": {"type": "integer", "minimum": 1000},
                "auth_type": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "services": {
            "type": "object",
            "properties": {
                "minio_url": {"type": "string", "minLength": 1},
                "rag_url": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "gateway_url": {"type": "string", "minLength": 1},
        "username": {"type": "string", "minLength": 1},
        "ssh": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "minLength": 1},
                "command_hint": {"type": "string"},
                "local_forwards": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["local_port", "remote_host", "remote_port"],
                        "properties": {
                            "local_port": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 65535,
                            },
                            "remote_host": {"type": "string", "minLength": 1},
                            "remote_port": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 65535,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


@dataclass
class ConnectionProfile:
    name: str
    mode: str
    server_host: str
    control_port: int = 5555
    data_port_range: DataPortRange = field(
        default_factory=lambda: DataPortRange.single(DEFAULT_DATA_PORT)
    )
    tunnel_mode: bool = False
    health_check_timeout_ms: int = 5000
    minio_url: str | None = None
    rag_url: str | None = None
    gateway_url: str | None = None
    ssh_host: str | None = None
    local_forwards: list[dict[str, Any]] = field(default_factory=list)
    command_hint: str | None = None
    path: str | None = None
    # ZMQ CURVE authentication type.  ``None`` (unset) is treated as
    # ``"none"`` by downstream consumers.  ``"session"`` triggers
    # auto-bootstrapping of ephemeral CURVE credentials via the gateway.
    zmq_auth_type: str | None = None

    @property
    def data_base_port(self) -> int:
        return self.data_port_range.base_port

    @property
    def data_port_range_label(self) -> str:
        return self.data_port_range.label

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_base_port"] = self.data_base_port
        payload["data_port_range_label"] = self.data_port_range_label
        return payload


@dataclass
class ResolvedConnection:
    profile_name: str | None
    profile_path: str | None
    profile_source: str | None
    mode: str
    server_host: str
    control_port: int
    data_port_range: DataPortRange
    tunnel_mode: bool
    health_check_timeout_ms: int
    minio_url: str
    rag_url: str
    gateway_url: str | None = None
    ssh_host: str | None = None
    local_forwards: list[dict[str, Any]] = field(default_factory=list)
    command_hint: str | None = None
    allocated_data_port: int | None = None
    sources: dict[str, str] = field(default_factory=dict)
    # ZMQ CURVE auth type (``"none"`` | ``"curve"`` | ``"session"``).
    # ``"session"`` triggers auto-bootstrapping of ephemeral CURVE
    # credentials via the gateway on the client side.
    zmq_auth_type: str = "none"

    @property
    def data_base_port(self) -> int:
        return self.data_port_range.base_port

    @property
    def data_port_range_label(self) -> str:
        return self.data_port_range.label

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_base_port"] = self.data_base_port
        payload["data_port_range_label"] = self.data_port_range_label
        return payload


def get_connections_dir() -> Path:
    return (
        Path(get_resolved_umrs_cfg_dir()).expanduser()
        / get_connections_default_cfg_subdir()
    )


def get_current_profile_path() -> Path | None:
    """Return the path to the current connection profile file.

    Returns:
        Path  — if get_current_connection_profile_file() returns a string.
        None  — if get_current_connection_profile_file() returns None
                (no profile selected yet).
    Raises:
        TypeError — if get_current_connection_profile_file() returns an
                    unexpected type (neither str nor None).
    """
    profile_file = get_current_connection_profile_file()
    if profile_file is None:
        return None
    if isinstance(profile_file, str):
        return Path(get_resolved_umrs_cfg_dir()).expanduser() / profile_file
    raise TypeError(
        f"get_current_connection_profile_file() returned unexpected type "
        f"{type(profile_file).__name__!r}: {profile_file!r}"
    )


def get_profile_path(name: str) -> Path:
    return get_connections_dir() / f"{name}.json"


def _load_profile_content(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Connection profile not found: {path}") from exc
    except json.decoder.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse connection profile {path}: {exc.msg} (char {exc.pos})"
        ) from exc

    try:
        validate(instance=content, schema=profile_schema)
    except ValidationError as exc:
        raise ValueError(f"Invalid connection profile {path}: {exc.message}") from exc

    return content


def _profile_from_content(content: dict[str, Any], path: Path) -> ConnectionProfile:
    zmq_cfg = content["zmq"]
    mode = content["mode"]
    tunnel_mode = zmq_cfg.get("tunnel_mode")
    if tunnel_mode is None:
        tunnel_mode = mode == "ssh_tunnel"

    services = content.get("services") or {}
    ssh_cfg = content.get("ssh") or {}
    data_port_range = parse_data_port_range(
        zmq_cfg.get("data_port_range"),
        default=parse_data_port_range(
            zmq_cfg.get("data_base_port"),
            default=DataPortRange.single(DEFAULT_DATA_PORT),
        ),
    )

    profile = ConnectionProfile(
        name=content["name"],
        mode=mode,
        server_host=zmq_cfg["host"],
        control_port=int(zmq_cfg.get("control_port", 5555)),
        data_port_range=data_port_range or DataPortRange.single(DEFAULT_DATA_PORT),
        tunnel_mode=bool(tunnel_mode),
        health_check_timeout_ms=int(zmq_cfg.get("health_check_timeout_ms", 5000)),
        minio_url=services.get("minio_url"),
        rag_url=services.get("rag_url"),
        gateway_url=content.get("gateway_url"),
        ssh_host=ssh_cfg.get("host"),
        local_forwards=list(ssh_cfg.get("local_forwards") or []),
        command_hint=ssh_cfg.get("command_hint"),
        path=str(path),
        zmq_auth_type=(zmq_cfg.get("auth_type") or "none").lower(),
    )
    if not profile.minio_url:
        profile.minio_url = _default_service_url(
            profile.server_host, 9000, profile.tunnel_mode
        )
    if not profile.rag_url:
        profile.rag_url = _default_service_url(
            profile.server_host, 8081, profile.tunnel_mode
        )
    return profile


def load_connection_profile(name: str) -> ConnectionProfile:
    path = get_profile_path(name)
    content = _load_profile_content(path)
    return _profile_from_content(content, path)


def load_connection_profile_file(
    path_like: str | os.PathLike[str],
) -> ConnectionProfile:
    path = Path(path_like).expanduser()
    content = _load_profile_content(path)
    return _profile_from_content(content, path)


def import_connection_profile(
    path_like: str | os.PathLike[str], overwrite: bool = False
) -> ConnectionProfile:
    source_path = Path(path_like).expanduser()
    content = _load_profile_content(source_path)
    profile = _profile_from_content(content, source_path)

    destination_path = get_profile_path(profile.name)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"Connection profile '{profile.name}' already exists at {destination_path}. "
            "Use --force to overwrite it."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)
        handle.write("\n")

    return _profile_from_content(content, destination_path)


def discover_connection_profiles() -> tuple[
    list[ConnectionProfile], list[dict[str, str]]
]:
    profiles: list[ConnectionProfile] = []
    errors: list[dict[str, str]] = []
    directory = get_connections_dir()
    if not directory.is_dir():
        return profiles, errors

    for path in sorted(directory.glob("*.json")):
        try:
            profiles.append(_profile_from_content(_load_profile_content(path), path))
        except Exception as exc:  # pragma: no cover - defensive listing path
            errors.append({"name": path.stem, "path": str(path), "error": str(exc)})

    return profiles, errors


def get_current_profile_name() -> str | None:
    path = get_current_profile_path()
    if path is None or not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def set_current_profile_name(name: str) -> Path:
    profile = load_connection_profile(name)
    path = get_current_profile_path()
    if path is None:
        raise RuntimeError(
            "Cannot set current profile: get_current_connection_profile_file() "
            "returned None."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.name + "\n", encoding="utf-8")
    return path


def resolve_profile_name(
    explicit_name: str | None = None,
) -> tuple[str | None, str | None]:
    if explicit_name:
        return explicit_name, "flag:--profile"

    env_name = os.environ.get(UMRS_PROFILE_ENV_VAR_NAME)
    if env_name:
        return env_name, f"env:{UMRS_PROFILE_ENV_VAR_NAME}"

    current_name = get_current_profile_name()
    if current_name:
        current_path = get_current_profile_path()
        return (
            current_name,
            f"current:{current_path}" if current_path else "current:unknown",
        )

    return None, None


def resolve_connection(
    explicit_profile_name: str | None = None,
    explicit_server_ip: str | None = None,
    explicit_control_port: int | None = None,
) -> ResolvedConnection:
    profile_name, profile_source = resolve_profile_name(explicit_profile_name)
    profile = load_connection_profile(profile_name) if profile_name else None

    sources: dict[str, str] = {}

    if explicit_server_ip:
        server_host = explicit_server_ip
        sources["server_host"] = "flag:--server-ip"
    elif os.environ.get("UMRS_SERVER_HOST"):
        server_host = os.environ["UMRS_SERVER_HOST"]
        sources["server_host"] = "env:UMRS_SERVER_HOST"
    elif profile:
        server_host = profile.server_host
        sources["server_host"] = f"profile:{profile.name}"
    else:
        server_host = "127.0.0.1"
        sources["server_host"] = "default"

    if explicit_control_port is not None:
        control_port = explicit_control_port
        sources["control_port"] = "flag:--control-port"
    elif _env_int("UMRS_CONTROL_PORT") is not None:
        control_port = _env_int("UMRS_CONTROL_PORT")
        sources["control_port"] = "env:UMRS_CONTROL_PORT"
    elif profile:
        control_port = profile.control_port
        sources["control_port"] = f"profile:{profile.name}"
    else:
        control_port = 5555
        sources["control_port"] = "default"

    env_data_port = _env_int("UMRS_DATA_PORT")
    env_data_port_range = _env_port_range("UMRS_DATA_PORT_RANGE")
    if env_data_port is not None:
        data_port_range = DataPortRange.single(env_data_port)
        sources["data_base_port"] = "env:UMRS_DATA_PORT"
        sources["data_port_range"] = "env:UMRS_DATA_PORT"
    elif env_data_port_range is not None:
        data_port_range = env_data_port_range
        sources["data_base_port"] = "env:UMRS_DATA_PORT_RANGE"
        sources["data_port_range"] = "env:UMRS_DATA_PORT_RANGE"
    elif _env_int("UMRS_DATA_BASE_PORT") is not None:
        data_port_range = DataPortRange.single(_env_int("UMRS_DATA_BASE_PORT"))
        sources["data_base_port"] = "env:UMRS_DATA_BASE_PORT"
        sources["data_port_range"] = "env:UMRS_DATA_BASE_PORT"
    elif profile:
        data_port_range = profile.data_port_range
        sources["data_base_port"] = f"profile:{profile.name}"
        sources["data_port_range"] = f"profile:{profile.name}"
    else:
        data_port_range = DataPortRange.single(DEFAULT_DATA_PORT)
        sources["data_base_port"] = "default"
        sources["data_port_range"] = "default"

    tunnel_override = _env_flag("UMRS_TUNNEL_MODE", None)
    if tunnel_override is not None:
        tunnel_mode = tunnel_override
        sources["tunnel_mode"] = "env:UMRS_TUNNEL_MODE"
    elif profile:
        tunnel_mode = profile.tunnel_mode
        sources["tunnel_mode"] = f"profile:{profile.name}"
    else:
        tunnel_mode = False
        sources["tunnel_mode"] = "default"

    health_timeout = _env_int("UMRS_HEALTH_CHECK_TIMEOUT_MS")
    if health_timeout is not None:
        health_check_timeout_ms = health_timeout
        sources["health_check_timeout_ms"] = "env:UMRS_HEALTH_CHECK_TIMEOUT_MS"
    elif profile:
        health_check_timeout_ms = profile.health_check_timeout_ms
        sources["health_check_timeout_ms"] = f"profile:{profile.name}"
    else:
        health_check_timeout_ms = 5000
        sources["health_check_timeout_ms"] = "default"

    base_profile = profile or ConnectionProfile(
        name=profile_name or "implicit-default",
        mode="ssh_tunnel" if tunnel_mode else "direct",
        server_host=server_host,
        control_port=control_port,
        data_port_range=data_port_range,
        tunnel_mode=tunnel_mode,
        health_check_timeout_ms=health_check_timeout_ms,
    )

    minio_url = _coerce_minio_url(base_profile)
    rag_url = _coerce_rag_url(base_profile)
    sources["minio_url"] = "env/profile/default"
    sources["rag_url"] = "env/profile/default"

    mode = "ssh_tunnel" if tunnel_mode else (profile.mode if profile else "direct")

    # ZMQ CURVE auth type: env var UMRS_ZMQ_AUTH_TYPE overrides profile.
    env_auth_type = os.environ.get("UMRS_ZMQ_AUTH_TYPE", "").strip().lower()
    if env_auth_type:
        zmq_auth_type = env_auth_type
    elif profile and profile.zmq_auth_type:
        zmq_auth_type = profile.zmq_auth_type
    else:
        zmq_auth_type = "none"

    return ResolvedConnection(
        profile_name=profile.name if profile else None,
        profile_path=profile.path if profile else None,
        profile_source=profile_source,
        mode=mode,
        server_host=server_host,
        control_port=control_port,
        data_port_range=data_port_range,
        tunnel_mode=tunnel_mode,
        health_check_timeout_ms=health_check_timeout_ms,
        minio_url=minio_url,
        rag_url=rag_url,
        gateway_url=profile.gateway_url if profile else None,
        ssh_host=profile.ssh_host if profile else None,
        local_forwards=list(profile.local_forwards) if profile else [],
        command_hint=profile.command_hint if profile else None,
        sources=sources,
        zmq_auth_type=zmq_auth_type,
    )


__all__ = [
    "ConnectionProfile",
    "DataPortRange",
    "ResolvedConnection",
    "UMRS_PROFILE_ENV_VAR_NAME",
    "discover_connection_profiles",
    "get_connections_dir",
    "get_current_profile_name",
    "get_current_profile_path",
    "get_profile_path",
    "import_connection_profile",
    "load_connection_profile",
    "load_connection_profile_file",
    "parse_data_port_range",
    "resolve_connection",
    "resolve_profile_name",
    "set_current_profile_name",
]
