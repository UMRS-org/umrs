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
            raise ValueError("Data port range end must be greater than or equal to start")

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
    if _empty_data_port_range_value(raw):
        return default

    if isinstance(raw, DataPortRange):
        return raw

    if isinstance(raw, int):
        return DataPortRange.single(raw)

    if isinstance(raw, str):
        return _parse_data_port_range_string(raw, default)

    if isinstance(raw, dict):
        return _parse_data_port_range_mapping(raw)

    raise ValueError(f"Unsupported data port range value: {raw!r}")


def _empty_data_port_range_value(raw: Any) -> bool:
    return raw is None or raw == ""


def _parse_data_port_range_string(
    raw: str,
    default: DataPortRange | None,
) -> DataPortRange | None:
    value = raw.strip()
    if not value:
        return default
    if "-" not in value:
        return DataPortRange.single(int(value))
    start_raw, end_raw = value.split("-", 1)
    return DataPortRange(start=int(start_raw.strip()), end=int(end_raw.strip()))


def _parse_data_port_range_mapping(raw: dict[str, Any]) -> DataPortRange:
    if "start" not in raw:
        raise ValueError("data_port_range requires a start value")
    start = int(raw["start"])
    end = int(raw.get("end", start))
    return DataPortRange(start=start, end=end)


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
    "required": ["name", "mode"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "mode": {"type": "string", "enum": ["ssh_tunnel", "direct", "gateway"]},
        "gateway_url": {"type": "string", "minLength": 1},
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
                "auth_type": {"type": "string", "enum": ["none", "curve_static", "session"]},
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
                            "local_port": {"type": "integer", "minimum": 1, "maximum": 65535},
                            "remote_host": {"type": "string", "minLength": 1},
                            "remote_port": {"type": "integer", "minimum": 1, "maximum": 65535},
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
    data_port_range: DataPortRange = field(default_factory=lambda: DataPortRange.single(DEFAULT_DATA_PORT))
    tunnel_mode: bool = False
    health_check_timeout_ms: int = 5000
    minio_url: str | None = None
    rag_url: str | None = None
    ssh_host: str | None = None
    local_forwards: list[dict[str, Any]] = field(default_factory=list)
    command_hint: str | None = None
    path: str | None = None
    gateway_url: str | None = None
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
    ssh_host: str | None = None
    local_forwards: list[dict[str, Any]] = field(default_factory=list)
    command_hint: str | None = None
    allocated_data_port: int | None = None
    sources: dict[str, str] = field(default_factory=dict)
    gateway_url: str | None = None
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


def get_connections_dir() -> Path:
    return Path(get_resolved_umrs_cfg_dir()).expanduser() / get_connections_default_cfg_subdir()


def get_current_profile_path() -> Path:
    return Path(get_resolved_umrs_cfg_dir()).expanduser() / get_current_connection_profile_file()


def get_profile_path(name: str) -> Path:
    return get_connections_dir() / f"{name}.json"


def _load_profile_content(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Connection profile not found: {path}") from exc
    except json.decoder.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse connection profile {path}: {exc.msg} (char {exc.pos})") from exc

    try:
        validate(instance=content, schema=profile_schema)
    except ValidationError as exc:
        raise ValueError(f"Invalid connection profile {path}: {exc.message}") from exc

    return content


def _profile_from_content(content: dict[str, Any], path: Path) -> ConnectionProfile:
    mode = content["mode"]
    zmq_cfg = content.get("zmq") or {}

    # Gateway-only mode: ZMQ config is optional.
    server_host = zmq_cfg.get("host", "127.0.0.1")

    tunnel_mode = zmq_cfg.get("tunnel_mode")
    if tunnel_mode is None:
        tunnel_mode = mode == "ssh_tunnel"

    services = content.get("services") or {}
    ssh_cfg = content.get("ssh") or {}
    data_port_range = parse_data_port_range(
        zmq_cfg.get("data_port_range"),
        default=parse_data_port_range(zmq_cfg.get("data_base_port"), default=DataPortRange.single(DEFAULT_DATA_PORT)),
    )

    profile = ConnectionProfile(
        name=content["name"],
        mode=mode,
        server_host=server_host,
        control_port=int(zmq_cfg.get("control_port", 5555)),
        data_port_range=data_port_range or DataPortRange.single(DEFAULT_DATA_PORT),
        tunnel_mode=bool(tunnel_mode),
        health_check_timeout_ms=int(zmq_cfg.get("health_check_timeout_ms", 5000)),
        minio_url=services.get("minio_url"),
        rag_url=services.get("rag_url"),
        ssh_host=ssh_cfg.get("host"),
        local_forwards=list(ssh_cfg.get("local_forwards") or []),
        command_hint=ssh_cfg.get("command_hint"),
        path=str(path),
        gateway_url=content.get("gateway_url"),
        zmq_auth_type=zmq_cfg.get("auth_type"),
    )
    return _with_default_service_urls(profile)


def _with_default_service_urls(profile: ConnectionProfile) -> ConnectionProfile:
    profile.minio_url = profile.minio_url or _default_service_url(profile.server_host, 9000, profile.tunnel_mode)
    profile.rag_url = profile.rag_url or _default_service_url(profile.server_host, 8081, profile.tunnel_mode)
    return profile


def load_connection_profile(name: str) -> ConnectionProfile:
    path = get_profile_path(name)
    content = _load_profile_content(path)
    return _profile_from_content(content, path)


def load_connection_profile_file(path_like: str | os.PathLike[str]) -> ConnectionProfile:
    path = Path(path_like).expanduser()
    content = _load_profile_content(path)
    return _profile_from_content(content, path)


def import_connection_profile(path_like: str | os.PathLike[str], overwrite: bool = False) -> ConnectionProfile:
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


def discover_connection_profiles() -> tuple[list[ConnectionProfile], list[dict[str, str]]]:
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
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def set_current_profile_name(name: str) -> Path:
    profile = load_connection_profile(name)
    path = get_current_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.name + "\n", encoding="utf-8")
    return path


def resolve_profile_name(explicit_name: str | None = None) -> tuple[str | None, str | None]:
    if explicit_name:
        return explicit_name, "flag:--profile"

    env_name = os.environ.get(UMRS_PROFILE_ENV_VAR_NAME)
    if env_name:
        return env_name, f"env:{UMRS_PROFILE_ENV_VAR_NAME}"

    # Fall back to R4_PROFILE so a unified profile name works across tools
    r4_profile = os.environ.get("R4_PROFILE")
    if r4_profile:
        return r4_profile, "env:R4_PROFILE"

    current_name = get_current_profile_name()
    if current_name:
        return current_name, f"current:{get_current_profile_path()}"

    return None, None


def resolve_connection(
    explicit_profile_name: str | None = None,
    explicit_server_ip: str | None = None,
    explicit_control_port: int | None = None,
) -> ResolvedConnection:
    profile_name, profile_source = resolve_profile_name(explicit_profile_name)
    profile: ConnectionProfile | None = None
    if profile_name:
        try:
            profile = load_connection_profile(profile_name)
        except FileNotFoundError:
            # Profile not in ~/.config/umrs/connections/ — try r4 config
            profile = _try_load_r4_profile(profile_name)
            if profile is not None:
                profile_source = f"r4:{profile_name}"
    sources: dict[str, str] = {}

    parts = _connection_parts(profile, profile_name, explicit_server_ip, explicit_control_port, sources)
    return _resolved_connection(profile, profile_source, parts)


def _connection_parts(
    profile: ConnectionProfile | None,
    profile_name: str | None,
    explicit_server_ip: str | None,
    explicit_control_port: int | None,
    sources: dict[str, str],
) -> dict[str, Any]:
    profile_source_label = f"profile:{profile.name}" if profile else ""
    server_host = _connection_value(
        sources, "server_host", [
            (explicit_server_ip, "flag:--server-ip"),
            (os.environ.get("UMRS_SERVER_HOST"), "env:UMRS_SERVER_HOST"),
            (_profile_attr(profile, "server_host"), profile_source_label),
        ], "127.0.0.1", "default",
    )
    control_port = _connection_value(
        sources, "control_port", [
            (explicit_control_port, "flag:--control-port"),
            (_env_int("UMRS_CONTROL_PORT"), "env:UMRS_CONTROL_PORT"),
            (_profile_attr(profile, "control_port"), profile_source_label),
        ], 5555, "default",
    )
    data_port_range = _resolve_data_port_range(profile, sources)
    tunnel_mode = _connection_value(
        sources, "tunnel_mode", [
            (_env_flag("UMRS_TUNNEL_MODE", None), "env:UMRS_TUNNEL_MODE"),
            (_profile_attr(profile, "tunnel_mode"), profile_source_label),
        ], False, "default",
    )
    health_check_timeout_ms = _connection_value(
        sources, "health_check_timeout_ms", [
            (_env_int("UMRS_HEALTH_CHECK_TIMEOUT_MS"), "env:UMRS_HEALTH_CHECK_TIMEOUT_MS"),
            (_profile_attr(profile, "health_check_timeout_ms"), profile_source_label),
        ], 5000, "default",
    )
    base_profile = _base_connection_profile(
        profile, profile_name, server_host, control_port, data_port_range, tunnel_mode, health_check_timeout_ms
    )

    minio_url = _coerce_minio_url(base_profile)
    rag_url = _coerce_rag_url(base_profile)
    sources["minio_url"] = "env/profile/default"
    sources["rag_url"] = "env/profile/default"
    return {
        "server_host": server_host, "control_port": control_port, "data_port_range": data_port_range,
        "tunnel_mode": tunnel_mode, "health_check_timeout_ms": health_check_timeout_ms,
        "minio_url": minio_url, "rag_url": rag_url, "sources": sources,
    }


def _resolved_connection(
    profile: ConnectionProfile | None,
    profile_source: str | None,
    parts: dict[str, Any],
) -> ResolvedConnection:
    return ResolvedConnection(
        profile_name=_profile_attr(profile, "name"),
        profile_path=_profile_attr(profile, "path"),
        profile_source=profile_source,
        mode=_connection_mode(profile, parts["tunnel_mode"]),
        server_host=parts["server_host"],
        control_port=parts["control_port"],
        data_port_range=parts["data_port_range"],
        tunnel_mode=parts["tunnel_mode"],
        health_check_timeout_ms=parts["health_check_timeout_ms"],
        minio_url=parts["minio_url"],
        rag_url=parts["rag_url"],
        ssh_host=_profile_attr(profile, "ssh_host"),
        local_forwards=list(profile.local_forwards) if profile else [],
        command_hint=_profile_attr(profile, "command_hint"),
        sources=parts["sources"],
        gateway_url=_profile_attr(profile, "gateway_url"),
        zmq_auth_type=_profile_attr(profile, "zmq_auth_type"),
    )


def _profile_attr(profile: ConnectionProfile | None, attr: str) -> Any:
    return getattr(profile, attr) if profile else None


def _connection_mode(profile: ConnectionProfile | None, tunnel_mode: bool) -> str:
    if tunnel_mode:
        return "ssh_tunnel"
    return profile.mode if profile else "direct"


def _connection_value(
    sources: dict[str, str],
    key: str,
    choices: list[tuple[Any, str]],
    default: Any,
    default_source: str,
) -> Any:
    for value, source in choices:
        if value is not None and value != "":
            sources[key] = source
            return value
    sources[key] = default_source
    return default


def _resolve_data_port_range(
    profile: ConnectionProfile | None,
    sources: dict[str, str],
) -> DataPortRange:
    env_data_port = _env_int("UMRS_DATA_PORT")
    env_data_port_range = _env_port_range("UMRS_DATA_PORT_RANGE")
    env_data_base_port = _env_int("UMRS_DATA_BASE_PORT")
    if env_data_port is not None:
        return _data_port_source(DataPortRange.single(env_data_port), "env:UMRS_DATA_PORT", sources)
    if env_data_port_range is not None:
        return _data_port_source(env_data_port_range, "env:UMRS_DATA_PORT_RANGE", sources)
    if env_data_base_port is not None:
        return _data_port_source(DataPortRange.single(env_data_base_port), "env:UMRS_DATA_BASE_PORT", sources)
    if profile:
        return _data_port_source(profile.data_port_range, f"profile:{profile.name}", sources)
    return _data_port_source(DataPortRange.single(DEFAULT_DATA_PORT), "default", sources)


def _data_port_source(
    data_port_range: DataPortRange,
    source: str,
    sources: dict[str, str],
) -> DataPortRange:
    sources["data_base_port"] = source
    sources["data_port_range"] = source
    return data_port_range


def _base_connection_profile(
    profile: ConnectionProfile | None,
    profile_name: str | None,
    server_host: str,
    control_port: int,
    data_port_range: DataPortRange,
    tunnel_mode: bool,
    health_check_timeout_ms: int,
) -> ConnectionProfile:
    if profile:
        return profile
    return ConnectionProfile(
        name=profile_name or "implicit-default",
        mode="ssh_tunnel" if tunnel_mode else "direct",
        server_host=server_host,
        control_port=control_port,
        data_port_range=data_port_range,
        tunnel_mode=tunnel_mode,
        health_check_timeout_ms=health_check_timeout_ms,
    )


# ---------------------------------------------------------------------------
# r4 config read-through (stdlib only — no r4 package import)
# ---------------------------------------------------------------------------

def _resolve_r4_config_dir() -> Path:
    """Return the r4 config directory (same logic as r4.config)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg, "r4") if xdg else Path.home() / ".config" / "r4"


def _try_load_r4_profile(name: str) -> ConnectionProfile | None:
    """Attempt to build a ConnectionProfile from r4's config files.

    Reads ``~/.config/r4/sources.json`` for the named source, extracting
    gateway_url, ZMQ host/port, and auth_type.  Uses only stdlib (no r4
    package import) to respect the import boundary.

    Returns None if r4 config doesn't exist or the profile isn't found.
    """
    r4_dir = _resolve_r4_config_dir()

    # Try sources.json first (has ZMQ details)
    sources_path = r4_dir / "sources.json"
    if sources_path.is_file():
        try:
            data = json.loads(sources_path.read_text(encoding="utf-8"))
            for src in data.get("sources", []):
                if src.get("name") != name or src.get("kind") != "platform":
                    continue
                url = src.get("url", "")
                host = src.get("zmq_host") or _r4_extract_host(url)
                port = src.get("zmq_control_port") or 5555
                tunnel = src.get("zmq_tunnel_mode") or False
                auth_type = src.get("zmq_auth_type")
                return ConnectionProfile(
                    name=name,
                    mode="ssh_tunnel" if tunnel else "direct",
                    server_host=host,
                    control_port=int(port),
                    data_port_range=DataPortRange(int(port) + 1, int(port) + 8),
                    tunnel_mode=bool(tunnel),
                    health_check_timeout_ms=5000,
                    gateway_url=url or None,
                    zmq_auth_type=auth_type,
                )
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    # Fallback: config.json profiles (simpler, may lack ZMQ details)
    config_path = r4_dir / "config.json"
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            profiles = data.get("profiles", {})
            if name in profiles:
                p = profiles[name]
                url = p.get("gateway_url", "")
                host = p.get("umrs_server_host") or _r4_extract_host(url)
                port = p.get("umrs_control_port") or 5555
                return ConnectionProfile(
                    name=name,
                    mode="direct",
                    server_host=host,
                    control_port=int(port),
                    data_port_range=DataPortRange(int(port) + 1, int(port) + 8),
                    tunnel_mode=False,
                    health_check_timeout_ms=5000,
                    gateway_url=url or None,
                )
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    return None


def _r4_extract_host(url: str) -> str:
    """Extract hostname from a URL (stdlib only)."""
    from urllib.parse import urlparse
    return urlparse(url).hostname or "127.0.0.1"


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
