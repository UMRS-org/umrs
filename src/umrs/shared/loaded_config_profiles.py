from __future__ import annotations

import sys
import configparser
import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from umrs.server.json_config import load_server_cfg_file
from umrs.shared.config import (
    get_current_s3_config_file,
    get_current_server_config_file,
    get_current_user_config_file,
    get_default_docker_s3cfg,
    get_default_json_server_file,
    get_resolved_umrs_cfg_dir,
    get_users_default_cfg_subdir,
)
from umrs.shared.json_config import load_client_cfg_file


@dataclass
class ConfigLayer:
    kind: str
    path: str | None = None
    source: str | None = None
    exists: bool = False
    valid: bool = False
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfigFileCandidate:
    kind: str
    path: str
    label: str
    exists: bool
    valid: bool
    current: bool = False
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedLoadedConfig:
    server: ConfigLayer
    user: ConfigLayer
    s3: ConfigLayer
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server": self.server.to_dict(),
            "user": self.user.to_dict(),
            "s3": self.s3.to_dict(),
            "warnings": list(self.warnings),
        }


def get_current_server_config_path() -> Path:
    return Path(get_resolved_umrs_cfg_dir()).expanduser() / get_current_server_config_file()


def get_current_user_config_path() -> Path:
    return Path(get_resolved_umrs_cfg_dir()).expanduser() / get_current_user_config_file()

def get_current_s3_config_path() -> Path:
    s3_config_file = get_current_s3_config_file()
    if s3_config_file is None:
        return Path()
    if isinstance(s3_config_file, str):
        return Path(get_resolved_umrs_cfg_dir()).expanduser() / s3_config_file
    raise TypeError(
        f"get_current_s3_config_file() returned unexpected type "
        f"{type(s3_config_file).__name__!r}: {s3_config_file!r}"
    )

def get_user_config_search_locations() -> list[str]:
    cfg_dir = Path(get_resolved_umrs_cfg_dir()).expanduser()
    return [
        str(Path.cwd() / "user*.json"),
        str(cfg_dir / get_users_default_cfg_subdir() / "*.json"),
    ]


def _normalize_path(path_like: str | Path) -> str:
    return str(Path(path_like).expanduser().resolve())


def _read_pointer(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _write_pointer(pointer_path: Path, target_path: Path) -> Path:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_path(target_path)
    pointer_path.write_text(normalized + "\n", encoding="utf-8")
    return pointer_path


def _project_candidates(pattern: str) -> list[Path]:
    cwd = Path.cwd()
    matches = []
    for path in cwd.rglob(pattern):
        if path.is_file():
            matches.append(path)
    return matches


def _config_root_candidates(pattern: str) -> list[Path]:
    cfg_dir = Path(get_resolved_umrs_cfg_dir()).expanduser()
    matches = []
    for path in cfg_dir.rglob(pattern):
        if path.is_file():
            matches.append(path)
    return matches


def _candidate_paths_for_server() -> list[Path]:
    cfg_dir = Path(get_resolved_umrs_cfg_dir()).expanduser()
    return [
        Path.cwd() / get_default_json_server_file(),
        cfg_dir / get_default_json_server_file(),
        *_project_candidates(get_default_json_server_file()),
        *_config_root_candidates(get_default_json_server_file()),
    ]


def _candidate_paths_for_user() -> list[Path]:
    cfg_dir = Path(get_resolved_umrs_cfg_dir()).expanduser()
    return [
        *Path.cwd().glob("user*.json"),
        *(cfg_dir / get_users_default_cfg_subdir()).glob("*.json"),
        *_project_candidates("user*.json"),
    ]


def _candidate_paths_for_s3() -> list[Path]:
    cfg_dir = Path(get_resolved_umrs_cfg_dir()).expanduser()
    docker_s3cfg = Path(get_default_docker_s3cfg())
    base_paths = [
        *Path.cwd().glob("s3cfg*"),
        *cfg_dir.glob("s3cfg*"),
        *Path.home().glob(".s3cfg*"),
        *_project_candidates("s3cfg*"),
    ]
    if docker_s3cfg.is_file():
        base_paths.append(docker_s3cfg)
    return base_paths


def _unique_paths(paths: list[Path | str | None]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        normalized = _normalize_path(path) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _validate_server(path: Path) -> ConfigLayer:
    exists = path.is_file()
    layer = ConfigLayer(kind="server", path=str(path), exists=exists)
    if not exists:
        layer.error = f"Server config file not found: {path}"
        return layer
    try:
        _, users, _ = load_server_cfg_file(str(path))
    except (SystemExit, ValueError, Exception) as exc:
        layer.error = str(exc)
        return layer
    layer.valid = True
    layer.details = {
        "user_config_paths": [user.get("config_file_path") for user in users],
        "usernames": [user.get("username") for user in users],
        "user_count": len(users),
    }
    return layer


def _validate_user(path: Path) -> ConfigLayer:
    exists = path.is_file()
    layer = ConfigLayer(kind="user", path=str(path), exists=exists)
    if not exists:
        layer.error = f"User config file not found: {path}"
        return layer
    try:
        user_cfg = load_client_cfg_file(str(path))
    except (SystemExit, ValueError) as exc:
        layer.error = str(exc)
        return layer
    layer.valid = True
    bucket_aliases = sorted({
        (mod.get("s3") or {}).get("bucket_alias")
        for xp in (user_cfg.get("experiences") or [])
        for mod in (xp.get("modalities") or [])
        if (mod.get("s3") or {}).get("bucket_alias")
    })
    layer.details = {
        "username": user_cfg.get("username"),
        "fullname": user_cfg.get("fullname"),
        "s3_configuration_file": user_cfg.get("s3_configuration_file"),
        "experience_count": len(user_cfg.get("experiences") or []),
        "experience_names": [xp.get("xp_name") for xp in user_cfg.get("experiences") or []],
        "bucket_aliases": bucket_aliases,
    }
    return layer


def _validate_s3(path: Path) -> ConfigLayer:
    exists = path.is_file()
    layer = ConfigLayer(kind="s3", path=str(path), exists=exists)
    if not exists:
        layer.error = f"S3 config file not found: {path}"
        return layer

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(path)
    if not parser.has_section("default"):
        layer.error = f"Invalid S3 config file {path}: missing [default] section"
        return layer

    missing = [
        key for key in ("host_base", "access_key", "secret_key", "bucket_location")
        if not parser.has_option("default", key)
    ]
    if missing:
        layer.error = f"Invalid S3 config file {path}: missing keys {', '.join(missing)}"
        return layer

    layer.valid = True
    layer.details = {
        "host_base": parser.get("default", "host_base"),
        "bucket_location": parser.get("default", "bucket_location"),
        "use_https": parser.getboolean("default", "use_https", fallback=False),
        "check_ssl_certificate": parser.getboolean("default", "check_ssl_certificate", fallback=True),
    }
    return layer


_VALIDATORS: dict[str, Callable[[Path], ConfigLayer]] = {
    "server": _validate_server,
    "user": _validate_user,
    "s3": _validate_s3,
}

_FLAG_SOURCES = {
    "server": "flag:--server-conf",
    "user": "flag:-c/--conf",
    "s3": "flag:--s3cfg",
}


def _make_candidate(kind: str, path: Path, current_path: str | None) -> ConfigFileCandidate:
    layer = _VALIDATORS[kind](path)
    current = bool(current_path and _normalize_path(path) == _normalize_path(current_path))
    return ConfigFileCandidate(
        kind=kind,
        path=str(path),
        label=path.name,
        exists=layer.exists,
        valid=layer.valid,
        current=current,
        error=layer.error,
        details=layer.details,
    )


def discover_loaded_config_profiles() -> dict[str, list[ConfigFileCandidate]]:
    current_server = _read_pointer(get_current_server_config_path())
    current_user = _read_pointer(get_current_user_config_path())
    current_s3 = _read_pointer(get_current_s3_config_path())

    server_paths = _unique_paths([*_candidate_paths_for_server(), current_server])
    server_candidates = [_make_candidate("server", path, current_server) for path in server_paths]

    referenced_user_paths: list[str] = []
    for candidate in server_candidates:
        referenced_user_paths.extend(candidate.details.get("user_config_paths") or [])

    user_paths = _unique_paths([*_candidate_paths_for_user(), *referenced_user_paths, current_user])
    user_candidates = [_make_candidate("user", path, current_user) for path in user_paths]

    referenced_s3_paths: list[str] = []
    for candidate in user_candidates:
        s3_path = candidate.details.get("s3_configuration_file")
        if s3_path:
            referenced_s3_paths.append(s3_path)

    s3_paths = _unique_paths([*_candidate_paths_for_s3(), *referenced_s3_paths, current_s3])
    s3_candidates = [_make_candidate("s3", path, current_s3) for path in s3_paths]

    return {
        "server": server_candidates,
        "user": user_candidates,
        "s3": s3_candidates,
    }


def _auto_select(candidates: list[ConfigFileCandidate], kind: str) -> ConfigLayer:
    for candidate in candidates:
        if candidate.valid:
            return ConfigLayer(
                kind=kind,
                path=candidate.path,
                source="auto-discovery",
                exists=candidate.exists,
                valid=candidate.valid,
                error=candidate.error,
                details=dict(candidate.details),
            )
    return ConfigLayer(kind=kind)


def _resolve_selector(
    kind: str,
    user_path: str | None,
    explicit_path: str | None,
    current_path: str | None,
    candidates: list[ConfigFileCandidate],
    pointer_path: Path,
) -> ConfigLayer:
    
    if user_path:
        layer = _VALIDATORS[kind](Path(user_path).expanduser())
        layer.source = _FLAG_SOURCES[kind]
        return layer

    if explicit_path:
        layer = _VALIDATORS[kind](Path(explicit_path).expanduser())
        layer.source = _FLAG_SOURCES[kind]
        return layer

    if current_path:
        layer = _VALIDATORS[kind](Path(current_path).expanduser())
        layer.source = f"current:{pointer_path}"
        return layer

    return _auto_select(candidates, kind)


def resolve_loaded_config(
    explicit_server_config_path: str | None = None,
    explicit_user_config_path: str | None = None,
    explicit_s3_config_path: str | None = None,
) -> ResolvedLoadedConfig:
    discovered = discover_loaded_config_profiles()
    current_server = _read_pointer(get_current_server_config_path())
    current_user = _read_pointer(get_current_user_config_path())
    current_s3 = _read_pointer(get_current_s3_config_path())

    server = _resolve_selector(
        "server",
        None,
        explicit_server_config_path,
        current_server,
        discovered["server"],
        get_current_server_config_path(),
    )

    derived_user_path = None
    if not explicit_user_config_path and not current_user and server.valid:
        server_user_paths = server.details.get("user_config_paths") or []
        if len(server_user_paths) == 1:
            derived_user_path = server_user_paths[0]

    user = _resolve_selector(
        "user",
        None,
        explicit_user_config_path or derived_user_path,
        None if explicit_user_config_path or derived_user_path else current_user,
        discovered["user"],
        get_current_user_config_path(),
    )


    if derived_user_path and user.path == str(Path(derived_user_path).expanduser()):
        user.source = f"server:{server.path}"

    derived_s3_path = None
    if user.valid:
        derived_s3_path = user.details.get("s3_configuration_file")
    
    user_s3_path = user.details.get("s3_configuration_file")

    
    user_s3_path = user.details.get("s3_configuration_file")

    s3 = _resolve_selector(
        "s3",
        user_s3_path,
        explicit_s3_config_path,
        current_s3,
        discovered["s3"],
        get_current_s3_config_path(),
    )

    if not s3.path and derived_s3_path:
        s3 = _VALIDATORS["s3"](Path(derived_s3_path).expanduser())
        s3.source = f"user:{user.path}"
    elif (
        derived_s3_path
        and s3.path == str(Path(derived_s3_path).expanduser())
        and not (s3.source and (s3.source.startswith("current:") or s3.source.startswith("flag:")))
    ):
        s3.source = f"user:{user.path}"

    warnings: list[str] = []
    if server.valid and user.valid:
        server_user_paths = {
            _normalize_path(path)
            for path in (server.details.get("user_config_paths") or [])
        }
        if server_user_paths and _normalize_path(user.path) not in server_user_paths:
            warnings.append("Selected user config is not listed in the selected server config.")

    if user.valid and s3.valid:
        user_s3_path = user.details.get("s3_configuration_file")
        if user_s3_path and _normalize_path(user_s3_path) != _normalize_path(s3.path):
            warnings.append("An explicit S3 config override is active for this client session.")
            sys.exit(f"this warning should never be raised: {warnings=}")

    if user.valid and not s3.valid and s3.path:
        warnings.append("The resolved S3 config is invalid; runtime operations may fail unless a fallback applies.")


    return ResolvedLoadedConfig(server=server, user=user, s3=s3, warnings=warnings)


def _select_candidate_path(kind: str, selection: str) -> Path:
    path = Path(selection).expanduser()
    if path.is_file():
        return path

    discovered = discover_loaded_config_profiles()[kind]
    matches = [Path(candidate.path) for candidate in discovered if candidate.label == selection]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {kind} config selection '{selection}'. Please use an explicit path instead."
        )
    raise FileNotFoundError(f"{kind.title()} config not found: {selection}")


def set_current_server_config(selection: str) -> Path:
    path = _select_candidate_path("server", selection)
    layer = _validate_server(path)
    if not layer.valid:
        raise ValueError(layer.error or f"Invalid server config: {path}")
    return _write_pointer(get_current_server_config_path(), path)


def set_current_user_config(selection: str) -> Path:
    path = _select_candidate_path("user", selection)
    layer = _validate_user(path)
    if not layer.valid:
        raise ValueError(layer.error or f"Invalid user config: {path}")
    return _write_pointer(get_current_user_config_path(), path)


def set_current_s3_config(selection: str) -> Path:
    path = _select_candidate_path("s3", selection)
    layer = _validate_s3(path)
    if not layer.valid:
        raise ValueError(layer.error or f"Invalid S3 config: {path}")
    return _write_pointer(get_current_s3_config_path(), path)


def apply_s3_config_override(user_cfg: dict[str, Any], s3_config_path: str | None) -> dict[str, Any]:
    if not s3_config_path:
        return user_cfg
    if user_cfg.get("s3_configuration_file") == s3_config_path:
        return user_cfg
    updated = copy.deepcopy(user_cfg)
    updated["s3_configuration_file"] = s3_config_path
    return updated


__all__ = [
    "ConfigFileCandidate",
    "ConfigLayer",
    "ResolvedLoadedConfig",
    "apply_s3_config_override",
    "discover_loaded_config_profiles",
    "get_current_s3_config_path",
    "get_current_server_config_path",
    "get_current_user_config_path",
    "get_user_config_search_locations",
    "resolve_loaded_config",
    "set_current_s3_config",
    "set_current_server_config",
    "set_current_user_config",
]
