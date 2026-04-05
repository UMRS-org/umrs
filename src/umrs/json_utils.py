"""Compatibility re-export for shared/server JSON configuration loaders."""

from umrs.shared.json_config import client_schema, load_client_cfg_file
from umrs.server.json_config import server_schema, load_server_cfg_file

__all__ = [
    "client_schema",
    "server_schema",
    "load_client_cfg_file",
    "load_server_cfg_file",
]
