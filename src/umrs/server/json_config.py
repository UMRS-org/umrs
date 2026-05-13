"""Server JSON configuration — re-exports from shared.json_config.

The canonical implementations of ``load_server_cfg_file``,
``load_server_cfg_json``, and ``server_schema`` live in
``umrs.shared.json_config`` so that ``shared/`` modules can use them
without importing from ``server/``.

This module re-exports everything for backward compatibility.
"""

from umrs.shared.json_config import (
    load_client_cfg_file,
    load_server_cfg_file,
    load_server_cfg_json,
    server_schema,
)

__all__ = ["server_schema", "load_server_cfg_json", "load_server_cfg_file", "load_client_cfg_file"]
