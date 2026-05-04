"""DEPRECATED: backward-compatibility shim.

All symbols have moved to their canonical modules:
  - umrs.server.maintenance: recompute_all_objects, update_objects, check_objects
  - umrs.server.processing: process_s3_objects
  - umrs.server.config: ENABLE_PRETTY_PRINT
  - umrs.server.provisioning: ServerCfg, configure_user

This module will be removed in a future release.
"""

import warnings as _warnings

_warnings.warn(
    "umrs.server.runtime is deprecated; import from canonical modules instead",
    DeprecationWarning,
    stacklevel=2,
)

from umrs.server.config import ENABLE_PRETTY_PRINT  # noqa: F401
from umrs.server.maintenance import (  # noqa: F401
    check_objects,
    recompute_all_objects,
    update_objects,
)
from umrs.server.processing import process_s3_objects  # noqa: F401
from umrs.server.provisioning import ServerCfg, configure_user  # noqa: F401

__all__ = [
    "ENABLE_PRETTY_PRINT",
    "ServerCfg",
    "check_objects",
    "configure_user",
    "process_s3_objects",
    "recompute_all_objects",
    "update_objects",
]
