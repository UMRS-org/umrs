"""Model path boot validation.

Ensures configured model paths exist and are readable before the UMRS
server starts accepting inference requests.  Fails fast with a clear
error message when any required path is missing or inaccessible.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Model path environment variables checked at boot.
_MODEL_PATH_VARS: List[str] = [
    "VLLM_MODEL_PATH",
    "VLLM_HOST_MODEL_PATH",
    "VLLM_PATH_TO_HUGGINGFACE",
]


def validate_model_paths() -> Tuple[bool, List[str]]:
    """Validate that configured model paths exist and are readable.

    Returns:
        (ok, missing) where ok is True when at least one configured
        path exists, and missing is a list of paths that failed.
    """
    missing: List[str] = []
    found: bool = False

    for var in _MODEL_PATH_VARS:
        path_str = os.environ.get(var, "").strip()
        if not path_str:
            continue  # Not configured — skip silently

        p = Path(path_str)
        if not p.exists():
            missing.append(f"{var}={path_str} (not found)")
            continue
        if not p.is_dir():
            missing.append(f"{var}={path_str} (not a directory)")
            continue
        # Check read permission on contents
        try:
            entries = list(p.iterdir())
            if entries:
                logger.debug("Model path %s (%s) has %d entries", var, path_str, len(entries))
        except PermissionError:
            missing.append(f"{var}={path_str} (permission denied)")
            continue
        found = True

    if missing:
        for msg in missing:
            logger.warning("Model path check FAILED: %s", msg)

    return found, missing


def enforce_model_paths(require_at_least_one: bool = True) -> None:
    """Validate model paths and raise on failure.

    When ``require_at_least_one`` is True (default for production),
    the function raises RuntimeError if no valid path is found.
    Useful for deployment checklists and boot-time gates.
    """
    ok, missing = validate_model_paths()
    if not ok and require_at_least_one:
        raise RuntimeError(
            f"No valid model path configured at boot. "
            f"Missing: {', '.join(missing)}. "
            f"Set one of {_MODEL_PATH_VARS} before starting UMRS."
        )
