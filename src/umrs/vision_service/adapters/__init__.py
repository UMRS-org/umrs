"""Vision model adapter registry.

Maps ``model_type`` strings (from the ``/infer`` request payload) to
adapter instances.

Adapters are loaded automatically at import time from:
  1. Python files in directories listed by ``VISION_ADAPTERS_DIR``
     (colon-separated, default: ``/adapters``).  Each file must define
     exactly one ``VisionModelAdapter`` subclass.  The registry key is
     the filename stem (e.g. ``my_model.py`` → ``"my_model"``).
     If a ``requirements.txt`` exists in the directory, its packages
     are auto-installed before importing the adapters.
  2. Dotted-path entries in the ``VISION_ADAPTERS`` env var
     (comma-separated, e.g. ``"my_pkg.yolo:YOLOAdapter,other:Foo"``).
     Format: ``module_path:ClassName``.

The built-in SAM3 and HMAE adapters ship as ``.py`` files in
``rivault/default_adapters/`` and are loaded via the same mechanism
(mounted at ``/adapters/system`` in the Docker container).

Example custom adapter file (``/adapters/user/super_res.py``)::

    from umrs.vision_service.adapters.base import VisionModelAdapter

    class SuperResAdapter(VisionModelAdapter):
        def cache_key(self, config):  return f"super_res:{config['checkpoint']}"
        def load(self, config):       ...  # return model object
        def infer(self, model, image, bucket, key, config, s3, output_bucket): ...
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import shutil
import subprocess
import inspect
from pathlib import Path
from typing import Dict

from .base import VisionModelAdapter

logger = logging.getLogger(__name__)

# ── Adapter registry (starts empty — populated by auto-discovery) ───────

ADAPTER_REGISTRY: Dict[str, VisionModelAdapter] = {}


def register_adapter(name: str, adapter: VisionModelAdapter) -> None:
    """Register a custom adapter at runtime.

    Args:
        name: Key that callers will pass as ``model_type``.
        adapter: An instance of a ``VisionModelAdapter`` subclass.
    """
    if name in ADAPTER_REGISTRY:
        logger.warning("Overwriting existing adapter '%s'", name)
    ADAPTER_REGISTRY[name] = adapter
    logger.info("Registered adapter '%s' (%s)", name, type(adapter).__name__)


# ── Auto-discovery helpers ──────────────────────────────────────────────


def _find_adapter_class(module):
    """Return the first VisionModelAdapter subclass defined in *module*."""
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, VisionModelAdapter)
            and obj is not VisionModelAdapter
            and obj.__module__ == module.__name__
        ):
            return obj
    return None


def _install_requirements(directory: Path) -> None:
    """Auto-install dependencies from ``requirements.txt`` if present."""
    req_file = directory / "requirements.txt"
    if not req_file.is_file():
        return

    installer = shutil.which("uv")
    if installer:
        cmd = [installer, "pip", "install", "-r", str(req_file)]
    else:
        installer = shutil.which("pip")
        if not installer:
            logger.warning(
                "Neither uv nor pip found — cannot install %s", req_file,
            )
            return
        cmd = [installer, "install", "-r", str(req_file)]

    logger.info("Installing adapter dependencies from %s", req_file)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Failed to install %s:\n%s", req_file, exc.stderr,
        )


def _load_adapters_from_dir(directory: str) -> None:
    """Import every ``*.py`` file in *directory* and register adapters.

    If a ``requirements.txt`` exists in the directory, its packages are
    installed automatically before importing any adapter scripts.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return

    _install_requirements(dir_path)

    for py_file in sorted(dir_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        name = py_file.stem.removesuffix("_adapter")
        try:
            spec = importlib.util.spec_from_file_location(name, py_file)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = _find_adapter_class(mod)
            if cls is not None:
                register_adapter(name, cls())
            else:
                logger.warning(
                    "No VisionModelAdapter subclass found in %s", py_file,
                )
        except Exception:
            logger.exception("Failed to load adapter from %s", py_file)
        except SystemExit as exc:
            logger.error(
                "Adapter %s called sys.exit(%s) — ignoring", py_file, exc.code,
            )


def _load_adapters_from_env(env_value: str) -> None:
    """Parse ``module:ClassName`` entries and register each adapter."""
    for entry in env_value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            logger.warning(
                "VISION_ADAPTERS entry '%s' missing ':ClassName' — skipped",
                entry,
            )
            continue
        module_path, class_name = entry.rsplit(":", 1)
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            if not (
                inspect.isclass(cls) and issubclass(cls, VisionModelAdapter)
            ):
                logger.warning(
                    "%s.%s is not a VisionModelAdapter subclass — skipped",
                    module_path, class_name,
                )
                continue
            name = class_name.lower().removesuffix("adapter")
            register_adapter(name, cls())
        except Exception:
            logger.exception(
                "Failed to load adapter from '%s'", entry,
            )


# ── Run auto-discovery on import ────────────────────────────────────────

# VISION_ADAPTERS_DIR supports colon-separated paths (like $PATH), e.g.
#   /adapters/system:/adapters/user
# Each directory is scanned for *.py adapter files.
def _default_adapter_dirs() -> str:
    candidate_dirs = []
    for adapter_dir in ("/adapters/system", "/adapters/user"):
        if Path(adapter_dir).is_dir():
            candidate_dirs.append(adapter_dir)
    if candidate_dirs:
        return ":".join(candidate_dirs)
    return "/adapters"


try:
    _adapters_dir = os.environ.get("VISION_ADAPTERS_DIR") or _default_adapter_dirs()
    for _dir in _adapters_dir.split(":"):
        _dir = _dir.strip()
        if _dir:
            _load_adapters_from_dir(_dir)

    _adapters_env = os.environ.get("VISION_ADAPTERS", "")
    if _adapters_env:
        _load_adapters_from_env(_adapters_env)
except Exception:
    logger.exception(
        "Adapter auto-discovery failed — service will start but reject "
        "all infer requests until adapters are loaded"
    )

if not ADAPTER_REGISTRY:
    logger.warning(
        "No vision adapters loaded. The service will reject all infer "
        "requests with 'Unknown model_type'. Check VISION_ADAPTERS_DIR "
        "and VISION_ADAPTERS environment variables."
    )
