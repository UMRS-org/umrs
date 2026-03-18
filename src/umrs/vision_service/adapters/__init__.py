"""Vision model adapter registry.

Maps ``model_type`` strings (from the ``/infer`` request payload) to
adapter instances.

Built-in adapters (sam3, hmae) are always available.

Custom adapters are loaded automatically at import time from:
  1. Python files in the directory pointed to by ``VISION_ADAPTERS_DIR``
     (default: ``/adapters``).  Each file must define exactly one
     ``VisionModelAdapter`` subclass.  The registry key is the filename
     stem (e.g. ``my_model.py`` → ``"my_model"``).
  2. Dotted-path entries in the ``VISION_ADAPTERS`` env var
     (comma-separated, e.g. ``"my_pkg.yolo:YOLOAdapter,other:Foo"``).
     Format: ``module_path:ClassName``.

Example custom adapter file (``/adapters/super_res.py``)::

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
import inspect
from pathlib import Path
from typing import Dict

from .base import VisionModelAdapter
from .sam3_adapter import SAM3Adapter
from .hmae_adapter import HMAEAdapter

logger = logging.getLogger(__name__)

# ── Built-in adapters ───────────────────────────────────────────────────

ADAPTER_REGISTRY: Dict[str, VisionModelAdapter] = {
    "sam3": SAM3Adapter(),
    "hmae": HMAEAdapter(),
}


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


def _load_adapters_from_dir(directory: str) -> None:
    """Import every ``*.py`` file in *directory* and register adapters."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return

    for py_file in sorted(dir_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        name = py_file.stem
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

_adapters_dir = os.environ.get("VISION_ADAPTERS_DIR", "/adapters")
_load_adapters_from_dir(_adapters_dir)

_adapters_env = os.environ.get("VISION_ADAPTERS", "")
if _adapters_env:
    _load_adapters_from_env(_adapters_env)
