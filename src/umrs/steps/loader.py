"""
Script loader — resolve, validate, and import user-provided step classes (U-203).

Supports three source types:
  * **local_path** — scripts live on the host filesystem (bind-mounted or
    directly accessible).
  * **s3** — a zip/tar archive or directory stored in MinIO; downloaded to
    a temporary directory before import.
  * **git** — (future) clone a repo at a pinned commit.

Security notes
--------------
* Only classes that inherit ``UMRSStep`` are accepted.
* Entrypoint format is strictly ``module:ClassName`` — no arbitrary
  expressions or shell metacharacters.
* ``importlib`` is used instead of ``exec``/``eval``.
"""

from __future__ import annotations

import importlib
import io
import logging
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from umrs.steps.interface import UMRSStep
from umrs.steps.models import (
    SourceType,
    StepSpec,
    WorkflowDefinition,
)

logger = logging.getLogger(__name__)

# Regex: "module_name:ClassName" — no whitespace, no shell chars.
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


class ScriptLoadError(Exception):
    """Raised when a user script cannot be loaded."""


class ScriptLoader:
    """Discover, validate, and instantiate user-defined step classes."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_workflow(source_uri: str) -> WorkflowDefinition:
        """Load a :class:`WorkflowDefinition` from *source_uri*.

        *source_uri* is the part after the ``user_script://`` scheme, e.g.::

            /home/user/my_models/workflow.json      (local)
            s3://bucket/workflows/workflow.json     (S3)
        """
        if source_uri.startswith("s3://"):
            raise NotImplementedError(
                "S3-hosted workflow files are not yet supported. "
                "Place workflow.json on a mounted local path for now."
            )

        path = Path(source_uri)
        return WorkflowDefinition.from_file(path)

    @staticmethod
    def load_steps(
        workflow: WorkflowDefinition,
        s3: Optional[Any] = None,
    ) -> List[UMRSStep]:
        """Import and instantiate every step declared in *workflow*.

        Returns an **ordered** list of ``UMRSStep`` instances in the same
        sequence as ``workflow.steps``.

        The ``source`` field of the workflow tells us where the Python
        files live.  For ``LOCAL_PATH`` the directory is added to
        ``sys.path`` so that ``importlib`` can find the modules.
        """
        script_dir = ScriptLoader._resolve_source(workflow, s3)
        instances: List[UMRSStep] = []
        for spec in workflow.steps:
            cls = ScriptLoader._import_step_class(spec, script_dir)
            instance = cls()
            instance.setup(spec.config)
            instances.append(instance)
        return instances

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_source(
        workflow: WorkflowDefinition,
        s3: Optional[Any] = None,
    ) -> Path:
        """Return a local directory containing the user's Python modules."""
        src = workflow.source

        if src.type == SourceType.LOCAL_PATH:
            p = Path(src.path)
            if not p.is_dir():
                raise ScriptLoadError(
                    f"Source directory does not exist: {p}"
                )
            return p

        if src.type == SourceType.S3:
            if s3 is None:
                raise ScriptLoadError(
                    "S3 client is required to download script source"
                )
            return ScriptLoader._download_s3_source(s3, src.path)

        if src.type == SourceType.GIT:
            raise NotImplementedError("Git source type is not yet implemented")

        raise ScriptLoadError(f"Unknown source type: {src.type}")

    @staticmethod
    def _download_s3_source(s3: Any, s3_uri: str) -> Path:
        """Download a zip archive from S3 and extract to a temp dir."""
        if not s3_uri.startswith("s3://"):
            raise ScriptLoadError(f"Invalid S3 URI: {s3_uri}")

        # Parse "s3://bucket/path/to/scripts.zip"
        without_scheme = s3_uri[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        if not key:
            raise ScriptLoadError(f"S3 URI missing key: {s3_uri}")

        tmp_dir = Path(tempfile.mkdtemp(prefix="umrs_steps_"))
        logger.info("Downloading user scripts from %s to %s", s3_uri, tmp_dir)

        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()

        if key.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(tmp_dir)
        else:
            # Single .py file — just write it
            dest = tmp_dir / Path(key).name
            dest.write_bytes(data)

        return tmp_dir

    @staticmethod
    def _import_step_class(spec: StepSpec, script_dir: Path) -> Type[UMRSStep]:
        """Import and validate a single step class.

        Raises :class:`ScriptLoadError` if the entrypoint is malformed,
        the module cannot be imported, or the class does not inherit
        ``UMRSStep``.
        """
        ep = spec.entrypoint
        if not _ENTRYPOINT_RE.match(ep):
            raise ScriptLoadError(
                f"Step '{spec.name}': invalid entrypoint format '{ep}'. "
                "Expected 'module_name:ClassName'."
            )

        module_name, class_name = ep.split(":", 1)

        # Temporarily add script_dir to sys.path so importlib can find it.
        str_dir = str(script_dir)
        added = False
        if str_dir not in sys.path:
            sys.path.insert(0, str_dir)
            added = True

        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            raise ScriptLoadError(
                f"Step '{spec.name}': cannot import module '{module_name}' "
                f"from {script_dir}: {exc}"
            ) from exc
        finally:
            if added:
                sys.path.remove(str_dir)

        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ScriptLoadError(
                f"Step '{spec.name}': class '{class_name}' not found in "
                f"module '{module_name}'"
            )

        if not (isinstance(cls, type) and issubclass(cls, UMRSStep)):
            raise ScriptLoadError(
                f"Step '{spec.name}': {class_name} does not inherit UMRSStep"
            )

        return cls
