"""
In-container step executor entrypoint.

Runs inside Singularity/Apptainer (or any OCI container) as::

    python -m umrs.plugins.plugin_runner.entrypoint

Reads step configuration from environment variables set by the step
adapter (``step_to_plugin_spec``) and forwarded through the Slurm/K8s
runner.  Executes ``setup → run → teardown`` and writes a machine-readable
``result_manifest.json`` to ``/workspace``.

Environment variables consumed
-------------------------------
- ``UMRS_STEP_NAME``        — human-readable step name
- ``UMRS_STEP_ENTRYPOINT``  — ``module:ClassName``
- ``UMRS_STEP_CONFIG``      — JSON-serialised step config dict
- ``UMRS_INPUT_BUCKET``     — S3 bucket for input data
- ``UMRS_INPUT_KEYS``       — JSON list of S3 keys to process
- ``UMRS_OUTPUT_BUCKET``    — S3 bucket for outputs
- ``UMRS_RUN_ID``           — unique run identifier

- ``PLUGIN_SOURCE_URI``     — path to step source code (mounted at /source)

Optional S3 connection vars (for building a StepContext):
- ``S3_ENDPOINT``           — MinIO / S3 endpoint URL
- ``S3_ACCESS_KEY``         — access key
- ``S3_SECRET_KEY``         — secret key
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("umrs.plugin_runner")

# Workspace directory inside the container — host job dir is bound here.
WORKSPACE = Path(os.environ.get("PLUGIN_WORKSPACE", "/workspace"))

# Source directory inside the container — step scripts are bound here.
SOURCE_DIR = Path(os.environ.get("PLUGIN_SOURCE_DIR", "/source"))


# ── Helpers ──────────────────────────────────────────────────────────────


def _read_env(name: str, *, required: bool = True) -> str:
    """Read an environment variable, raising on missing required ones."""
    val = os.environ.get(name, "")
    if required and not val:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return val


def _import_step_class(entrypoint: str, script_dir: Path):
    """Import a Step subclass from ``module:ClassName`` syntax.

    Mirrors the logic in ``pipeline_sdk.workflow._import_step_class``
    but avoids importing the full UMRS server stack.
    """
    if ":" not in entrypoint:
        raise ValueError(
            f"Invalid entrypoint format {entrypoint!r}. "
            "Expected 'module_name:ClassName'."
        )

    module_name, class_name = entrypoint.split(":", 1)

    # Resolve module file on disk
    rel_parts = module_name.replace(".", "/")
    module_file = script_dir / f"{rel_parts}.py"
    if not module_file.is_file():
        module_file = script_dir / rel_parts / "__init__.py"

    if module_file.is_file():
        loader_spec = importlib.util.spec_from_file_location(
            module_name,
            str(module_file),
            submodule_search_locations=[str(script_dir)],
        )
        if loader_spec is None or loader_spec.loader is None:
            raise RuntimeError(
                f"Cannot create module spec for {module_name!r} at {module_file}"
            )
        mod = importlib.util.module_from_spec(loader_spec)
        sys.modules[module_name] = mod
        try:
            loader_spec.loader.exec_module(mod)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise RuntimeError(
                f"Cannot load module {module_name!r} from {module_file}: {exc}"
            ) from exc
    else:
        # Fallback — the module may be installed as a package
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot import module {module_name!r} "
                f"(source dir: {script_dir}): {exc}"
            ) from exc

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"Class {class_name!r} not found in module {module_name!r}"
        )
    return cls


def _build_s3_client():
    """Build a boto3 S3 client from environment variables."""
    try:
        import boto3
        from botocore.config import Config

        endpoint = os.environ.get("S3_ENDPOINT", "http://minio:9000")
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY", ""),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY", ""),
            config=Config(signature_version="s3v4"),
        )
    except ImportError:
        logger.warning("boto3 not available — S3 operations will fail")
        return None


def _write_manifest(manifest: dict[str, Any]) -> Path:
    """Write the result manifest to /workspace/result_manifest.json."""
    manifest_path = WORKSPACE / "result_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Wrote result manifest to %s", manifest_path)
    return manifest_path


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    """Execute the step and write result_manifest.json.

    Returns 0 on success, 1 on failure.
    """
    started_at = datetime.now(timezone.utc)

    try:
        # ── Read configuration from env ──────────────────────────────
        step_name = _read_env("UMRS_STEP_NAME")
        entrypoint = _read_env("UMRS_STEP_ENTRYPOINT")
        step_config = json.loads(_read_env("UMRS_STEP_CONFIG", required=False) or "{}")
        input_bucket = _read_env("UMRS_INPUT_BUCKET")
        input_keys = json.loads(_read_env("UMRS_INPUT_KEYS", required=False) or "[]")
        output_bucket = _read_env("UMRS_OUTPUT_BUCKET")
        run_id = _read_env("UMRS_RUN_ID", required=False) or "unknown"

        logger.info(
            "Step executor starting: name=%s entrypoint=%s run_id=%s "
            "input_keys=%d",
            step_name, entrypoint, run_id, len(input_keys),
        )

        # ── Determine source directory ───────────────────────────────
        # The runner binds the source dir at /source. If it doesn't
        # exist, fall back to PLUGIN_SOURCE_URI (which is the host-side
        # path — may work for subprocess mode).
        if SOURCE_DIR.is_dir():
            script_dir = SOURCE_DIR
        else:
            source_uri = os.environ.get("PLUGIN_SOURCE_URI", "")
            if source_uri and Path(source_uri).is_dir():
                script_dir = Path(source_uri)
            else:
                raise RuntimeError(
                    f"Step source directory not found: tried {SOURCE_DIR} "
                    f"and PLUGIN_SOURCE_URI={source_uri!r}"
                )

        # ── Import and instantiate the step ──────────────────────────
        step_cls = _import_step_class(entrypoint, script_dir)
        step = step_cls()
        logger.info("Loaded step class: %s", step_cls.__name__)

        # ── setup ────────────────────────────────────────────────────
        step.setup(step_config)

        # ── run ──────────────────────────────────────────────────────
        from umrs.pipeline_sdk.step import StepContext

        s3 = _build_s3_client()

        ctx = StepContext(
            s3=s3,
            input_bucket=input_bucket,
            input_keys=input_keys,
            output_bucket=output_bucket,
            config=step_config,
            metadata={},
            run_id=run_id,
        )

        result = step.run(ctx)
        logger.info(
            "Step completed: output_keys=%d embeddings=%s",
            len(result.output_keys),
            result.embeddings is not None,
        )

        # ── teardown ─────────────────────────────────────────────────
        try:
            step.teardown()
        except Exception:
            logger.warning("teardown failed", exc_info=True)

        # ── Write result manifest ────────────────────────────────────
        ended_at = datetime.now(timezone.utc)

        # Serialise embeddings — convert numpy arrays to lists for JSON.
        embeddings_serialised: list | None = None
        if result.embeddings is not None:
            try:
                import numpy as np
                if isinstance(result.embeddings, np.ndarray):
                    embeddings_serialised = result.embeddings.tolist()
                else:
                    embeddings_serialised = [
                        e.tolist() if isinstance(e, np.ndarray) else e
                        for e in result.embeddings
                    ]
            except ImportError:
                embeddings_serialised = list(result.embeddings)

        manifest = {
            "status": "succeeded",
            "step_name": step_name,
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "output_keys": result.output_keys,
            "embeddings": embeddings_serialised,
            "checksums": result.checksums,
            "metadata": result.metadata,
            "extra": result.extra,
        }
        _write_manifest(manifest)

        return 0

    except Exception as exc:
        ended_at = datetime.now(timezone.utc)
        logger.error("Step execution failed: %s", exc, exc_info=True)

        manifest = {
            "status": "failed",
            "step_name": _read_env("UMRS_STEP_NAME", required=False) or "unknown",
            "run_id": _read_env("UMRS_RUN_ID", required=False) or "unknown",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "output_keys": [],
            "embeddings": None,
            "checksums": [],
            "metadata": [],
            "extra": {},
        }
        _write_manifest(manifest)

        return 1


if __name__ == "__main__":
    sys.exit(main())
