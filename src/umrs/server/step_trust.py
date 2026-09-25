"""Which workflow steps are uploaded code, decided without importing them.

ADR-074 (Amendment 1): normal users upload their own workflows to run on
connected HPCs, and the UMRS server never imports or runs that code — an
import alone runs the module's top-level statements inside the server, with
every platform credential in reach.

A step is *uploaded code* when its entrypoint module lives in the workflow's
uploaded source tree (the worker puts that tree first on ``sys.path``, so any
top-level name it carries is the tree's), or when it is a platform template
that runs a caller-chosen command or binary (``ShellStep``, ``VideoStep``).
Everything else is code the operator installed in the server image.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

logger = logging.getLogger(__name__)

#: Platform templates whose config names the command/binary they run.
_CALLER_COMMAND_CLASSES = frozenset({"ShellStep", "VideoStep"})
#: The lanes that run a step on a connected HPC (the ADR-061 ``gpu-job`` lane).
HPC_LANES = frozenset({"slurm", "slurm-remote"})


def _module_in_tree(module_name: str, source_path: str | None) -> bool:
    top = module_name.split(".", 1)[0]
    # No module (a gate step) is no code; a bare root would match the tree itself.
    if not top or not source_path or not os.path.isdir(source_path):
        return False
    root = os.path.realpath(source_path)
    return os.path.isdir(os.path.join(root, top)) or os.path.isfile(
        os.path.join(root, f"{top}.py")
    )


def is_uploaded_step(entrypoint: str, source_path: str | None) -> bool:
    module_name, _, class_name = entrypoint.partition(":")
    if class_name in _CALLER_COMMAND_CLASSES and module_name.startswith("umrs."):
        return True
    return _module_in_tree(module_name, source_path)


def workflow_has_uploaded_code(
    steps: list[dict[str, Any]], source_path: str | None
) -> bool:
    return any(
        is_uploaded_step(str(s.get("entrypoint") or ""), source_path) for s in steps
    )


def hpc_lane_available() -> bool:
    """Whether this server can submit a step to a connected HPC."""
    try:
        from umrs.orchestrator.site_config import get_site_config

        get_site_config().inject_env(overwrite=False)
    except Exception:
        logger.debug("site config unavailable for the HPC-lane check", exc_info=True)
    if os.environ.get("UMRS_SLURM_STEP_SUBMIT") == "remote":
        return bool(os.environ.get("UMRS_SLURM_WORKDIR_ROOT", "").strip())
    return shutil.which("sbatch") is not None


def uploaded_code_lane(requested: str | None) -> str | None:
    """The executor for a workflow carrying uploaded code.

    A requested HPC lane stands; any other lane (``in_process``, the
    deployment's ``UMRS_STEP_EXECUTOR``, the ``inproc`` default) becomes the
    site's HPC lane; None when the site has none (the run is refused).
    """
    effective = (
        (requested or os.environ.get("UMRS_STEP_EXECUTOR") or "inproc").strip().lower()
    )
    if effective in HPC_LANES:
        return effective
    return "slurm" if hpc_lane_available() else None
