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

The site policy for that code lives here too: whether the hardened HPC lane
is available, the ``uploaded_code`` transition switch for sites without one,
and the step-image allow-list (ADR-074 D-7; build spec 2026-09-26 §1-§2).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from umrs.shared.uploaded_code import (  # noqa: F401 — re-exported
    LEGACY_IN_PROCESS,
    UPLOADED_CODE_REFUSE,
    uploaded_code_policy,
)
from umrs.shared.uploaded_code import load_site_env as _load_site_env

logger = logging.getLogger(__name__)


#: Platform templates whose config names the command/binary they run.
_CALLER_COMMAND_CLASSES = frozenset({"ShellStep", "VideoStep"})
#: The lanes that run a step on a connected HPC (the ADR-061 ``gpu-job`` lane).
HPC_LANES = frozenset({"slurm", "slurm-remote"})
#: Lanes that run a step without a container image (an image hint is moot).
IMAGELESS_LANES = frozenset({"inproc", "subprocess"})
#: ``step_executor.uploaded_code`` values (``UMRS_STEP_UPLOADED_CODE``).
#: What the HPC lane needs besides SSH submit and ``remote_enabled``.
_LANE_REQUIRED_ENV = ("UMRS_SLURM_WORKDIR_ROOT", "UMRS_STEP_HELPER", "UMRS_STEP_IMAGE")
_TRUE = frozenset({"1", "true", "yes", "on"})


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


def _env_set(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def hpc_lane_available() -> bool:
    """Whether uploaded step code may go to this site's connected HPC.

    Only the hardened front-end lane qualifies (build spec 2026-09-26 §1): SSH
    submit, a workdir root, the site's explicit ``remote_enabled``, the
    login-node helper and a platform step image. A local ``sbatch`` alone
    does not: that lane has neither the staged data plane nor the posture.
    """
    _load_site_env()
    if os.environ.get("UMRS_SLURM_STEP_SUBMIT") != "remote":
        return False
    if os.environ.get("UMRS_STEP_REMOTE_ENABLED", "").strip().lower() not in _TRUE:
        return False
    return all(map(_env_set, _LANE_REQUIRED_ENV))


def uploaded_code_lane(requested: str | None) -> str | None:
    """The executor for a workflow carrying uploaded code.

    When the site's HPC lane is available, a requested HPC lane stands and
    any other lane (``in_process``, the deployment's ``UMRS_STEP_EXECUTOR``,
    the ``inproc`` default) becomes ``slurm``. Otherwise the transition switch
    decides: :data:`LEGACY_IN_PROCESS` (the caller runs the pre-ADR-074
    lanes) or None (the run is refused by name) — a requested HPC lane does
    not stand while the site keeps its lane closed.
    """
    if hpc_lane_available():
        effective = (
            (requested or os.environ.get("UMRS_STEP_EXECUTOR") or "inproc")
            .strip()
            .lower()
        )
        return effective if effective in HPC_LANES else "slurm"
    if uploaded_code_policy() == LEGACY_IN_PROCESS:
        return LEGACY_IN_PROCESS
    return None


def allowed_step_images() -> frozenset[str]:
    """The platform step image plus the site's ``allowed_images`` (D-7)."""
    _load_site_env()
    raw = os.environ.get("UMRS_STEP_ALLOWED_IMAGES", "").split(",")
    images = {image.strip() for image in raw}
    images.add(os.environ.get("UMRS_STEP_IMAGE", "").strip())
    images.discard("")
    return frozenset(images)


def step_image_refusal(image: Any, lane: str) -> str | None:
    """Why a step's ``routing_hints.image`` may not run here, or None.

    ADR-074 D-7: a step names the platform step image or one on the site's
    admin-curated ``allowed_images``; on a lane that runs no image the hint
    is moot and passes.
    """
    if image is None or lane in IMAGELESS_LANES:
        return None
    if isinstance(image, str) and image in allowed_step_images():
        return None
    return (
        f"routing_hints.image {image!r} is not an allowed step image on this "
        "site — name the platform step image or one listed in "
        "step_executor.allowed_images (ADR-074 D-7)"
    )
