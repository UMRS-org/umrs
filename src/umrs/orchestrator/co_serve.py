"""Co-served entries: one Slurm job, several endpoint rows (ADR-109 D-2).

A roster entry may declare ``co_serve:``, extra servings that share its job::

    - name: qwen3.6-35b-a3b-nvfp4
      model: Qwen3.6-35B-A3B-NVFP4
      aliases: [svc-vision]
      co_serve:
        - {name: sam3.1, model: SAM3.1, aliases: [svc-vision2],
           model_type: vision, adapter: sam3}

The job then starts the BentoML vision sidecar beside vLLM and registers one
row per serving: the primary on its LiteLLM port and one row per co-serving on
the sidecar's port. Every row carries ``job_group`` = the primary's job id;
a co-served row also carries ``co_served_by``. Lifecycle (drain, teardown,
renew, failure) acts on the job, so the registry moves the co-served rows
with their primary. A co-served row has its own role probe and READY state:
a failed sidecar never fails the primary.

Only ``model_type: vision`` co-servings exist today (one BentoML sidecar per
job, so at most one). Any other type is rejected by name and skipped.

Foundation module: stdlib only, no ``umrs.api`` / ``umrs.server`` imports.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Roster key on an entry, and the provision-request metadata key that
#: carries the resolved co-servings to the orchestrator.
CO_SERVE_KEY = "co_serve"
#: Every row of a co-served job: the primary's job id.
JOB_GROUP_KEY = "job_group"
#: A co-served row only: the primary's job id (its lifecycle owner).
CO_SERVED_BY_KEY = "co_served_by"
#: Separator between the primary's job id and the serving name in a
#: co-served row's job id (``<primary>::<name>``; registry rows are keyed by
#: job id, so each serving needs its own).
CO_JOB_SEP = "::"

SUPPORTED_TYPES = ("vision",)
VISION_PROFILE = "vision-bentoml"

#: ADR-109 D-2 role probe of a vision co-serving: one segmentation of the
#: service's built-in probe image, at least one mask.
SEGMENT_PROBE = "segment"

#: ADR-109 D-2 memory budget of the sidecar on a GB10 (vLLM keeps its own
#: ``VLLM_GPU_MEM_UTIL``). The roster's co_serve ``env`` overrides each key.
VISION_ENV_DEFAULTS: dict[str, str] = {
    "VISION_MAX_BATCH": "1",
    "MAX_VISION_MODELS": "1",
    "VISION_GPU_MEMORY_BUDGET_GIB": "16",
    "VISION_PYTORCH_CUDA_ALLOC_CONF": (
        "expandable_segments:True,garbage_collection_threshold:0.8"
    ),
    # A sidecar that does not come up must not end the job: the primary
    # keeps serving and the relay rolls the replica (ADR-109 D-2).
    "RIVAULT_VISION_STRICT": "0",
}

#: The sidecar listens this far above the job's LiteLLM port. Ports are
#: unique per host for primaries, so the offset keeps co-ports unique too.
CO_PORT_OFFSET_ENV = "UMRS_CO_SERVE_PORT_OFFSET"
_DEFAULT_PORT_OFFSET = 1000


@dataclass(frozen=True)
class CoServe:
    """One validated co-serving of a roster entry."""

    name: str
    model: str
    model_type: str
    adapter: str
    aliases: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def job_env(self) -> dict[str, str]:
        """The serving's env for the shared job: defaults, then the roster's."""
        return {**VISION_ENV_DEFAULTS, **self.env}

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "model_type": self.model_type,
            "adapter": self.adapter,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_metadata(cls, raw: dict[str, Any]) -> CoServe:
        return cls(
            name=str(raw["name"]),
            model=str(raw["model"]),
            model_type=str(raw.get("model_type") or "vision"),
            adapter=str(raw.get("adapter") or ""),
            aliases=tuple(str(a) for a in raw.get("aliases") or ()),
        )


def _text(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    return value.strip() if isinstance(value, str) else ""


def _aliases(item: dict[str, Any]) -> tuple[str, ...] | None:
    raw = item.get("aliases") or []
    if not isinstance(raw, list) or not all(isinstance(a, str) and a for a in raw):
        return None
    return tuple(raw)


def _env(item: dict[str, Any]) -> dict[str, str] | None:
    raw = item.get("env") or {}
    if not isinstance(raw, dict):
        return None
    return {str(k): str(v) for k, v in raw.items()}


def _item_error(item: Any, where: str) -> str | None:
    """Why one ``co_serve`` item is invalid, or ``None``."""
    if not isinstance(item, dict):
        return f"{where}: each co_serve item must be a mapping"
    name = _text(item, "name") or "?"
    if not _text(item, "name") or not _text(item, "model"):
        return f"{where}: co_serve '{name}' needs a name and a model"
    model_type = _text(item, "model_type").lower()
    if model_type not in SUPPORTED_TYPES:
        return (
            f"{where}: co_serve '{name}' has model_type '{model_type or '-'}'; "
            f"only {', '.join(SUPPORTED_TYPES)} co-serving is supported"
        )
    return _shape_error(item, f"{where}: co_serve '{name}'")


def _shape_error(item: dict[str, Any], label: str) -> str | None:
    """Why a vision item's adapter, aliases or env are unusable, or ``None``."""
    if not _text(item, "adapter"):
        return f"{label} (vision) needs an adapter"
    if _aliases(item) is None or _env(item) is None:
        return f"{label} aliases must be names and env a mapping"
    return None


def _spec(item: dict[str, Any]) -> CoServe:
    return CoServe(
        name=_text(item, "name"),
        model=_text(item, "model"),
        model_type=_text(item, "model_type").lower(),
        adapter=_text(item, "adapter"),
        aliases=_aliases(item) or (),
        env=_env(item) or {},
    )


def _accept(spec: CoServe, specs: list[CoServe], where: str) -> str | None:
    """Why a valid item still cannot join the job, or ``None``."""
    if any(s.name == spec.name for s in specs):
        return f"{where}: co_serve '{spec.name}' is declared twice"
    if any(s.model_type == spec.model_type for s in specs):
        return (
            f"{where}: co_serve '{spec.name}' is a second {spec.model_type} "
            "serving; a job runs one BentoML sidecar"
        )
    return None


def parse_co_serve(entry: dict[str, Any]) -> tuple[list[CoServe], list[str]]:
    """The entry's valid co-servings and the reasons each other one was rejected."""
    raw = entry.get(CO_SERVE_KEY)
    where = str(entry.get("name") or entry.get("model") or "?")
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], [f"{where}: co_serve must be a list"]
    specs: list[CoServe] = []
    errors: list[str] = []
    for item in raw:
        spec, error = _checked(item, specs, where)
        if spec is not None:
            specs.append(spec)
        else:
            errors.append(str(error))
    return specs, errors


def _checked(
    item: Any, specs: list[CoServe], where: str
) -> tuple[CoServe | None, str | None]:
    """The item as a spec, or the reason it is rejected."""
    error = _item_error(item, where)
    if error:
        return None, error
    spec = _spec(item)
    error = _accept(spec, specs, where)
    return (None, error) if error else (spec, None)


def co_serves(entry: dict[str, Any] | None) -> list[CoServe]:
    """The entry's valid co-servings; each rejected one is logged by name."""
    if not entry:
        return []
    specs, errors = parse_co_serve(entry)
    for error in errors:
        logger.warning("co_serve skipped: %s", error)
    return specs


def from_metadata(metadata: dict[str, Any] | None) -> list[CoServe]:
    """The co-servings a provision request carries (``metadata["co_serve"]``)."""
    raw = (metadata or {}).get(CO_SERVE_KEY) or []
    return [CoServe.from_metadata(item) for item in raw if isinstance(item, dict)]


def co_job_id(primary_job_id: str, name: str) -> str:
    """The registry job id of a co-serving's row."""
    return f"{primary_job_id}{CO_JOB_SEP}{name}"


def primary_job_id(job_id: str) -> str:
    """The backend job behind a row: a co-served row's primary, else itself."""
    return str(job_id).split(CO_JOB_SEP, 1)[0]


def is_co_served(endpoint: Any) -> bool:
    """A co-served row (its job belongs to another row)."""
    return bool((getattr(endpoint, "metadata", None) or {}).get(CO_SERVED_BY_KEY))


def co_port(primary_port: int) -> int:
    """The sidecar's port for a job whose LiteLLM listens on ``primary_port``."""
    try:
        offset = int(os.getenv(CO_PORT_OFFSET_ENV, "") or _DEFAULT_PORT_OFFSET)
    except ValueError:
        offset = _DEFAULT_PORT_OFFSET
    return int(primary_port) + offset


def job_env(specs: list[CoServe]) -> dict[str, str]:
    """Env the shared job needs for its co-servings (later specs win)."""
    env: dict[str, str] = {}
    for spec in specs:
        env.update(spec.job_env())
    return env


def submit_env(metadata: dict[str, Any] | None, primary_port: int) -> dict[str, str]:
    """Submit-time env that depends on the allocated port ({} without co-serving)."""
    if not from_metadata(metadata):
        return {}
    return {"BENTOML_VISION_DOCKER_PORT": str(co_port(primary_port))}


def _profile_list(value: str | None) -> list[str]:
    return [p for p in str(value or "").split() if p]


def sidecar_profiles(profiles: str | None, *, co_served: bool) -> str:
    """``profiles`` with the vision sidecar profile added or removed.

    ADR-109 D-2: an LLM job starts ``vision-bentoml`` only when its entry
    declares a vision co-serving; otherwise the profile is stripped, even when
    a roster or site default lists it.
    """
    names = [p for p in _profile_list(profiles) if p != VISION_PROFILE]
    if co_served:
        names.append(VISION_PROFILE)
    return " ".join(names)


def row_metadata(
    spec: CoServe, primary_job: str, primary_metadata: dict[str, Any], host: str, port: int
) -> dict[str, Any]:
    """Registry metadata of a co-served row (ownership copied from the primary)."""
    inherited = {
        key: primary_metadata[key]
        for key in ("owner_type", "tier", "owner_id", "access", "requesting_user")
        if key in primary_metadata
    }
    return {
        **inherited,
        JOB_GROUP_KEY: primary_job,
        CO_SERVED_BY_KEY: primary_job,
        "co_serve_name": spec.name,
        "aliases": list(spec.aliases),
        "model_type": spec.model_type,
        "serving_backend": "bentoml",
        "service_kind": "bentoml_vision",
        "capabilities": {"supports_vision": True},
        "vision_adapter": spec.adapter,
        # The janitor's drift probe stays the cheap /readyz; the role probe
        # (one segmentation) runs once, before the row turns READY.
        "readiness_probe": "http",
        "readiness_path": "/readyz",
        "role_probe": SEGMENT_PROBE,
        "allocated_port": port,
        "service_url": f"http://{host}:{port}" if host and host != "pending" else "",
        "detail": "waiting for the primary job",
    }
