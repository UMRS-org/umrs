"""Typed ServingPlan (``ResolvedEndpointSpec``).

Canonical internal contract for both ``r4 endpoint up`` and ``r4 pool up``:

    model → site recipe → *typed plan* → Service.provision (unchanged).

Field policy classes — enforced at resolve time, never at launch time:

    USER_TUNABLE       partition, node_count, gpu_count, max_model_len
    RECIPE_CONTROLLED  image, engine.*, patches[], env — curated recipes only
    SITE_CONTROLLED    cluster, backend, account/qos, partition allow-list
    SECURITY_CONTROLLED mounts, egress, trust-remote-code, patch provenance

Invariant: a raw user request can NEVER write a recipe-controlled key
(image/env/patch/extra_args). The curated path is the only channel for those.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


# ---------------------------------------------------------------------------
# Policy: field classifications
# ---------------------------------------------------------------------------

USER_TUNABLE_FIELDS = frozenset(
    {"partition", "node_count", "gpu_count", "max_model_len", "ttl_sec"}
)

RECIPE_CONTROLLED_FIELDS = frozenset(
    {
        "image",
        "engine",
        "patches",
        "env",
        "parallel_strategy",
        "served_model_name",
        "readiness",
    }
)

SITE_CONTROLLED_FIELDS = frozenset({"cluster", "backend", "account", "qos"})

# Every request-named override a non-admin user may pass.
USER_OVERRIDE_ALLOWLIST = USER_TUNABLE_FIELDS

# Engine keys accepted from recipes (curated). Unknown keys get a warning,
# and are dropped — the launcher must never see unknown engine args.
ENGINE_ALLOWED_KEYS = frozenset(
    {
        "max_model_len",
        "gpu_mem_util",
        "kv_cache_memory_bytes",
        "max_num_seqs",
        "max_num_batched_tokens",
        "tokenizer",
        "tool_call_parser",
        "reasoning_parser",
        "speculative_config",
        "language_model_only",
        "extra_args",
    }
)

STRATEGIES = frozenset({"tp_pp", "tep"})


class ServingSpecError(ValueError):
    """Base schema/policy error for serving specs."""


class RecipePolicyError(ServingSpecError):
    """User request attempted to override a non-user-tunable field."""


class RecipeSchemaError(ServingSpecError):
    """Recipe content failed validation."""


class RecipeNotFoundError(ServingSpecError):
    """No recipe matched (model, variant[, fallback])."""


# ---------------------------------------------------------------------------
# Typed spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchSpec:
    """Read-only bind-mount of an operator-authored patched module file."""

    source: str
    target: str
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    served_name: str
    storage_ref: str
    owner_id: str = ""
    project_id: str = "default"


@dataclass(frozen=True)
class SiteSpec:
    cluster: str
    backend: str
    partition: str
    account: str | None = None
    qos: str | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    image: str
    nodes: int
    gpus_per_node: int
    tensor_parallel: int
    pipeline_parallel: int
    strategy: str = "tp_pp"


@dataclass(frozen=True)
class EngineSpec:
    max_model_len: int
    gpu_mem_util: float
    kv_cache_memory_bytes: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    tokenizer: str | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    speculative_config: dict[str, Any] | None = None
    language_model_only: bool = False
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProvenanceSpec:
    recipe_file: str | None
    recipe_hash: str  # sha256 of recipe file bytes ("" for generic profiles)
    resolved_at: str  # ISO-8601; excluded from plan_hash


@dataclass(frozen=True)
class ResolvedEndpointSpec:
    """A complete, validated, deterministic serving plan."""

    model: ModelSpec
    site: SiteSpec
    runtime: RuntimeSpec
    engine: EngineSpec
    patches: tuple[PatchSpec, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    ttl_sec: int = 0
    readiness_timeout_sec: int = 3600
    provenance: ProvenanceSpec = field(
        default_factory=lambda: ProvenanceSpec(None, "", "")
    )

    def to_plan_dict(self) -> dict[str, Any]:
        """Canonical (stable) material used for hashing/audit. Excludes resolved_at."""
        d: dict[str, Any] = {
            "model": vars(self.model),
            "site": vars(self.site),
            "runtime": vars(self.runtime),
            "engine": _engine_dict(self.engine),
            "patches": [vars(p) for p in self.patches],
            "env": dict(self.env),
            "ttl_sec": self.ttl_sec,
            "readiness_timeout_sec": self.readiness_timeout_sec,
            "provenance": {
                "recipe_file": self.provenance.recipe_file,
                "recipe_hash": self.provenance.recipe_hash,
            },
        }
        return d

    def plan_hash(self) -> str:
        canonical = json.dumps(self.to_plan_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _engine_dict(e: EngineSpec) -> dict[str, Any]:
    d = vars(e)
    d["extra_args"] = list(e.extra_args)
    return d


def patches_to_env(patches: Sequence[PatchSpec]) -> dict[str, str]:
    """Serialize PatchSpec entries for the launcher.

    Wire format: space-separated ``src:target:sha256`` triplets in
    ``VLLM_PATCHES``. Spaces force the M5 hex transport (naturally), so the
    value is both mobile and provenance-safe; paths carry no spaces by the
    recipe validator (targets are checked to be module paths at startup).
    """
    if not patches:
        return {}
    line = " ".join(f"{p.source}:{p.target}:{p.sha256}" for p in patches)
    return {
        "VLLM_PATCHES": line,
        "VLLM_PATCHES_COUNT": str(len(patches)),
    }


# ---------------------------------------------------------------------------
# Recipe loading (stdlib + PyYAML — foundation-safe, matches provision_roster)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_roster_recipes(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load roster recipe files. Each file ≈ {"defaults": {...}, "endpoints": [...]}.

    Returns a flat list of {**defaults, **entry, "_recipe_file": str} dicts.
    """
    entries: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        data = yaml.safe_load(path.read_text()) or {}
        defaults = data.get("defaults", {}) if isinstance(data, dict) else {}
        for entry in data.get("endpoints", []) or []:
            if not isinstance(entry, dict) or "model" not in entry:
                raise RecipeSchemaError(f"{path}: every entry needs a 'model' field")
            merged = {**defaults, **entry}
            merged["_recipe_file"] = str(path)
            entries.append(merged)
    return entries


def _match_variant(
    candidates: Sequence[Mapping[str, Any]], variant: str
) -> Mapping[str, Any] | None:
    for e in candidates:
        if e.get("name") == variant:
            return e
    return None


def _match_canonical(
    candidates: Sequence[Mapping[str, Any]], model: str
) -> Mapping[str, Any] | None:
    for e in candidates:
        if e.get("name") in (None, model):
            return e
    return None


def resolve_recipe_entry(
    entries: Sequence[Mapping[str, Any]],
    *,
    model: str,
    variant: str | None,
) -> Mapping[str, Any]:
    """Exact (model, variant) then canonical (model, name==model)."""
    candidates = [e for e in entries if e.get("model") == model]
    if variant is not None:
        match = _match_variant(candidates, variant)
        if match is not None:
            return match
    canonical = _match_canonical(candidates, model)
    if canonical is not None:
        return canonical
    if candidates and variant is None:
        return candidates[0]
    raise RecipeNotFoundError(f"no recipe for model={model!r} variant={variant!r}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_str(spec: Mapping[str, Any], key: str, owner: str) -> str:
    v = spec.get(key)
    if not isinstance(v, str) or not v.strip():
        raise RecipeSchemaError(f"{owner}: '{key}' must be a non-empty string")
    return v.strip()


def _require_int(spec: Mapping[str, Any], key: str, owner: str, *, minimum: int) -> int:
    v = spec.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
        raise RecipeSchemaError(f"{owner}: '{key}' must be int >= {minimum}")
    return v


def _require_util(spec: Mapping[str, Any], owner: str) -> float:
    v = spec.get("gpu_mem_util")
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 < float(v) < 1):
        raise RecipeSchemaError(f"{owner}: 'gpu_mem_util' must be a float in (0,1)")
    return float(v)


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------


def _optional_int(v: Any) -> int | None:
    return int(v) if v else None


def _parse_extra_args(raw: Any) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, str):
        return tuple(raw.split())
    return tuple(str(a) for a in raw)


def build_engine_spec(
    engine: Mapping[str, Any], owner: str
) -> tuple[EngineSpec, list[str]]:
    """Validate + normalize an engine block; returns (spec, dropped_unknown_keys)."""
    unknown = sorted(set(engine) - ENGINE_ALLOWED_KEYS)
    eng = {k: v for k, v in engine.items() if k in ENGINE_ALLOWED_KEYS}
    spec = EngineSpec(
        max_model_len=_require_int(eng, "max_model_len", owner, minimum=512),
        gpu_mem_util=_require_util(eng, owner),
        kv_cache_memory_bytes=_optional_int(eng.get("kv_cache_memory_bytes")),
        max_num_seqs=_optional_int(eng.get("max_num_seqs")),
        max_num_batched_tokens=_optional_int(eng.get("max_num_batched_tokens")),
        tokenizer=eng.get("tokenizer"),
        tool_call_parser=eng.get("tool_call_parser"),
        reasoning_parser=eng.get("reasoning_parser"),
        speculative_config=(
            dict(eng["speculative_config"]) if eng.get("speculative_config") else None
        ),
        language_model_only=bool(eng.get("language_model_only", False)),
        extra_args=_parse_extra_args(eng.get("extra_args")),
    )
    return spec, unknown


def build_patch_specs(
    patches: Sequence[Mapping[str, Any]], owner: str
) -> tuple[PatchSpec, ...]:
    out: list[PatchSpec] = []
    for i, raw in enumerate(patches or []):
        sub = f"{owner}.patches[{i}]"
        source = _require_str(raw, "source", sub)
        target = _require_str(raw, "target", sub)
        sha = raw.get("sha256") or ""
        if sha:
            if not re.fullmatch(r"[0-9a-f]{64}", str(sha).lower()):
                raise RecipeSchemaError(
                    f"{sub}: 'sha256' must be a 64-hex sha256 or omitted"
                )
        out.append(PatchSpec(source=source, target=target, sha256=str(sha).lower()))
    return tuple(out)


def _runtime_from_recipe(recipe: Mapping[str, Any], model: str) -> RuntimeSpec:
    owner = f"recipe[{model}]"
    image = _require_str(recipe, "image", owner)
    nodes = _require_int(recipe, "node_count", owner, minimum=1)
    gpus = _require_int(recipe, "gpu_count", owner, minimum=1)
    tp = (
        _require_int(recipe, "tensor_parallel", owner, minimum=1)
        if recipe.get("tensor_parallel")
        else gpus
    )
    pp = int(recipe.get("pipeline_parallel") or 1)
    strategy = str(recipe.get("parallel_strategy") or "tp_pp")
    if strategy not in STRATEGIES:
        raise RecipeSchemaError(f"{owner}: unknown strategy {strategy!r}")
    return RuntimeSpec(
        image=image,
        nodes=nodes,
        gpus_per_node=gpus,
        tensor_parallel=tp,
        pipeline_parallel=pp,
        strategy=strategy,
    )


def _engine_block_from_recipe(recipe: Mapping[str, Any], model: str) -> dict[str, Any]:
    block = dict(recipe.get("engine") or {})
    mml = _require_int(recipe, "max_model_len", f"recipe[{model}]", minimum=512)
    block.setdefault("max_model_len", mml)
    raw_util = recipe.get("gpu_mem_util")
    block.setdefault(
        "gpu_mem_util", raw_util if isinstance(raw_util, (int, float)) else 0.9
    )
    return block


def build_spec_from_recipe(
    recipe: Mapping[str, Any],
    *,
    site: SiteSpec,
    project_id: str,
    owner_id: str,
) -> ResolvedEndpointSpec:
    model = _require_str(recipe, "model", "recipe")
    served = str(recipe.get("served_model_name") or recipe.get("name") or model)
    storage_ref = _require_str(recipe, "path", f"recipe[{model}]")
    engine, dropped = build_engine_spec(
        _engine_block_from_recipe(recipe, model), f"recipe[{model}].engine"
    )
    if dropped:
        # Unknown curated recipe fields are tolerated-but-dropped: the plan
        # stays clean; the dropped names are reported via build_engine_spec()
        # for callers that care (resolver logs them).
        pass
    patches = build_patch_specs(recipe.get("patches") or [], f"recipe[{model}]")
    recipe_file = recipe.get("_recipe_file")
    return ResolvedEndpointSpec(
        model=ModelSpec(
            name=model,
            served_name=served,
            storage_ref=storage_ref,
            owner_id=owner_id,
            project_id=project_id,
        ),
        site=site,
        runtime=_runtime_from_recipe(recipe, model),
        engine=engine,
        patches=patches,
        ttl_sec=int(recipe.get("ttl_sec") or 0),
        readiness_timeout_sec=int(recipe.get("timeout") or 3600),
        provenance=ProvenanceSpec(
            recipe_file=str(recipe_file) if recipe_file else None,
            recipe_hash=(recipe.get("_recipe_hash") or ""),
            resolved_at="",
        ),
    )


def build_generic_spec(
    *,
    model: str,
    served_name: str | None,
    storage_ref: str,
    site: SiteSpec,
    project_id: str,
    owner_id: str,
    serving_profile_defaults: Mapping[str, Any] | None = None,
) -> ResolvedEndpointSpec:
    """Fallback plan for a user-owned model with NO recipe.

    Hard constraints (custom-model integrity): no engine extra_args, no
    patches, no env injection, site-default image only.
    """
    defaults = dict(serving_profile_defaults or {})
    engine_block = {
        "max_model_len": int(defaults.get("max_model_len") or 4096),
        "gpu_mem_util": float(defaults.get("gpu_mem_util") or 0.85),
    }
    engine, _ = build_engine_spec(engine_block, f"generic[{model}]")
    image = str(defaults.get("image") or "docker.io/vllm/vllm-openai:latest")
    return ResolvedEndpointSpec(
        model=ModelSpec(
            name=model,
            served_name=served_name or model,
            storage_ref=storage_ref,
            owner_id=owner_id,
            project_id=project_id,
        ),
        site=replace(site, partition=site.partition),
        runtime=RuntimeSpec(
            image=image,
            nodes=1,
            gpus_per_node=1,
            tensor_parallel=1,
            pipeline_parallel=1,
        ),
        engine=engine,
        provenance=ProvenanceSpec(recipe_file=None, recipe_hash="", resolved_at=""),
    )


# ---------------------------------------------------------------------------
# User overrides (policy gate)
# ---------------------------------------------------------------------------


def apply_user_overrides(
    spec: ResolvedEndpointSpec,
    overrides: Mapping[str, Any] | None,
    *,
    is_admin: bool = False,
) -> ResolvedEndpointSpec:
    """Apply … to the plan; rejects non-USER_TUNABLE for non-admins.

    Admin callers may alter RECIPE_CONTROLLED material only through explicit
    `admin_overrides` — this function deliberately does NOT provide such a lane
    (curated material belongs in recipes files, not hand-set CLI env).
    """
    applied = dict(overrides or {})
    violations = sorted(set(applied) - USER_TUNABLE_FIELDS)
    if violations and not is_admin:
        raise RecipePolicyError(
            "non-admin may only override user-tunable fields; got: "
            + ", ".join(violations)
        )
    applied = {k: v for k, v in applied.items() if k in USER_TUNABLE_FIELDS}
    if not applied:
        return spec
    return _apply_tunable_fields(spec, applied)


def _apply_tunable_fields(
    spec: ResolvedEndpointSpec, applied: dict[str, Any]
) -> ResolvedEndpointSpec:
    if "partition" in applied:
        spec = replace(
            spec, site=replace(spec.site, partition=str(applied["partition"]))
        )
    if "node_count" in applied or "gpu_count" in applied:
        nodes = int(applied.get("node_count", spec.runtime.nodes))
        gpus = int(applied.get("gpu_count", spec.runtime.gpus_per_node))
        spec = replace(
            spec,
            runtime=replace(
                spec.runtime,
                nodes=nodes,
                gpus_per_node=gpus,
                tensor_parallel=min(spec.runtime.tensor_parallel, nodes * gpus),
            ),
        )
    if "max_model_len" in applied:
        mml = int(applied["max_model_len"])
        if mml < 512:
            raise RecipePolicyError("max_model_len must be >= 512")
        spec = replace(spec, engine=replace(spec.engine, max_model_len=mml))
    if "ttl_sec" in applied:
        spec = replace(spec, ttl_sec=int(applied["ttl_sec"]))
    return spec


# ---------------------------------------------------------------------------
# Legacy-roster compat (M2): build a ServingPlan from an existing pool entry
# ---------------------------------------------------------------------------

_MISSING = object()  # sentinel for absent engine keys in legacy rosters


def _env_int(env: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for k in keys:
        v = env.get(k)
        if v is not None and str(v).isdigit():
            return int(str(v))
    return None


def _env_float(env: Mapping[str, Any], key: str) -> float | None:
    v = env.get(key)
    if v is None:
        return None
    try:
        return float(str(v))
    except ValueError:
        return None


def _env_json(env: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    v = env.get(key)
    if not v:
        return None
    try:
        data = json.loads(str(v))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _legacy_engine_block(entry: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Translate legacy `env_vars` into the typed engine block."""
    env = entry.get("env_vars") or {}
    block: dict[str, Any] = {}
    if entry.get("max_model_len") is not None:
        block["max_model_len"] = int(entry["max_model_len"])
    else:
        block["max_model_len"] = _env_int(env, ("VLLM_MAX_MODEL_LEN",)) or 4096
    util = _env_float(env, "VLLM_GPU_MEM_UTIL")
    if util is not None:
        block["gpu_mem_util"] = util
    else:
        tol = entry.get("gpu_mem_util")
        block["gpu_mem_util"] = float(tol) if isinstance(tol, (int, float)) else 0.9
    _legacy_engine_numeric(env, block)
    block["tokenizer"] = env.get("VLLM_TOKENIZER")
    block["tool_call_parser"] = env.get("VLLM_TOOL_CALL_PARSER")
    block["reasoning_parser"] = env.get("VLLM_REASONING_PARSER")
    block["speculative_config"] = _env_json(env, "VLLM_SPECULATIVE_CONFIG")
    lmo = str(env.get("VLLM_LANGUAGE_MODEL_ONLY", "")).lower()
    block["language_model_only"] = lmo in {"1", "true", "yes"}
    block["extra_args"] = env.get("VLLM_EXTRA_ARGS") or ""
    return block


def _legacy_engine_numeric(env: Mapping[str, Any], block: dict[str, Any]) -> None:
    for key in ("VLLM_KV_CACHE_MEMORY_BYTES", "VLLM_KV_CACHE_BYTES"):
        val = _env_int(env, (key,))
        if val is not None:
            block["kv_cache_memory_bytes"] = val
            break
    seqs = _env_int(env, ("VLLM_MAX_NUM_SEQS",))
    if seqs is not None:
        block["max_num_seqs"] = seqs
    batched = _env_int(env, ("VLLM_MAX_NUM_BATCHED_TOKENS",))
    if batched is not None:
        block["max_num_batched_tokens"] = batched


def _legacy_runtime(entry: Mapping[str, Any], env: Mapping[str, Any]) -> RuntimeSpec:
    default_image = env.get("VLLM_DOCKER_IMAGE") or "docker.io/vllm/vllm-openai:latest"
    tp = _env_int(env, ("VLLM_TENSOR_PARALLEL_SIZE", "VLLM_TP")) or int(
        entry.get("gpu_count") or 1
    )
    return RuntimeSpec(
        image=str(entry.get("image") or default_image),
        nodes=int(entry.get("node_count") or 1),
        gpus_per_node=int(entry.get("gpu_count") or 1),
        tensor_parallel=tp,
        pipeline_parallel=int(entry.get("pipeline_parallel") or 1),
        strategy=str(entry.get("parallel_strategy") or "tp_pp"),
    )


def _legacy_model_spec(
    entry: Mapping[str, Any], project_id: str, owner_id: str, storage_ref: str
) -> ModelSpec:
    model = str(entry["model"])
    served = str(entry.get("served_model_name") or entry.get("name") or model)
    ref = storage_ref or str(entry.get("path") or "")
    return ModelSpec(
        name=model,
        served_name=served,
        storage_ref=ref,
        owner_id=owner_id,
        project_id=project_id,
    )


def build_spec_from_legacy_entry(
    entry: Mapping[str, Any],
    *,
    site: SiteSpec,
    project_id: str,
    owner_id: str,
    storage_ref: str = "",
) -> ResolvedEndpointSpec:
    """Build a ServingPlan from a production `pool.yaml` entry as-is (M2)."""
    model = str(entry["model"])
    env = {str(k): str(v) for k, v in (entry.get("env_vars") or {}).items()}
    engine, _dropped = build_engine_spec(
        _legacy_engine_block(entry, model), f"legacy[{model}].env_vars"
    )
    return ResolvedEndpointSpec(
        model=_legacy_model_spec(entry, project_id, owner_id, storage_ref),
        site=site,
        runtime=_legacy_runtime(entry, env),
        engine=engine,
        patches=build_patch_specs(entry.get("patches") or [], f"legacy[{model}]"),
        env=env,
        ttl_sec=int(entry.get("ttl_sec") or 0),
        readiness_timeout_sec=int(entry.get("timeout") or 3600),
        provenance=ProvenanceSpec(
            recipe_file=str(entry.get("_recipe_file") or "") or None,
            recipe_hash=str(entry.get("_recipe_hash") or ""),
            resolved_at="",
        ),
    )


# ---------------------------------------------------------------------------
# Inversion (used by the golden-diff harness): plan → legacy defaults dict
# ---------------------------------------------------------------------------


def spec_to_legacy_defaults(spec: ResolvedEndpointSpec) -> dict[str, Any]:
    """Recreate the OLD resolver's return shape from a plan (test-of-equivalence).

    Legacy = field subset a caller expects from `resolve_model_provision_defaults`
    through the pool rosters. The env union re-materialises engine-representable
    keys so the bucket matches the original verbatim env content.
    """
    out: dict[str, Any] = {}
    if spec.site.partition:
        out["partition"] = spec.site.partition
    out["node_count"] = spec.runtime.nodes
    out["gpu_count"] = spec.runtime.gpus_per_node
    out["max_model_len"] = spec.engine.max_model_len
    out["ttl_sec"] = spec.ttl_sec
    env: dict[str, str] = dict(spec.env)
    _engine_to_env(spec, env)
    if env:
        out["env_vars"] = env
    if spec.runtime.strategy != "tp_pp":
        out["parallel_strategy"] = spec.runtime.strategy
    return out


def _engine_to_env(spec: ResolvedEndpointSpec, env: dict[str, str]) -> None:
    env.setdefault("VLLM_TENSOR_PARALLEL_SIZE", str(spec.runtime.tensor_parallel))
    env.setdefault("VLLM_GPU_MEM_UTIL", str(spec.engine.gpu_mem_util))
    env.setdefault("VLLM_MAX_MODEL_LEN", str(spec.engine.max_model_len))
    env.setdefault("VLLM_DOCKER_IMAGE", spec.runtime.image)
    for key, val in _opt_engine_str_env(spec):
        env.setdefault(key, val)
    for key, val in _opt_engine_num_env(spec):
        env.setdefault(key, val)


def _opt_engine_str_env(spec: ResolvedEndpointSpec) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if spec.engine.tool_call_parser:
        out.append(("VLLM_TOOL_CALL_PARSER", spec.engine.tool_call_parser))
    if spec.engine.reasoning_parser:
        out.append(("VLLM_REASONING_PARSER", spec.engine.reasoning_parser))
    if spec.engine.tokenizer:
        out.append(("VLLM_TOKENIZER", spec.engine.tokenizer))
    if spec.engine.extra_args:
        out.append(("VLLM_EXTRA_ARGS", " ".join(spec.engine.extra_args)))
    if spec.engine.language_model_only:
        out.append(("VLLM_LANGUAGE_MODEL_ONLY", "1"))
    return out


def _opt_engine_num_env(spec: ResolvedEndpointSpec) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if spec.engine.kv_cache_memory_bytes:
        out.append(
            ("VLLM_KV_CACHE_MEMORY_BYTES", str(spec.engine.kv_cache_memory_bytes))
        )
    if spec.engine.max_num_seqs:
        out.append(("VLLM_MAX_NUM_SEQS", str(spec.engine.max_num_seqs)))
    if spec.engine.max_num_batched_tokens:
        out.append(
            ("VLLM_MAX_NUM_BATCHED_TOKENS", str(spec.engine.max_num_batched_tokens))
        )
    if spec.engine.speculative_config:
        out.append(
            ("VLLM_SPECULATIVE_CONFIG", json.dumps(spec.engine.speculative_config))
        )
    return out


def spec_env_vars(spec: ResolvedEndpointSpec) -> dict[str, str]:
    """Authoritative env view: engine-materialized defaults + verbatim set.

    Used by the provision handler's env propagation. Verbatim plan-env keys
    stay untouched; engine fields materialize into the standard VLLM_* names
    without overriding anything already present.
    """
    env = dict(spec.env)
    _engine_to_env(spec, env)
    if spec.runtime.strategy != "tp_pp":
        env.setdefault("VLLM_PARALLEL_STRATEGY", spec.runtime.strategy)
    return env
