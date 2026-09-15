"""
DeploymentSpec — immutable, authoritative deployment plan for one endpoint.

A DeploymentSpec is generated at provision time from the registry + roster
defaults and flows through to the Slurm job as explicit environment variables.
serve.slurm receives these vars and executes them — no re-derivation, no model-
name pattern matching, no hidden bash logic.

Design invariant: generated once, consumed everywhere.  Immutable after creation.

Lifecycle:
  Registry (ModelEntry) + Roster (pool-ai4s.yaml) + Site Config
      │
      ▼
  DeploymentSpec.from_registry(model, profile, overrides)
      │
      ├──▶ validate()  → list of error strings (empty = valid)
      │
      ├──▶ generate_sbatch_env()  → dict[str, str] for --export
      │
      └──▶ generate_vllm_args()  → list[str] for serve.slurm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from umrs.model_registry.runtime_config import RuntimeConfig

if TYPE_CHECKING:
    from umrs.model_registry.models import ModelEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeploymentSpec:
    """Immutable deployment plan for a single mini-RiVault endpoint.

    All fields are populated from the authoritative registry (ModelEntry),
    roster defaults (pool-ai4s.yaml), and site config.  No field is derived
    at runtime.
    """

    # ── Identity ──
    alias: str                          # "Kimi-K2.7-Code" (registered UMRS name)
    huggingface_id: str                 # "moonshotai/Kimi-K2.7-Code"
    model_family: str                   # "kimi_k2"
    served_names: tuple[str, ...]       # ("Kimi-K2.7-Code", "moonshotai/Kimi-K2.7-Code")

    # ── Storage ──
    storage_ref: str                    # /work/hps0/.../umrs_models/Kimi-K2.7-Code

    # ── Topology ──
    tp: int                             # 4
    pp: int                             # 1
    ep: bool                            # False (expert-parallel)
    parallel_strategy: str              # "tp_pp" | "tep"
    gpus_per_node: int                  # 4
    node_count: int                     # 1

    # ── Runtime ──
    tool_call_parser: str | None        # None = no parser (or "hermes")
    reasoning_parser: str | None        # None = no reasoning parser
    kv_cache_dtype: str | None          # None = default (bf16)
    enforce_eager: bool                 # True = no CUDA graphs
    flashinfer_moe_fp16: bool           # False = fall back to Triton MoE
    chat_template_format: str | None    # None = "auto"
    mm_encoder_tp_mode: str | None      # None = not needed
    trust_remote_code: bool
    enable_auto_tool_choice: bool
    enable_prefix_caching: bool

    # ── Capacity ──
    max_model_len: int                  # 245760
    max_num_seqs: int                   # 32
    max_num_batched_tokens: int         # 8192
    gpu_memory_utilization: float       # 0.90

    # ── Infrastructure ──
    partition: str                      # "4n4gpu-p"
    slurm_account: str | None           # "ea" (site master) or per-user
    container_image: str                # /work/.../mini-rivault-aarch64-v4.sif
    slurm_time_limit: str               # "96:00:00"
    slurm_cpus: int                     # 32
    slurm_memory: str                   # "1400G"

    # ── Health ──
    readiness_probe: str                # "models" | "chat" | "http"
    readiness_probe_model: str          # "Kimi-K2.7-Code" (alias for /v1/models check)
    readiness_path: str | None          # "/readyz" for BentoML vision

    # ── Extras ──
    env_overrides: dict[str, str]       # Pool/env var overrides (VLLM_MAX_NUM_SEQS etc.)
    ttl_sec: int                        # 0 = no expiry
    tier: str                           # "system" | "project" | "user"

    # ── Traceability ──
    spec_version: str = "1.0"
    generated_at: str = ""              # ISO timestamp
    registry_model_id: str = ""         # model_catalog_entries.model_id

    def validate(self) -> list[str]:
        """Validate this spec is deployable. Returns list of errors."""
        errors: list[str] = []

        # Identity
        if not self.alias:
            errors.append("alias is required")
        if not self.huggingface_id:
            errors.append("huggingface_id is required")
        if not self.storage_ref:
            errors.append("storage_ref is required")
        if not self.served_names:
            errors.append("served_names must be non-empty (at least the alias)")

        # Topology
        if self.tp < 1:
            errors.append(f"tp must be >= 1, got {self.tp}")
        if self.pp < 1:
            errors.append(f"pp must be >= 1, got {self.pp}")
        if self.gpus_per_node < 1:
            errors.append(f"gpus_per_node must be >= 1, got {self.gpus_per_node}")
        if self.node_count < 1:
            errors.append(f"node_count must be >= 1, got {self.node_count}")
        total_gpus = self.tp * self.pp
        alloc_gpus = self.gpus_per_node * self.node_count
        if total_gpus > alloc_gpus:
            errors.append(
                f"TP×PP ({self.tp}×{self.pp}={total_gpus}) exceeds allocated "
                f"GPUs ({self.gpus_per_node}×{self.node_count}={alloc_gpus})"
            )
        if self.pp > 1 and self.node_count < self.pp:
            errors.append(
                f"PP={self.pp} requires at least {self.pp} nodes, "
                f"got {self.node_count}"
            )

        # Parallel strategy
        if self.parallel_strategy not in ("tp_pp", "tep"):
            errors.append(f"Unknown parallel_strategy: {self.parallel_strategy}")
        if self.parallel_strategy == "tep" and self.node_count < 2:
            errors.append("TEP requires at least 2 nodes")
        if self.parallel_strategy == "tep" and self.ep != True:
            errors.append("TEP requires expert-parallel (ep=True)")

        # Runtime
        if self.tool_call_parser == "":
            # Explicit empty = disabled; valid.
            pass
        if self.reasoning_parser == "":
            # Explicit empty = disabled; valid.
            pass

        # Capacity
        if self.max_model_len < 512:
            errors.append(f"max_model_len too small: {self.max_model_len}")
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1.0:
            errors.append(
                f"gpu_memory_utilization must be in (0, 1.0], "
                f"got {self.gpu_memory_utilization}"
            )
        if self.max_num_seqs < 1:
            errors.append(f"max_num_seqs must be >= 1, got {self.max_num_seqs}")

        # Infrastructure
        if not self.partition:
            errors.append("partition is required")
        if not self.container_image:
            errors.append("container_image (SIF path) is required")

        return errors

    def generate_sbatch_env(self) -> dict[str, str]:
        """Generate the --export environment variables for sbatch.

        These are consumed by serve.slurm as explicit VLLM_* env vars.
        serve.slurm no longer derives them from model name patterns.
        """
        env: dict[str, str] = {
            # Identity
            "MODEL_NAME": self.alias,
            "VLLM_SERVED_MODEL_NAME": self.huggingface_id,
            # Storage
            "VLLM_MODEL_PATH": self.storage_ref,
            # Topology
            "VLLM_TENSOR_PARALLEL_SIZE": str(self.tp),
            "VLLM_PIPELINE_PARALLEL_SIZE": str(self.pp),
            "VLLM_PARALLEL_STRATEGY": self.parallel_strategy,
            # Runtime (explicit — serve.slurm uses ${VAR:-$(derive)} fallback)
            "VLLM_MAX_MODEL_LEN": str(self.max_model_len),
            "VLLM_GPU_MEMORY_UTILIZATION": str(self.gpu_memory_utilization),
            # Health
            "VLLM_LITELLM_MASTER_KEY_PLACEHOLDER": "",  # Set by orchestrator
        }

        # Optional runtime settings (only set when non-default)
        if self.tool_call_parser is not None:
            env["VLLM_TOOL_CALL_PARSER"] = self.tool_call_parser
        if self.reasoning_parser is not None:
            env["VLLM_REASONING_PARSER"] = self.reasoning_parser
        if self.kv_cache_dtype is not None:
            env["VLLM_KV_CACHE_DTYPE"] = self.kv_cache_dtype
        if self.enforce_eager:
            env["VLLM_ENFORCE_EAGER"] = "1"
        if not self.flashinfer_moe_fp16:
            env["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
        if self.chat_template_format is not None:
            env["VLLM_CHAT_TEMPLATE_FORMAT"] = self.chat_template_format
        if self.mm_encoder_tp_mode is not None:
            env["VLLM_MM_ENCODER_TP_MODE"] = self.mm_encoder_tp_mode
        if self.max_num_seqs:
            env["VLLM_MAX_NUM_SEQS"] = str(self.max_num_seqs)
        if self.max_num_batched_tokens:
            env["VLLM_MAX_NUM_BATCHED_TOKENS"] = str(self.max_num_batched_tokens)
        if self.ep:
            env["VLLM_EXPERT_PARALLEL"] = "1"

        # Merge user/pool env overrides (they win over spec defaults)
        env.update(self.env_overrides)

        return env

    def generate_vllm_args(self) -> list[str]:
        """Generate the vLLM CLI arguments.

        Used for dry-run validation and for future direct-launch paths.
        serve.slurm can also consume this to reduce its own arg construction.
        """
        args: list[str] = [
            "--model", "/models",  # container-internal path
            "--served-model-name", self.huggingface_id,
            "--host", "0.0.0.0",
            "-tp", str(self.tp),
            "-pp", str(self.pp),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--trust-remote-code",
        ]

        # Additional served names
        for name in self.served_names:
            if name != self.huggingface_id:
                args += ["--served-model-name", name]

        # Context length
        if self.max_model_len > 0:
            args += ["--max-model-len", str(self.max_model_len)]

        # Tool-call parser
        if self.tool_call_parser and self.tool_call_parser != "":
            args += ["--tool-call-parser", self.tool_call_parser]

        # Reasoning parser
        if self.reasoning_parser and self.reasoning_parser != "":
            args += ["--reasoning-parser", self.reasoning_parser]

        # KV-cache dtype
        if self.kv_cache_dtype:
            args += ["--kv-cache-dtype", self.kv_cache_dtype]

        # Eager mode
        if self.enforce_eager:
            args.append("--enforce-eager")

        # Expert parallelism
        if self.ep:
            args.append("--enable-expert-parallel")

        # Chat template format
        if self.chat_template_format:
            args += ["--chat-template-content-format", self.chat_template_format]

        # MM encoder TP mode
        if self.mm_encoder_tp_mode:
            args += ["--mm-encoder-tp-mode", self.mm_encoder_tp_mode]

        # Concurrency
        if self.max_num_seqs:
            args += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.max_num_batched_tokens:
            args += ["--max-num-batched-tokens", str(self.max_num_batched_tokens)]

        # Auto tool choice
        if self.enable_auto_tool_choice:
            args.append("--enable-auto-tool-choice")

        # Prefix caching
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")

        return args

    def with_overrides(self, **kwargs: Any) -> "DeploymentSpec":
        """Return a new DeploymentSpec with specific fields overridden.

        Because DeploymentSpec is frozen, this creates a copy.
        """
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/audit."""
        return {
            "alias": self.alias,
            "huggingface_id": self.huggingface_id,
            "model_family": self.model_family,
            "served_names": list(self.served_names),
            "storage_ref": self.storage_ref,
            "tp": self.tp,
            "pp": self.pp,
            "ep": self.ep,
            "parallel_strategy": self.parallel_strategy,
            "gpus_per_node": self.gpus_per_node,
            "node_count": self.node_count,
            "tool_call_parser": self.tool_call_parser,
            "reasoning_parser": self.reasoning_parser,
            "kv_cache_dtype": self.kv_cache_dtype,
            "enforce_eager": self.enforce_eager,
            "flashinfer_moe_fp16": self.flashinfer_moe_fp16,
            "chat_template_format": self.chat_template_format,
            "mm_encoder_tp_mode": self.mm_encoder_tp_mode,
            "trust_remote_code": self.trust_remote_code,
            "enable_auto_tool_choice": self.enable_auto_tool_choice,
            "enable_prefix_caching": self.enable_prefix_caching,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "partition": self.partition,
            "slurm_account": self.slurm_account,
            "container_image": self.container_image,
            "slurm_time_limit": self.slurm_time_limit,
            "slurm_cpus": self.slurm_cpus,
            "slurm_memory": self.slurm_memory,
            "readiness_probe": self.readiness_probe,
            "readiness_probe_model": self.readiness_probe_model,
            "readiness_path": self.readiness_path,
            "env_overrides": self.env_overrides,
            "ttl_sec": self.ttl_sec,
            "tier": self.tier,
            "spec_version": self.spec_version,
            "generated_at": self.generated_at,
            "registry_model_id": self.registry_model_id,
        }

    @classmethod
    def from_registry(
        cls,
        entry: "ModelEntry",
        *,
        profile: dict[str, Any] | None = None,
        site_config: Any | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "DeploymentSpec":
        """Build a DeploymentSpec from the authoritative registry entry.

        This is the ONLY path to creating a DeploymentSpec.
        No JSON catalog. No bash derivation. No pool YAML as primary source.

        Args:
            entry: ModelEntry from the Postgres registry.
            profile: Optional deployment profile dict (from pool roster or API).
            site_config: SiteConfig for infrastructure defaults.
            overrides: Map of field overrides from the API request.

        Returns:
            A validated DeploymentSpec ready for sbatch.
        """
        from datetime import datetime, timezone

        overrides = overrides or {}
        profile = profile or {}
        sp = entry.serving_profile
        rc = entry.runtime_config

        # ── Identity ──
        alias = entry.name
        hf_id = overrides.get("huggingface_id") or entry.huggingface_id or alias
        served_names = (alias, hf_id) if alias != hf_id else (alias,)

        # ── Storage ──
        storage_ref = entry.storage_ref

        # ── Topology ──
        tp = overrides.get("tp") or sp.tensor_parallel_size or 1
        pp = overrides.get("pp") or sp.pipeline_parallel_size or 1
        gpus_per_node = profile.get("gpu_count") or overrides.get("gpu_count") or sp.recommended_gpu_count or 1
        node_count = profile.get("node_count") or overrides.get("node_count") or sp.recommended_node_count or 1
        parallel_strategy = profile.get("parallel_strategy") or overrides.get("parallel_strategy") or "tp_pp"
        ep = parallel_strategy == "tep"

        # ── Runtime ──
        tool_call_parser = rc.effective_tool_call_parser()
        reasoning_parser = rc.effective_reasoning_parser()
        kv_cache_dtype = rc.kv_cache_dtype
        enforce_eager = rc.enforce_eager or False
        flashinfer_moe_fp16 = rc.flashinfer_moe_fp16 if rc.flashinfer_moe_fp16 is not None else True
        chat_template_format = rc.chat_template_format
        mm_encoder_tp_mode = rc.mm_encoder_tp_mode
        trust_remote_code = rc.trust_remote_code if rc.trust_remote_code is not None else True
        enable_auto_tool_choice = rc.enable_auto_tool_choice if rc.enable_auto_tool_choice is not None else True
        enable_prefix_caching = rc.enable_prefix_caching if rc.enable_prefix_caching is not None else True

        # ── Capacity ──
        ctx_from_profile = profile.get("max_model_len")
        ctx_from_sp = sp.max_model_len
        ctx_from_override = overrides.get("max_model_len")
        max_model_len = ctx_from_override or ctx_from_profile or ctx_from_sp or 8192
        max_num_seqs = profile.get("max_num_seqs") or 16
        max_num_batched_tokens = profile.get("max_num_batched_tokens") or 0
        gpu_mem_util = overrides.get("gpu_memory_utilization") or sp.gpu_memory_utilization or 0.90

        # ── Infrastructure ──
        partition = profile.get("partition") or ""
        slurm_account = profile.get("slurm_account")
        container_image = ""
        slurm_time_limit = "96:00:00"
        slurm_cpus = 32
        slurm_memory = "1400G"
        if site_config:
            container_image = getattr(site_config.runtime, "rivault_dir", "") or ""
            slurm_time_limit = getattr(site_config.scheduler, "time_limit", "96:00:00")
            slurm_cpus = getattr(site_config.scheduler, "cpus", 32)
            slurm_memory = getattr(site_config.scheduler, "memory", "1400G")
            if not partition:
                partition = site_config.scheduler.partition.split(",")[0]
            if not slurm_account:
                slurm_account = getattr(site_config.scheduler, "account", None)

        # ── Health ──
        readiness_probe = profile.get("readiness_probe") or sp.readiness_probe or "models"
        readiness_probe_model = alias
        readiness_path = sp.readiness_path

        # ── Extras ──
        env_overrides = {}
        if isinstance(profile.get("env_vars"), dict):
            env_overrides.update(
                {str(k): str(v) for k, v in profile["env_vars"].items()}
            )
        if isinstance(overrides.get("env_vars"), dict):
            env_overrides.update(
                {str(k): str(v) for k, v in overrides["env_vars"].items()}
            )
        if sp.quantization and "VLLM_QUANTIZATION" not in env_overrides:
            env_overrides["VLLM_QUANTIZATION"] = sp.quantization
        ttl_sec = profile.get("ttl_sec") or overrides.get("ttl_sec") or 0
        tier = profile.get("tier") or overrides.get("tier") or "user"

        return cls(
            alias=alias,
            huggingface_id=hf_id,
            model_family=rc.model_family or "unknown",
            served_names=served_names,
            storage_ref=storage_ref,
            tp=tp,
            pp=pp,
            ep=ep,
            parallel_strategy=parallel_strategy,
            gpus_per_node=gpus_per_node,
            node_count=node_count,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            flashinfer_moe_fp16=flashinfer_moe_fp16,
            chat_template_format=chat_template_format,
            mm_encoder_tp_mode=mm_encoder_tp_mode,
            trust_remote_code=trust_remote_code,
            enable_auto_tool_choice=enable_auto_tool_choice,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_mem_util,
            partition=partition,
            slurm_account=slurm_account,
            container_image=container_image,
            slurm_time_limit=slurm_time_limit,
            slurm_cpus=slurm_cpus,
            slurm_memory=slurm_memory,
            readiness_probe=readiness_probe,
            readiness_probe_model=readiness_probe_model,
            readiness_path=readiness_path,
            env_overrides=env_overrides,
            ttl_sec=ttl_sec,
            tier=tier,
            generated_at=datetime.now(timezone.utc).isoformat(),
            registry_model_id=entry.model_id,
        )
