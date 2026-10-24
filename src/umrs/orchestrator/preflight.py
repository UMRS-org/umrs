"""
Preflight Validation — validate deployment readiness before sbatch.

Provides ``umrs endpoint validate``: a comprehensive pre-submission check
that validates registry entry, storage, parser availability, tokenizer,
topology, capacity, partition, and container image.

Target: most deployment failures occur before sbatch. No GPU allocation required.

Usage:
    from umrs.orchestrator.preflight import PreflightValidator

    validator = PreflightValidator(spec=deployment_spec, site_config=site)
    result = await validator.run_all()
    print(result.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .partition_choice import split_partitions

if TYPE_CHECKING:
    from umrs.orchestrator.deployment_spec import DeploymentSpec
    from umrs.orchestrator.launcher_stack import LauncherStack
    from umrs.orchestrator.ssh_control import SSHControlManager

logger = logging.getLogger(__name__)


@dataclass
class PreflightCheck:
    """Result of a single preflight check."""

    name: str  # "registry", "storage", "parser", etc.
    status: str  # "pass", "fail", "warn", "skip"
    message: str  # Human-readable result
    duration_ms: float = 0.0
    detail: str | None = None  # Extra detail for failures

    def symbol(self) -> str:
        if self.status == "pass":
            return "\u2713"  # ✓
        if self.status == "fail":
            return "\u2717"  # ✗
        if self.status == "warn":
            return "\u26a0"  # ⚠
        return "\u2014"  # —


@dataclass
class PreflightResult:
    """Aggregate result of all preflight checks."""

    passed: bool
    checks: list[PreflightCheck]
    total_duration_ms: float = 0.0

    @property
    def failure_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    def summary(self) -> str:
        """Produce a human-readable summary."""
        lines = [
            f"Preflight Validation for {self.checks[0].message if self.checks else 'unknown'}",
            "\u2500" * 60,
        ]
        for check in self.checks:
            lines.append(f"  {check.symbol()} {check.name:<15} {check.message}")
            if check.detail and check.status != "pass":
                lines.append(f"     {check.detail}")
        lines.append("\u2500" * 60)
        if self.passed:
            lines.append(
                f"READY TO DEPLOY \u2014 {len(self.checks)} checks passed "
                f"({self.total_duration_ms:.0f}ms)"
            )
        else:
            lines.append(
                f"VALIDATION FAILED \u2014 {self.failure_count} failure(s), "
                f"{self.warn_count} warning(s)"
            )
        return "\n".join(lines)


class PreflightValidator:
    """Validates a DeploymentSpec before sbatch submission.

    Checks are ordered from cheapest (local checks) to most expensive
    (remote SSH checks).  This ensures fast failures for obvious errors.
    """

    def __init__(
        self,
        spec: DeploymentSpec,
        site_config: Any | None = None,
        launcher_stack: LauncherStack | None = None,
        ssh_manager: SSHControlManager | None = None,
    ):
        self.spec = spec
        self.site = site_config
        self._stack = launcher_stack
        self._ssh = ssh_manager

    # ── Public API ───────────────────────────────────────────────────────────

    async def run_all(self) -> PreflightResult:
        """Run all preflight checks and return aggregate result."""
        t0 = time.monotonic()
        checks: list[PreflightCheck] = []

        # Phase A: Local checks (no SSH needed)
        checks.append(self._check_registry())
        checks.append(self._check_topology())

        # Phase B: Parser check (runs locally if LauncherStack provided)
        checks.append(self._check_parser_availability())

        # Phase C: Remote checks (SSH to login node)
        if self._ssh:
            checks.append(await self._check_storage())
            checks.append(await self._check_partition())
            checks.append(await self._check_container_image())
        else:
            checks.append(
                PreflightCheck(
                    name="remote",
                    status="warn",
                    message="No SSH connection — skipping storage, partition, image checks",
                )
            )

        # Phase C: Capacity estimation (local, after remote data gathered)
        checks.append(self._check_capacity())

        total_ms = (time.monotonic() - t0) * 1000
        passed = all(c.status != "fail" for c in checks)

        return PreflightResult(
            passed=passed,
            checks=checks,
            total_duration_ms=total_ms,
        )

    # ── Individual Checks ────────────────────────────────────────────────────

    def _check_registry(self) -> PreflightCheck:
        """Registry: model has complete metadata with runtime config populated."""
        t0 = time.monotonic()
        issues: list[str] = []

        if not self.spec.alias:
            issues.append("alias is missing")
        if not self.spec.huggingface_id:
            issues.append("huggingface_id is missing")
        if not self.spec.model_family or self.spec.model_family == "unknown":
            issues.append("model_family is unknown — runtime config may be incomplete")
        if not self.spec.registry_model_id:
            issues.append("not linked to a registry entry (registry_model_id is empty)")

        # Runtime config completeness — only warn when values are missing,
        # since serve.slurm will fall back to derivation. When all models have
        # runtime_config populated, this should become a hard failure.
        runtime_warnings: list[str] = []
        if not self.spec.tool_call_parser or self.spec.tool_call_parser == "":
            runtime_warnings.append(
                "tool_call_parser not set in registry — serve.slurm will derive from model name"
            )
        if self.spec.reasoning_parser is not None and self.spec.reasoning_parser == "":
            # Explicit empty = disabled; valid
            pass
        elif not self.spec.reasoning_parser or self.spec.reasoning_parser == "":
            runtime_warnings.append(
                "reasoning_parser not set in registry — serve.slurm will derive from model name"
            )
        if not self.spec.kv_cache_dtype:
            runtime_warnings.append(
                "kv_cache_dtype not set in registry — serve.slurm will derive from parser"
            )

        ms = (time.monotonic() - t0) * 1000

        if issues:
            return PreflightCheck(
                name="registry",
                status="fail",
                message=f"Registry entry incomplete: {len(issues)} issue(s)",
                detail="; ".join(issues + runtime_warnings),
                duration_ms=ms,
            )

        if runtime_warnings:
            return PreflightCheck(
                name="registry",
                status="warn",
                message=f"Registry entry OK but runtime config incomplete: "
                f"{len(runtime_warnings)} field(s) will fall back to serve.slurm derivation",
                detail="; ".join(runtime_warnings),
                duration_ms=ms,
            )

        return PreflightCheck(
            name="registry",
            status="pass",
            message=f"Model found ({self.spec.model_family} family) with complete runtime config",
            duration_ms=ms,
        )

    def _check_topology(self) -> PreflightCheck:
        """Topology: TP/PP/EP are valid for the target nodes."""
        t0 = time.monotonic()
        spec_errors = self.spec.validate()
        # Filter to topology-related errors only
        topo_errors = [
            e
            for e in spec_errors
            if any(
                kw in e.lower() for kw in ("tp", "pp", "node", "gpu", "parallel", "tep")
            )
        ]

        if not spec_errors:
            # Also validate against site partition profiles if available
            site_errors = self._validate_partition_profile()
            ms = (time.monotonic() - t0) * 1000
            if site_errors:
                return PreflightCheck(
                    name="topology",
                    status="warn",
                    message=f"Topology valid but partition concerns: {len(site_errors)}",
                    detail="; ".join(site_errors),
                    duration_ms=ms,
                )
            return PreflightCheck(
                name="topology",
                status="pass",
                message=f"TP={self.spec.tp} PP={self.spec.pp} EP={self.spec.ep} "
                f"({self.spec.parallel_strategy}) on "
                f"{self.spec.node_count}×{self.spec.gpus_per_node} GPU",
                duration_ms=ms,
            )

        ms = (time.monotonic() - t0) * 1000
        return PreflightCheck(
            name="topology",
            status="fail",
            message=f"Topology invalid: {len(topo_errors)} error(s)",
            detail="; ".join(topo_errors),
            duration_ms=ms,
        )

    def _validate_partition_profile(self) -> list[str]:
        """Check that the requested partition can satisfy the GPU/node request.

        A comma list (the submit picks one) is checked element by element.
        """
        if not self.site:
            return []
        profiles = getattr(self.site, "partition_profiles", {}) or {}
        errors = []
        for partition in split_partitions(self.spec.partition):
            profile_gpus = getattr(profiles.get(partition), "gpus", 0)
            if profile_gpus and self.spec.gpus_per_node > profile_gpus:
                errors.append(
                    f"Partition '{partition}' has {profile_gpus} GPU/node, "
                    f"but spec requests {self.spec.gpus_per_node}"
                )
        return errors

    async def _check_storage(self) -> PreflightCheck:
        """Storage: model weights and required files exist on the shared FS."""
        t0 = time.monotonic()
        storage = self.spec.storage_ref

        if not storage:
            return PreflightCheck(
                name="storage",
                status="fail",
                message="storage_ref is empty",
                duration_ms=0,
            )

        # Check existence of model directory
        result = await self._ssh.run_remote(
            f"test -d {_sh_quote(storage)} && echo OK || echo MISSING",
            timeout=15,
        )
        ms = (time.monotonic() - t0) * 1000

        if not result.ok or "MISSING" in result.stdout:
            return PreflightCheck(
                name="storage",
                status="fail",
                message=f"Model directory not found: {storage}",
                detail=result.stderr[:200] if result.stderr else None,
                duration_ms=ms,
            )

        # Check for required files
        result2 = await self._ssh.run_remote(
            f"for f in config.json tokenizer.json tokenizer_config.json; do "
            f'test -f {_sh_quote(storage)}/$f && echo "$f:OK" || echo "$f:MISSING"; done',
            timeout=15,
        )
        ms = (time.monotonic() - t0) * 1000

        missing = [
            line.split(":")[0]
            for line in result2.stdout.strip().split("\n")
            if line.endswith(":MISSING")
        ]

        if missing:
            return PreflightCheck(
                name="storage",
                status="fail",
                message=f"Missing required files: {', '.join(missing)}",
                duration_ms=ms,
            )

        return PreflightCheck(
            name="storage",
            status="pass",
            message="All required files present (config.json, tokenizer.json, tokenizer_config.json)",
            duration_ms=ms,
        )

    def _check_parser_availability(self) -> PreflightCheck:
        """Runtime: tool-call and reasoning parsers are available in the SIF."""
        t0 = time.monotonic()
        if not self._stack:
            ms = (time.monotonic() - t0) * 1000
            # LauncherStack not wired — cannot verify parser availability.
            # Treat as a warning so operators know the check was skipped,
            # but don't block deployments during rollout.
            return PreflightCheck(
                name="parser",
                status="warn",
                message="No LauncherStack available — cannot verify parser exists in SIF",
                detail="Wire LauncherStack into PreflightValidator for full parser validation.",
                duration_ms=ms,
            )

        issues: list[str] = []

        # Check tool-call parser
        parser = self.spec.tool_call_parser
        if parser and parser != "":
            if self._stack.available_tool_parsers is None:
                issues.append(f"Cannot verify parser '{parser}' — probe failed")
            elif not self._stack.has_parser(parser):
                issues.append(
                    f"Tool parser '{parser}' not in SIF. "
                    f"Available: {self._stack.available_tool_parsers}"
                )

        # Check reasoning parser
        reasoning = self.spec.reasoning_parser
        if reasoning and reasoning != "":
            if self._stack.available_reasoning_parsers is None:
                issues.append(
                    f"Cannot verify reasoning parser '{reasoning}' — probe failed"
                )
            elif not self._stack.has_reasoning_parser(reasoning):
                issues.append(
                    f"Reasoning parser '{reasoning}' not in SIF. "
                    f"Available: {self._stack.available_reasoning_parsers}"
                )

        ms = (time.monotonic() - t0) * 1000

        if issues:
            return PreflightCheck(
                name="parser",
                status="fail",
                message=f"Parser availability issue: {len(issues)} problem(s)",
                detail="; ".join(issues),
                duration_ms=ms,
            )

        if parser or reasoning:
            msg_parts = []
            if parser and parser != "":
                msg_parts.append(f"Tool parser '{parser}' available")
            if reasoning and reasoning != "":
                msg_parts.append(f"Reasoning parser '{reasoning}' available")
            return PreflightCheck(
                name="parser",
                status="pass",
                message=", ".join(msg_parts),
                duration_ms=ms,
            )

        return PreflightCheck(
            name="parser",
            status="pass",
            message="No special parser required",
            duration_ms=ms,
        )

    async def _check_partition(self) -> PreflightCheck:
        """Infrastructure: Slurm partition exists and is available."""
        t0 = time.monotonic()
        partition = self.spec.partition

        if not self._ssh:
            return PreflightCheck(
                name="partition", status="skip", message="No SSH", duration_ms=0
            )
        if not partition:
            ms = (time.monotonic() - t0) * 1000
            return PreflightCheck(
                name="partition",
                status="fail",
                message="No partition specified",
                duration_ms=ms,
            )

        result = await self._ssh.run_remote(
            f"sinfo -p {_sh_quote(partition)} -h -o '%a' 2>&1 | head -1",
            timeout=15,
        )
        ms = (time.monotonic() - t0) * 1000

        state = result.stdout.strip()
        if not state or "error" in result.stderr.lower():
            return PreflightCheck(
                name="partition",
                status="fail",
                message=f"Partition '{partition}' not found or sinfo failed",
                detail=result.stderr[:200] if result.stderr else None,
                duration_ms=ms,
            )

        if state != "up":
            return PreflightCheck(
                name="partition",
                status="warn",
                message=f"Partition '{partition}' state is '{state}' (not 'up')",
                duration_ms=ms,
            )

        return PreflightCheck(
            name="partition",
            status="pass",
            message=f"Partition '{partition}' exists and is up",
            duration_ms=ms,
        )

    async def _check_container_image(self) -> PreflightCheck:
        """SIF: container image exists and is accessible."""
        t0 = time.monotonic()
        image = self.spec.container_image

        if not self._ssh:
            return PreflightCheck(
                name="image", status="skip", message="No SSH", duration_ms=0
            )
        if not image:
            ms = (time.monotonic() - t0) * 1000
            return PreflightCheck(
                name="image",
                status="fail",
                message="No container image specified",
                duration_ms=ms,
            )

        result = await self._ssh.run_remote(
            f"test -f {_sh_quote(image)} && echo OK || echo MISSING",
            timeout=15,
        )
        ms = (time.monotonic() - t0) * 1000

        if not result.ok or "MISSING" in result.stdout:
            return PreflightCheck(
                name="image",
                status="fail",
                message=f"SIF image not found: {image}",
                duration_ms=ms,
            )

        return PreflightCheck(
            name="image",
            status="pass",
            message=f"SIF found: {image}",
            duration_ms=ms,
        )

    def _check_capacity(self) -> PreflightCheck:
        """Capacity: context length fits in estimated KV-cache memory."""
        t0 = time.monotonic()

        ctx = self.spec.max_model_len
        tp = self.spec.tp
        gpu_mem_util = self.spec.gpu_memory_utilization
        gpus_per_node = self.spec.gpus_per_node
        node_count = self.spec.node_count
        total_gpus = gpus_per_node * node_count

        # GPU memory per B200 (GB200 NVL4): 173.2 GiB
        mem_per_gpu_gb = 173.2  # Hardcoded for AI4S; site_config-driven in future

        # Simple KV-cache estimate:
        #   KV ≈ 2 × num_layers × num_kv_heads × head_dim × context × dtype_bytes
        # For a generic large MoE (e.g., DeepSeek-V4-Pro):
        #   ~64 layers, ~8 KV heads, ~128 head_dim, fp8 (1 byte)
        #   KV = 2 × 64 × 8 × 128 × ctx × 1 = 131,072 × ctx bytes
        #   = 0.125 × ctx KiB ≈ 0.12 × ctx MiB per element
        # This is approximate — a real calculation would read config.json.
        kv_bytes_per_token = 2 * 64 * 8 * 128 * 1  # ~131 KiB
        total_kv_gib = (kv_bytes_per_token * ctx) / (1024**3)
        kv_per_gpu_gib = total_kv_gib / tp  # KV is sharded by TP

        usable_mem_gib = mem_per_gpu_gb * gpu_mem_util
        # Rough weight memory for large MoE models: ~40-60 GiB per GPU at tp4+
        weight_mem_gib = 50.0  # Conservative estimate

        available_for_kv = usable_mem_gib - weight_mem_gib
        headroom_gib = available_for_kv - kv_per_gpu_gib

        ms = (time.monotonic() - t0) * 1000

        if headroom_gib < 0:
            return PreflightCheck(
                name="capacity",
                status="fail",
                message=(
                    f"Context {ctx} too large: KV cache ~{kv_per_gpu_gib:.1f} GiB/GPU "
                    f"exceeds available ~{available_for_kv:.1f} GiB/GPU "
                    f"({mem_per_gpu_gb:.0f} GiB × {gpu_mem_util} util − ~{weight_mem_gib:.0f} GiB weights)"
                ),
                detail=(
                    f"Total KV: {total_kv_gib:.1f} GiB across {tp} TP shards. "
                    f"Consider reducing max_model_len or increasing node_count."
                ),
                duration_ms=ms,
            )

        if headroom_gib < 20:
            return PreflightCheck(
                name="capacity",
                status="warn",
                message=(
                    f"Low KV headroom: ~{headroom_gib:.1f} GiB/GPU "
                    f"(KV uses ~{kv_per_gpu_gib:.1f} GiB/GPU at {ctx} ctx)"
                ),
                detail="May be OK but leaves little margin for batch concurrency.",
                duration_ms=ms,
            )

        return PreflightCheck(
            name="capacity",
            status="pass",
            message=(
                f"KV estimate: ~{kv_per_gpu_gib:.1f} GiB/GPU ({ctx} ctx, {tp} TP), "
                f"~{headroom_gib:.0f} GiB/GPU headroom "
                f"({mem_per_gpu_gb:.0f} GiB × {gpu_mem_util} util)"
            ),
            duration_ms=ms,
        )


def _sh_quote(value: str) -> str:
    """Single-quote a string for safe use in shell commands."""
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"
