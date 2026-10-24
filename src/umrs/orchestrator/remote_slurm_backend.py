"""
Remote Slurm Backend — SSH-based Slurm orchestration for front-end deployments.

Extends the standard SlurmBackend to route all Slurm CLI commands (sbatch,
squeue, sacct, scancel) through an SSH ControlMaster connection to the login
node. Used when UMRS runs on a front-end server that lacks direct Slurm access.

Configuration in site.yaml::

    scheduler:
      kind: slurm-remote
      partition: ng-dgx-m0,ng-dgx-m1,ng-dgx-m2,ng-dgx-m3
      time_limit: "04:00:00"
      extra_sbatch: []

    extra:
      ssh_target: rccs-cloud
      ssh_user: ""
      ssh_control_path: ""
      ssh_connect_timeout: 15
      ssh_command_timeout: 30
      ssh_reverse_tunnels:
        - "127.0.0.1:9000:127.0.0.1:9000"
      slurm_job_info_dir: /lvs0/rccs-dmp/masaru.nagaso/ai4s-platform/umrs/deploy/rccs
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from datetime import datetime
from typing import Any

from . import partition_choice as pc
from .base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
)
from .slurm_backend import SlurmBackend
from .ssh_control import SSHControlConfig, SSHControlManager
from .tunnel_pool import TunnelPool, TunnelPoolConfig

logger = logging.getLogger(__name__)

# Memory units understood in scheduler.memory (case-insensitive suffix → MB).
_MEM_UNIT_TO_MB = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}

# Serve-wrapper job names the orphan listing matches: the in-tree job script
# (mini-rivault) and the per-cluster shims deploy/<cluster>/serve-model.slurm
# (umrs-rivault). squeue -n takes a comma list.
_WRAPPER_JOB_NAMES = "mini-rivault,umrs-rivault"

# squeue answered, and the job is not in the answer.
_SQUEUE_NOT_LISTED = "not listed by squeue"
# squeue does not list the job and sacct errors (live bdrt 2026-09-27:
# AccountingStorageHost=localhost, so every sacct fails). The controller no
# longer knows the job; how it ended is unknowable.
_SLURM_VANISHED = "slurm-vanished"
_SLURM_VANISHED_ERROR = "Slurm no longer lists the job and sacct is unavailable"

# Per-candidate bound on one `sbatch --test-only` probe; a timeout counts as
# a failed candidate. At most _TEST_ONLY_CONCURRENCY probes run at once (they
# share the ControlMaster with squeue polls; sshd MaxSessions defaults to
# 10), so five candidates wait at most two rounds of this bound.
_TEST_ONLY_TIMEOUT_SEC = 30
_TEST_ONLY_CONCURRENCY = 3


def _squeue_omits_job(result: Any) -> bool:
    """True when squeue answered and the job is not in the answer.

    Exit 0 with no row, or Slurm's own "Invalid job id specified" once the
    controller purged the job (MinJobAge). Any other failure (SSH, a
    controller timeout) is no answer at all.
    """
    if result.ok:
        return not result.stdout.strip()
    return "invalid job id" in (result.stderr or "").lower()


def _parse_mem_to_mb(value: str) -> float | None:
    """Parse a Slurm-style memory string (e.g. ``1400G``, ``400000M``) to MB.

    Returns ``None`` when the value cannot be parsed.
    """
    text = str(value).strip()
    if not text:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KkMmGgTt]?)[Bb]?", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "M").upper()
    return number * _MEM_UNIT_TO_MB.get(unit, 1.0)


def _scale_memory_for_gpus(
    memory: str, effective_gpus: int | None, max_node_gpus: int | None
) -> str:
    """Scale a full-node memory request down to a partial-GPU allocation.

    ``scheduler.memory`` is sized for a full node (all GPUs). Sub-node Slurm
    partitions (e.g. AI4S ``1n1gpu``) enforce a ``MaxMemPerNode`` proportional
    to the GPU share, so a full-node ``--mem`` trips ``MaxMemPerLimit`` and the
    job never starts. When fewer GPUs than a full node are requested, scale the
    memory by the GPU fraction. Full-node and multi-node-per-node requests are
    returned unchanged.
    """
    if not effective_gpus or not max_node_gpus or effective_gpus >= max_node_gpus:
        return memory
    total_mb = _parse_mem_to_mb(memory)
    if total_mb is None:
        return memory
    scaled_mb = int(total_mb * effective_gpus / max_node_gpus)
    if scaled_mb <= 0:
        return memory
    return f"{scaled_mb}M"


def _resolve_sbatch_mem_arg(site: Any, effective_gpus: int | None) -> str:
    """Return the ``--mem=`` sbatch argument for a wrapper job.

    Uses ``--mem=0`` (suppress DefMemPerCPU) when no memory is configured;
    otherwise scales the configured full-node memory by the requested GPU
    fraction (see :func:`_scale_memory_for_gpus`).
    """
    if not site.scheduler.memory:
        return "--mem=0"
    max_node_gpus = max(
        (p.gpus for p in site.partition_profiles.values()),
        default=0,
    )
    mem = _scale_memory_for_gpus(site.scheduler.memory, effective_gpus, max_node_gpus)
    return f"--mem={mem}"


class RemoteSlurmBackend(SlurmBackend):
    """
    SSH-based remote Slurm backend for front-end server deployments.

    All Slurm CLI commands are executed remotely via SSH ControlMaster.
    Job info files on the shared filesystem are read via SSH cat.

    When ``slurm_delegation`` is ``per_user_ssh``, job submission and cancellation
    are executed as the requesting user via a per-user SSH connection using the
    UMRS service identity key. The user must have enrolled the service public key
    in their ~/.ssh/authorized_keys (via ``r4 onboard``).
    """

    def __init__(self, config: Any, ssh_config: SSHControlConfig | None = None) -> None:
        super().__init__(config)
        # Build SSH config from site config extra section or explicit param.
        if ssh_config:
            self._ssh_config = ssh_config
        else:
            extra = getattr(config, "extra", {}) or {}
            if not extra.get("ssh_target"):
                # Fallback: load extra from site config directly
                try:
                    from .site_config import load_site_config

                    site = load_site_config()
                    extra = site.extra or {}
                except Exception:
                    pass
            self._ssh_config = SSHControlConfig.from_site_config(extra)

        self._ssh: SSHControlManager | None = None
        self._tunnel_pool: TunnelPool | None = None

        # Per-user SSH delegation state
        self._delegation_mode: str = "service_account"
        self._service_identity_file: str = ""
        self._user_ssh_pool: dict[str, SSHControlManager] = {}
        try:
            from .site_config import load_site_config

            site = load_site_config()
            self._delegation_mode = site.scheduler.slurm_delegation
            # Derive private key path from pubkey path (strip .pub suffix)
            pubkey_path = site.scheduler.service_pubkey_path
            if pubkey_path.endswith(".pub"):
                self._service_identity_file = pubkey_path[:-4]
        except Exception:
            pass

    @property
    def backend_name(self) -> str:
        return "slurm-remote"

    async def _get_ssh(self) -> SSHControlManager:
        """Get or create the SSH control manager."""
        if self._ssh is None:
            self._ssh = SSHControlManager(self._ssh_config)
        await self._ssh.ensure_connected()
        return self._ssh

    async def _get_ssh_for_user(self, username: str) -> SSHControlManager:
        """Get or create a per-user SSH control manager for delegated submission.

        Uses the UMRS service identity key to connect as the target user.
        The user must have enrolled the service pubkey in their authorized_keys.
        Falls back to the service-account connection if delegation is not
        configured **or** if the per-user connection fails (user has not
        enrolled yet).
        """
        if self._delegation_mode != "per_user_ssh" or not self._service_identity_file:
            return await self._get_ssh()

        if username not in self._user_ssh_pool:
            # Build a user-specific SSH config based on service config
            user_config = SSHControlConfig(
                ssh_target=self._ssh_config.ssh_target,
                ssh_user=username,
                ssh_port=self._ssh_config.ssh_port,
                connect_timeout=self._ssh_config.connect_timeout,
                command_timeout=self._ssh_config.command_timeout,
                control_path="",  # Auto-generated per-user socket
                server_alive_interval=self._ssh_config.server_alive_interval,
                server_alive_count_max=self._ssh_config.server_alive_count_max,
                reverse_tunnels=[],  # No tunnels for per-user connections
                proxy_jump=self._ssh_config.proxy_jump,
                identity_file=self._service_identity_file,
                remote_env=self._ssh_config.remote_env,
            )
            self._user_ssh_pool[username] = SSHControlManager(user_config)

        mgr = self._user_ssh_pool[username]
        try:
            await mgr.ensure_connected()
        except Exception:
            # Per-user SSH failed (user likely hasn't enrolled their key).
            # Fall back to service account so the job can still proceed
            # under the site master Slurm account.
            logger.warning(
                "Per-user SSH connection failed for %s; falling back to "
                "service account.  The user may not have enrolled the UMRS "
                "service key (r4 auth login / r4 onboard).",
                username,
            )
            return await self._get_ssh()
        return mgr

    def _get_wrapper_script(self, env: dict | None = None) -> str | None:
        """Get the remote wrapper script path.

        Checks (in order):
        1. env["SLURM_WRAPPER_SCRIPT"] — per-job override from pool env_vars
        2. site.extra["slurm_wrapper_script"] — site-wide default
        """
        # Per-job override via env_vars (e.g. from pool YAML).
        if env and env.get("SLURM_WRAPPER_SCRIPT"):
            return str(env["SLURM_WRAPPER_SCRIPT"])
        extra = getattr(self.config, "extra", {}) or {}
        if not extra.get("slurm_wrapper_script"):
            try:
                from .site_config import load_site_config

                site = load_site_config()
                extra = site.extra or {}
            except Exception:
                pass
        return extra.get("slurm_wrapper_script")

    def _job_info_dir(self) -> str:
        """Return the remote shared directory for Slurm job info files."""
        info_dir = getattr(self.config, "slurm_job_info_dir", "/tmp")
        extra = getattr(self.config, "extra", {}) or {}
        if extra.get("slurm_job_info_dir"):
            return str(extra["slurm_job_info_dir"])
        try:
            from .site_config import load_site_config

            site = load_site_config()
            return str(site.extra.get("slurm_job_info_dir", info_dir))
        except Exception:
            return str(info_dir)

    def _control_dir(self) -> str:
        """Return the remote shared directory for graceful stop requests."""
        return f"{self._job_info_dir().rstrip('/')}/control"

    def _stop_request_path(self, job_id: str) -> str:
        """Return the remote shared stop request path for a Slurm job."""
        return f"{self._control_dir()}/stop-{job_id}.request"

    async def submit_job(
        self,
        model_path: str | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Submit a Slurm job via remote SSH sbatch.

        When per-user delegation is active and ``requesting_user`` is provided,
        the sbatch is executed as that user via the UMRS service identity key.
        """
        # Wrapper jobs (loop jobs, vendor wrappers) carry their own runtime —
        # a model requirement is only valid for the legacy direct-submit path.
        requesting_user = kwargs.pop("requesting_user", None)
        _peek_env = kwargs.get("env") or {}
        wrapper_script = self._get_wrapper_script(env=_peek_env)

        model = model_path or self.config.model_path
        if not model and not wrapper_script:
            raise ValueError(
                "model_path is required for direct-submit jobs (no wrapper declares one)"
            )

        if wrapper_script:
            return await self._submit_wrapper_job(
                model or "", wrapper_script, requesting_user=requesting_user, **kwargs
            )

        # Legacy: generate Singularity-based sbatch script and pipe via stdin.
        script_content = self._build_sbatch_script(model, **kwargs)

        if requesting_user and self._delegation_mode == "per_user_ssh":
            ssh = await self._get_ssh_for_user(requesting_user)
        else:
            ssh = await self._get_ssh()
        result = await ssh.run_remote_with_stdin(
            "sbatch --parsable",
            script_content.encode(),
            timeout=60,
        )

        if not result.ok:
            raise RuntimeError(f"Remote sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0]

        return JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_path=model,
            created_at=datetime.now(),
            backend=self.backend_name,
        )

    @staticmethod
    def _select_partition(
        site: Any, gpu_count: int | None, node_count: int | None
    ) -> str:
        """Pick a Slurm partition from the requested GPU/node sizing.

        AI4S declares partitions following a ``<nodes>n<gpus>gpu`` naming
        convention (e.g. ``1n4gpu``, ``2n4gpu``). When such a profile exists for
        the requested sizing, route to it so multi-GPU/multi-node jobs land on a
        partition that can satisfy the allocation. Sites that do not follow this
        convention (e.g. rccs-cloud) fall through to the configured default:
        the whole candidate list on a site that chooses per submit
        (``scheduler.partition_choice: earliest_start``), else the first.
        """
        default = pc.default_partition(site)
        if not gpu_count:
            return default
        nodes = node_count if node_count and node_count > 1 else 1
        candidate = f"{nodes}n{gpu_count}gpu"
        if candidate in getattr(site, "partition_profiles", {}):
            return candidate
        return default

    async def _submit_wrapper_job(
        self,
        model_path: str,
        wrapper_script: str,
        requesting_user: str | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Submit a site-provided wrapper script (e.g. serve-model.slurm) via remote sbatch."""
        # Build sbatch CLI args from site config.
        from .site_config import load_site_config

        site = load_site_config()

        output_dir = self._job_info_dir()

        # Resource sizing (popped up-front so partition selection can use them).
        gpu_count = kwargs.pop("gpu_count", None)
        node_count = kwargs.pop("node_count", None)
        candidates = self._partition_candidates(
            site, kwargs.pop("partition_override", None), gpu_count, node_count
        )
        slurm_account = kwargs.pop("slurm_account", None) or site.scheduler.account
        nodelist_hint = kwargs.pop("nodelist_hint", None)

        sbatch_args = self._build_wrapper_sbatch_args(
            site=site,
            output_dir=output_dir,
            gpu_count=gpu_count,
            node_count=node_count,
            partition_override=",".join(candidates) or None,
            slurm_account=slurm_account,
            nodelist_hint=nodelist_hint,
        )

        # Pop unused kwargs before building env
        kwargs.pop("min_gpu_memory_gb", None)

        if requesting_user and self._delegation_mode == "per_user_ssh":
            ssh = await self._get_ssh_for_user(requesting_user)
        else:
            ssh = await self._get_ssh()
        # One partition per submission (RCCS refuses a list); the probes run
        # through the same ssh as the real submit.
        ranked = await self._rank_partitions(
            ssh, [*sbatch_args, shlex.quote(wrapper_script)], candidates
        )

        # Build environment exports + sacct attribution for the job. The cap
        # is the strictest over every partition the submit may still land on.
        requested_env = _cap_gpu_mem_util_env(
            site, ",".join(ranked), kwargs.pop("env", None)
        )
        project_id = kwargs.pop("project_id", None)
        if isinstance(requested_env, dict) and not project_id:
            project_id = requested_env.get("PROJECT_ID", "")
        sbatch_args.extend(
            self._build_wrapper_job_exports(
                model_path=model_path,
                wrapper_script=wrapper_script,
                output_dir=output_dir,
                requesting_user=requesting_user,
                project_id=project_id,
                requested_env=requested_env,
            )
        )

        # Append the wrapper script path
        sbatch_args.append(shlex.quote(wrapper_script))

        result = await self._sbatch_ranked(ssh, sbatch_args, ranked)

        if not result.ok:
            raise RuntimeError(f"Remote sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0]
        logger.info("Submitted wrapper job %s", job_id)

        return JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_path=model_path,
            created_at=datetime.now(),
            backend=self.backend_name,
        )

    def _partition_candidates(
        self,
        site: Any,
        partition_override: str | None,
        gpu_count: int | None,
        node_count: int | None,
    ) -> list[str]:
        """The partitions this submit may use, in configured order.

        A list collapses to its first candidate unless the site chooses per
        submit (``scheduler.partition_choice: earliest_start``).
        """
        raw = partition_override or self._select_partition(site, gpu_count, node_count)
        candidates = pc.split_partitions(raw)
        if len(candidates) > 1 and not pc.spreads_partitions(site):
            return candidates[:1]
        return candidates

    async def _rank_partitions(
        self, ssh: Any, sbatch_args: list[str], candidates: list[str]
    ) -> list[str]:
        """Candidates ranked by `sbatch --test-only` start, earliest first.

        A single candidate (or none) is returned untouched, without a probe.
        Candidates whose probe fails (quota, invalid, unavailable, timeout)
        are dropped; ties (see ``partition_choice``) go to the partition this
        process picked least recently. When none passes (e.g. every probe
        timed out), the whole list is kept in configured order so the real
        submit still moves on after a quota refusal.
        """
        if len(candidates) <= 1:
            return candidates
        gate = asyncio.Semaphore(_TEST_ONLY_CONCURRENCY)

        async def _probe(cand: str) -> datetime | None:
            async with gate:
                return await self._test_only_start(ssh, sbatch_args, cand)

        starts = await asyncio.gather(*(_probe(cand) for cand in candidates))
        by_name = dict(zip(candidates, starts, strict=True))
        ranked = pc.choose(candidates, by_name)
        if not ranked:
            logger.info(
                "No Slurm partition of %s passed --test-only; submitting in "
                "configured order",
                ",".join(candidates),
            )
            return list(candidates)
        logger.info(
            "Selected Slurm partition %s from %s (earliest start %s)",
            ranked[0],
            ",".join(candidates),
            by_name[ranked[0]].isoformat(),  # type: ignore[union-attr]
        )
        return ranked

    async def _test_only_start(
        self, ssh: Any, sbatch_args: list[str], partition: str
    ) -> datetime | None:
        """When Slurm would start this job on ``partition``; None if refused."""
        cmd = " ".join(pc.probe_args(sbatch_args, partition))
        try:
            result = await ssh.run_remote(cmd, timeout=_TEST_ONLY_TIMEOUT_SEC)
        except (asyncio.TimeoutError, OSError) as exc:
            logger.info("Partition %s skipped: --test-only failed: %s", partition, exc)
            return None
        output = f"{result.stdout}\n{result.stderr}"
        parsed = pc.parse_test_only(output) if result.ok else None
        if parsed is None:
            logger.info(
                "Partition %s skipped: %s",
                partition,
                pc.first_line(result.stderr) or pc.first_line(result.stdout),
            )
            return None
        return parsed[0]

    async def _sbatch_ranked(
        self, ssh: Any, sbatch_args: list[str], ranked: list[str]
    ) -> Any:
        """Submit to the first ranked partition; on a quota refusal, the next.

        Any other failure (and the quota refusal of the last candidate) is
        returned to the caller unchanged, as a single submit was before.
        """
        result = None
        for index, partition in enumerate(ranked or [""]):
            args = (
                pc.with_partition(sbatch_args, partition) if partition else sbatch_args
            )
            logger.info("Submitting wrapper job: %s", pc.redact_exports(args))
            result = await ssh.run_remote(" ".join(args), timeout=60)
            last = index >= len(ranked) - 1
            if result.ok or last or not pc.quota_exceeded_partition(result.stderr):
                return result
            logger.info(
                "Partition %s refused by quota (%s); trying %s",
                partition,
                pc.first_line(result.stderr),
                ranked[index + 1],
            )
        return result

    def _build_wrapper_sbatch_args(
        self,
        *,
        site: Any,
        output_dir: str,
        gpu_count: int | None,
        node_count: int | None,
        partition_override: str | None,
        slurm_account: str | None,
        nodelist_hint: str | None,
    ) -> list[str]:
        """Assemble the resource-directive portion of a wrapper-job sbatch CLI.

        Returns the leading ``sbatch`` argument list (output/error paths plus
        partition/time/GPU/CPU/memory/account/qos/nodelist/nodes/extras). The
        caller appends ``--comment``/``--export`` attribution and the wrapper
        script path. ``scheduler.memory`` is full-node sized; it is scaled by
        the requested GPU fraction so sub-node partitions (e.g. AI4S ``1n1gpu``)
        do not trip ``MaxMemPerLimit``.
        """
        sched = site.scheduler
        # Explicit GPU request wins; otherwise the site default.
        effective_gpus = gpu_count if gpu_count is not None else sched.gpus
        # Partition auto-selection honours the AI4S "<n>n<g>gpu" convention.
        partition = partition_override or self._select_partition(
            site, gpu_count, node_count
        )
        nodes = node_count if (node_count or 0) > 1 else None

        sbatch_args = [
            "sbatch",
            "--parsable",
            f"--output={output_dir}/slurm-%j.out",
            f"--error={output_dir}/slurm-%j.err",
            _resolve_sbatch_mem_arg(site, effective_gpus),
        ]
        # Optional scalar directives appended only when truthy.
        optional = [
            ("--partition=", partition),
            ("--time=", sched.time_limit),
            ("--time-min=", getattr(sched, "time_min", "")),
            ("--gres=gpu:", effective_gpus if sched.gpus != 0 else None),
            ("--cpus-per-task=", sched.cpus),
            ("--account=", slurm_account),
            ("--qos=", sched.qos),
            ("--nodelist=", nodelist_hint),
            ("--nodes=", nodes),
        ]
        for flag, value in optional:
            if value:
                sbatch_args.append(f"{flag}{value}")
        sbatch_args.extend(getattr(sched, "extra_sbatch", []))
        return sbatch_args

    def _build_wrapper_job_exports(
        self,
        *,
        model_path: str,
        wrapper_script: str,
        output_dir: str,
        requesting_user: str | None,
        project_id: str | None,
        requested_env: Any,
    ) -> list[str]:
        """Build the ``--comment`` and ``--export`` sbatch args for a wrapper job.

        The ``--export`` arg is shell-quoted because values may contain spaces
        (e.g. ``RIVAULT_PROFILES="harness vllm"``); otherwise the remote shell
        word-splits the comma list and sbatch treats the tail as a stray file.

        Slurm ``--export`` is a **comma-delimited** ``KEY=VALUE`` list with no
        escaping mechanism, so any value that itself contains a comma (e.g. the
        ``VLLM_SPECULATIVE_CONFIG`` JSON ``{"method":"dflash",...}``) is
        silently truncated at its first comma on the compute node. Encoding is
        now a hex round-trip (``<KEY>_HEX``) — characters from the hexadecimal
        alphabet only (``0-9a-f``), so the line passes the per-user SSH proxy's
        charset whitelist, which rejected earlier comma-safe base64 ('+') and
        any shell quoting ('\\'', '"'). The serve wrapper decodes every
        ``*_HEX`` variable back to its original name, and keeps decoding the
        legacy ``*_B64`` lane for backward compatibility.
        """
        env_overrides: dict[str, str] = {}
        if isinstance(requested_env, dict):
            env_overrides.update(
                {
                    k: str(v)
                    for k, v in requested_env.items()
                    if isinstance(v, (str, int, float))
                }
            )
        env_overrides["VLLM_MODEL_PATH"] = model_path
        env_overrides.setdefault(
            "VLLM_SERVED_MODEL_NAME", env_overrides.get("MODEL_NAME", model_path)
        )
        env_overrides.setdefault("RIVAULT_WRAPPER_DIR", os.path.dirname(wrapper_script))
        env_overrides.setdefault("JOB_INFO_DIR", output_dir)
        env_overrides.setdefault("RIVAULT_CONTROL_DIR", self._control_dir())

        # Attribution: export the requesting user/project identity for Slurm
        # accounting (UMRS_* env vars) and sacct (--comment), so containers can
        # log the originating user even when submitted via the master account.
        args: list[str] = []
        attribution_parts = []
        if requesting_user:
            env_overrides.setdefault("UMRS_USER", requesting_user)
            attribution_parts.append(f"user={requesting_user}")
        if project_id:
            env_overrides.setdefault("UMRS_PROJECT", project_id)
            attribution_parts.append(f"project={project_id}")
        if attribution_parts:
            args.append(f"--comment={','.join(attribution_parts)}")

        args.append(f"--export={','.join(_hex_export_parts(env_overrides))}")
        return args

    async def get_job_status(self, job_id: str) -> JobInfo:
        """Get status of a Slurm job via SSH."""
        return await self._job_status(job_id, before_ready=False)

    async def get_job_status_before_ready(self, job_id: str) -> JobInfo:
        """Get status for an endpoint that is not READY yet.

        Unlike ``get_job_status``, a job squeue does not list while sacct
        errors is reported FAILED: before READY no healthy endpoint can be
        lost, and on a cluster without working accounting it is the only
        trace an externally cancelled job leaves.
        """
        return await self._job_status(job_id, before_ready=True)

    async def _job_status(self, job_id: str, *, before_ready: bool) -> JobInfo:
        # Try job info file first (read remotely).
        info_content = await self._read_remote_job_info(job_id)
        if info_content is not None:
            info = self._parse_info_content(info_content)
            # ADR-048 D-3: budget_exhausted is a terminal marker in the wrapper's
            # info drop (job lineage respawn budget gone) — surface it verbatim
            # through state_reason BEFORE the ghost-file sweep, otherwise a
            # wrapper exit=125 + stale RIVAULT_STATUS=running would ghost-map to
            # generic terminal without the marker name, and _remove_stale_info_file
            # would delete the only evidence. Marker stays byte-identical to the
            # wrapper's write shape ('budget_exhausted').
            if (
                str(info.get("RIVAULT_TERMINAL_REASON", "")).strip()
                == "budget_exhausted"
            ):
                return JobInfo(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    backend=self.backend_name,
                    metadata=info,
                    state_reason="budget_exhausted",
                )
            status = self._status_from_info(info)
            terminal = await self._slurm_terminal_override(
                job_id, status, before_ready=before_ready
            )
            if terminal is not None:
                return terminal

            endpoint = self._endpoint_from_info(status, info)
            if endpoint is not None:
                endpoint = await self._endpoint_via_tunnel(endpoint)
            return JobInfo(
                job_id=job_id,
                status=status,
                endpoint=endpoint,
                model_path=info.get("RIVAULT_MODEL_PATH", ""),
                backend=self.backend_name,
                metadata=info,
            )

        # Fallback to squeue/sacct.
        status, reason = await self._query_slurm_job_status(job_id)
        if reason == _SLURM_VANISHED:
            if before_ready:
                return self._vanished_job_info(job_id, status)
            reason = None  # an UNKNOWN miss, as for any other caller
        return JobInfo(
            job_id=job_id,
            status=status,
            backend=self.backend_name,
            state_reason=reason,
        )

    async def _slurm_terminal_override(
        self, job_id: str, status: JobStatus, *, before_ready: bool
    ) -> JobInfo | None:
        """Terminal JobInfo when Slurm contradicts a live info-file status.

        The info file is a startup artifact written to the shared
        filesystem; it is only deleted by ``cleanup()``. If the job died by
        walltime, preemption, node failure, or an external ``scancel``, the
        file is orphaned and still reports the last written status
        (``running`` or ``starting``). Trusting it verbatim produces a ghost
        endpoint. Cross-check live states against Slurm so a dead job is
        reported as terminal. Only override on a *positively* terminal Slurm
        state — a squeue miss / ``UNKNOWN`` must not delete a healthy
        endpoint. Before READY there is none to delete, so there (only) a
        job squeue does not list while sacct errors counts as ended.
        """
        if status not in (JobStatus.READY, JobStatus.RUNNING, JobStatus.STARTING):
            return None
        slurm_status, reason = await self._query_slurm_job_status(job_id)
        if slurm_status in (JobStatus.FAILED, JobStatus.STOPPED):
            logger.warning(
                "Job %s info file reports %s but Slurm reports %s; "
                "treating as terminal and removing stale info file",
                job_id,
                status.value,
                slurm_status.value,
            )
            await self._remove_stale_info_file(job_id)
            return JobInfo(
                job_id=job_id,
                status=slurm_status,
                backend=self.backend_name,
            )
        if before_ready and reason == _SLURM_VANISHED:
            # The info file stays: cleanup() reads it to close the tunnel.
            return self._vanished_job_info(job_id, status)
        return None

    def _vanished_job_info(self, job_id: str, last_status: JobStatus) -> JobInfo:
        logger.warning(
            "Job %s (last %s) is not listed by squeue and sacct is unavailable; "
            "treating it as ended (endpoint not READY yet)",
            job_id,
            last_status.value,
        )
        return JobInfo(
            job_id=job_id,
            status=JobStatus.FAILED,
            backend=self.backend_name,
            error_message=_SLURM_VANISHED_ERROR,
        )

    async def _remove_stale_info_file(self, job_id: str) -> None:
        """Best-effort removal of an orphaned job info file via SSH."""
        try:
            ssh = await self._get_ssh()
            info_path = f"{self._job_info_dir()}/rivault-job-{job_id}.info"
            await ssh.run_remote(f"rm -f {shlex.quote(info_path)}")
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logger.warning(
                "Failed to remove stale info file for job %s: %s", job_id, exc
            )

    async def _endpoint_via_tunnel(self, endpoint: EndpointInfo) -> EndpointInfo:
        """Return a front-end local endpoint backed by an SSH forward tunnel."""
        local_port = await self._get_tunnel_pool().open_tunnel(
            endpoint.node,
            endpoint.litellm_port,
            local_port=endpoint.litellm_port,
        )
        return EndpointInfo(
            node="127.0.0.1",
            litellm_port=local_port,
            vllm_port=endpoint.vllm_port,
        )

    def _get_tunnel_pool(self) -> TunnelPool:
        if self._tunnel_pool is None:
            local_port_min = 12000
            local_port_max = 12999
            try:
                from .site_config import load_site_config

                site = load_site_config()
                local_port_min = site.network.port_range_min
                local_port_max = site.network.port_range_max
            except Exception:
                pass

            self._tunnel_pool = TunnelPool(
                TunnelPoolConfig(
                    ssh_target=self._ssh_config.ssh_target,
                    ssh_user=self._ssh_config.ssh_user,
                    ssh_port=self._ssh_config.ssh_port,
                    local_port_min=local_port_min,
                    local_port_max=local_port_max,
                    control_path=self._ssh_config.control_path,
                )
            )
        return self._tunnel_pool

    async def _read_remote_job_info(self, job_id: str) -> str | None:
        """Read job info file from remote filesystem via SSH."""
        info_dir = self._job_info_dir()
        info_path = f"{info_dir}/rivault-job-{job_id}.info"
        ssh = await self._get_ssh()
        return await ssh.read_remote_file(info_path)

    def _parse_info_content(self, content: str) -> dict[str, str]:
        """Parse job info file content into a dict."""
        info: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
        return info

    async def _query_slurm_job_status(
        self, job_id: str
    ) -> tuple[JobStatus, str | None]:
        """Query Slurm job status via remote squeue/sacct.

        Returns ``(status, reason)`` where ``reason`` is the scheduler's
        human-readable explanation for the current state (squeue ``%R`` — e.g.
        ``MaxMemPerLimit``, ``Resources``, ``Priority``) when the job is still
        queued, or ``None`` otherwise. A job squeue answers it does not list
        while sacct errors comes back as ``(UNKNOWN, _SLURM_VANISHED)``; the
        caller decides whether that is terminal.
        """
        squeue_result = await self._query_squeue_status(job_id)
        if squeue_result is not None and squeue_result[1] != _SQUEUE_NOT_LISTED:
            return squeue_result
        sacct_status = await self._query_sacct_status(job_id)
        if sacct_status is not None:
            return sacct_status, None
        return JobStatus.UNKNOWN, _SLURM_VANISHED if squeue_result else None

    async def _query_squeue_status(
        self, job_id: str
    ) -> tuple[JobStatus, str | None] | None:
        """Query squeue for active job status + pending reason via SSH.

        ``None`` when squeue could not answer; ``(UNKNOWN, _SQUEUE_NOT_LISTED)``
        when it answered without the job.
        """
        ssh = await self._get_ssh()
        # %T = job state, %R = reason (for PENDING) or node list (for RUNNING).
        result = await ssh.run_remote(f'squeue -j {job_id} -h -o "%T|%R"')

        if _squeue_omits_job(result):
            return JobStatus.UNKNOWN, _SQUEUE_NOT_LISTED
        if not result.ok:
            return None
        raw = result.stdout.strip().splitlines()[0]
        state, _, reason = raw.partition("|")
        state = state.strip()
        reason = reason.strip() or None
        status = self._status_from_slurm_state(state)
        # Only surface the reason while the job is still queued/pending. For a
        # RUNNING job, %R holds the node list, not a diagnostic.
        if status != JobStatus.PENDING:
            reason = None
        return status, reason

    async def time_left_sec(self, job_id: str) -> int | None:
        """Seconds until Slurm ends a running job (``squeue %L``), else ``None``.

        A job started with a shortened limit (``--time-min``, e.g. to finish
        before a reservation) ends earlier than ``created_at + time_limit``;
        the walltime relay hands it over by this answer instead. ``%L`` is a
        duration, so the cluster's time zone never enters.
        """
        from .endpoint_relay import parse_slurm_duration

        if not re.fullmatch(r"\d+", str(job_id)):
            return None
        ssh = await self._get_ssh()
        result = await ssh.run_remote(f'squeue -j {job_id} -h -t RUNNING -o "%L"')
        if not result.ok or not result.stdout.strip():
            return None
        return parse_slurm_duration(result.stdout.strip().splitlines()[0])

    async def _query_sacct_status(self, job_id: str) -> JobStatus | None:
        """Query sacct for completed job status via SSH (``None`` if sacct errors)."""
        ssh = await self._get_ssh()
        result = await ssh.run_remote(f"sacct -j {job_id} -n -o State")
        if not result.ok:
            return None

        state = result.stdout.strip().split("\n")[0].strip() if result.stdout else ""
        return self._status_from_accounting_state(state)

    async def cleanup(self, job_id: str, force: bool = False) -> None:
        """Cancel a Slurm job via remote scancel."""
        from .readiness import cancel_readiness

        cancel_readiness(job_id)
        info_content = await self._read_remote_job_info(job_id)
        if info_content is not None and self._tunnel_pool is not None:
            info = self._parse_info_content(info_content)
            endpoint = self._endpoint_from_info(JobStatus.RUNNING, info)
            if endpoint is not None:
                await self._tunnel_pool.close_tunnel(
                    endpoint.node, endpoint.litellm_port
                )

        ssh = await self._get_ssh()
        info_dir = self._job_info_dir()
        info_path = f"{info_dir}/rivault-job-{job_id}.info"

        if not force:
            stop_request = self._stop_request_path(job_id)
            control_dir = self._control_dir()
            result = await ssh.run_remote(
                f"mkdir -p {shlex.quote(control_dir)} && touch {shlex.quote(stop_request)}"
            )
            if result.ok:
                graceful_timeout = int(
                    os.environ.get("RIVAULT_GRACEFUL_STOP_TIMEOUT", "20")
                )
                logger.info(
                    "requested graceful remote compute-node stop for %s via %s",
                    job_id,
                    stop_request,
                )
                if await self._wait_for_remote_job_exit(job_id, graceful_timeout):
                    await ssh.run_remote(f"rm -f {shlex.quote(info_path)}")
                    return
            else:
                logger.warning(
                    "failed to create remote graceful stop request for %s: %s",
                    job_id,
                    result.stderr.strip(),
                )

        cmd = f"scancel -Q {shlex.quote(job_id)}"
        if force:
            cmd = f"scancel -Q --full --signal=KILL {shlex.quote(job_id)}"
        result = await ssh.run_remote(cmd)
        if not result.ok:
            logger.warning(
                "remote scancel failed for %s (force=%s): %s",
                job_id,
                force,
                result.stderr.strip(),
            )

        # Clean up remote info file.
        await ssh.run_remote(f"rm -f {shlex.quote(info_path)}")

    async def _wait_for_remote_job_exit(self, job_id: str, timeout_sec: int) -> bool:
        """Wait briefly for a remote Slurm job to leave the active queue."""
        for _ in range(max(timeout_sec, 0)):
            status, _reason = await self._query_slurm_job_status(job_id)
            if status in (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.UNKNOWN):
                return True
            await asyncio.sleep(1)
        return False

    async def list_jobs(
        self,
        status_filter: list[JobStatus] | None = None,
    ) -> list[JobInfo]:
        """List all Mini-RiVault jobs via SSH."""
        jobs = await self._remote_jobs_from_info_files(status_filter)
        await self._remote_append_squeue_jobs(jobs, status_filter)
        return jobs

    async def _remote_jobs_from_info_files(
        self,
        status_filter: list[JobStatus] | None,
    ) -> list[JobInfo]:
        """List job info files on remote filesystem."""
        info_dir = self._job_info_dir()
        ssh = await self._get_ssh()
        result = await ssh.run_remote(
            f"ls {info_dir}/rivault-job-*.info 2>/dev/null || true"
        )

        jobs: list[JobInfo] = []
        if not result.stdout.strip():
            return jobs

        for line in result.stdout.strip().splitlines():
            match = re.search(r"rivault-job-(\d+)\.info", line)
            if match:
                job_id = match.group(1)
                job = await self.get_job_status(job_id)
                if status_filter is None or job.status in status_filter:
                    jobs.append(job)
        return jobs

    async def _remote_append_squeue_jobs(
        self,
        jobs: list[JobInfo],
        status_filter: list[JobStatus] | None,
    ) -> None:
        """Query squeue for running jobs via SSH."""
        user = os.environ.get("USER", "")
        ssh = await self._get_ssh()
        result = await ssh.run_remote(
            f"squeue -u {user} -n {_WRAPPER_JOB_NAMES} -h -o %i"
        )

        if not result.stdout.strip():
            return

        for job_id in result.stdout.strip().split("\n"):
            job_id = job_id.strip()
            # Job ids are strictly numeric (array ids contain '_'). The remote
            # squeue flavor that does not know the '%i' format prints its full
            # usage help — historically those words flowed downstream as
            # "stop-Usage: squeue [-A account]..." pickle files in the control
            # dir. Filter defensively at the choke point (P3 hygiene fix).
            if not re.fullmatch(r"\d+(?:_.+)?", job_id):
                continue
            if any(j.job_id == job_id for j in jobs):
                continue
            job = await self.get_job_status(job_id)
            if status_filter is None or job.status in status_filter:
                jobs.append(job)

    async def cleanup_orphans(self) -> int:
        """Clean up orphaned job info files via SSH."""
        info_dir = self._job_info_dir()
        ssh = await self._get_ssh()

        result = await ssh.run_remote(
            f"ls {info_dir}/rivault-job-*.info 2>/dev/null || true"
        )
        if not result.stdout.strip():
            return 0

        cleaned = 0
        for line in result.stdout.strip().splitlines():
            match = re.search(r"rivault-job-(\d+)\.info", line)
            if not match:
                continue
            job_id = match.group(1)

            # Check if job still exists in Slurm.
            check = await ssh.run_remote(f"squeue -j {job_id} -h")
            if not check.stdout.strip():
                await ssh.run_remote(f"rm -f {info_dir}/rivault-job-{job_id}.info")
                cleaned += 1

        return cleaned

    async def health_check(self) -> dict[str, Any]:
        """Check if Slurm is accessible via SSH."""
        try:
            ssh = await self._get_ssh()
            partition = self.config.slurm_partition
            result = await ssh.run_remote(f"sinfo -h -p {partition} -o '%P %a %D'")

            if not result.ok:
                return {
                    "backend": self.backend_name,
                    "healthy": False,
                    "message": f"Remote sinfo failed: {result.stderr}",
                    "ssh_target": self._ssh_config.ssh_target,
                }

            return {
                "backend": self.backend_name,
                "healthy": True,
                "message": "Remote Slurm cluster accessible via SSH",
                "ssh_target": self._ssh_config.ssh_target,
                "partition_info": result.stdout.strip(),
            }
        except ConnectionError as exc:
            return {
                "backend": self.backend_name,
                "healthy": False,
                "message": f"SSH connection failed: {exc}",
                "ssh_target": self._ssh_config.ssh_target,
            }

    async def check_readiness(self, endpoint: EndpointInfo) -> bool:
        """Check endpoint readiness via SSH curl (front-end can't reach compute directly)."""
        ssh = await self._get_ssh()
        url = f"http://{endpoint.node}:{endpoint.litellm_port}/v1/models"
        result = await ssh.run_remote(
            f"curl -sf --max-time 5 {url}",
            timeout=10,
        )
        return result.ok

    async def shutdown(self) -> None:
        """Clean shutdown of SSH connection."""
        if self._tunnel_pool is not None:
            await self._tunnel_pool.shutdown()
            self._tunnel_pool = None
        # Close per-user SSH connections
        for username, mgr in list(self._user_ssh_pool.items()):
            await mgr.close()
        self._user_ssh_pool.clear()
        if self._ssh:
            await self._ssh.close()
            self._ssh = None


_PROXY_PASSTHRU = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:/%-"
)


# The mini-rivault wrapper's own default when VLLM_GPU_MEM_UTIL is unset
# (serve-model.yaml / serve-model.slurm: ${VLLM_GPU_MEM_UTIL:-0.95}).
_WRAPPER_GPU_MEM_UTIL_DEFAULT = 0.95


def _partition_util_cap(site: Any, partition: str | None) -> float:
    """Strictest ``gpu_mem_util_max`` across the listed partitions (0 = none)."""
    profiles = getattr(site, "partition_profiles", None) or {}
    caps = [
        float(getattr(profiles.get(part.strip()), "gpu_mem_util_max", 0) or 0)
        for part in (partition or "").split(",")
        if part.strip()
    ]
    return min((c for c in caps if c > 0), default=0.0)


def _requested_util(env: dict[str, Any]) -> float:
    try:
        return float(env.get("VLLM_GPU_MEM_UTIL") or _WRAPPER_GPU_MEM_UTIL_DEFAULT)
    except (TypeError, ValueError):
        return _WRAPPER_GPU_MEM_UTIL_DEFAULT


def _cap_gpu_mem_util_env(site: Any, partition: str | None, env: Any) -> Any:
    """Apply the partition's ``gpu_mem_util_max`` to the launch env.

    Every wrapper launch (``r4 endpoint up``, pool reconcile, warm reuse)
    passes through this submit, so the cap lives here, not only in the
    DeploymentSpec path (live rccs 2026-09-24: an endpoint-up launch still
    requested 0.9 against ~98 of 121.6 GiB free and failed at startup). For
    a comma list the strictest listed cap applies; an unset value counts as
    the wrapper's own default (0.95), so it is capped too.
    """
    cap = _partition_util_cap(site, partition)
    if cap <= 0 or not (env is None or isinstance(env, dict)):
        return env
    capped = dict(env or {})
    requested = _requested_util(capped)
    if requested > cap:
        capped["VLLM_GPU_MEM_UTIL"] = f"{cap:g}"
        logger.info(
            "VLLM_GPU_MEM_UTIL %.2f capped to %.2f by partition %s",
            requested,
            cap,
            partition,
        )
    return capped


def _hex_export_parts(env: dict[str, str]) -> list[str]:
    """Build the comma-joined KEY=VALUE parts, hex-encoding unsafe values.

    Two emission traps decide the encoding:
      • sbatch-assembly is a plain ``" ".join`` — a value with a SPACE
        would be tokenized; quotes were previously added but the RCCS
        per-user SSH proxy whitelist (``[^a-zA-Z0-9\\ _./:=,%-]``) rejects
        them,
      • ``sbatch --export`` is comma-delimited — a comma in the value
        silently truncates (base64's ``+`` was also proxy-blocked).
    The hex alphabet (``0-9a-f``) is the lowest common denominator that both
    layers tolerate, so values are now hex-encoded whenever they contain any
    byte outside the passthru set (alnum + ``._:/%``), including spaces.

    NOTE: must live at module scope AFTER the class. Inserting this function
    mid-class silently closes the class body and strands every following
    method as unreachable code inside this function — discoverable only at
    dispatch time (M5 regression, 2026-08-12).
    """
    out = ["ALL"]
    for k, v in env.items():
        if v and all(c in _PROXY_PASSTHRU for c in v):
            out.append(f"{k}={v}")
        else:
            out.append(f"{k}_HEX={v.encode().hex()}")
    return out
