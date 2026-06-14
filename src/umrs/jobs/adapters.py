"""
Backend Adapters - Bridge between JobService and existing orchestrators.

This module provides adapters that wrap existing orchestration backends
(Compose, Kubernetes, Slurm) to work with the durable job tracking system.

Architecture:
    JobService → Scheduler → BackendAdapter → OrchestrationBackend
                 Monitor → StatusChecker → ReadinessPoller
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..orchestrator.endpoint_mounts import build_mount_environment
from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

if TYPE_CHECKING:
    from umrs.orchestrator.base import OrchestratorBackend as BaseOrchestrator
    from umrs.orchestrator.endpoint_registry import EndpointRegistry
    from umrs.orchestrator.readiness import ReadinessPoller

logger = logging.getLogger(__name__)


def _parse_backend_ref(backend_ref: str) -> dict[str, str]:
    parts = backend_ref.split("|")
    if len(parts) < 2:
        raise ValueError(f"Malformed backend reference: {backend_ref}")

    parsed: dict[str, str] = {"kind": parts[0]}
    for item in parts[1:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def _resolve_registered_model(model_name: str) -> dict[str, Any] | None:
    """Resolve a model name from the UMRS catalog when no path is supplied.

    Checks both the legacy JSON catalog (``model_catalog.json``) and the
    durable Postgres-backed model store (``model_catalog_entries``).
    Returns the first match found, preferring system-scope entries.
    """
    # ── 1. Legacy JSON catalog ───────────────────────────────────────────────
    try:
        from umrs.server.model_registry_handler import _load_catalog
    except Exception:
        pass
    else:
        models = _load_catalog().get("models", [])
        matches = [entry for entry in models if entry.get("name") == model_name]
        if matches:
            scope_order = {"system": 0, "project": 1, "user": 2}
            matches.sort(
                key=lambda entry: scope_order.get(str(entry.get("scope", "system")), 99)
            )
            return matches[0]

    # ── 2. Durable Postgres model store ──────────────────────────────────────
    try:
        from umrs.model_registry.store import AccessContext, ModelStore, PathValidator
        from umrs.server.job_tracking_handler import build_durable_job_db_pool_from_env

        # Use the same DB pool builder as model registration (http_sidecar.py)
        # and the durable job runtime.  This guarantees a single source of truth
        # — both writes (register) and reads (resolve) hit the same database.
        db_pool = build_durable_job_db_pool_from_env()
        conn = db_pool.getconn()
        try:
            store = ModelStore(conn, path_validator=PathValidator())
            entries = store.list_models(
                AccessContext(user_id="", is_admin=True),
                limit=500,
            )
            matches = [entry.to_dict() for entry in entries if entry.name == model_name]
            if matches:
                scope_order = {"system": 0, "project": 1, "user": 2}
                matches.sort(
                    key=lambda e: scope_order.get(str(e.get("scope", "system")), 99)
                )
                logger.info(
                    "Resolved model %s from durable store (db=%s): storage_ref=%s",
                    model_name,
                    getattr(db_pool, "dsn", "unknown"),
                    matches[0].get("storage_ref"),
                )
                return matches[0]
        finally:
            db_pool.putconn(conn)
            db_pool.closeall()
    except Exception:
        logger.debug(
            "Durable model store unavailable while resolving %s",
            model_name,
            exc_info=True,
        )

    return None


def _short_progress_text(value: str | None, *, limit: int = 96) -> str:
    """Normalize and shorten progress text stored in job/endpoint metadata."""
    cleaned = " ".join((value or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _backend_progress_detail(job_info: Any) -> str:
    """Describe the current backend-side provisioning phase."""
    from umrs.orchestrator.base import JobStatus as OrchestratorJobStatus

    endpoint = getattr(job_info, "endpoint", None)
    target = None
    if endpoint and getattr(endpoint, "node", None) and endpoint.node != "pending":
        target = f"{endpoint.node}:{endpoint.litellm_port}"

    if job_info.status == OrchestratorJobStatus.PENDING:
        return "waiting for scheduler allocation"
    if job_info.status == OrchestratorJobStatus.STARTING:
        if target:
            return f"starting container / loading model at {target}"
        return "starting container / loading model"
    if job_info.status in (OrchestratorJobStatus.RUNNING, OrchestratorJobStatus.READY):
        if target:
            return f"services up at {target}; waiting for readiness"
        return "services starting; waiting for readiness"
    if job_info.status == OrchestratorJobStatus.STOPPED:
        return "backend job stopped"
    if job_info.error_message:
        return _short_progress_text(job_info.error_message)
    return "waiting for backend status"


def _readiness_progress_detail(endpoint: Any, error: str | None) -> str:
    """Describe readiness-probe progress in a user-facing way."""
    target = f"{endpoint.host}:{endpoint.port}"
    if not error:
        return f"waiting for endpoint readiness ({target})"

    lowered = error.lower()
    if "connection error" in lowered or "connection refused" in lowered:
        return "starting LiteLLM / waiting for service port"
    if "chat probe timeout" in lowered or "request timeout" in lowered:
        return "loading model weights / waiting for inference warmup"
    if any(
        code in lowered for code in ("http 404", "http 500", "http 502", "http 503")
    ):
        return "starting LiteLLM / waiting for proxy readiness"
    if "no models in response" in lowered or "expected model" in lowered:
        return "registering served model / loading model"
    if "chat probe http" in lowered or "chat probe error" in lowered:
        return "endpoint reachable; waiting for inference readiness"

    return f"waiting for endpoint readiness ({target}): {_short_progress_text(error, limit=72)}"


class EndpointProvisionBackend(OrchestratorBackend):
    """
    Backend adapter for endpoint provisioning jobs.

    Wraps the existing orchestration backend to submit vLLM/LiteLLM
    endpoint provisioning jobs via the durable job tracking system.
    """

    def __init__(
        self,
        orchestrator: BaseOrchestrator,
        registry: EndpointRegistry,
        port_manager,
        user_store=None,
    ):
        """
        Initialize adapter.

        Args:
            orchestrator: Underlying orchestrator (Compose/K8s/Slurm)
            registry: Endpoint registry for registration
            port_manager: Port allocation manager
            user_store: Optional UserStore for resolving per-user Slurm accounts
        """
        self.orchestrator = orchestrator
        self.registry = registry
        self.port_manager = port_manager
        self._user_store = user_store

    def _resolve_slurm_account(self, user_id: str) -> str | None:
        """Look up the submitting user's Slurm account from their admin record.

        Uses the per-site ``slurm_accounts`` dict when available, falling back
        to the flat ``slurm_account`` field.

        Returns ``None`` when no UserStore is configured, the user is not
        enrolled, or the record has no account set for the current site.
        """
        if not self._user_store or not user_id:
            return None
        try:
            from umrs.orchestrator.site_config import load_site_config

            site_name: str | None = None
            try:
                site_name = load_site_config().name
            except Exception:
                pass
            record = self._user_store.get(user_id)
            return record.resolve_slurm_account(site_name)
        except Exception:
            logger.debug("Could not resolve slurm_account for user %s", user_id)
            return None

    @property
    def backend_name(self) -> str:
        """Return orchestrator type name."""
        backend_name = getattr(self.orchestrator, "backend_name", None)
        if isinstance(backend_name, str):
            return backend_name

        backend_type = getattr(self.orchestrator, "backend_type", None)
        if hasattr(backend_type, "value"):
            return backend_type.value
        if isinstance(backend_type, str):
            return backend_type

        return "unknown"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        """
        Submit endpoint provisioning to the orchestrator.

        Extracts parameters from job.spec and calls the underlying
        orchestrator.submit_job().

        Returns:
            Backend reference (container ID, k8s job name, slurm job ID)
        """
        spec = job.spec

        # Extract endpoint parameters from job spec
        model_path = spec.get("model_path")
        project_id = job.project_id
        model_name = spec.get("model", "default")
        gpu_count = spec.get("gpu_count", 1)
        node_count = spec.get("node_count")  # multi-node (e.g. GLM-5.1, DeepSeek-V4)
        max_model_len = spec.get("max_model_len", 4096)
        ttl_sec = spec.get("ttl_sec", 3600)
        idle_ttl_sec = spec.get("idle_ttl_sec")
        parallel_strategy = spec.get("parallel_strategy")  # tp_pp | tep
        # Explicit compute placement requested via `endpoint up --partition` /
        # `--spec`. The Slurm wrapper backend pops these from kwargs; without
        # forwarding them here, provisioning falls back to auto-selected
        # partitions and ignores the operator's explicit sizing.
        partition_override = spec.get("partition_override")
        nodelist_hint = spec.get("nodelist_hint")
        requested_env = spec.get("env_vars") or {}
        mount_env, mount_metadata = build_mount_environment(
            tier=str(spec.get("tier") or "user"),
            mounts=spec.get("mounts") if isinstance(spec.get("mounts"), dict) else None,
            egress_domains=spec.get("egress_domains")
            if isinstance(spec.get("egress_domains"), list)
            else None,
        )

        if not model_path and model_name:
            model_entry = _resolve_registered_model(model_name)
            if model_entry:
                model_path = model_entry.get("storage_ref")
                served_model_name = model_entry.get("huggingface_id") or model_name
                requested_env = dict(requested_env)
                requested_env.setdefault(
                    "VLLM_SERVED_MODEL_NAME",
                    served_model_name,
                )
                logger.info(
                    "Resolved endpoint model %s to %s (served as %s)",
                    model_name,
                    model_path,
                    served_model_name,
                )
            else:
                # Fallback: use model name as sentinel path when catalog
                # resolution fails.  The slurm wrapper (serve.slurm) receives
                # VLLM_MODEL alongside VLLM_MODEL_PATH and can resolve from
                # the HF cache or site-wide model directory.
                model_path = model_name
                logger.error(
                    "Failed to resolve storage_ref for model '%s' from both "
                    "JSON catalog and durable Postgres store.  Using model "
                    "name as path.  The Slurm wrapper will receive "
                    "VLLM_MODEL_PATH=%s — this must map to a real filesystem "
                    "directory for vLLM to start.",
                    model_name,
                    model_name,
                )

        # Determine target host
        target_host = self._determine_target_host()

        # Allocate port using attempt ID for tracking
        port = self.port_manager.allocate(attempt.attempt_id, target_host)
        if port is None:
            raise RuntimeError("Failed to allocate port for endpoint")

        try:
            submit_env = {
                key: str(value)
                for key, value in requested_env.items()
                if isinstance(value, (str, int, float))
            }
            submit_env.update(
                {
                    "LITELLM_PORT": str(port),
                    "VLLM_MAX_MODEL_LEN": str(max_model_len),
                    "PROJECT_ID": project_id,
                    "MODEL_NAME": model_name,
                }
            )
            # Single-node tensor parallelism defaults to the requested GPU
            # count (TP == GPUs on the node). Honor any explicit override the
            # request already carried in env_vars.
            submit_env.setdefault("VLLM_TENSOR_PARALLEL_SIZE", str(gpu_count))
            # Forward the LiteLLM master key from the orchestrator process
            # environment so the mini-RiVault wrapper can authenticate the
            # in-job LiteLLM proxy. Never overwrite an explicit request value.
            litellm_master_key = os.environ.get("LITELLM_MASTER_KEY")
            if litellm_master_key:
                submit_env.setdefault("LITELLM_MASTER_KEY", litellm_master_key)
            # When no registered model was found (model_path is None), forward
            # the requested model name as VLLM_MODEL so the slurm wrapper can
            # resolve it from the HF cache instead of falling back to the site
            # config default.
            if not model_path and model_name:
                submit_env.setdefault("VLLM_MODEL", model_name)
            submit_env.update(mount_env)

            # Resolve per-user Slurm account from the user record so sbatch
            # attributes the job to the submitting user's fairshare group.
            slurm_account = self._resolve_slurm_account(job.user_id)

            # Submit to underlying orchestrator
            submit_kwargs: dict[str, Any] = {
                "model_path": model_path,
                "env": submit_env,
                "gpu_count": gpu_count,
                "requesting_user": job.user_id,
            }
            if slurm_account:
                submit_kwargs["slurm_account"] = slurm_account
            self._apply_advanced_layout_kwargs(
                submit_kwargs,
                node_count=node_count,
                parallel_strategy=parallel_strategy,
                partition_override=partition_override,
                nodelist_hint=nodelist_hint,
                requested_env=requested_env,
            )
            job_info = await self.orchestrator.submit_job(**submit_kwargs)

            if not job_info or not job_info.job_id:
                # Release port on failure
                self.port_manager.release(port, target_host)
                raise RuntimeError("Backend failed to submit job")

            # Get actual host from job info
            actual_host = job_info.endpoint.node if job_info.endpoint else target_host

            # Register endpoint in registry (status=provisioning)
            # Use attempt_id as endpoint ID for correlation
            endpoint_metadata = {
                "durable_job_id": job.job_id,
                "attempt_id": attempt.attempt_id,
                "attempt_no": attempt.attempt_no,
            }
            if idle_ttl_sec is not None:
                with contextlib.suppress(TypeError, ValueError):
                    parsed_idle_ttl = int(idle_ttl_sec)
                    if parsed_idle_ttl > 0:
                        endpoint_metadata["idle_ttl_sec"] = parsed_idle_ttl
            endpoint_metadata.update(mount_metadata)
            # Store the served model name so the gateway can discover by either name
            env_served_name = submit_env.get("VLLM_SERVED_MODEL_NAME")
            if env_served_name and env_served_name != model_name:
                endpoint_metadata["served_model_name"] = env_served_name
            endpoint_metadata.setdefault("job_status", job_info.status.value)
            endpoint_metadata.setdefault("detail", _backend_progress_detail(job_info))
            if parallel_strategy:
                endpoint_metadata["parallel_strategy"] = parallel_strategy

            self.registry.register(
                project_id=project_id,
                model=model_name,
                job_id=job_info.job_id,
                host=actual_host,
                port=port,
                backend=self.backend_name,
                ttl_sec=ttl_sec,
                metadata=endpoint_metadata,
            )

            logger.info(
                f"Submitted endpoint provision: job={job.job_id} "
                f"attempt={attempt.attempt_id} backend_ref={job_info.job_id}"
            )

            return job_info.job_id

        except Exception:
            # Release port on any error
            self.port_manager.release(port, target_host)
            raise

    @staticmethod
    def _apply_advanced_layout_kwargs(
        submit_kwargs: dict[str, Any],
        *,
        node_count: int | None,
        parallel_strategy: str | None,
        partition_override: str | None,
        nodelist_hint: str | None,
        requested_env: dict[str, Any] | None,
    ) -> None:
        """Inject multi-node layout and explicit placement flags.

        Forwards multi-node sizing (``node_count``/``parallel_strategy``) and
        operator-requested placement (``partition_override``/``nodelist_hint``)
        to the orchestrator. Without the placement flags, the Slurm wrapper
        backend auto-selects a partition and ignores explicit ``--partition``
        sizing requested at provision time.
        """
        if node_count is not None:
            submit_kwargs["node_count"] = node_count
        if partition_override:
            submit_kwargs["partition_override"] = partition_override
        if nodelist_hint:
            submit_kwargs["nodelist_hint"] = nodelist_hint
        eff_strategy = parallel_strategy or (requested_env or {}).get(
            "VLLM_PARALLEL_STRATEGY", ""
        )
        if eff_strategy:
            submit_kwargs["env"]["VLLM_PARALLEL_STRATEGY"] = eff_strategy
            is_tep = eff_strategy.lower() == "tep"
            if is_tep and node_count and node_count > 1:
                submit_kwargs["env"]["VLLM_EXPERT_PARALLEL"] = "1"

    def _determine_target_host(self) -> str:
        """Determine target host based on orchestrator type."""
        backend_name = self.backend_name.lower()
        if backend_name in ("compose", "docker-compose"):
            return "localhost"
        if backend_name in ("kubernetes", "k8s"):
            return "mini-rivault"  # K8s Service DNS
        if backend_name in ("slurm", "slurm-podman"):
            return "pending"  # Updated after allocation
        return "localhost"


class EndpointProvisionStatusChecker(BackendStatusChecker):
    """
    Status checker for endpoint provisioning jobs.

    Uses ReadinessPoller to check if endpoints are up and healthy.
    """

    def __init__(
        self,
        readiness_poller: ReadinessPoller,
        registry: EndpointRegistry,
        orchestrator: BaseOrchestrator | None = None,
    ):
        """
        Initialize checker.

        Args:
            readiness_poller: Poller for health/readiness checks
            registry: Endpoint registry to lookup endpoints
            orchestrator: Optional backend for status refresh and host discovery
        """
        self.readiness_poller = readiness_poller
        self.registry = registry
        self.orchestrator = orchestrator
        self._readiness_attempts: dict[str, int] = {}
        self._last_readiness_error: dict[str, str] = {}
        self._last_readiness_target: dict[str, tuple[str, int]] = {}

    @property
    def job_type(self) -> JobType:
        return JobType.ENDPOINT_PROVISION

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """
        Check status of an endpoint provisioning attempt.

        Looks up the endpoint in registry and checks readiness.
        """
        # Get backend ref (container/k8s/slurm job ID)
        backend_ref = attempt.backend_ref
        if not backend_ref:
            return AttemptCheckResult.failed("No backend reference")

        backend_error = await self._refresh_from_backend(backend_ref)
        if backend_error:
            return AttemptCheckResult.failed(backend_error)

        # Lookup endpoint in registry
        endpoint = self.registry.get_by_job_id(backend_ref)
        if not endpoint:
            return AttemptCheckResult.running("waiting for endpoint registration")

        from umrs.orchestrator.endpoint_registry import EndpointStatus

        if endpoint.status == EndpointStatus.FAILED:
            self._clear_readiness_tracking(backend_ref)
            return AttemptCheckResult.failed(
                endpoint.error_message
                or endpoint.metadata.get("error", "endpoint failed")
            )

        if endpoint.host in ("", "pending"):
            return AttemptCheckResult.running(
                str(endpoint.metadata.get("detail") or "waiting for backend allocation")
            )

        # Check if endpoint is ready
        try:
            readiness_probe = str(endpoint.metadata.get("readiness_probe") or "chat")
            (
                is_ready,
                is_degraded,
                models,
                error,
            ) = await self.readiness_poller.check_ready_detail(
                endpoint.host,
                endpoint.port,
                probe_model=endpoint.model,
                probe_kind=readiness_probe,
            )

            check_no = self._next_readiness_check_no(
                backend_ref,
                endpoint.host,
                endpoint.port,
            )

            if check_no == 1:
                logger.info(
                    "Starting readiness polling for %s at %s:%s via durable monitor "
                    "(auth=%s)",
                    backend_ref,
                    endpoint.host,
                    endpoint.port,
                    "yes" if self.readiness_poller.config.auth_token else "no",
                )

            if is_ready:
                logger.info(
                    "Endpoint %s ready after durable monitor check %s at %s:%s. "
                    "Models: %s",
                    backend_ref,
                    check_no,
                    endpoint.host,
                    endpoint.port,
                    models,
                )
                self._clear_readiness_tracking(backend_ref)
                # Transition the endpoint registry status to READY
                self.registry.update_metadata(
                    backend_ref,
                    {
                        "detail": "endpoint ready",
                        "readiness_error": None,
                    },
                )
                self.registry.mark_ready(backend_ref)
                return AttemptCheckResult.succeeded(
                    metrics={
                        "host": endpoint.host,
                        "port": endpoint.port,
                        "endpoint_status": "ready",
                    },
                )

            if is_degraded:
                logger.warning(
                    "Endpoint %s degraded at check %s — reachable but unhealthy: %s",
                    backend_ref,
                    check_no,
                    error or "no error detail",
                )
                progress_detail = _readiness_progress_detail(endpoint, error)
                self.registry.update_metadata(
                    backend_ref,
                    {
                        "detail": f"degraded: {progress_detail}"
                        if progress_detail
                        else "degraded",
                        "readiness_error": _short_progress_text(error)
                        if error
                        else None,
                    },
                )
                self.registry.mark_degraded(backend_ref)
                return AttemptCheckResult.running(
                    "degraded: endpoint reachable but unhealthy"
                )

            last_error = self._last_readiness_error.get(backend_ref)
            if check_no == 1 or check_no % 5 == 0 or error != last_error:
                logger.info(
                    "Readiness pending for %s at %s:%s after durable monitor check %s: %s",
                    backend_ref,
                    endpoint.host,
                    endpoint.port,
                    check_no,
                    error or "not ready",
                )
            self._last_readiness_error[backend_ref] = error or ""

            progress_detail = _readiness_progress_detail(endpoint, error)
            self.registry.update_metadata(
                backend_ref,
                {
                    "detail": progress_detail,
                    "readiness_error": _short_progress_text(error) if error else None,
                    "readiness_check_no": check_no,
                },
            )

            return AttemptCheckResult.running(progress_detail)

        except Exception as e:
            logger.warning(f"Readiness check failed for {backend_ref}: {e}")

            # Don't mark as failed on transient errors
            # Only fail if endpoint explicitly reports failure
            if endpoint.status == EndpointStatus.FAILED:
                self._clear_readiness_tracking(backend_ref)
                return AttemptCheckResult.failed(
                    endpoint.error_message or endpoint.metadata.get("error", str(e))
                )

            progress_detail = _readiness_progress_detail(endpoint, str(e))
            self.registry.update_metadata(
                backend_ref,
                {
                    "detail": progress_detail,
                    "readiness_error": _short_progress_text(str(e)),
                },
            )

            return AttemptCheckResult.running(progress_detail)

    async def check_attempt(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """Backward-compatible wrapper for older tests/callers."""
        return await self.check_status(attempt, job)

    async def _refresh_from_backend(self, backend_ref: str) -> str | None:
        """Refresh endpoint coordinates and coarse status from backend state."""
        if self.orchestrator is None:
            return None

        try:
            job_info = await self.orchestrator.get_job_status(backend_ref)
        except Exception as exc:
            logger.warning(
                "Failed to refresh backend status for %s: %s", backend_ref, exc
            )
            return None

        from umrs.orchestrator.base import JobStatus as OrchestratorJobStatus
        from umrs.orchestrator.endpoint_registry import EndpointStatus

        progress_metadata = {
            "job_status": job_info.status.value,
            "detail": _backend_progress_detail(job_info),
        }
        if job_info.error_message:
            progress_metadata["backend_error"] = _short_progress_text(
                job_info.error_message
            )

        if job_info.endpoint:
            self.registry.update_location(
                backend_ref,
                host=job_info.endpoint.node,
                port=job_info.endpoint.litellm_port,
                metadata=progress_metadata,
            )
        else:
            self.registry.update_metadata(backend_ref, progress_metadata)

        if job_info.status == OrchestratorJobStatus.PENDING:
            self.registry.update_status(backend_ref, EndpointStatus.PROVISIONING)
        elif job_info.status in (
            OrchestratorJobStatus.STARTING,
            OrchestratorJobStatus.RUNNING,
            OrchestratorJobStatus.READY,
        ):
            self.registry.update_status(backend_ref, EndpointStatus.STARTING)
        elif job_info.status in (
            OrchestratorJobStatus.FAILED,
            OrchestratorJobStatus.STOPPED,
        ):
            self._clear_readiness_tracking(backend_ref)
            error = job_info.error_message or (
                f"Backend job {backend_ref} {job_info.status.value}"
            )
            self.registry.update_metadata(
                backend_ref,
                {
                    "job_status": job_info.status.value,
                    "detail": _short_progress_text(error),
                },
            )
            self.registry.mark_failed(backend_ref, error)
            return error

        return None

    def _next_readiness_check_no(self, backend_ref: str, host: str, port: int) -> int:
        target = (host, port)
        if self._last_readiness_target.get(backend_ref) != target:
            self._last_readiness_target[backend_ref] = target
            self._readiness_attempts[backend_ref] = 0
            self._last_readiness_error.pop(backend_ref, None)

        check_no = self._readiness_attempts.get(backend_ref, 0) + 1
        self._readiness_attempts[backend_ref] = check_no
        return check_no

    def _clear_readiness_tracking(self, backend_ref: str) -> None:
        self._readiness_attempts.pop(backend_ref, None)
        self._last_readiness_error.pop(backend_ref, None)
        self._last_readiness_target.pop(backend_ref, None)


class IndexBuildBackend(OrchestratorBackend):
    """
    Backend adapter for index build jobs.

    Executes pgvector/HNSW index maintenance synchronously against the target
    database using the same admin connection settings as the server runtime.
    """

    @property
    def backend_name(self) -> str:
        return "index_build"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        spec = job.spec
        database = (spec.get("database") or "").strip()
        table = (spec.get("table") or "").strip()
        if not database:
            raise ValueError("index_build requires spec.database")
        if not table:
            raise ValueError("index_build requires spec.table")

        rebuild = bool(spec.get("rebuild", False))
        index_name = spec.get("index_name") or None
        from umrs.server.sql import maintain_pgvector_index  # noqa: boundary — lazy composition

        metrics = maintain_pgvector_index(
            database,
            table,
            rebuild=rebuild,
            index_name=index_name,
        )

        logger.info(
            "Completed index maintenance: job=%s attempt=%s db=%s table=%s action=%s",
            job.job_id,
            attempt.attempt_id,
            database,
            table,
            metrics.get("action", "unknown"),
        )

        return (
            f"index_build|database={database}|table={table}|"
            f"index={metrics['index_name']}|action={metrics['action']}"
        )


class IndexBuildStatusChecker(BackendStatusChecker):
    """Status checker for synchronous index-build jobs."""

    @property
    def job_type(self) -> JobType:
        return JobType.INDEX_BUILD

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        backend_ref = attempt.backend_ref
        if not backend_ref:
            return AttemptCheckResult.failed("No backend reference")

        try:
            parsed = _parse_backend_ref(backend_ref)
        except ValueError as exc:
            return AttemptCheckResult.failed(str(exc))

        database = parsed.get("database") or job.spec.get("database")
        table = parsed.get("table") or job.spec.get("table")
        if not database or not table:
            return AttemptCheckResult.failed("Missing database/table metadata")

        try:
            from umrs.server.sql import maintain_pgvector_index  # noqa: boundary — lazy composition

            metrics = maintain_pgvector_index(
                database,
                table,
                rebuild=False,
                index_name=parsed.get("index"),
            )
        except Exception as exc:
            return AttemptCheckResult.failed(str(exc))

        metrics["verified_after_submit"] = True
        return AttemptCheckResult.succeeded(metrics=metrics)

    async def check_attempt(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        return await self.check_status(attempt, job)


class CleanupBackend(OrchestratorBackend):
    """Backend adapter for explicit durable cleanup jobs."""

    def __init__(
        self,
        orchestrator,
        registry: EndpointRegistry,
        get_job_with_attempts: Callable[[str], tuple[Job, list[JobAttempt]]]
        | None = None,
    ):
        self.orchestrator = orchestrator
        self.registry = registry
        self.get_job_with_attempts = get_job_with_attempts

    @property
    def backend_name(self) -> str:
        return "cleanup"

    def _resolve_endpoint_job_id(self, target_job_id: str) -> str:
        endpoint = self.registry.get_by_job_id(target_job_id)
        if endpoint is not None:
            return target_job_id

        if self.get_job_with_attempts is None:
            return target_job_id

        try:
            target_job, attempts = self.get_job_with_attempts(target_job_id)
        except Exception:
            return target_job_id

        if target_job.job_type != JobType.ENDPOINT_PROVISION:
            raise ValueError(
                "cleanup target=endpoint requires an endpoint_provision job_id "
                "or a backend endpoint job id"
            )

        for prior_attempt in reversed(attempts):
            if prior_attempt.backend_ref:
                return prior_attempt.backend_ref

        raise ValueError(
            f"cleanup target job {target_job_id} has no provisioned backend reference yet"
        )

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        spec = job.spec
        target = (spec.get("target") or "").strip().lower()
        if target != "endpoint":
            raise ValueError(
                f"Unsupported cleanup target: {target or '<missing>'}. "
                "Supported targets: ['endpoint']"
            )

        target_job_id = (spec.get("job_id") or "").strip()
        if not target_job_id:
            raise ValueError("cleanup target=endpoint requires spec.job_id")

        resolved_job_id = self._resolve_endpoint_job_id(target_job_id)
        force = bool(spec.get("force", False))
        endpoint = self.registry.get_by_job_id(resolved_job_id)
        if endpoint is None:
            logger.info(
                "Cleanup job %s treated endpoint %s (requested=%s) as already absent",
                job.job_id,
                resolved_job_id,
                target_job_id,
            )
        else:
            await self.orchestrator.cleanup(resolved_job_id, force=force)
            self.registry.remove(resolved_job_id)

        logger.info(
            "Completed cleanup: job=%s attempt=%s target=%s resolved_target=%s force=%s",
            job.job_id,
            attempt.attempt_id,
            target_job_id,
            resolved_job_id,
            force,
        )
        backend_ref = f"cleanup|target=endpoint|job_id={resolved_job_id}"
        if resolved_job_id != target_job_id:
            backend_ref += f"|requested_job_id={target_job_id}"
        backend_ref += f"|force={int(force)}"
        return backend_ref


class CleanupStatusChecker(BackendStatusChecker):
    """Status checker for synchronous cleanup jobs."""

    def __init__(self, registry: EndpointRegistry):
        self.registry = registry

    @property
    def job_type(self) -> JobType:
        return JobType.CLEANUP

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        backend_ref = attempt.backend_ref
        if not backend_ref:
            return AttemptCheckResult.failed("No backend reference")

        try:
            parsed = _parse_backend_ref(backend_ref)
        except ValueError as exc:
            return AttemptCheckResult.failed(str(exc))

        target = parsed.get("target") or (job.spec.get("target") or "")
        if target != "endpoint":
            return AttemptCheckResult.failed(
                f"Unsupported cleanup target in backend state: {target or '<missing>'}"
            )

        target_job_id = parsed.get("job_id") or job.spec.get("job_id")
        if not target_job_id:
            return AttemptCheckResult.failed("Missing cleanup target job_id")

        if self.registry.get_by_job_id(target_job_id) is None:
            metrics = {
                "target": "endpoint",
                "target_job_id": target_job_id,
            }
            requested_job_id = parsed.get("requested_job_id")
            if requested_job_id:
                metrics["requested_job_id"] = requested_job_id
            return AttemptCheckResult.succeeded(
                metrics=metrics,
                commit=False,
            )

        return AttemptCheckResult.failed(
            f"Cleanup target still present after submit: {target_job_id}"
        )

    async def check_attempt(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        return await self.check_status(attempt, job)


def create_backend_factory(
    orchestrator: BaseOrchestrator,
    registry: EndpointRegistry,
    port_manager,
    get_job_with_attempts: Callable[[str], tuple[Job, list[JobAttempt]]] | None = None,
    embedding_worker=None,
    s3_client=None,
    model_store=None,
    path_validator=None,
    user_store=None,
):
    """
    Create a backend factory function for JobService.

    Args:
        orchestrator: Underlying orchestrator for endpoint jobs
        registry: Endpoint registry
        port_manager: Port allocation manager
        embedding_worker: Worker for index jobs (optional)
        s3_client: S3 client (optional)
        model_store: Model registry store for download jobs (optional)
        path_validator: Path validator for download jobs (optional)
        user_store: Optional UserStore for per-user Slurm account resolution

    Returns:
        Factory function: (JobType) -> OrchestratorBackend
    """
    # Pre-create backends
    endpoint_backend = EndpointProvisionBackend(
        orchestrator=orchestrator,
        registry=registry,
        port_manager=port_manager,
        user_store=user_store,
    )
    index_backend = IndexBuildBackend()
    cleanup_backend = CleanupBackend(
        orchestrator=orchestrator,
        registry=registry,
        get_job_with_attempts=get_job_with_attempts,
    )
    from .workflow_run import WorkflowRunBackend

    workflow_run_backend = WorkflowRunBackend()

    model_download_backend = None
    if model_store and path_validator:
        from .model_download import ModelDownloadBackend

        model_download_backend = ModelDownloadBackend(
            model_store=model_store,
            path_validator=path_validator,
        )

    def factory(job_type: JobType) -> OrchestratorBackend:
        if job_type == JobType.ENDPOINT_PROVISION:
            return endpoint_backend
        elif job_type == JobType.INDEX_BUILD:
            return index_backend
        elif job_type == JobType.CLEANUP:
            return cleanup_backend
        elif job_type == JobType.WORKFLOW_RUN:
            return workflow_run_backend
        elif job_type == JobType.MODEL_DOWNLOAD:
            if model_download_backend is None:
                raise RuntimeError(
                    "Model download backend not configured. "
                    "Provide model_store and path_validator."
                )
            return model_download_backend
        else:
            raise ValueError(f"Unsupported job type: {job_type}")

    return factory


def create_status_checker(
    readiness_poller: ReadinessPoller,
    registry: EndpointRegistry,
    orchestrator: BaseOrchestrator | None = None,
    embedding_worker=None,
    include_model_download: bool = True,
):
    """
    Create a composite status checker for JobService.

    Args:
        readiness_poller: For endpoint provisioning checks
        registry: Endpoint registry
        orchestrator: Underlying orchestrator for backend status refresh
        embedding_worker: For index build checks (optional)
        include_model_download: Whether to include model download checker

    Returns:
        CompositeStatusChecker instance
    """
    from .monitor import CompositeStatusChecker

    checker = CompositeStatusChecker()
    checker.register(
        JobType.ENDPOINT_PROVISION,
        EndpointProvisionStatusChecker(
            readiness_poller=readiness_poller,
            registry=registry,
            orchestrator=orchestrator,
        ),
    )
    from .workflow_run import WorkflowRunStatusChecker

    checker.register(JobType.WORKFLOW_RUN, WorkflowRunStatusChecker())
    checker.register(JobType.INDEX_BUILD, IndexBuildStatusChecker())
    checker.register(JobType.CLEANUP, CleanupStatusChecker(registry=registry))

    if include_model_download:
        from .model_download import ModelDownloadStatusChecker

        checker.register(JobType.MODEL_DOWNLOAD, ModelDownloadStatusChecker())

    return checker
