"""Plan -> commit for workflow runs (ADR-105 D-4, slice S3).

A plan is a pure computation, never a row: ``POST /umrs/workflows/{name}/plan``
resolves what a run would use (lane, partition, account, GPU steps, a wall
bound) and signs it; ``.../commit`` verifies the signature, re-resolves and
starts the run with the plan id as the durable job's idempotency key. No
table, no job row: ``JobStatus.PLANNED`` is the status every job is created
in and may run without a commit, so it is not a plan marker.

This module holds the pure parts: the signature (:func:`sign_plan`,
:func:`verify_plan`), the quote (:func:`quote_workflow`) and the project's
plan policy (:func:`read_project_plan_policy`, migration 0131).

Signing key: HMAC-SHA256(``UMRS_SECRET_KEY``, ``b"umrs-workflow-plan-v1"``).
``UMRS_SECRET_KEY`` is server-only (the bundle-signing key); the gateway
handshake secret is not, since the gateway and cluster loop jobs hold it, so
it is never used here. Without ``UMRS_SECRET_KEY`` a per-process random key
signs: plans then do not survive a restart, nor reach another worker process.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

PLAN_TTL_SECONDS = 900
PLAN_ID_PREFIX = "plan-"
_KEY_LABEL = b"umrs-workflow-plan-v1"
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_DEFAULT_STEP_SECONDS = 3600.0
_TRUE_HINTS = ("1", "true", "yes")
#: A shorter ``UMRS_SECRET_KEY`` is treated as unset (never a key).
MIN_SECRET_CHARS = 16
#: Generic text for an unreadable policy: the database error is logged only.
POLICY_UNAVAILABLE_MESSAGE = "the project's plan policy is unavailable; try again"

#: The fields a plan signs, in the order the response lists them.
SIGNED_FIELDS: tuple[str, ...] = (
    "project_id",
    "user_id",
    "workflow_name",
    "version",
    "source_sha256",
    "parameters",
    "input_files",
    "lane",
    "partition",
    "account",
    "gpu_steps",
    "max_wall_seconds",
    "max_gpu_hours",
    "expires_at",
)
#: A commit re-resolves these; any change refuses it as stale.
STALE_FIELDS: tuple[str, ...] = (
    "version",
    # ADR-113 D-4: the digest of the version's files (closes the ADR-105
    # known limit "signs the catalog version, not a hash").
    "source_sha256",
    "lane",
    "partition",
    "account",
    "gpu_steps",
    "max_wall_seconds",
    "max_gpu_hours",
)

_process_key: bytes | None = None
_process_key_lock = threading.Lock()


# ── signature ─────────────────────────────────────────────────────────────


def canonical_json(value: Any) -> str:
    """The byte form a digest covers: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _per_process_key() -> bytes:
    global _process_key
    with _process_key_lock:
        if _process_key is None:
            _process_key = secrets.token_bytes(32)
            logger.warning(
                "UMRS_SECRET_KEY is not set (or shorter than %d characters): "
                "workflow plans are signed with a per-process key and do not "
                "survive a restart",
                MIN_SECRET_CHARS,
            )
        return _process_key


def _configured_secret() -> str:
    """``UMRS_SECRET_KEY`` stripped; a value under :data:`MIN_SECRET_CHARS`
    (a blank, a placeholder) counts as unset, so it never becomes the key."""
    secret = os.environ.get("UMRS_SECRET_KEY", "").strip()
    return secret if len(secret) >= MIN_SECRET_CHARS else ""


def _signing_key() -> bytes:
    secret = _configured_secret()
    if secret:
        return hmac.new(secret.encode("utf-8"), _KEY_LABEL, hashlib.sha256).digest()
    return _per_process_key()


def plan_digest(signed: dict[str, Any]) -> str:
    """Hex HMAC-SHA256 of the canonical signed dict."""
    body = canonical_json(signed).encode("utf-8")
    return hmac.new(_signing_key(), body, hashlib.sha256).hexdigest()


def plan_id_for(digest: str) -> str:
    return PLAN_ID_PREFIX + digest[:24]


def expires_at_iso(now: float, ttl: int = PLAN_TTL_SECONDS) -> str:
    return datetime.fromtimestamp(now + ttl, tz=timezone.utc).strftime(_ISO_FORMAT)


def sign_plan(
    fields: dict[str, Any], *, now: float | None = None
) -> tuple[dict[str, Any], str, str]:
    """``(signed, digest, plan_id)`` for *fields* plus a fresh ``expires_at``."""
    issued = time.time() if now is None else now
    signed = {name: fields.get(name) for name in SIGNED_FIELDS[:-1]}
    signed["expires_at"] = expires_at_iso(issued)
    digest = plan_digest(signed)
    return signed, digest, plan_id_for(digest)


def verify_plan(signed: Any, digest: Any) -> bool:
    """Whether *digest* is this server's signature over *signed*."""
    if not isinstance(signed, dict) or not isinstance(digest, str):
        return False
    try:
        return hmac.compare_digest(plan_digest(signed), digest)
    except (TypeError, ValueError):  # a non-ASCII digest, an unencodable dict
        return False


def plan_expired(signed: dict[str, Any], *, now: float | None = None) -> bool:
    """True past ``expires_at``; an unreadable or zone-less stamp reads as
    expired (this server writes ``...Z``)."""
    try:
        stamp = datetime.fromisoformat(str(signed.get("expires_at")))
    except ValueError:
        return True
    if stamp.tzinfo is None:
        return True
    current = time.time() if now is None else now
    return current >= stamp.timestamp()


def stale_fields(signed: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """The :data:`STALE_FIELDS` whose re-resolved value differs."""
    return [name for name in STALE_FIELDS if signed.get(name) != fresh.get(name)]


# ── quote: what a run would use, resolved without running it ─────────────


@dataclass(frozen=True)
class StepQuote:
    """One step's resolved lane, GPU request and wall bound."""

    name: str
    lane: str
    gate: bool
    gpu: bool
    gpus: int
    attempt_seconds: int
    attempts: int

    @property
    def max_wall_seconds(self) -> int:
        return self.attempt_seconds * self.attempts

    @property
    def gpu_seconds(self) -> int:
        return self.max_wall_seconds * self.gpus if self.gpu else 0

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["max_wall_seconds"] = self.max_wall_seconds
        return row


@dataclass(frozen=True)
class WorkflowQuote:
    """The run's lane, scheduler target and bounds (or why it cannot run)."""

    lane: str | None
    partition: str | None
    account: str | None
    steps: tuple[StepQuote, ...]
    refusal: str = ""

    @property
    def gpu_steps(self) -> int:
        return sum(1 for step in self.steps if step.gpu)

    @property
    def max_wall_seconds(self) -> int:
        return sum(step.max_wall_seconds for step in self.steps)

    @property
    def max_gpu_hours(self) -> float:
        return round(sum(step.gpu_seconds for step in self.steps) / 3600.0, 4)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _default_lane(worker: str | None) -> str:
    """The executor name a run without uploaded code builds (env fallback)."""
    chosen = worker or os.environ.get("UMRS_STEP_EXECUTOR") or "inproc"
    return chosen.strip().lower()


def _run_lanes(
    workflow_json: dict[str, Any], steps: list[dict[str, Any]], source_path: str
) -> tuple[str | None, str, str]:
    """(uploaded-step lane, platform-template lane, refusal), as the runner
    decides them (``_workflow_executor_name``, ``_split_step_lanes``) but
    without the legacy lane's side effects."""
    from umrs.server.step_trust import (
        LEGACY_IN_PROCESS,
        uploaded_code_lane,
        workflow_has_uploaded_code,
    )
    from umrs.server.workflow_runner import (
        _NO_HPC_LANE,
        _executor_name_from_execution_mode,
    )

    worker = _executor_name_from_execution_mode(workflow_json.get("execution_mode"))
    template = _default_lane(worker)
    if not workflow_has_uploaded_code(steps, source_path):
        return template, template, ""
    lane = uploaded_code_lane(worker)
    if lane is None:
        return None, template, _NO_HPC_LANE
    if lane == LEGACY_IN_PROCESS:
        return template, template, ""
    return lane, template, ""


def _remote_submit(lane: str) -> bool:
    return (
        lane == "slurm-remote" or os.environ.get("UMRS_SLURM_STEP_SUBMIT") == "remote"
    )


def _hpc_attempt_seconds(step_timeout: float | None, lane: str) -> float:
    """The ``--time`` one attempt of a step gets on an HPC lane."""
    from umrs.server.slurm_step_executor import slurm_duration_seconds

    site = slurm_duration_seconds(os.environ.get("UMRS_SLURM_TIME", "01:00:00"))
    site_limit = site or _DEFAULT_STEP_SECONDS
    if not _remote_submit(lane):
        # Local sbatch: the executor timeout is --time; 0 falls to the site's.
        return float(_env_int("UMRS_STEP_EXECUTOR_TIMEOUT", 3600) or site_limit)
    limits = [
        site_limit,
        float(_env_int("UMRS_STEP_EXECUTOR_TIMEOUT", 0) or site_limit),
    ]
    if step_timeout:
        limits.append(step_timeout)
    return min(limits)


def _attempt_seconds(step_timeout: float | None, lane: str) -> int:
    from umrs.server.step_trust import HPC_LANES

    if lane in HPC_LANES:
        return math.ceil(_hpc_attempt_seconds(step_timeout, lane))
    # No scheduler bounds these lanes: a step without timeout_seconds
    # counts the 3600 s default.
    return math.ceil(step_timeout or _DEFAULT_STEP_SECONDS)


def _truthy_hint(hint: Any) -> bool:
    return hint is True or str(hint).strip().lower() in _TRUE_HINTS


def _step_needs_gpu(spec: dict[str, Any], lane: str) -> bool:
    """As the executors decide: the remote lane needs the site's GPU switch
    AND the step's ``routing_hints.gpu``; elsewhere the hint, else the switch."""
    from umrs.server.step_trust import HPC_LANES

    hints = spec.get("routing_hints")
    hint = hints.get("gpu") if isinstance(hints, dict) else None
    site_gpu = os.environ.get("UMRS_STEP_GPU", "0").strip() == "1"
    if lane in HPC_LANES and _remote_submit(lane):
        return site_gpu and _truthy_hint(hint)
    return site_gpu if hint is None else _truthy_hint(hint)


def _step_quote(spec: dict[str, Any], lane: str) -> StepQuote:
    from umrs.pipeline_sdk.step import StepPolicy

    name = str(spec.get("name") or "")
    if spec.get("gate"):
        return StepQuote(name, lane, True, False, 0, 0, 1)
    policy = StepPolicy.from_mapping(spec)
    gpu = _step_needs_gpu(spec, lane)
    return StepQuote(
        name=name,
        lane=lane,
        gate=False,
        gpu=gpu,
        gpus=max(1, _env_int("UMRS_SLURM_GPUS", 1)) if gpu else 0,
        attempt_seconds=_attempt_seconds(policy.timeout_seconds, lane),
        attempts=1 + policy.max_retries,
    )


def _step_lane(
    spec: dict[str, Any], source_path: str, uploaded: str, template: str
) -> str:
    from umrs.server.step_trust import is_uploaded_step

    entrypoint = str(spec.get("entrypoint") or "")
    return uploaded if is_uploaded_step(entrypoint, source_path) else template


def _scheduler_target(steps: tuple[StepQuote, ...]) -> tuple[str | None, str | None]:
    """(partition, account) when any step goes to an HPC lane."""
    from umrs.server.step_trust import HPC_LANES

    if not any(step.lane in HPC_LANES for step in steps):
        return None, None
    partition = os.environ.get("UMRS_SLURM_PARTITION", "gpu")
    return partition, os.environ.get("UMRS_SLURM_ACCOUNT", "")


def quote_workflow(workflow_json: dict[str, Any], source_path: str) -> WorkflowQuote:
    """Resolve the run's lanes, GPU steps and wall bound from the site config.

    Reads the site's step settings (``load_site_env``) and the workflow's
    steps; imports no step code and starts nothing.
    """
    from umrs.shared.uploaded_code import load_site_env

    load_site_env()
    specs = [s for s in workflow_json.get("steps") or [] if isinstance(s, dict)]
    uploaded, template, refusal = _run_lanes(workflow_json, specs, source_path)
    if refusal or uploaded is None:
        return WorkflowQuote(None, None, None, (), refusal or "no lane")
    steps = tuple(
        _step_quote(spec, _step_lane(spec, source_path, uploaded, template))
        for spec in specs
    )
    partition, account = _scheduler_target(steps)
    return WorkflowQuote(uploaded, partition, account, steps)


# ── project plan policy (project_budgets, migration 0131) ─────────────────


class PlanPolicyUnavailable(RuntimeError):
    """The project's plan policy could not be read (fail closed)."""


@dataclass(frozen=True)
class PlanPolicy:
    """A project's plan settings; the defaults are a project without a row."""

    auto_commit_max_gpu_hours: float | None = None
    require_plan: bool = False

    def auto_commit_allows(self, max_gpu_hours: float) -> bool:
        """Fail closed: no threshold refuses every ``auto`` commit."""
        limit = self.auto_commit_max_gpu_hours
        return limit is not None and max_gpu_hours <= limit


_POLICY_SQL = (
    "SELECT auto_commit_max_gpu_hours, require_plan"
    " FROM project_budgets WHERE project_id = %s"
)


def _policy_from_row(row: Any) -> PlanPolicy:
    if not row:
        return PlanPolicy()
    limit = None if row[0] is None else float(row[0])
    return PlanPolicy(auto_commit_max_gpu_hours=limit, require_plan=bool(row[1]))


def _fetch_policy_row(pool: Any, project_id: str) -> Any:
    conn = pool.getconn()
    try:
        with conn:  # commits or rolls back: the read never stays open
            with conn.cursor() as cur:
                cur.execute(_POLICY_SQL, (project_id,))
                return cur.fetchone()
    finally:
        pool.putconn(conn)


def read_project_plan_policy(pool: Any, project_id: str) -> PlanPolicy:
    """The project's policy over the shared ``umrs_db`` pool.

    No pool (a process without the durable store) answers the defaults; a
    database error raises :class:`PlanPolicyUnavailable`.
    """
    if pool is None or not project_id:
        return PlanPolicy()
    try:
        row = _fetch_policy_row(pool, project_id)
    except Exception as exc:  # any database failure fails closed
        # The database text stays in the log, never in a reply.
        logger.warning("plan policy of project %r is unreadable: %s", project_id, exc)
        raise PlanPolicyUnavailable(POLICY_UNAVAILABLE_MESSAGE) from exc
    if not isinstance(row, (tuple, list)) and row is not None:
        logger.warning("plan policy of project %r: unexpected row %r", project_id, row)
        raise PlanPolicyUnavailable(POLICY_UNAVAILABLE_MESSAGE)
    return _policy_from_row(row)


def plan_required_message(project_id: str, workflow_name: str) -> str:
    return (
        f"project '{project_id}' requires a plan: use POST "
        f"/umrs/workflows/{workflow_name}/plan then POST "
        f"/umrs/workflows/{workflow_name}/commit"
    )


__all__ = [
    "PLAN_TTL_SECONDS",
    "SIGNED_FIELDS",
    "STALE_FIELDS",
    "PlanPolicy",
    "PlanPolicyUnavailable",
    "StepQuote",
    "WorkflowQuote",
    "canonical_json",
    "plan_digest",
    "plan_expired",
    "plan_id_for",
    "plan_required_message",
    "quote_workflow",
    "read_project_plan_policy",
    "sign_plan",
    "stale_fields",
    "verify_plan",
]
