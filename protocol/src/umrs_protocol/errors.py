"""Shared platform error taxonomy (G-P12-5, WP-16).

Transport surfaces (gateway/UMRS) built errors as free-form strings; nothing
machine-readable tied a failure to a stable identity. This module is the
zero-dependency contract for typed platform errors:

- codes are namespaced, dot-separated, and append-only (removals need a
  major-version decision — consumers rely on codes for programmatic
  discrimination that message strings cannot carry);
- ``message`` stays the human-facing detail string, byte-identical to the
  legacy string it replaces (additive: existing clients see no drift);
- ``to_http_parts()`` projects the error into FastAPI-compatible kwargs with
  the code on an ``x-platform-error-code`` response header — the transport
  detail body never changes shape, the taxonomy rides the header.

Consumers surface the header; they never parse it. R4 clients gain a typed
discriminator now; richer problem-details bodies stay open (problem+json is
a later contract when route shapes converge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ERROR_CODE_HEADER = "x-platform-error-code"

# Initial code families. Append-only by contract.
SESSION_ERRORS = {
    "session.not_found": 404,
    "session.expired": 410,
    "session.expired_or_not_found": 404,
    "session.forbidden": 403,
    # WP-16 round 2 (research_sessions family conversion): additive only.
    "session.store_unavailable": 503,
    "session.archive_cascade_failed": 500,
}

AUTH_ERRORS = {
    "auth.missing_header": 401,
    "auth.bad_scheme": 401,
    "auth.bad_token": 401,
    # WP-16 round 2 (files family): verified project chain details.
    "auth.verified_project_mismatch": 401,
    "auth.verified_project_required": 401,
    "auth.project_required": 401,
    # WP-16 round 2 (jobs family): gateway-verified identity chain.
    "auth.gateway_signature_rejected": 401,
    "auth.gateway_identity_required": 401,
    "auth.caller_identity_required": 401,
    # WP-16 round 2 (ops family): authorization-not-authentication (403).
    "auth.admin_required": 403,
}

PROVIDER_ERRORS = {
    "provider.unavailable": 503,
}

CURVE_ERRORS = {
    "curve.disabled": 404,
    "curve.key_unconfigured": 500,
}

PRESIGN_ERRORS = {
    "presign.out_of_scope": 403,
    "presign.expired": 403,
    "presign.not_minted": 409,
}

JOBS_ERRORS = {
    # WP-16 round 2 (jobs family conversion): additive only.
    "jobs.not_found": 404,
    "jobs.attempt_not_found": 404,
    "jobs.conflict": 409,
    "jobs.forbidden": 403,
    "jobs.bad_request": 400,
    "jobs.unprocessable": 422,
    "jobs.unavailable": 503,
    "jobs.manifest_missing": 404,
}

PIPELINES_ERRORS = {
    # WP-16 round 2 (pipelines family conversion): additive only.
    "pipelines.not_found": 404,
    "pipelines.forbidden": 403,
    "pipelines.conflict": 409,
    "pipelines.bad_spec": 400,
    "pipelines.rag_profile_not_found": 404,
}

WORKFLOWS_ERRORS = {
    # WP-16 round 2 (workflows family conversion): additive only.
    "workflows.not_found": 404,
    "workflows.bad_request": 400,
    "workflows.forbidden": 403,
    "workflows.conflict": 409,
    "workflows.failure": 500,
    "workflows.bad_gateway": 502,
    "workflows.upload_too_large": 413,
    # ADR-105 D-4 (S3, 2026-11-02): plan → commit refusals.
    "workflows.plan_invalid": 403,
    "workflows.plan_expired": 409,
    "workflows.plan_mismatch": 403,
    "workflows.plan_stale": 409,
    "workflows.auto_commit_refused": 403,
    "workflows.plan_required": 403,
    "workflows.policy_unavailable": 503,
    "workflows.lane_unavailable": 409,
    "workflows.commit_unavailable": 503,
}

OPS_ERRORS = {
    # WP-16 round 2 (ops family conversion): additive only.
    "ops.governance_unavailable": 503,
    "ops.reload_unavailable": 503,
    "ops.not_found": 404,
    "ops.bad_request": 400,
    "ops.unprocessable": 422,
}

ENDPOINTS_ERRORS = {
    # WP-16 round 2 (endpoints family conversion): additive only.
    "endpoints.not_found": 404,
    "endpoints.forbidden": 403,
    "endpoints.conflict": 409,
    "endpoints.bad_request": 400,
    "endpoints.unavailable": 503,
    "endpoints.failure": 500,
    "endpoints.not_implemented": 501,
    "endpoints.bad_gateway": 502,
    # ADR-058 D-3 (WP-22): cancel issued but termination could not be
    # guaranteed (surrogate identity family; detail names both ids).
    "endpoints.cancellation_unachieved": 409,
    # 2026-10-23 (bdrt failing re-provision storm): the last attempts of this
    # model on this venue failed within the cap window; Retry-After says when.
    "endpoints.provision_capped": 429,
}

MODELS_ERRORS = {
    # WP-16 round 2 (models family conversion): additive only.
    "models.not_found": 404,
    "models.version_not_found": 404,
    "models.forbidden": 403,
    "models.conflict": 409,
    "models.bad_request": 400,
    "models.unavailable": 503,
    "models.failure": 500,
}

LOOPS_ERRORS = {
    # WP-16 round 2 (loops family conversion): additive only.
    "loops.not_found": 404,
    "loops.forbidden": 403,
    "loops.conflict": 409,
    "loops.bad_request": 400,
    "loops.unprocessable": 422,
    "loops.unavailable": 503,
}

COMPANION_ERRORS = {
    # WP-16 round 2 (companion family conversion): additive only.
    "companion.not_found": 404,
    "companion.forbidden": 403,
    "companion.unauthorized": 401,
    "companion.conflict": 409,
    "companion.gone": 410,
    "companion.unavailable": 503,
    "companion.unprocessable": 422,
}

RAGSERVICE_ERRORS = {
    # WP-16 round 2 (rag_service family conversion): additive only.
    "ragservice.unavailable": 500,
    "ragservice.not_found": 404,
    "ragservice.bad_request": 400,
    "ragservice.failure": 500,
}

RAG_ERRORS = {
    # WP-16 round 2 (rag family conversion): additive only.
    "rag.unavailable": 503,
    "rag.forbidden": 403,
    "rag.not_found": 404,
    "rag.bad_request": 400,
}

ARTIFACTS_ERRORS = {
    # WP-16 round 2 (artifacts family conversion): additive only.
    "artifacts.not_found": 404,
    "artifacts.unavailable": 503,
    "artifacts.bad_request": 400,
}

MCP_ERRORS = {
    # WP-16 round 2 (mcp family conversion): additive only.
    "mcp.not_found": 404,
    "mcp.unavailable": 503,
    "mcp.unprocessable": 422,
    # D-19/D-49: placement/trust-class validation at registration.
    "mcp.bad_request": 400,
    # D-19 call-time enforcement: a trusted-host-only server invoked from
    # any existing lane (no trusted:<host> venue rail exists yet) —
    # 403 by name, never approximated.
    "mcp.trust_class_denied": 403,
}

KNOWLEDGE2_ERRORS = {
    # WP-16 round 2 (knowledge_v2 family conversion): additive only.
    "knowledge2.unknown_graph": 400,
    "knowledge2.graph_not_active": 400,
    "knowledge2.bad_request": 400,
    "knowledge2.unavailable": 503,
    "knowledge2.failure": 500,
    # 2026-09-11 (knowledge_upload module): body/env/site fault chain.
    "knowledge2.upload_too_large": 413,
    # 2026-09-24: the HTTP ingest route accepts only remote-safe source kinds;
    # server-path kinds (local/local_file/local_dir) and caller-named s3_uri
    # would read the SERVER's filesystem / any bucket key with server creds.
    "knowledge2.source_kind_refused": 400,
}

SOURCEGRANTS_ERRORS = {
    # WP-16 round 2 (source_grants family conversion): additive only.
    "sourcegrants.unavailable": 503,
    "sourcegrants.unauthorized": 401,
    "sourcegrants.bad_request": 400,
    "sourcegrants.not_found": 404,
    # 2026-09-25: sharing to a retired group (umrs/api/source_grants.py) —
    # raised but unregistered (KeyError -> 500).
    "sourcegrants.conflict": 409,
}

SCIMEM_ERRORS = {
    # WP-16 round 2 (scientific_memory family conversion): additive only.
    "scimem.unavailable": 503,
    "scimem.not_found": 404,
    "scimem.bad_request": 400,
}

PROVENANCE_ERRORS = {
    # WP-16 round 2 (provenance family conversion): additive only.
    "provenance.not_found": 404,
    "provenance.forbidden": 403,
    "provenance.bad_request": 400,
}

KNOWLEDGEMODEL_ERRORS = {
    # WP-16 round 2 (knowledge_model family conversion): additive only.
    "knowledgemodel.unavailable": 503,
    "knowledgemodel.not_found": 404,
    "knowledgemodel.bad_request": 400,
}

GRANTS_ERRORS = {
    # WP-16 round 2 (grants family conversion): additive only.
    "grants.unauthorized": 401,
    "grants.unavailable": 503,
    "grants.not_found": 404,
    "grants.conflict": 409,
    "grants.bad_request": 400,
}

TRACES_ERRORS = {
    # WP-16 round 2 (traces family conversion): additive only.
    "traces.not_found": 404,
    "traces.forbidden": 403,
}

KNOWLEDGEGRAPHS_ERRORS = {
    # WP-16 round 2 (knowledge_graphs family conversion): additive only.
    "knowledgegraphs.unavailable": 503,
    "knowledgegraphs.not_found": 404,
    "knowledgegraphs.bad_request": 400,
    # D-35 two-tier: promotion is a human decision — agents are refused.
    "knowledgegraphs.forbidden": 403,
}

USERSOURCES_ERRORS = {
    # WP-16 round 2 (user_sources family conversion): additive only.
    "usersources.unavailable": 503,
    "usersources.unauthorized": 401,
    "usersources.bad_request": 400,
    "usersources.failure": 500,
    "usersources.not_found": 404,
}

VERIFICATION_ERRORS = {
    "verification.unavailable": 503,
    "verification.forbidden": 403,
    "verification.bad_request": 400,
}

SECRETBROKER_ERRORS = {
    "secretbroker.not_found": 404,
    "secretbroker.unauthorized": 401,
    "secretbroker.forbidden": 403,
    "secretbroker.bad_request": 400,
}

COMPANIONSESSIONS_ERRORS = {
    "companionsessions.unavailable": 503,
    "companionsessions.unauthorized": 401,
    "companionsessions.not_found": 404,
    "companionsessions.bad_request": 400,
    # ADR-057 leg 3: chat-turn approval intents
    "companionsessions.invalid_decision": 400,
    "companionsessions.conflict": 409,
    # F9 owner-scope queue (2026-09-21): action/content validation
    "companionsessions.invalid_action": 400,
    "companionsessions.invalid_value": 400,
    # ADR-114 S2.1: proven identity, receiver policy and queue replay/read rules
    "companionsessions.identity_mismatch": 409,
    "companionsessions.receiver_refused": 409,
    "companionsessions.invalid_recipient": 409,
    "companionsessions.replay_conflict": 409,
    "companionsessions.invalid_reply": 400,
    "companionsessions.inbox_full": 409,
}

# Remote-session (frontend venue) runtime routes — start/status/stop for the
# session owner + the lease-token worker rail (wave 2026-09-21, ADR-046 A2).
COMPANIONVENUE_ERRORS = {
    "companionvenue.unavailable": 503,
    "companionvenue.unauthorized": 401,
    "companionvenue.forbidden": 403,
    "companionvenue.not_found": 404,
    "companionvenue.bad_request": 400,
    # design F-2 step 1: a source_ref without a live sealed row
    "companionvenue.source_not_pushed": 409,
    "companionvenue.already_running": 409,
    "companionvenue.provision_failed": 502,
    # D-09 pause/resume verbs: state-transition conflicts (pause from
    # terminal, resume from unpaused) answer 409, never a smudged 500.
    "companionvenue.conflict": 409,
    # ADR-062 (D-51 gate): a sensitive project's session refused
    # public-kind inference sources — 403, by name, never smudged.
    "companionvenue.source_class_refused": 403,
    # ADR-099 D-8: an answer to a remote ask_user question that is no longer
    # pending (answered from another client, expired, cancelled).
    "companionvenue.question_closed": 409,
}

# Session policy (ADR-099 D-2, D-3; remote-session plan RS-1): limit rules,
# the user's web choice at session open, and their resolution at start.
SESSIONPOLICY_ERRORS = {
    "sessionpolicy.invalid": 400,
    "sessionpolicy.web_choice_required": 400,
    "sessionpolicy.widening_refused": 403,
    "sessionpolicy.forbidden": 403,
    "sessionpolicy.not_found": 404,
    "sessionpolicy.unavailable": 503,
}

CATALOG_ERRORS = {
    # 503 — pre-taxonomy HTTPException status was 503 (WP-16 r2 wave-2
    # mis-mapped 500 on conversion; the contract is byte-identical status).
    "catalog.unavailable": 503,
    "catalog.not_found": 404,
    "catalog.forbidden": 403,
    # 2026-09-25: raised by umrs/api/catalog.py (search without q/filters, bad
    # source=) but never registered — the handler died with KeyError -> 500.
    "catalog.unprocessable": 422,
}

A2L_ERRORS = {
    "a2l.not_found": 404,
    "a2l.bad_request": 400,
    "a2l.conflict": 409,
    "a2l.forbidden": 403,
}

USAGE_ERRORS = {
    "usage.unavailable": 503,
}

WORKFLOWFORK_ERRORS = {
    "workflowfork.unavailable": 503,
    "workflowfork.bad_request": 400,
    "workflowfork.not_found": 404,
}

WORKLOADIDENTITY_ERRORS = {
    "workloadidentity.unavailable": 503,
    "workloadidentity.bad_request": 400,
    # 2026-09-12 tail sites in workload_identity routes.
    "workloadidentity.unauthorized": 401,
    "workloadidentity.not_found": 404,
}

ADAPTIVE_ERRORS = {"adaptive.not_found": 404}

ASYNCPGPOOL_ERRORS = {"asyncpgpool.unavailable": 503}

# Cycle-3 (2026-10-22): a UMRS database outage (an unhandled psycopg2/asyncpg
# connection error) or a server with no DSN. UMRS sets this code as a literal
# (umrs.api.error_responses.DATABASE_UNAVAILABLE_CODE) so a server never
# needs a newer r4-contracts; registered here so the taxonomy is complete.
DATABASE_ERRORS = {"database.unavailable": 503}

BUNDLE_ERRORS = {
    "bundle.bad_request": 400,
    # 2026-09-24: server-path restore is an operator facility (admin only) —
    # a caller-named bundle_dir reads the SERVER's filesystem, and
    # write_upgrade rewrites a manifest there.
    "bundle.forbidden": 403,
}

CHATAPI_ERRORS = {"chatapi.bad_gateway": 502, "chatapi.unavailable": 503}

CHECKPOINTS_ERRORS = {"checkpoints.not_found": 404}

CONTEXTAPI_ERRORS = {"contextapi.not_found": 404}

DASHBOARD_ERRORS = {"dashboard.not_found": 404}

EVIDENCE_ERRORS = {"evidence.not_found": 404, "evidence.unavailable": 503}

EXPERIMENTS_ERRORS = {
    "experiments.not_found": 404,
    "experiments.unavailable": 503,
    # H-1 (2026-10-18): an unknown status filter was a plain, untyped 400.
    "experiments.bad_request": 400,
}

INTAKE_ERRORS = {
    "intake.bad_request": 400,
    "intake.conflict": 409,
    "intake.forbidden": 403,
    "intake.not_found": 404,
}

KNOWLEDGEGUARD_ERRORS = {
    "knowledge_guard.conflict": 409,
    "knowledge_guard.forbidden": 403,
    "knowledge_guard.not_found": 404,
    "knowledge_guard.unavailable": 503,
}

KNOWLEDGEQUERY_ERRORS = {"knowledgequery.bad_request": 400, "knowledgequery.unavailable": 503}

LLM_ERRORS = {
    "llm.bad_gateway": 502,
    "llm.bad_request": 400,
    "llm.gateway_timeout": 504,
    "llm.unavailable": 503,
}

NAMESPACES_ERRORS = {"namespaces.not_found": 404, "namespaces.unavailable": 503}

PROPOSALS_ERRORS = {
    "proposals.conflict": 409,
    "proposals.forbidden": 403,
    "proposals.not_found": 404,
    "proposals.unavailable": 503,
}

RECOVERY_ERRORS = {"recovery.not_found": 404, "recovery.unavailable": 503}

REFLECTIVE_ERRORS = {"reflective.bad_request": 400, "reflective.not_found": 404}

ROUTING_ERRORS = {"routing.not_found": 404}

SCOPEIDENTITY_ERRORS = {"scopeidentity.unauthorized": 401}

# ADR-108 D-3: signed scope tokens of work levels (umrs/scopes).
SCOPES_ERRORS = {
    "scopes.bad_request": 400,
    "scopes.invalid": 401,
    "scopes.revoked": 401,
    "scopes.forbidden": 403,
    "scopes.grant_missing": 403,
    "scopes.unavailable": 503,
}

SKILLS_ERRORS = {
    "skills.bad_request": 400,
    "skills.forbidden": 403,
    "skills.not_found": 404,
    "skills.unavailable": 503,
    # D-27 merge-on-green (2026-09-24): loop-proposed skills merge only when
    # tests pass; a re-review of an already-decided proposal is a conflict.
    "skills.merge_not_green": 403,
    "skills.conflict": 409,
}

# ADR-050 §3 groundwork (groups + membership authority; 2026-09-16).
GROUPS_ERRORS = {
    "groups.bad_request": 400,
    "groups.forbidden": 403,
    "groups.not_found": 404,
    "groups.unavailable": 503,
    "groups.conflict": 409,
}

# ADR-086 route family (project-scoped circles; supersedes the groups family).
CIRCLES_ERRORS = {
    "circles.bad_request": 400,
    "circles.forbidden": 403,
    "circles.not_found": 404,
    "circles.unavailable": 503,
    "circles.conflict": 409,
}

# ADR-051 route family (self-service durable projects; appended 2026-09-14).
PROJECTS_ERRORS = {
    "projects.bad_request": 400,
    "projects.forbidden": 403,
    "projects.not_found": 404,
    "projects.unavailable": 503,
    "projects.conflict": 409,
}

# Scientific Catalog family (gap row D-39, wave 2026-09-22): the catalog as a
# named, versioned object beside projects.
SCIENTIFIC_CATALOGS_ERRORS = {
    "scientificcatalogs.bad_request": 400,
    "scientificcatalogs.forbidden": 403,
    "scientificcatalogs.not_found": 404,
    "scientificcatalogs.unavailable": 503,
    "scientificcatalogs.conflict": 409,
}

# ADR-051 quota family (resource poles at spawn/upload gates; never accounting).
QUOTA_ERRORS = {
    "quota.cpu_process_exceeded": 503,
    "quota.disk_bytes_exceeded": 503,
    "quota.unavailable": 503,
    # ADR-099 D-5/D-6 (RS-1): the inference proxy's daily token and per-user
    # rate caps, and the live remote-session cap at start.
    "quota.tokens_per_day_exceeded": 429,
    "quota.rpm_exceeded": 429,
    "quota.concurrent_sessions_exceeded": 429,
}

TOOLCACHE_ERRORS = {
    "toolcache.bad_request": 400,
    "toolcache.forbidden": 403,
    "toolcache.not_found": 404,
    "toolcache.unavailable": 503,
}

VECTORS_ERRORS = {"vectors.not_found": 404, "vectors.unavailable": 503}

VERIFIED_ERRORS = {
    "verified.bad_request": 400,
    "verified.forbidden": 403,
    "verified.not_found": 404,
    "verified.unauthorized": 401,
}

VISION_ERRORS = {"vision.bad_request": 400, "vision.not_found": 404}

WORKFLOWAUTH_ERRORS = {"workflowauth.forbidden": 403, "workflowauth.not_found": 404}

WORKFLOWS2_ERRORS = {"workflows2.bad_request": 400}

KNOWLEDGE_ERRORS = {
    # WP-16 r2 tail (knowledge.py family conversion): additive only.
    "knowledge.unavailable": 503,
    "knowledge.not_found": 404,
    "knowledge.bad_request": 400,
    "knowledge.failure": 500,
    # 2026-09-24: export/import act on the caller's verified project only.
    "knowledge.project_mismatch": 409,
    # 2026-09-24 (ADR-032 raw-materials amendment): raw bytes leaving the
    # installation — owner/admin only, sensitive needs acknowledgement,
    # the catalog must state a licence.
    "knowledge.raw_forbidden": 403,
    "knowledge.raw_unacknowledged": 409,
    "knowledge.raw_licence_required": 409,
}

FILES_ERRORS = {
    # WP-16 round 2 (files family conversion): additive only.
    "files.traversal": 400,
    "files.filename_required": 400,
    "files.upload_failed": 500,
    "files.presign_failed": 500,
    "files.not_found": 404,
    "files.job_not_found": 404,
    "files.auth_unavailable": 503,
    "files.metadata_failed": 500,
    "files.delete_failed": 500,
    "files.download_failed": 500,
    "files.bucket_unreachable": 500,
    "files.bucket_denied": 403,
    # ADR-079 personal buckets + owned-bucket listing (2026-10-02).
    "files.list_failed": 500,
    "files.bucket_name_invalid": 400,
    "files.bucket_taken": 409,
    "files.bucket_create_failed": 500,
    "files.bucket_registry_unavailable": 503,
}

HPCACCESS_ERRORS = {
    # ADR-112 (2026-11-08): declared HPC access modes and gated delegation.
    # Wave A raises the first three; waves B-D raise the rest. Append-only.
    "hpcaccess.mode_undeclared": 403,
    "hpcaccess.project_not_allowed": 403,
    "hpcaccess.not_enrolled": 403,
    "hpcaccess.allocation_not_linked": 403,
    "hpcaccess.allocation_not_at_site": 403,
    "hpcaccess.gate_refused": 403,
    "hpcaccess.gate_version": 409,
    "hpcaccess.signature_required": 428,
    # Wave B1 review (2026-11-08): an agent-marked request to change the
    # account (D-8: the person's own act).
    "hpcaccess.person_required": 403,
    # Wave B2 (2026-11-08, plan §5 B2.0 §10.2): signed intents and the lanes
    # that have no gated path yet.
    "hpcaccess.intent_invalid": 422,
    "hpcaccess.intent_used": 409,
    "hpcaccess.lane_not_gated": 403,
    # Waves C and D (2026-11-09, plan §6 "C/D.0" C4): standing grants, the
    # project GPU budget and the usage ledger.
    "hpcaccess.grant_invalid": 422,
    "hpcaccess.grant_not_found": 404,
    "hpcaccess.budget_exceeded": 403,
    "hpcaccess.ledger_unavailable": 503,
}

ONTOLOGY_ERRORS = {
    # ADR-116 D-5 (2026-11-12): the ontology registry routes
    # (umrs/api/knowledge_ontologies.py). ``not_found`` is 400 when a graph
    # binds an unknown coordinate (the request body is wrong, not the URL).
    "ontology.invalid": 400,
    "ontology.forbidden": 403,
    "ontology.not_found": 404,
    "ontology.conflict": 409,
}


@dataclass(frozen=True)
class PlatformError:
    """One typed platform failure, transport-agnostic."""

    code: str
    message: str
    http_status: int
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code or "." not in self.code:
            raise ValueError(f"error codes must be namespaced ('family.name'): {self.code!r}")
        if not 400 <= self.http_status < 600:
            raise ValueError(f"http_status must be a 4xx/5xx class: {self.http_status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "http_status": self.http_status,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PlatformError:
        return cls(
            code=str(value["code"]),
            message=str(value["message"]),
            http_status=int(value["http_status"]),
            detail=dict(value.get("detail") or {}),
        )

    def to_http_parts(self) -> dict[str, Any]:
        """FastAPI-compatible kwargs: status body UNCHANGED, code on header."""
        return {
            "status_code": self.http_status,
            "detail": self.message,
            "headers": {ERROR_CODE_HEADER: self.code},
        }

    @classmethod
    def session(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"session.{name}",
            message=message,
            http_status=SESSION_ERRORS[f"session.{name}"],
            detail=detail,
        )

    @classmethod
    def auth(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"auth.{name}",
            message=message,
            http_status=AUTH_ERRORS[f"auth.{name}"],
            detail=detail,
        )

    @classmethod
    def provider(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"provider.{name}",
            message=message,
            http_status=PROVIDER_ERRORS[f"provider.{name}"],
            detail=detail,
        )

    @classmethod
    def curve(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"curve.{name}",
            message=message,
            http_status=CURVE_ERRORS[f"curve.{name}"],
            detail=detail,
        )

    @classmethod
    def presign(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"presign.{name}",
            message=message,
            http_status=PRESIGN_ERRORS[f"presign.{name}"],
            detail=detail,
        )

    @classmethod
    def files(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"files.{name}",
            message=message,
            http_status=FILES_ERRORS[f"files.{name}"],
            detail=detail,
        )

    @classmethod
    def jobs(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"jobs.{name}",
            message=message,
            http_status=JOBS_ERRORS[f"jobs.{name}"],
            detail=detail,
        )

    @classmethod
    def pipelines(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"pipelines.{name}",
            message=message,
            http_status=PIPELINES_ERRORS[f"pipelines.{name}"],
            detail=detail,
        )

    @classmethod
    def ops(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"ops.{name}",
            message=message,
            http_status=OPS_ERRORS[f"ops.{name}"],
            detail=detail,
        )

    @classmethod
    def models(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"models.{name}",
            message=message,
            http_status=MODELS_ERRORS[f"models.{name}"],
            detail=detail,
        )

    @classmethod
    def endpoints(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"endpoints.{name}",
            message=message,
            http_status=ENDPOINTS_ERRORS[f"endpoints.{name}"],
            detail=detail,
        )

    @classmethod
    def loops(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"loops.{name}",
            message=message,
            http_status=LOOPS_ERRORS[f"loops.{name}"],
            detail=detail,
        )

    @classmethod
    def companion(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"companion.{name}",
            message=message,
            http_status=COMPANION_ERRORS[f"companion.{name}"],
            detail=detail,
        )

    @classmethod
    def rag(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"rag.{name}",
            message=message,
            http_status=RAG_ERRORS[f"rag.{name}"],
            detail=detail,
        )

    @classmethod
    def ragservice(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"ragservice.{name}",
            message=message,
            http_status=RAGSERVICE_ERRORS[f"ragservice.{name}"],
            detail=detail,
        )

    @classmethod
    def usersources(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"usersources.{name}",
            message=message,
            http_status=USERSOURCES_ERRORS[f"usersources.{name}"],
            detail=detail,
        )

    @classmethod
    def sourcegrants(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"sourcegrants.{name}",
            message=message,
            http_status=SOURCEGRANTS_ERRORS[f"sourcegrants.{name}"],
            detail=detail,
        )

    @classmethod
    def scimem(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"scimem.{name}",
            message=message,
            http_status=SCIMEM_ERRORS[f"scimem.{name}"],
            detail=detail,
        )

    @classmethod
    def provenance(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"provenance.{name}",
            message=message,
            http_status=PROVENANCE_ERRORS[f"provenance.{name}"],
            detail=detail,
        )

    @classmethod
    def knowledgemodel(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"knowledgemodel.{name}",
            message=message,
            http_status=KNOWLEDGEMODEL_ERRORS[f"knowledgemodel.{name}"],
            detail=detail,
        )

    @classmethod
    def grants(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"grants.{name}",
            message=message,
            http_status=GRANTS_ERRORS[f"grants.{name}"],
            detail=detail,
        )

    @classmethod
    def traces(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"traces.{name}",
            message=message,
            http_status=TRACES_ERRORS[f"traces.{name}"],
            detail=detail,
        )

    @classmethod
    def knowledgegraphs(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"knowledgegraphs.{name}",
            message=message,
            http_status=KNOWLEDGEGRAPHS_ERRORS[f"knowledgegraphs.{name}"],
            detail=detail,
        )

    @classmethod
    def verification(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"verification.{name}",
            message=message,
            http_status=VERIFICATION_ERRORS[f"verification.{name}"],
            detail=detail,
        )

    @classmethod
    def secretbroker(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"secretbroker.{name}",
            message=message,
            http_status=SECRETBROKER_ERRORS[f"secretbroker.{name}"],
            detail=detail,
        )

    @classmethod
    def companionsessions(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"companionsessions.{name}",
            message=message,
            http_status=COMPANIONSESSIONS_ERRORS[f"companionsessions.{name}"],
            detail=detail,
        )

    @classmethod
    def companionvenue(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"companionvenue.{name}",
            message=message,
            http_status=COMPANIONVENUE_ERRORS[f"companionvenue.{name}"],
            detail=dict(detail),
        )

    @classmethod
    def sessionpolicy(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"sessionpolicy.{name}",
            message=message,
            http_status=SESSIONPOLICY_ERRORS[f"sessionpolicy.{name}"],
            detail=dict(detail),
        )

    @classmethod
    def catalog(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"catalog.{name}",
            message=message,
            http_status=CATALOG_ERRORS[f"catalog.{name}"],
            detail=detail,
        )

    @classmethod
    def a2l(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"a2l.{name}",
            message=message,
            http_status=A2L_ERRORS[f"a2l.{name}"],
            detail=detail,
        )

    @classmethod
    def usage(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"usage.{name}",
            message=message,
            http_status=USAGE_ERRORS[f"usage.{name}"],
            detail=detail,
        )

    @classmethod
    def workflowfork(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"workflowfork.{name}",
            message=message,
            http_status=WORKFLOWFORK_ERRORS[f"workflowfork.{name}"],
            detail=detail,
        )

    @classmethod
    def workloadidentity(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"workloadidentity.{name}",
            message=message,
            http_status=WORKLOADIDENTITY_ERRORS[f"workloadidentity.{name}"],
            detail=detail,
        )

    @classmethod
    def artifacts(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"artifacts.{name}",
            message=message,
            http_status=ARTIFACTS_ERRORS[f"artifacts.{name}"],
            detail=detail,
        )

    @classmethod
    def mcp(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"mcp.{name}",
            message=message,
            http_status=MCP_ERRORS[f"mcp.{name}"],
            detail=detail,
        )

    @classmethod
    def knowledge2(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"knowledge2.{name}",
            message=message,
            http_status=KNOWLEDGE2_ERRORS[f"knowledge2.{name}"],
            detail=detail,
        )

    @classmethod
    def workflows(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"workflows.{name}",
            message=message,
            http_status=WORKFLOWS_ERRORS[f"workflows.{name}"],
            detail=detail,
        )

    @classmethod
    def hpcaccess(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"hpcaccess.{name}",
            message=message,
            http_status=HPCACCESS_ERRORS[f"hpcaccess.{name}"],
            detail=detail,
        )

    @classmethod
    def ontology(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"ontology.{name}",
            message=message,
            http_status=ONTOLOGY_ERRORS[f"ontology.{name}"],
            detail=detail,
        )

    @classmethod
    def adaptive(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"adaptive.{name}",
            message=message,
            http_status=ADAPTIVE_ERRORS[f"adaptive.{name}"],
            detail=detail,
        )

    @classmethod
    def projects(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"projects.{name}",
            message=message,
            http_status=PROJECTS_ERRORS[f"projects.{name}"],
            detail=detail,
        )

    @classmethod
    def scientificcatalogs(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"scientificcatalogs.{name}",
            message=message,
            http_status=SCIENTIFIC_CATALOGS_ERRORS[f"scientificcatalogs.{name}"],
            detail=detail,
        )

    @classmethod
    def groups(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"groups.{name}",
            message=message,
            http_status=GROUPS_ERRORS[f"groups.{name}"],
            detail=detail,
        )

    @classmethod
    def circles(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"circles.{name}",
            message=message,
            http_status=CIRCLES_ERRORS[f"circles.{name}"],
            detail=detail,
        )

    @classmethod
    def quota(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"quota.{name}",
            message=message,
            http_status=QUOTA_ERRORS[f"quota.{name}"],
            detail=detail,
        )

    @classmethod
    def asyncpgpool(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"asyncpgpool.{name}",
            message=message,
            http_status=ASYNCPGPOOL_ERRORS[f"asyncpgpool.{name}"],
            detail=detail,
        )

    @classmethod
    def bundle(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"bundle.{name}",
            message=message,
            http_status=BUNDLE_ERRORS[f"bundle.{name}"],
            detail=detail,
        )

    @classmethod
    def chatapi(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"chatapi.{name}",
            message=message,
            http_status=CHATAPI_ERRORS[f"chatapi.{name}"],
            detail=detail,
        )

    @classmethod
    def checkpoints(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"checkpoints.{name}",
            message=message,
            http_status=CHECKPOINTS_ERRORS[f"checkpoints.{name}"],
            detail=detail,
        )

    @classmethod
    def contextapi(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"contextapi.{name}",
            message=message,
            http_status=CONTEXTAPI_ERRORS[f"contextapi.{name}"],
            detail=detail,
        )

    @classmethod
    def dashboard(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"dashboard.{name}",
            message=message,
            http_status=DASHBOARD_ERRORS[f"dashboard.{name}"],
            detail=detail,
        )

    @classmethod
    def evidence(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"evidence.{name}",
            message=message,
            http_status=EVIDENCE_ERRORS[f"evidence.{name}"],
            detail=detail,
        )

    @classmethod
    def experiments(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"experiments.{name}",
            message=message,
            http_status=EXPERIMENTS_ERRORS[f"experiments.{name}"],
            detail=detail,
        )

    @classmethod
    def intake(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"intake.{name}",
            message=message,
            http_status=INTAKE_ERRORS[f"intake.{name}"],
            detail=detail,
        )

    @classmethod
    def scopes(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"scopes.{name}",
            message=message,
            http_status=SCOPES_ERRORS[f"scopes.{name}"],
            detail=detail,
        )

    @classmethod
    def knowledge_guard(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"knowledge_guard.{name}",
            message=message,
            http_status=KNOWLEDGEGUARD_ERRORS[f"knowledge_guard.{name}"],
            detail=detail,
        )

    @classmethod
    def knowledge(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"knowledge.{name}",
            message=message,
            http_status=KNOWLEDGE_ERRORS[f"knowledge.{name}"],
            detail=detail,
        )

    @classmethod
    def knowledgequery(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"knowledgequery.{name}",
            message=message,
            http_status=KNOWLEDGEQUERY_ERRORS[f"knowledgequery.{name}"],
            detail=detail,
        )

    @classmethod
    def llm(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"llm.{name}",
            message=message,
            http_status=LLM_ERRORS[f"llm.{name}"],
            detail=detail,
        )

    @classmethod
    def namespaces(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"namespaces.{name}",
            message=message,
            http_status=NAMESPACES_ERRORS[f"namespaces.{name}"],
            detail=detail,
        )

    @classmethod
    def proposals(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"proposals.{name}",
            message=message,
            http_status=PROPOSALS_ERRORS[f"proposals.{name}"],
            detail=detail,
        )

    @classmethod
    def recovery(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"recovery.{name}",
            message=message,
            http_status=RECOVERY_ERRORS[f"recovery.{name}"],
            detail=detail,
        )

    @classmethod
    def reflective(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"reflective.{name}",
            message=message,
            http_status=REFLECTIVE_ERRORS[f"reflective.{name}"],
            detail=detail,
        )

    @classmethod
    def routing(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"routing.{name}",
            message=message,
            http_status=ROUTING_ERRORS[f"routing.{name}"],
            detail=detail,
        )

    @classmethod
    def scopeidentity(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"scopeidentity.{name}",
            message=message,
            http_status=SCOPEIDENTITY_ERRORS[f"scopeidentity.{name}"],
            detail=detail,
        )

    @classmethod
    def skills(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"skills.{name}",
            message=message,
            http_status=SKILLS_ERRORS[f"skills.{name}"],
            detail=detail,
        )

    @classmethod
    def toolcache(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"toolcache.{name}",
            message=message,
            http_status=TOOLCACHE_ERRORS[f"toolcache.{name}"],
            detail=detail,
        )

    @classmethod
    def vectors(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"vectors.{name}",
            message=message,
            http_status=VECTORS_ERRORS[f"vectors.{name}"],
            detail=detail,
        )

    @classmethod
    def verified(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"verified.{name}",
            message=message,
            http_status=VERIFIED_ERRORS[f"verified.{name}"],
            detail=detail,
        )

    @classmethod
    def vision(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"vision.{name}",
            message=message,
            http_status=VISION_ERRORS[f"vision.{name}"],
            detail=detail,
        )

    @classmethod
    def workflowauth(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"workflowauth.{name}",
            message=message,
            http_status=WORKFLOWAUTH_ERRORS[f"workflowauth.{name}"],
            detail=detail,
        )

    @classmethod
    def workflows2(cls, name: str, message: str, **detail: Any) -> PlatformError:
        return cls(
            code=f"workflows2.{name}",
            message=message,
            http_status=WORKFLOWS2_ERRORS[f"workflows2.{name}"],
            detail=detail,
        )


# Convenience named constructors for the taxonomy's known members. Keeping
# them in ONE table makes the append-only review point a one-file diff.
def known_error_codes() -> dict[str, int]:
    """Return the full code → default-status registry."""
    out: dict[str, int] = {}
    for fam in (
        SESSION_ERRORS,
        AUTH_ERRORS,
        PROVIDER_ERRORS,
        CURVE_ERRORS,
        PRESIGN_ERRORS,
        FILES_ERRORS,
        JOBS_ERRORS,
        PIPELINES_ERRORS,
        OPS_ERRORS,
        WORKFLOWS_ERRORS,
        MODELS_ERRORS,
        ENDPOINTS_ERRORS,
        LOOPS_ERRORS,
        COMPANION_ERRORS,
        RAG_ERRORS,
        ARTIFACTS_ERRORS,
        MCP_ERRORS,
        KNOWLEDGE2_ERRORS,
        RAGSERVICE_ERRORS,
        USERSOURCES_ERRORS,
        SOURCEGRANTS_ERRORS,
        SCIMEM_ERRORS,
        PROVENANCE_ERRORS,
        KNOWLEDGEMODEL_ERRORS,
        GRANTS_ERRORS,
        TRACES_ERRORS,
        KNOWLEDGEGRAPHS_ERRORS,
        KNOWLEDGE_ERRORS,
        VERIFICATION_ERRORS,
        SECRETBROKER_ERRORS,
        COMPANIONSESSIONS_ERRORS,
        COMPANIONVENUE_ERRORS,
        SESSIONPOLICY_ERRORS,
        CATALOG_ERRORS,
        A2L_ERRORS,
        USAGE_ERRORS,
        WORKFLOWFORK_ERRORS,
        WORKLOADIDENTITY_ERRORS,
        ADAPTIVE_ERRORS,
        ASYNCPGPOOL_ERRORS,
        DATABASE_ERRORS,
        BUNDLE_ERRORS,
        CHATAPI_ERRORS,
        CHECKPOINTS_ERRORS,
        CONTEXTAPI_ERRORS,
        DASHBOARD_ERRORS,
        EVIDENCE_ERRORS,
        EXPERIMENTS_ERRORS,
        INTAKE_ERRORS,
        KNOWLEDGEGUARD_ERRORS,
        KNOWLEDGEQUERY_ERRORS,
        LLM_ERRORS,
        NAMESPACES_ERRORS,
        PROPOSALS_ERRORS,
        RECOVERY_ERRORS,
        REFLECTIVE_ERRORS,
        ROUTING_ERRORS,
        SCOPEIDENTITY_ERRORS,
        SCOPES_ERRORS,
        SKILLS_ERRORS,
        TOOLCACHE_ERRORS,
        VECTORS_ERRORS,
        VERIFIED_ERRORS,
        VISION_ERRORS,
        WORKFLOWAUTH_ERRORS,
        WORKFLOWS2_ERRORS,
        HPCACCESS_ERRORS,
        ONTOLOGY_ERRORS,
    ):
        out.update(fam)
    return out
