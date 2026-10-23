"""H-1 static scope guard (hardening plan H-1 Do-5/6, 2026-10-18).

Walks every mounted route of the real sidecar app and keeps the H-1
authorization sweep fixed. A route's closure is the handler, its ``Depends``
targets and the ``umrs.*`` helpers they call (``_DEPTH`` levels, function-local
imports included). Guards and resolvers are matched by the referenced
object's ``module.qualname``, so a same-named helper does not count.

1. **Requested project.** A route taking a Query/Path/Body field whose name
   contains "project" or is corpus/namespace(s) (nested body models
   included, ``_MODEL_DEPTH`` deep) must reach a real project predicate
   (``SCOPE_GUARDS``, ``METHOD_GUARDS``). Resolving the caller alone is not a
   guard: such routes are in ``ALLOWLIST`` as IDENTITY-ONLY with the reason.
   Only the legacy inputs of ``IDENTITY_DEPENDENCIES`` are not requested
   projects.
2. **Identity.** Every route resolves the caller through an HMAC-verifying
   resolver or another authenticated principal (``IDENTITY_RESOLVERS``:
   signed gateway headers, JWT, lease token, admin JWT, or an ``_identity``
   call with the literal ``verify_hmac=True``), unless its module is in
   ``MODULE_ALLOWLIST`` (every route there lacks one) or the route is in
   ``ROUTE_IDENTITY_ALLOWLIST`` (probes, public metadata, admin-token routes,
   deferred gaps).
3. **Annotation.** A handler whose docstring says ``# auth: verified`` must
   resolve the identity (plan H-1 Do-6).
4. **Signed identity.** Identity is read only through the HMAC: a callable in
   the closure that reads ``X-User-ID`` / ``X-Verified-Project-ID`` /
   ``X-Verified-Is-Admin`` (constants or ``Header`` params) or calls
   ``_identity`` without ``verify_hmac=True``, and verifies nothing in its
   own helpers, is a finding unless a ``Depends`` target verified the tuple
   for the request. A resolver reached only from a deeper, possibly
   conditional helper does not excuse it (the pre-fix
   ``rag._require_rag_scope`` on /upload and /retrieve). Headers no HMAC
   covers (role, legacy ``X-Project-ID`` outside a verifying resolver, and
   the memberships list outside a verifier-module reader that reaches a
   ``MEMBERSHIP_VERIFIERS`` check) are always findings. Only an
   ``IDENTITY_RESOLVERS`` entry authenticates when it lives in a verifier
   module: its other helpers (the signed memberships reader, the mode probe)
   check a header or nothing and name nobody, so as a Depends they do not
   excuse raw identity reads.
   ``UNSIGNED_READER_ALLOWLIST`` is keyed by the reader, where the fix lands.

Not proven by any rule: object-id routes (Appendix A.2 class: load the row,
compare its project or owner, a foreign id reads as not found). The fix
lanes' tests cover the ones they fixed; the known open ones are listed by
hand in ``KNOWN_GAPS`` or the allowlists. A guard anywhere in the closure
covers every project field of the route (rule 1 cannot tie a guard to a
field: POST /umrs/endpoints/{job_id}/upgrade passed on
``_forbid_if_project_mismatch`` while ``request.model_path`` was unchecked,
until 2026-10-18). A guard that checks something other than the route's
requested project is dropped for that route in ``NOT_A_PROJECT_GUARD_FOR``,
so the route stays in ``ALLOWLIST`` with its real reason. Raw ``request.query_params`` /
``request.json()`` reads are not discovered as fields, and ``Header``
project params are skipped (strict-mode resolvers ignore ``X-Project-ID``).

Allowlist hygiene: an entry that no longer matches a finding fails the
suite, so a fixed gap is removed from the list in the same change.
``KNOWN_GAPS`` (``DEFERRED H-1: ...``) must still name mounted routes.
"""

from __future__ import annotations

import dis
import importlib
import inspect
import types
import typing
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.routing import APIRoute
from pydantic import BaseModel
from test_surface_characterization import _iter_api_routes

# ── vocabulary ─────────────────────────────────────────────────────────────
# Guards and resolvers are matched by the referenced object's
# ``module.qualname``, never by bare name: a same-named helper elsewhere (a
# raw-header ``get_caller``, ``server.commands.job._caller_is_admin``) does
# not count.

PROJECT_WORDS = frozenset({"corpus", "corpora", "namespace", "namespaces"})
# Names containing "project" that name no project to read or write.
NOT_PROJECT_FIELDS: frozenset[str] = frozenset()

# Real project (or object) predicates. The caller-only resolvers
# (request_caller_scope, get_project_id) are NOT guards: a route that only
# resolves the caller and then uses a requested project must be allowlisted
# as IDENTITY-ONLY with the reason it is safe.
SCOPE_GUARDS: dict[str, str] = {
    "umrs.knowledge.access_scope.project_readable": "three-tier read predicate",
    "umrs.knowledge.access_scope.project_writable": "three-tier write predicate",
    "umrs.knowledge.access_scope.writable_project_for": "requested project if writable",
    "umrs.knowledge.access_scope.graph_readable": "graph tier predicate",
    "umrs.knowledge.access_scope.graph_writable": "graph tier predicate",
    "umrs.api._access.graph_read_allowed": "graph tier or ADR-080 access list",
    "umrs.api._access.graph_meta_readable": "graph_read_allowed on a registry row",
    "umrs.api.knowledge._readable_project": "fail-closed 403 (project_readable)",
    "umrs.api.knowledge._scoped_project": "fail-closed 403 on a predicate",
    "umrs.api.projects._readable_project": "admin, owner, signed project, circle",
    "umrs.api.a2l._scoped_project": "project_readable on the named project",
    "umrs.api.ops._scoped_project": "non-admins pinned via scope_filter",
    "umrs.api.rag._require_rag_scope": "three-tier predicate on the HMAC-verified caller",
    "umrs.api.scientific_catalogs._require_project_scope": "body project == caller",
    "umrs.api.catalog._enforce_catalog_scope": "project_readable",
    "umrs.knowledge.events_sse._authorize_project": "project_readable",
    "umrs.api.loop._enforce_loop_project": "project_writable",
    "umrs.api.workflows._authorize_declared_workflow_project": "declared project in verified set",
    "umrs.api.mcp._enforce_registration_scope": "own project unless admin",
    "umrs.api.mcp._enforce_unregister_scope": "server bound to own project",
    "umrs.api.mcp._server_visible": "drops servers bound to another project",
    "umrs.api.artifact_routes._scope_checked_project": "non-admins pinned to own project",
    "umrs.api.artifact_routes._require_artifact_scope": "foreign artifact reads as 404",
    "umrs.api.endpoints._authorize_model_entry_for_caller": "model scope vs caller",
    "umrs.api.endpoints._forbid_if_project_mismatch": "row project vs caller (Appendix A)",
    "umrs.api.jobs._enforce_job_access": "job owner/project vs caller",
    "umrs.api.research_sessions._authorize_loop_link": "loop attachable by the caller",
    "umrs.api.projects._authorize_patch": "project owner or admin",
    "umrs.api.projects._require_owner_or_admin": "storage binding: owner or admin",
    "umrs.api.circles._visible_circle": "admin, owner, affiliated, circle member",
    "umrs.api.circles._owner_or_admin": "retire: circle owner or admin",
    "umrs.api.circles._manager_or_admin": "membership: owner/circle-admin/admin",
    "umrs.api.circles._require_create_right": "named circle: owner/admin/affiliated",
    "umrs.api.companion._owns_turn": "turn owner is the worker owner",
    "umrs.api.companion._turn_worker": "turns begin only on the caller's worker",
    "umrs.api.companion._ownerless_approval_open": "ownerless intents follow the loop rule",
    "umrs.api.companion._authorize_scope_write": "project:<pid> must be the caller's",
    "umrs.api.companion._enforce_ownership": "registry row owner",
    "umrs.api.companion._enforce_approval_ownership": "approval owner / loop project",
    # H-1 fix lanes, 2026-10-18 (each read before listing):
    "umrs.api.companion._caller_worker_job": "body job_id must be the caller's own worker job",
    "umrs.api.circles._roster_visible": "roster: admin, project owner or affiliated",
    "umrs.api.circles._own_rows_or_404": "own rows; 404 unless verified into the project",
    "umrs.api.endpoints._enforce_mount_scope": "non-admin mount sources under caller roots",
    "umrs.api.endpoints._enforce_model_path_scope": "caller-named model path under caller roots",
    "umrs.api.files._job_visible": "job project vs caller (or verified admin)",
    "umrs.api.files._visible_job": "foreign job reads as 404 (Appendix A.2)",
    "umrs.api.grants._may_revoke": "grantor, asset writer or verified admin",
    "umrs.api.models._member_may_write": "signed member project, project_writable",
    "umrs.api.models._caller_job_partition": "job partition via writable_project_for",
}
# CallerIdentity methods (``_verified_headers``): matched by attribute name,
# since ``caller.scope_filter(...)`` names no module object.
METHOD_GUARDS: dict[str, str] = {
    "scope_filter": "CallerIdentity: non-admins read only their own project",
    "bind_for_write": "CallerIdentity: declared project must be the caller's",
}

# Per route: guards in the closure that check something other than the
# requested project. Rule 1 ignores them for that route, so the route keeps
# its ALLOWLIST entry and its real reason (review 2026-10-18).
NOT_A_PROJECT_GUARD_FOR: dict[str, dict[str, str]] = {
    "POST /umrs/jobs": {
        "umrs.api.endpoints._authorize_model_entry_for_caller": (
            "catalog-model scope of an endpoint_provision spec; the job's project and user come from"
            " _resolve_job_create_identity_primitive (IDENTITY-ONLY)"
        ),
    },
}

# HMAC-verified gateway identity or another authenticated principal.
HMAC_KEYWORD = "umrs.api._identity:verify_hmac=True"
IDENTITY_RESOLVERS: dict[str, str] = {
    "umrs.api._scope_identity.request_caller_scope": "HMAC-verified CallerScope",
    "umrs.api._scope_identity.get_caller_scope": "Depends wrapper of request_caller_scope",
    "umrs.api.files.get_project_id": "JWT or HMAC-verified tuple",
    "umrs.api.endpoints.get_project_id": "JWT or HMAC-verified tuple",
    "umrs.api.jobs.get_caller_identity": "HMAC-verified CallerIdentity",
    "umrs.api._verified_headers.get_caller": "HMAC-verified, 401 in strict",
    "umrs.api._verified_headers.get_verified_caller": "identity required in every mode",
    "umrs.api._verified_headers.resolve_caller": "core HMAC resolution",
    "umrs.api._verified_headers.verified_headers_trusted": "the HMAC check itself",
    "umrs.api._verified_headers.workflow_author_trusted": "HMAC check for workflow authors",
    "umrs.api._verified_headers.verify_admin_flag": "HMAC-bound admin flag",
    "umrs.api.endpoints._caller_is_admin": "admin JWT or HMAC-verified admin",
    "umrs.api.models.get_access_context": "HMAC-verified user/admin, gateway-signed memberships",
    "umrs.api.workflows._require_verified_workflow_identity": "HMAC-verified tuple",
    "umrs.api._rail_identity.resolve_rail_principal": "venue rail lease-token principal",
    "umrs.shared.auth.validate_jwt": "bearer JWT",
    HMAC_KEYWORD: "_identity.request_identity / verified_identity with verify_hmac=True",
}

# ``_identity`` resolves the gateway headers but checks the HMAC only when
# called with ``verify_hmac=True``; it is not walked into, and a call without
# that keyword is an unsigned identity read.
_IDENTITY_CALLS = frozenset(
    {
        "umrs.api._identity.request_identity",
        "umrs.api._identity.verified_identity",
        "umrs.api._identity.request_caller",
    }
)
_HMAC_CAPABLE = frozenset(
    {"umrs.api._identity.request_identity", "umrs.api._identity.verified_identity"}
)
NO_RESOLVER = "<scope guard without a resolver>"
_NOT_FOLLOWED_MODULES = frozenset({"umrs.api._identity"})
# The verifier modules themselves read the raw headers to check them.
_VERIFIER_MODULES = frozenset({"umrs.api._verified_headers", "umrs.api._scope_identity"})
# The gateway HMAC covers (user, project, admin); nothing covers the rest.
SIGNED_HEADERS = frozenset({"x-user-id", "x-verified-project-id", "x-verified-is-admin"})
MEMBERSHIPS_HEADER = "x-verified-project-memberships"
UNSIGNED_HEADERS = frozenset({MEMBERSHIPS_HEADER, "x-user-role", "x-project-id"})
# Since 2026-10-18 the gateway signs the memberships list too
# (X-Verified-Memberships-HMAC over the signed tuple and the sorted list). A
# verifier-module callable whose own helpers reach one of these reads the list
# signed; anywhere else a raw read stays a finding, because reaching the check
# does not prove the verified value is the one used (a fallback branch).
MEMBERSHIP_VERIFIERS: dict[str, str] = {
    "r4_contracts.handshake_hmac.verify_memberships": "memberships HMAC bound to the signed tuple",
}

# Dependencies whose project-named inputs are identity, not a requested
# project: the legacy mvp ``caller_project_id`` / ``x_project_id`` inputs,
# ignored or rejected in strict mode in favour of the signed headers.
IDENTITY_DEPENDENCIES: dict[str, str] = {
    "umrs.api.jobs.get_caller_identity": "caller_project_id: legacy mvp input, strict uses the signed tuple",
    "umrs.api.ops.require_ops_admin": "caller_project_id: deleted unread (legacy query surface)",
    "umrs.api.models.get_access_context": "project_memberships: legacy mvp query, strict raises 401 without headers",
}

_DEPTH = 3  # route closure: handler, Depends targets and their helpers
_VERIFY_DEPTH = 2  # a header reader verifies through its own helpers only
_MODEL_DEPTH = 3  # nested body models

# ── rule 1 allowlist: project-taking routes without a listed guard ─────────
# Classes: ADMIN-ONLY, IDENTITY-ONLY (the caller's own project/user wins),
# OWNER-OR-ADMIN / SCOPED (inline check on the loaded row), NOT-SCOPE (the
# value reads no project data), DEFERRED H-1 (a known gap, not yet fixed).

_ADMIN_TOKEN = "ADMIN-ONLY: x-admin-token must equal UMRS_ADMIN_TOKEN"
_MODEL_ADMIN = "ADMIN-ONLY: ctx.is_admin from the HMAC-verified get_access_context"

ALLOWLIST: dict[str, str] = {
    "POST /umrs/buckets": (
        "NOT-SCOPE: project is a free experiment label in the caller's own <user>.<project>.<alias> bucket name"
        " (ADR-079 D-2); owner is the verified caller, and access changes audit under the caller's verified"
        " project, not the label (H-1 2026-10-18)"
    ),
    "POST /umrs/a2l/ledger/artifacts": (
        "ADMIN-ONLY: root artifacts are admin/ingestion only (is_admin from the HMAC-verified get_caller_identity)"
    ),
    "GET /umrs/assets": (
        "IDENTITY-ONLY: a non-admin's project_id is replaced by the verified project; rows pass caller visibility"
    ),
    "POST /umrs/assets/patch": (
        "IDENTITY-ONLY: a non-admin's body project_id is replaced by the verified project (same rule as GET /umrs/assets)"
    ),
    "GET /umrs/admin/sessions": (
        "ADMIN-ONLY: require_session_admin needs an admin JWT or an HMAC-verified admin identity; project"
        " only filters every user's sessions (RS-0 C-1, 2026-10-22)"
    ),
    "GET /umrs/admin/session-policy/effective": (
        "ADMIN-ONLY: require_session_admin needs an admin JWT or an HMAC-verified admin identity; project"
        " only names whose limits to compute (ADR-099 lane P, 2026-10-23)"
    ),
    "GET /umrs/endpoints": (
        "ADMIN-ONLY: all_projects drops the project scope only when _caller_is_admin (admin JWT or"
        " HMAC-verified admin); a non-admin stays on get_project_id_optional's project (RS-0, 2026-10-22)"
    ),
    "GET /umrs/companion/sessions": (
        "IDENTITY-ONLY: project only filters the caller's own snapshots (WHERE user_id = the HMAC-verified _uid)"
    ),
    "POST /umrs/egress/audit": (
        "ADMIN-ONLY: _require_audit_writer needs an HMAC-verified admin (H-1 registries lane)"
    ),
    "POST /umrs/endpoints": (
        "ADMIN-ONLY: _require_internal_endpoint_writer needs an admin JWT or HMAC-verified admin in strict mode"
    ),
    "GET /umrs/jobs": (
        "IDENTITY-ONLY: non-admin callers are pinned to their verified project and their own user_id"
    ),
    "POST /umrs/jobs": (
        "IDENTITY-ONLY: strict mode submits under the HMAC-verified project and user; the body project is the"
        " mvp fallback only"
    ),
    "POST /umrs/k2/verify": (
        "NOT-SCOPE: the body project_id is ignored; the job runs on the get_project_id (HMAC-verified) project"
    ),
    "POST /umrs/knowledge/corpus-attrition-report": (
        "IDENTITY-ONLY: reports only the get_project_id project; all_projects adds no project (its raw admin"
        " check runs after get_project_id verified the tuple)"
    ),
    "POST /umrs/knowledge/export": (
        "SCOPED: a body project_id other than the get_project_id project is refused inline (project_mismatch)"
    ),
    "POST /umrs/knowledge/external/search": (
        "NOT-SCOPE: the body project_id reads no project data; the route only queries public academic sources"
        " (caller resolved through get_project_id and logged since 2026-10-18)"
    ),
    "POST /umrs/knowledge/restore": (
        "ADMIN-ONLY: _require_operator needs an HMAC-verified admin (request_caller_scope) in every mode"
    ),
    "GET /umrs/models": (
        "SCOPED: rows pass the model ACL on ctx (HMAC-verified user/admin, the signed project plus the"
        " gateway-signed memberships list); project_id only narrows them"
    ),
    "POST /umrs/models": (
        "SCOPED: ACLEnforcer.enforce_create admits a project scope only for a project in ctx.project_memberships"
        " (signed project plus the gateway-signed list)"
    ),
    "POST /umrs/models/system": _MODEL_ADMIN,
    "PATCH /umrs/models/{model_name}/policy": _MODEL_ADMIN,
    "PUT /umrs/ops/governance/quotas": _ADMIN_TOKEN,
    "GET /umrs/ops/governance/analytics": _ADMIN_TOKEN + " (_check_governance_admin)",
    "POST /umrs/ops/governance/ownership/transfer": (
        _ADMIN_TOKEN + " (new_project_id is the target)"
    ),
    "GET /umrs/provenance": (
        "SCOPED: a non-admin project_id other than the verified project is refused inline; rows also pass can_access_job"
    ),
    "GET /umrs/usage/summary": (
        "IDENTITY-ONLY: non-admins are pinned to their own HMAC-verified user rows; project_id only narrows them"
    ),
    "GET /umrs/usage/records": (
        "IDENTITY-ONLY: non-admins are pinned to their own HMAC-verified user rows; project_id only narrows them"
    ),
    "GET /umrs/usage/billing": "ADMIN-ONLY: _require_admin on the HMAC-verified caller",
    "PUT /umrs/projects/{project_id}/storage": (
        "OWNER-OR-ADMIN: binding_refusal on the loaded project row (owners also need UMRS_PROJECT_STORAGE_ALLOWED)"
    ),
    "GET /umrs/scientific-catalogs": (
        "IDENTITY-ONLY: non-admin scope is the verified project; shared rows pass _catalog_read_allowed (ADR-080)"
    ),
    "GET /umrs/scientific-catalogs/pins/by-ref/{ref}": (
        "IDENTITY-ONLY: non-admin scope is the verified project"
    ),
    "GET /umrs/scientific-catalogs/versions/list": (
        "SCOPED: every row passes _catalog_read_allowed (own project or ADR-080 share)"
    ),
}

# ── rule 2 allowlists: routes that resolve no verified caller ─────────────

# Whole modules: every route lacks a resolver (stale once one route has one).
MODULE_ALLOWLIST: dict[str, str] = {
    "umrs.server.http_sidecar": "INFRA: /health, /metrics and /readiness probes, open by design",
    # 2026-10-18 (H-10.1 rag-profiles wave): only the three PUBLIC static
    # component catalogs remain (ROUTE_IDENTITY_ALLOWLIST); the CRUD routes
    # that resolved the caller were deleted with their clients.
    "umrs.api.pipelines": "PUBLIC: static component catalogs only (X-003..X-005)",
}

# Single routes in modules that otherwise resolve the caller.
ROUTE_IDENTITY_ALLOWLIST: dict[str, str] = {
    "GET /umrs/a2l/ledger/health": "PROBE: static liveness for r4 artifact",
    "GET /umrs/mcp/health": "PROBE: server and tool counts only",
    "GET /umrs/ops/ping": "PROBE: readiness for umrs ping, no data",
    "GET /umrs/web/dashboard/health": "PROBE: dashboard liveness, open by design",
    "GET /umrs/web/dashboard/metrics": "PROBE: Prometheus text, open by design",
    "GET /umrs/zmq/server-key": "PUBLIC: the CURVE server public key is not secret",
    # GET /umrs/components/* are covered by the MODULE_ALLOWLIST entry for
    # umrs.api.pipelines since 2026-10-18 (the H-10.1 rag-profiles wave left
    # them the module's only routes).
    "GET /umrs/models/system": "PUBLIC: system-scope models are readable by every user (U-817)",
    "DELETE /umrs/ops/audit/retention": _ADMIN_TOKEN,
    "GET /umrs/ops/governance/analytics": _ADMIN_TOKEN,
    "GET /umrs/ops/governance/orphans": _ADMIN_TOKEN,
    "DELETE /umrs/ops/governance/orphans": _ADMIN_TOKEN,
    "POST /umrs/ops/governance/ownership/transfer": _ADMIN_TOKEN,
    "PUT /umrs/ops/governance/quotas": _ADMIN_TOKEN,
}

# ── rule 3 allowlist: '# auth: verified' handlers without a resolver ──────

ANNOTATION_ALLOWLIST: dict[str, str] = {}

# ── rule 4 allowlist: identity read without the HMAC ──────────────────────
# Keyed by the reader (module.qualname): the defect lives there, and one fix
# clears every route it serves. A NO_RESOLVER finding is keyed by route.

UNSIGNED_READER_ALLOWLIST: dict[str, str] = {}

# ── deferred route gaps the rules above cannot see ─────────────────────────
# Object-id routes (Appendix A.2 class: load by id, compare the row's project
# or owner, a foreign id reads as not found) are not proven by any rule; the
# fix lanes' tests cover the ones they fixed, and the known open ones are
# listed here by hand.
# Checked and closed, so not listed: PATCH /umrs/verification/findings/{key}
# (knowledge-objects lane question): get_verified_caller + is_admin, so only a
# verified admin transitioned there, in every mode (2026-10-18; mvp anonymous
# refused); later that day it answered 410 and was then deleted with its
# module (H-10.1 follow-up). Transitions go through PATCH
# /umrs/knowledge/findings/{key} (a project the caller may write). Closed 2026-10-18 (h1-leftovers lane, tests/test_h1x_*.py): the
# upgrade model_path and POST /umrs/jobs endpoint_provision host paths take
# the per-caller root rule (orchestrator/caller_roots.py), and the jobs spec
# also the provision recipe policy (env_vars, container_image, tier=system
# admin-only) and a model that must resolve (review round); upgrade, PUT
# .../access, DELETE .../deprovision and PUT .../used check identity before
# the lookup; PUT /umrs/buckets/{bucket}/access answers a caller the list does
# not admit files.not_found; nginx /gw/ops/connect-info forwards the signed
# tuple; rag._get_caller_identity is deleted.

_PROJECT_403 = (
    "DEFERRED H-1: answers 403 (owner or admin only) to a caller who cannot read the project, while"
    " GET /umrs/projects/{project_id} hides it as 404; refuse non-readers with not_found via"
    " _may_read_project first (projects lane)"
)

KNOWN_GAPS: dict[str, str] = {
    "GET /umrs/web/dashboard/jobs": (
        "DEFERRED H-1: an admin's ?user_id= page is filtered in memory over a window of all jobs, so a page can"
        " be empty while pagination.total is positive (use _job_count_owner in _fetch_jobs); nginx has no"
        " 'location = /gw/web' redirect for the bare shell path (web-shell lane)"
    ),
    "GET /umrs/web/dashboard/models": (
        "DEFERRED H-1: shows only the active signed project's models; the gateway now signs memberships,"
        " so take them from Depends(_verified_headers.get_verified_memberships) (web-shell lane)"
    ),
    "GET /umrs/projects/{project_id}": (
        "DEFERRED H-1: a project reachable only through the memberships list reads 404 (_may_read_project"
        " ignores the now gateway-signed list); the r4 client fails closed on 403/404 since 2026-10-18 except"
        " for a gateway-less local record (projects lane)"
    ),
    "POST /umrs/circles": (
        "DEFERRED H-1: ESCALATED (ADR-086 §3): any verified caller may create its personal cell in any"
        " project, which is an existence oracle (404 vs 201) and self-affiliates, defeating the 404/409 hiding"
        " of GET /umrs/circles/{project_id}, POST /umrs/projects and PUT .../budget; maintainer decision"
    ),
    "PATCH /umrs/projects/{project_id}": _PROJECT_403,
    "GET /umrs/projects/{project_id}/storage": _PROJECT_403,
    "PUT /umrs/projects/{project_id}/storage": _PROJECT_403,
    "DELETE /umrs/projects/{project_id}/storage": _PROJECT_403,
    "POST /umrs/projects/{project_id}/storage/check": _PROJECT_403,
    "POST /umrs/endpoints/provision": (
        "DEFERRED H-1: residual: the caller-root check on mount sources and model paths is lexical at request"
        " time on the UMRS host (symlink TOCTOU, paths only the compute node sees); re-check realpaths in the"
        " compute-node launcher (serve.slurm / endpoint_mounts) or run under the caller's own account"
    ),
    "POST /umrs/files/presign": (
        "DEFERRED H-1: the gateway presigns downloads with its own S3_BUCKET; where it differs from UMRS's"
        " (legacy login_node role) that is now 403 bucket_denied; align or add S3_DOWNLOAD_EXTRA_BUCKETS (deploy)"
    ),
}

# Deferred fix-lane items that name no route (kept here so the guard shows
# the whole remaining H-1 surface in one place).
DEFERRED_ELSEWHERE: tuple[str, ...] = (
    "DEFERRED H-1: knowledge/events_sse.py imports umrs.api lazily under a boundary exemption that expires"
    " 2026-12; move request_caller_scope (or an adapter) into a non-api module (streams lane)",
    "DEFERRED H-1: parent tests/test_ops_introspection.py must land in the same parent commit as the umrs"
    " pointer bump that carries the new ops.py (admin-identity lane)",
    "DEFERRED H-1: chat.py must land in the same umrs commit as rag.py (rag._verified_caller /"
    " _verified_admin), never before it (chat-llm lane)",
    "DEFERRED H-1: _identity.request_identity compares X-Verified-Is-Admin case-sensitively while other"
    " readers lowercase it; add .lower() so every reader agrees (owner of _identity.py; chat-llm lane note)",
    "DEFERRED H-1: rag/search.py posts to /gw/llm/v1/chat/completions with no credentials (nginx rejects it);"
    " send a service bearer token or call the endpoint's litellm_url directly (owner of rag/search.py)",
    "DEFERRED H-1: jobs/knowledge_adapters._bridge_a2l_ref imports the deleted knowledge/a2l_bridge.py inside"
    " a blanket try; delete it, its 4 call sites and the 3 test monkeypatches (knowledge lane)",
    "DEFERRED H-1: http_sidecar._check_job_store calls the non-existent count_by_status, so /health reports"
    " pending_jobs 'unknown'; use count_jobs(status='queued') (owner of http_sidecar.py)",
    "DEFERRED H-1: the endpoint caller-root knob UMRS_ENDPOINT_CALLER_MOUNT_ROOTS (and the mount prefix"
    " knobs) are undocumented in umrs/deploy/_common/README.md; set it for /lvs0 sites via converge (deploy)",
    "DEFERRED H-1: parent tools/authz_canaries.yaml 'ops-admin' probes /gw/ops/connect-info with"
    " expect_denied: true, but connect-info is open discovery; point it at an admin-only ops path (parent)",
    "DEFERRED H-1: r4 bucket create documents PROJECT as a free name; the server now requires the caller's"
    " verified project (or none); make it optional and fix the ADR-079 and opossum guides (r4 lane)",
    "DEFERRED H-1: r4 ADR-062 gate residual: a server project recorded locally without a gateway still"
    " admits public routes on 404, and 401/5xx still fail open (r4 lane, policy decision)",
    "DEFERRED H-1: docs: ADR-062 amendment Known limits note (client gate now fails closed on denial),"
    " ADR-034 / adr README note that knowledge/a2l_bridge.py was deleted, STANDALONE guide /jobs/stats curl,"
    " security.md / apis.md signed memberships, mvp caller_project_id note (orchestrator commit)",
    "DEFERRED H-1: nginx -t on gateway/nginx after the /gw/web/ and memberships-HMAC blocks (maintainer,"
    " converge check before apply)",
    "DEFERRED H-1: deploy coupling: _verified_headers imports r4_contracts.handshake_hmac.verify_memberships"
    " (new, parent r4-contracts), so an older r4-contracts install stops every router at import; and strict"
    " UMRS drops an unsigned memberships list. r4-contracts handshake_hmac.py, gateway/sidecar/main.py and"
    " gateway/nginx server.conf land in the parent commit with the umrs pointer bump; converge the gateway no"
    " later than UMRS; the converge check confirms the installed r4-contracts exports verify_memberships"
    " (maintainer, escalated)",
    "DEFERRED H-1: plan status row, gap analysis and wave ledger for the H-1 fixes (orchestrator commit)",
)


# ── closure walk ───────────────────────────────────────────────────────────


def _qualified(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '')}"


def _code_target(fn: Any) -> Any:
    """The function whose bytecode a dependency or handler runs."""
    target = inspect.unwrap(fn)
    if inspect.isclass(target):
        return target.__init__
    if hasattr(target, "__code__"):
        return target
    return inspect.getattr_static(type(target), "__call__", None)


def _code_objects(code: types.CodeType) -> Iterator[types.CodeType]:
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def _import_module(name: str, level: Any, package: str) -> Any:
    relative = isinstance(level, int) and level > 0
    target = "." * level + name if relative else name
    try:
        return importlib.import_module(target, package if relative else None)
    except Exception:  # noqa: BLE001 - an unimportable lazy import is just not followed
        return None


def _lazy_imports(instructions: list[dis.Instruction], package: str) -> dict[str, Any]:
    """``from x import y`` inside function bodies (ADR-024 lazy imports)."""
    found: dict[str, Any] = {}
    module = None
    for index, ins in enumerate(instructions):
        if ins.opname == "IMPORT_NAME":
            level = instructions[index - 2].argval if index >= 2 else 0
            module = _import_module(ins.argval, level, package)
        elif ins.opname == "IMPORT_FROM" and module is not None:
            found[ins.argval] = getattr(module, ins.argval, None)
    return found


def _kwnames(ins: dis.Instruction, consts: tuple[Any, ...]) -> tuple[str, ...] | None:
    """The keyword-names tuple an instruction carries, if any.

    3.10 and 3.13 push it as a ``LOAD_CONST`` tuple; 3.11/3.12 ``KW_NAMES``
    holds a ``co_consts`` index that ``dis`` does not resolve (its argval is
    ``<unknown>`` on 3.11), so it is read from the code object's constants.
    """
    if ins.opname == "KW_NAMES":
        value = consts[ins.arg] if ins.arg is not None and ins.arg < len(consts) else None
        return value if isinstance(value, tuple) else None
    if ins.opname == "LOAD_CONST" and isinstance(ins.argval, tuple):
        return ins.argval
    return None


def _constant_keyword(
    instructions: list[dis.Instruction], index: int, kwnames: tuple[str, ...], name: str
) -> Any:
    """The constant value pushed for keyword ``name`` of the call whose
    keyword-names tuple is ``instructions[index]`` (``None`` if not constant)."""
    values = instructions[max(index - len(kwnames), 0) : index]
    if len(values) != len(kwnames) or any(v.opname != "LOAD_CONST" for v in values):
        return None
    return values[kwnames.index(name)].argval


def _passes_verify_hmac(instructions: list[dis.Instruction], consts: tuple[Any, ...]) -> bool:
    """A call in this block passes the constant keyword ``verify_hmac=True``.

    The keyword-names tuple follows the pushed keyword values (3.10
    ``CALL_FUNCTION_KW``, 3.11/3.12 ``KW_NAMES``, 3.13 ``CALL_KW``); only
    constant keyword values are read, so write the call with literals.
    """
    for index, ins in enumerate(instructions):
        names = _kwnames(ins, consts)
        if (
            names
            and "verify_hmac" in names
            and _constant_keyword(instructions, index, names, "verify_hmac") is True
        ):
            return True
    return False


def _resolve(names: set[str], lookup: Callable[[str], Any]) -> list[Any]:
    """Functions and classes the names bind to, also as ``module.attr``."""
    direct = [lookup(name) for name in names]
    modules = [obj for obj in direct if inspect.ismodule(obj)]
    via_module = [vars(module).get(name) for module in modules for name in names]
    return [
        obj for obj in (*direct, *via_module) if inspect.isfunction(obj) or inspect.isclass(obj)
    ]


@dataclass(frozen=True)
class _Scan:
    """What one callable's own bytecode references."""

    qualname: str
    module: str
    refs: frozenset[str]  # its own and every resolved callee's module.qualname
    names: frozenset[str]  # bare names (attribute and method calls)
    callees: tuple[Any, ...]
    headers: frozenset[str]  # identity header names among its string constants


_EMPTY_CODE = (frozenset(), frozenset(), (), frozenset())


def _block_facts(
    code: types.CodeType, package: str
) -> tuple[set[str], dict[str, Any], set[str], bool]:
    names: set[str] = set()
    lazy: dict[str, Any] = {}
    consts: set[str] = set()
    hmac_kw = False
    for block in _code_objects(code):
        instructions = list(dis.get_instructions(block))
        names.update(block.co_names)
        lazy.update(_lazy_imports(instructions, package))
        consts.update(c.lower() for c in block.co_consts if isinstance(c, str))
        hmac_kw = hmac_kw or _passes_verify_hmac(instructions, block.co_consts)
    return names, lazy, consts, hmac_kw


_SCANS: dict[int, tuple[Any, _Scan]] = {}


def _scan(fn: Any) -> _Scan:
    if id(fn) not in _SCANS:
        _SCANS[id(fn)] = (fn, _scan_uncached(fn))
    return _SCANS[id(fn)][1]


def _scan_uncached(fn: Any) -> _Scan:
    own = inspect.unwrap(fn)
    target = _code_target(fn)
    code = getattr(target, "__code__", None)
    module = getattr(own, "__module__", "") or ""
    if code is None:
        return _Scan(_qualified(own), module, frozenset({_qualified(own)}), *_EMPTY_CODE)
    namespace = getattr(target, "__globals__", {})
    names, lazy, consts, hmac_kw = _block_facts(code, module.rpartition(".")[0])
    callees = _resolve(names, lambda name: lazy.get(name, namespace.get(name)))
    refs = {_qualified(own)} | {_qualified(obj) for obj in callees}
    if hmac_kw and refs & _HMAC_CAPABLE:
        refs.add(HMAC_KEYWORD)
    return _Scan(
        _qualified(own),
        module,
        frozenset(refs),
        frozenset(names),
        tuple(callees),
        frozenset(consts & (SIGNED_HEADERS | UNSIGNED_HEADERS)),
    )


def _followable(obj: Any) -> bool:
    if not inspect.isfunction(obj):
        return False
    module = obj.__module__ or ""
    if module in _NOT_FOLLOWED_MODULES:
        return False
    return module.startswith("umrs.") or module == __name__  # this file's throwaways


def _helper_refs(fn: Any, depth: int = _VERIFY_DEPTH) -> set[str]:
    """``module.qualname`` of everything one helper reaches (``depth`` deep)."""
    refs: set[str] = set()
    seen: set[int] = set()
    frontier = [(fn, 0)]
    while frontier:
        current, level = frontier.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        scan = _scan(current)
        refs |= scan.refs
        if level < depth:
            frontier.extend((obj, level + 1) for obj in scan.callees if _followable(obj))
    return refs


def _verifies(fn: Any) -> bool:
    """The callable checks the gateway HMAC (itself or in the helpers it
    calls), or it is a verifier module."""
    if _scan(fn).module in _VERIFIER_MODULES:
        return True
    return bool(_helper_refs(fn) & IDENTITY_RESOLVERS.keys())


def _authenticates(fn: Any) -> bool:
    """A ``Depends`` target that verifies the caller's tuple for the request.
    An allowlist inside the verifier modules: only their ``IDENTITY_RESOLVERS``
    name a caller (``get_verified_memberships`` checks a header but names
    nobody; a new helper there must be listed before it counts)."""
    if _scan(fn).module in _VERIFIER_MODULES:
        return _qualified(inspect.unwrap(fn)) in IDENTITY_RESOLVERS
    return _verifies(fn)


def _signs_memberships(fn: Any) -> bool:
    """A verifier-module callable that checks the memberships HMAC in its own
    helpers (only those modules may touch the raw list)."""
    if _scan(fn).module not in _VERIFIER_MODULES:
        return False
    return bool(_helper_refs(fn) & MEMBERSHIP_VERIFIERS.keys())


def _unsigned_headers(fn: Any, headers: frozenset[str], verifies: bool) -> frozenset[str]:
    """Headers no HMAC covers. The legacy ``X-Project-ID`` input of a
    callable that verifies the tuple is its mvp fallback (strict resolvers
    refuse it), not a read; the memberships list of a callable that checks
    its own HMAC (``_signs_memberships``) is signed."""
    unsigned = headers & UNSIGNED_HEADERS
    if verifies:
        unsigned -= {"x-project-id"}
    if MEMBERSHIPS_HEADER in unsigned and _signs_memberships(fn):
        unsigned -= {MEMBERSHIPS_HEADER}
    return unsigned


@dataclass
class _Facts:
    """Everything one route's closure references.

    ``unchecked``: callables that read the signed identity headers (or call
    ``_identity`` without ``verify_hmac=True``) and verify nothing; safe only
    when a ``Depends`` target already verified this request's tuple.
    ``unsigned``: callables reading headers no HMAC covers (role, the legacy
    ``X-Project-ID``, memberships without their own HMAC); never safe.
    """

    refs: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    unchecked: set[str] = field(default_factory=set)
    unsigned: set[str] = field(default_factory=set)

    def add(self, fn: Any, params: frozenset[str] = frozenset()) -> _Scan:
        scan = _scan(fn)
        self.refs |= scan.refs
        self.names |= scan.names
        headers = scan.headers | params
        if not headers and not scan.refs & _IDENTITY_CALLS:
            return scan
        verifies = _verifies(fn)
        if not verifies:
            self.unchecked.add(scan.qualname)
        if _unsigned_headers(fn, headers, verifies):
            self.unsigned.add(scan.qualname)
        return scan


def _walk(roots: Iterable[tuple[Any, frozenset[str]]], depth: int = _DEPTH) -> _Facts:
    """The roots (callable, identity header params) and the umrs helpers
    they call, ``depth`` levels deep."""
    facts = _Facts()
    seen: set[int] = set()
    frontier = [(fn, params, 0) for fn, params in roots]
    while frontier:
        fn, params, level = frontier.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        scan = facts.add(fn, params)
        if level < depth:
            frontier.extend(
                (obj, frozenset(), level + 1) for obj in scan.callees if _followable(obj)
            )
    return facts


def _sub_dependants(dependant: Any) -> Iterator[Any]:
    for sub in dependant.dependencies:
        yield sub
        yield from _sub_dependants(sub)


def _header_params(dependant: Any) -> frozenset[str]:
    aliases = {p.alias.lower() for p in dependant.header_params}
    return frozenset(aliases & (SIGNED_HEADERS | UNSIGNED_HEADERS))


@dataclass
class _RouteFacts:
    facts: _Facts
    depends_verified: bool  # a Depends target verifies the signed tuple


_FACTS: dict[int, tuple[APIRoute, _RouteFacts]] = {}


def _route_facts(route: APIRoute) -> _RouteFacts:
    if id(route) not in _FACTS:
        subs = [sub for sub in _sub_dependants(route.dependant) if sub.call]
        roots = [(d.call, _header_params(d)) for d in (route.dependant, *subs)]
        verified = any(_authenticates(sub.call) for sub in subs)
        _FACTS[id(route)] = (route, _RouteFacts(_walk(roots), verified))
    return _FACTS[id(route)][1]


def _is_guarded(route: APIRoute) -> bool:
    facts = _route_facts(route).facts
    refs = facts.refs - NOT_A_PROJECT_GUARD_FOR.get(_route_key(route), {}).keys()
    return bool(refs & SCOPE_GUARDS.keys() or facts.names & METHOD_GUARDS.keys())


def _resolves_identity(route: APIRoute) -> bool:
    return bool(_route_facts(route).facts.refs & IDENTITY_RESOLVERS.keys())


def _identity_findings(route: APIRoute) -> set[str]:
    """Rule 4 findings for one route (see ``unsigned_identity_routes``)."""
    route_facts = _route_facts(route)
    found = set(route_facts.facts.unsigned)
    if not route_facts.depends_verified:
        found |= route_facts.facts.unchecked
    if not found and _is_guarded(route) and not _resolves_identity(route):
        found.add(NO_RESOLVER)
    return found


# ── field discovery ────────────────────────────────────────────────────────


def _is_project_field(name: str) -> bool:
    lowered = name.lower()
    if lowered in NOT_PROJECT_FIELDS:
        return False
    return "project" in lowered or lowered in PROJECT_WORDS


def _model_field_names(annotation: Any, depth: int = _MODEL_DEPTH) -> set[str]:
    """Field names of a body model, nested models included (``depth`` deep)."""
    names: set[str] = set()
    if depth <= 0:
        return names
    for candidate in (annotation, *typing.get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            for name, info in candidate.model_fields.items():
                names |= {name, info.alias or name}
                names |= _model_field_names(info.annotation, depth - 1)
        elif typing.get_args(candidate):
            names |= _model_field_names(candidate, depth - 1)
    return names


def _declared_fields(dependant: Any) -> set[str]:
    params = [*dependant.query_params, *dependant.path_params]
    names = {p.name for p in params} | {p.alias for p in params}
    for body in dependant.body_params:
        names.add(body.name)
        names |= _model_field_names(body.field_info.annotation)
    return {name for name in names if _is_project_field(name)}


def _requested_project_fields(route: APIRoute) -> set[str]:
    fields = _declared_fields(route.dependant)
    for sub in _sub_dependants(route.dependant):
        if (
            sub.call is not None
            and _qualified(inspect.unwrap(sub.call)) not in IDENTITY_DEPENDENCIES
        ):
            fields |= _declared_fields(sub)
    return fields


def _route_key(route: APIRoute) -> str:
    return f"{','.join(sorted(route.methods))} {route.path}"


# ── the four rules ─────────────────────────────────────────────────────────


def unguarded_project_routes(routes: Iterable[APIRoute]) -> dict[str, list[str]]:
    """Rule 1: route key -> project fields, for routes with no scope guard."""
    found: dict[str, list[str]] = {}
    for route in routes:
        fields = _requested_project_fields(route)
        if fields and not _is_guarded(route):
            found[_route_key(route)] = sorted(fields)
    return found


def routes_without_identity(routes: Iterable[APIRoute]) -> set[str]:
    """Rule 2: routes that resolve no verified caller (allowlisted modules aside)."""
    return {
        _route_key(route)
        for route in routes
        if route.endpoint.__module__ not in MODULE_ALLOWLIST and not _resolves_identity(route)
    }


def modules_without_identity(routes: Iterable[APIRoute]) -> set[str]:
    """Rule 2 (module tier): router modules none of whose routes resolve the caller."""
    resolved: dict[str, bool] = defaultdict(bool)
    for route in routes:
        module = route.endpoint.__module__
        resolved[module] = resolved[module] or _resolves_identity(route)
    return {module for module, ok in resolved.items() if not ok}


def unverified_annotations(routes: Iterable[APIRoute]) -> set[str]:
    """Rule 3: '# auth: verified' handlers that resolve no identity."""
    found: set[str] = set()
    for route in routes:
        doc = inspect.getdoc(inspect.unwrap(route.endpoint)) or ""
        if "# auth: verified" in doc and not _resolves_identity(route):
            found.add(_route_key(route))
    return found


def unsigned_identity_routes(routes: Iterable[APIRoute]) -> dict[str, list[str]]:
    """Rule 4: route key -> callables that read identity without the HMAC.

    A read of an unsigned header (memberships, role, legacy project) always
    counts; a read of the signed headers counts unless a ``Depends`` target
    verified them for this request (a resolver reached only from a deeper,
    possibly conditional helper does not); a scope guard on a route that
    resolves no identity counts as ``<scope guard without a resolver>``.
    """
    found: dict[str, list[str]] = {}
    for route in routes:
        readers = _identity_findings(route)
        if readers:
            found[_route_key(route)] = sorted(readers)
    return found


# ── the real app ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def routes() -> list[APIRoute]:
    """The served route table (same factory and walker as S4 characterization)."""
    from umrs.server import http_sidecar

    original = http_sidecar._build_db_pool
    http_sidecar._build_db_pool = lambda _app: None
    try:
        return _iter_api_routes(http_sidecar._create_app())
    finally:
        http_sidecar._build_db_pool = original


def _mounted_keys(routes: list[APIRoute]) -> set[str]:
    keys = {_route_key(route) for route in routes}
    return keys | {f"{method} {route.path}" for route in routes for method in route.methods}


def test_every_project_taking_route_is_guarded_or_allowlisted(routes):
    found = unguarded_project_routes(routes)
    missing = {key: fields for key, fields in found.items() if key not in ALLOWLIST}
    assert not missing, (
        "routes take a project but reference no scope guard; check the caller "
        "against the requested project (access_scope helpers) or add an "
        f"ALLOWLIST entry with a reason: {missing}"
    )


def test_allowlist_has_no_stale_entries(routes):
    stale = sorted(set(ALLOWLIST) - set(unguarded_project_routes(routes)))
    assert not stale, f"guarded or gone now; remove from ALLOWLIST: {stale}"


def _reached_refs(routes: list[APIRoute], key: str) -> set[str]:
    found = [route for route in routes if _route_key(route) == key]
    return set(_route_facts(found[0]).facts.refs) if found else set()


def test_not_a_project_guard_entries_are_live(routes):
    """Each entry names a mounted route whose closure still reaches the guard."""
    stale = [
        f"{key}: {ref}"
        for key, refs in NOT_A_PROJECT_GUARD_FOR.items()
        for ref in sorted(set(refs) - _reached_refs(routes, key))
    ]
    assert not stale, f"remove from NOT_A_PROJECT_GUARD_FOR: {stale}"
    excluded = {ref for refs in NOT_A_PROJECT_GUARD_FOR.values() for ref in refs}
    assert excluded <= SCOPE_GUARDS.keys()


def test_every_router_module_resolves_identity(routes):
    missing = sorted(modules_without_identity(routes) - set(MODULE_ALLOWLIST))
    assert not missing, f"router modules resolve no verified identity: {missing}"


def test_module_allowlist_has_no_stale_entries(routes):
    stale = sorted(set(MODULE_ALLOWLIST) - modules_without_identity(routes))
    assert not stale, (
        f"a route resolves identity now; list the rest in ROUTE_IDENTITY_ALLOWLIST and drop: {stale}"
    )


def test_every_route_resolves_identity(routes):
    missing = sorted(routes_without_identity(routes) - set(ROUTE_IDENTITY_ALLOWLIST))
    assert not missing, (
        "routes resolve no verified caller (HMAC resolver, JWT, lease or admin JWT); "
        f"add one or a ROUTE_IDENTITY_ALLOWLIST entry with a reason: {missing}"
    )


def test_route_identity_allowlist_has_no_stale_entries(routes):
    stale = sorted(set(ROUTE_IDENTITY_ALLOWLIST) - routes_without_identity(routes))
    assert not stale, f"resolves identity now; remove from ROUTE_IDENTITY_ALLOWLIST: {stale}"


def test_auth_verified_annotations_are_true(routes):
    missing = sorted(unverified_annotations(routes) - set(ANNOTATION_ALLOWLIST))
    assert not missing, f"'# auth: verified' but no identity resolver: {missing}"


def test_annotation_allowlist_has_no_stale_entries(routes):
    stale = sorted(set(ANNOTATION_ALLOWLIST) - unverified_annotations(routes))
    assert not stale, f"verified now; remove from ANNOTATION_ALLOWLIST: {stale}"


def _unlisted_readers(found: dict[str, list[str]]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for key, readers in found.items():
        unlisted = [r for r in readers if r not in UNSIGNED_READER_ALLOWLIST]
        if NO_RESOLVER in unlisted and key in UNSIGNED_READER_ALLOWLIST:
            unlisted.remove(NO_RESOLVER)
        if unlisted:
            missing[key] = unlisted
    return missing


def test_identity_is_read_only_through_the_hmac(routes):
    missing = _unlisted_readers(unsigned_identity_routes(routes))
    assert not missing, (
        "identity read without the gateway HMAC (use request_caller_scope / get_caller, or "
        f"verify_hmac=True) or an UNSIGNED_READER_ALLOWLIST entry: {missing}"
    )


def test_unsigned_reader_allowlist_has_no_stale_entries(routes):
    found = unsigned_identity_routes(routes)
    seen = {reader for readers in found.values() for reader in readers}
    seen |= {key for key, readers in found.items() if NO_RESOLVER in readers}
    stale = sorted(set(UNSIGNED_READER_ALLOWLIST) - seen)
    assert not stale, f"verified now; remove from UNSIGNED_READER_ALLOWLIST: {stale}"


def test_known_gaps_name_mounted_routes(routes):
    mounted = _mounted_keys(routes)
    gone = sorted(key for key in KNOWN_GAPS if key not in mounted)
    assert not gone, f"KNOWN_GAPS entries name no mounted route; revisit: {gone}"


def test_vocabulary_names_real_functions():
    """Every guard and resolver key names an importable function, so a
    rename cannot silently empty the vocabulary."""
    keys = [
        *SCOPE_GUARDS,
        *IDENTITY_RESOLVERS,
        *IDENTITY_DEPENDENCIES,
        *_IDENTITY_CALLS,
        *MEMBERSHIP_VERIFIERS,
    ]
    missing = [key for key in keys if key != HMAC_KEYWORD and not _importable(key)]
    assert not missing, f"no such function: {missing}"


def _importable(qualname: str) -> bool:
    module, _, name = qualname.rpartition(".")
    try:
        return inspect.isfunction(getattr(importlib.import_module(module), name, None))
    except ImportError:
        return False


_TABLES = (
    ALLOWLIST,
    MODULE_ALLOWLIST,
    ROUTE_IDENTITY_ALLOWLIST,
    ANNOTATION_ALLOWLIST,
    UNSIGNED_READER_ALLOWLIST,
    KNOWN_GAPS,
)


def test_every_allowlist_reason_is_one_line():
    bad = [key for table in _TABLES for key, why in table.items() if not why.strip() or "\n" in why]
    assert not bad, f"each entry needs a one-line reason: {bad}"


def _unmarked(reasons) -> list[str]:
    return [why for why in reasons if not why.startswith("DEFERRED H-1: ")]


def test_deferred_entries_are_marked():
    gaps = [why for table in _TABLES for why in table.values() if "DEFERRED" in why]
    assert _unmarked(gaps) == []
    assert _unmarked(DEFERRED_ELSEWHERE) == []
    assert _unmarked(KNOWN_GAPS.values()) == []


# ── the guard catches what it should (throwaway routers) ──────────────────


def _routes_of(router: APIRouter) -> list[APIRoute]:
    app = FastAPI()
    app.include_router(router)
    return _iter_api_routes(app)


class _NamespaceBody(BaseModel):
    namespace: str
    text: str = ""


class _Scope(BaseModel):
    project_id: str


class _NestedBody(BaseModel):
    scope: _Scope


class _ManyBody(BaseModel):
    project_ids: list[str]
    namespaces: list[str] = []


def _project_from_query(project_id: str = Query(...)) -> str:
    return project_id


def _identified_project(request: Request, project_id: str = Query(...)) -> str:
    """Resolves the caller, then hands the requested project on unchecked."""
    from umrs.api._scope_identity import request_caller_scope

    request_caller_scope(request)
    return project_id


def _enforce_rate_limit() -> None:
    """A guard-shaped name that checks no project."""


def _leaky_router() -> APIRouter:
    router = APIRouter(prefix="/umrs/throwaway")

    @router.get("/{project_id}/rows")
    def leak_rows(project_id: str) -> dict[str, str]:
        return {"project_id": project_id}

    @router.post("/notes")
    def leak_write(body: _NamespaceBody) -> dict[str, str]:
        return {"namespace": body.namespace}

    @router.get("/via-dependency")
    def leak_dependency(project: str = Depends(_project_from_query)) -> dict[str, str]:
        return {"project": project}

    @router.get("/identified-dependency")
    def leak_identified(project: str = Depends(_identified_project)) -> dict[str, str]:
        return {"project": project}

    @router.get("/identity-only")
    def leak_after_identity(request: Request, project_id: str) -> dict[str, str]:
        from umrs.api._scope_identity import request_caller_scope

        request_caller_scope(request)
        return {"project_id": project_id}

    @router.get("/rate-limited")
    def leak_rate_limited(project_id: str) -> dict[str, str]:
        _enforce_rate_limit()
        return {"project_id": project_id}

    @router.post("/nested")
    def leak_nested(body: _NestedBody) -> dict[str, str]:
        return {"project_id": body.scope.project_id}

    @router.post("/many")
    def leak_many(body: _ManyBody) -> dict[str, list[str]]:
        return {"project_ids": body.project_ids}

    return router


def _throwaway_routes() -> list[APIRoute]:
    router = _leaky_router()

    @router.get("/guarded/{project_id}")
    def guarded(project_id: str, request: Request) -> dict[str, bool]:
        from umrs.api._scope_identity import request_caller_scope
        from umrs.knowledge.access_scope import project_readable

        return {"ok": project_readable(project_id, request_caller_scope(request))}

    return _routes_of(router)


def test_guard_flags_a_new_unguarded_project_route():
    found = unguarded_project_routes(_throwaway_routes())
    assert found == {
        "GET /umrs/throwaway/{project_id}/rows": ["project_id"],
        "POST /umrs/throwaway/notes": ["namespace"],
        "GET /umrs/throwaway/via-dependency": ["project_id"],
        "GET /umrs/throwaway/identified-dependency": ["project_id"],
        "GET /umrs/throwaway/identity-only": ["project_id"],
        "GET /umrs/throwaway/rate-limited": ["project_id"],
        "POST /umrs/throwaway/nested": ["project_id"],
        "POST /umrs/throwaway/many": ["namespaces", "project_ids"],
    }


def test_guard_flags_a_router_module_without_identity():
    unguarded = [
        r for r in _throwaway_routes() if "/guarded/" not in r.path and "identi" not in r.path
    ]
    assert modules_without_identity(unguarded) == {__name__}


def test_guard_flags_a_bare_route_in_an_identified_module():
    found = routes_without_identity(_throwaway_routes())
    assert "GET /umrs/throwaway/{project_id}/rows" in found
    assert "GET /umrs/throwaway/guarded/{project_id}" not in found
    assert "GET /umrs/throwaway/identity-only" not in found


def get_caller(request: Request) -> str:
    """A raw-header helper that borrows a trusted resolver's name."""
    return request.headers.get("x-user-id", "")


def project_readable(project_id: str, caller: str) -> bool:
    """A predicate that borrows a trusted guard's name and checks nothing."""
    return bool(project_id and caller)


def test_guard_matches_objects_not_names():
    router = APIRouter()

    @router.get("/umrs/throwaway/{project_id}/namesake")
    def namesake(project_id: str, caller: str = Depends(get_caller)) -> dict[str, bool]:
        return {"ok": project_readable(project_id, caller)}

    found = _routes_of(router)
    key = "GET /umrs/throwaway/{project_id}/namesake"
    assert unguarded_project_routes(found) == {key: ["project_id"]}
    assert routes_without_identity(found) == {key}
    assert unsigned_identity_routes(found) == {key: [f"{__name__}.get_caller"]}


def test_guard_flags_a_false_auth_verified_annotation():
    router = APIRouter()

    @router.get("/umrs/throwaway/claim")
    def claims_verified() -> dict[str, str]:
        """Says it is verified but reads no identity.
        # auth: verified
        """
        return {}

    assert unverified_annotations(_routes_of(router)) == {"GET /umrs/throwaway/claim"}


def test_guard_flags_a_scope_check_on_unsigned_headers():
    router = APIRouter()

    @router.get("/umrs/throwaway/{project_id}/trusting")
    def trusting(project_id: str, request: Request) -> dict[str, bool]:
        from umrs.api._identity import request_caller
        from umrs.knowledge.access_scope import project_readable

        return {"ok": project_readable(project_id, request_caller(request))}

    trusting_routes = _routes_of(router)
    assert unguarded_project_routes(trusting_routes) == {}
    assert unsigned_identity_routes(trusting_routes) == {
        "GET /umrs/throwaway/{project_id}/trusting": [f"{__name__}.{trusting.__qualname__}"]
    }
    assert unsigned_identity_routes(_throwaway_routes()) == {}


def _raw_scope(request: Request, project_id: str) -> bool:
    """The rag._require_rag_scope shape: identity from raw headers."""
    from umrs.knowledge.access_scope import CallerScope, project_readable

    caller = CallerScope(
        user_id=request.headers.get("x-user-id", ""),
        project_id=request.headers.get("x-verified-project-id", ""),
    )
    return project_readable(project_id, caller)


def _graph_check(request: Request, graph: str) -> bool:
    """A verified resolver reached only on one branch."""
    from umrs.api._scope_identity import request_caller_scope

    return bool(graph) and bool(request_caller_scope(request).user_id)


def test_guard_flags_a_resolver_reached_only_on_a_branch():
    router = APIRouter()

    @router.get("/umrs/throwaway/{project_id}/branchy")
    def branchy(project_id: str, request: Request, graph: str = "") -> dict[str, bool]:
        if graph:
            _graph_check(request, graph)
        return {"ok": _raw_scope(request, project_id)}

    found = _routes_of(router)
    assert _resolves_identity(found[0])  # the old rule 4 passed this route
    assert unsigned_identity_routes(found) == {
        "GET /umrs/throwaway/{project_id}/branchy": [f"{__name__}._raw_scope"]
    }


def _verified_project(request: Request) -> str:
    from umrs.api._scope_identity import request_caller_scope

    return request_caller_scope(request).project_id


def test_raw_reads_after_a_verifying_dependency_are_signed():
    router = APIRouter()

    @router.get("/umrs/throwaway/{project_id}/after-depends")
    def after_depends(
        project_id: str, request: Request, own: str = Depends(_verified_project)
    ) -> dict[str, bool]:
        return {"ok": _raw_scope(request, project_id) and bool(own)}

    @router.get("/umrs/throwaway/memberships")
    def memberships(request: Request, own: str = Depends(_verified_project)) -> dict[str, str]:
        return {"m": request.headers.get("x-verified-project-memberships", "") + own}

    found = unsigned_identity_routes(_routes_of(router))
    assert found == {"GET /umrs/throwaway/memberships": [f"{__name__}.{memberships.__qualname__}"]}


def test_verify_hmac_true_counts_as_identity():
    router = APIRouter()

    @router.get("/umrs/throwaway/signed")
    def signed(request: Request) -> dict[str, str]:
        from umrs.api._identity import verified_identity

        return {"user": verified_identity(request, require_project=True, verify_hmac=True)[0]}

    @router.get("/umrs/throwaway/unsigned")
    def unsigned(request: Request) -> dict[str, str]:
        from umrs.api._identity import verified_identity

        return {"user": verified_identity(request, require_project=True)[0]}

    found = _routes_of(router)
    assert routes_without_identity(found) == {"GET /umrs/throwaway/unsigned"}
    assert unsigned_identity_routes(found) == {
        "GET /umrs/throwaway/unsigned": [f"{__name__}.{unsigned.__qualname__}"]
    }


def _raw_user(request: Request) -> str:
    return request.headers.get("x-user-id", "")


def test_signed_memberships_are_not_a_principal():
    """The gateway-signed memberships reader is signed for the list, but it
    authenticates nobody: a raw X-User-ID read beside it is still a finding,
    and a memberships read that skips the HMAC still is too (2026-10-18)."""
    from umrs.api._verified_headers import get_verified_memberships

    router = APIRouter()

    @router.get("/umrs/throwaway/signed-list")
    def signed_list(
        request: Request,
        own: str = Depends(_verified_project),
        listed: frozenset[str] = Depends(get_verified_memberships),
    ) -> dict[str, str]:
        return {"m": ",".join(sorted(listed)) + own}

    @router.get("/umrs/throwaway/list-only")
    def list_only(
        request: Request, listed: frozenset[str] = Depends(get_verified_memberships)
    ) -> dict[str, str]:
        return {"user": _raw_user(request), "m": ",".join(sorted(listed))}

    @router.get("/umrs/throwaway/raw-list")
    def raw_list(
        request: Request,
        own: str = Depends(_verified_project),
        listed: frozenset[str] = Depends(get_verified_memberships),
    ) -> dict[str, str]:
        return {"m": request.headers.get("x-verified-project-memberships", "") + own}

    found = unsigned_identity_routes(_routes_of(router))
    assert found == {
        "GET /umrs/throwaway/list-only": [f"{__name__}._raw_user"],
        "GET /umrs/throwaway/raw-list": [f"{__name__}.{raw_list.__qualname__}"],
    }


def _fallback_memberships(request: Request) -> str:
    """Verify the list when its HMAC is present, else trust it raw (the
    'old gateways' fallback): reaching verify_memberships is not enough."""
    from umrs.api._verified_headers import trusted_memberships

    raw = request.headers.get("x-verified-project-memberships", "")
    if request.headers.get("x-verified-memberships-hmac"):
        signed = trusted_memberships(
            strict=True,
            secret=None,
            presented_memberships=raw,
            presented_hmac=request.headers.get("x-verified-memberships-hmac"),
            user_id=None,
            project_id=None,
            is_admin=False,
        )
        return ",".join(sorted(signed))
    return raw


def test_a_conditional_memberships_verifier_is_unsigned():
    """Only the verifier modules may read the raw memberships list: a helper
    elsewhere that verifies on one branch and returns the raw list on the
    other is a finding (2026-10-18 review)."""
    router = APIRouter()

    @router.get("/umrs/throwaway/fallback-list")
    def fallback_list(request: Request, own: str = Depends(_verified_project)) -> dict[str, str]:
        return {"m": _fallback_memberships(request) + own}

    found = unsigned_identity_routes(_routes_of(router))
    assert found == {"GET /umrs/throwaway/fallback-list": [f"{__name__}._fallback_memberships"]}


def test_only_listed_resolvers_in_a_verifier_module_authenticate():
    """A verifier-module helper outside ``IDENTITY_RESOLVERS`` (here the
    strict-mode probe) used as a Depends does not excuse a raw identity read
    (2026-10-18 review: the principal rule is an allowlist)."""
    from umrs.api._verified_headers import _strict_mode_env

    router = APIRouter()

    @router.get("/umrs/throwaway/mode-only")
    def mode_only(request: Request, strict: bool = Depends(_strict_mode_env)) -> dict[str, str]:
        return {"user": _raw_user(request), "strict": str(strict)}

    found = unsigned_identity_routes(_routes_of(router))
    assert found == {"GET /umrs/throwaway/mode-only": [f"{__name__}._raw_user"]}
