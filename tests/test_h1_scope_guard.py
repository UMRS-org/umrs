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
   conditional helper does not excuse it (``rag._require_rag_scope`` on
   /upload and /retrieve). Headers no HMAC covers (memberships, role, legacy
   ``X-Project-ID`` outside a verifying resolver) are always findings.
   ``UNSIGNED_READER_ALLOWLIST`` is keyed by the reader, where the fix lands.

Not proven by any rule: object-id routes (Appendix A.2 class: load the row,
compare its project or owner, a foreign id reads as not found). The fix
lanes' tests cover the ones they fixed; the known open ones are listed by
hand in ``KNOWN_GAPS`` or the allowlists. A guard anywhere in the closure
covers every project field of the route (rule 1 cannot tie a guard to a
field; ``KNOWN_GAPS`` names one such case). Raw ``request.query_params`` /
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
    "umrs.api.rag._require_rag_scope": "project compare (identity: see rule 4)",
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
}
# CallerIdentity methods (``_verified_headers``): matched by attribute name,
# since ``caller.scope_filter(...)`` names no module object.
METHOD_GUARDS: dict[str, str] = {
    "scope_filter": "CallerIdentity: non-admins read only their own project",
    "bind_for_write": "CallerIdentity: declared project must be the caller's",
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
    "umrs.api.models.get_access_context": "HMAC-verified user/admin",
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
UNSIGNED_HEADERS = frozenset({"x-verified-project-memberships", "x-user-role", "x-project-id"})

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

_UNSIGNED_MEMBERSHIPS = (
    "trusts the unsigned X-Verified-Project-Memberships header (sign_claims covers"
    " user, project and admin only); needs gateway-signed memberships"
)
_UNCHECKED_JOB_PROJECT = (
    "Appendix A.1: ?caller_project_id (the durable job partition) is not checked"
    " against the verified caller, and the body project drives the policy check;"
    " no H-1 lane took models.py"
)
_ADMIN_TOKEN = "ADMIN-ONLY: x-admin-token must equal UMRS_ADMIN_TOKEN"
_MODEL_ADMIN = "ADMIN-ONLY: ctx.is_admin from the HMAC-verified get_access_context"

ALLOWLIST: dict[str, str] = {
    "POST /umrs/a2l/ledger/artifacts": (
        "ADMIN-ONLY: root artifacts are admin/ingestion only (is_admin from the HMAC-verified get_caller_identity)"
    ),
    "GET /umrs/assets": (
        "IDENTITY-ONLY: a non-admin's project_id is replaced by the verified project; rows pass caller visibility"
    ),
    "POST /umrs/assets/patch": (
        "IDENTITY-ONLY: a non-admin's body project_id is replaced by the verified project (same rule as GET /umrs/assets)"
    ),
    "POST /umrs/buckets": (
        "IDENTITY-ONLY: project is a name segment of the caller's own <user>.<project>.<alias> bucket; owner is the"
        " verified caller (the segment is not checked: see KNOWN_GAPS)"
    ),
    "GET /umrs/circles/{project_id}": (
        "IDENTITY-ONLY: unaffiliated callers get only their own circle rows; the roster needs admin, project owner"
        " or affiliation (existence still shows: see KNOWN_GAPS)"
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
    "GET /umrs/jobs/stats": (
        "DEFERRED H-1: Appendix A.1 ?project_id unchecked with no identity; unreachable today because"
        " GET /umrs/jobs/{job_id} is registered first; delete the route or scope it"
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
        "NOT-SCOPE: project_id is accepted but unused; the route only queries public academic sources"
    ),
    "POST /umrs/knowledge/restore": (
        "ADMIN-ONLY: _require_operator needs an HMAC-verified admin (request_caller_scope) in every mode"
    ),
    "GET /umrs/models": "DEFERRED H-1: the model ACL context " + _UNSIGNED_MEMBERSHIPS,
    "POST /umrs/models": "DEFERRED H-1: acl.enforce_create " + _UNSIGNED_MEMBERSHIPS,
    "POST /umrs/models/system": _MODEL_ADMIN,
    "PATCH /umrs/models/{model_name}/policy": _MODEL_ADMIN,
    "POST /umrs/models/{model_id}/publish": (
        "DEFERRED H-1: _check_publish_allowed admits target_project_id and " + _UNSIGNED_MEMBERSHIPS
    ),
    "POST /umrs/models/stage": "DEFERRED H-1: " + _UNCHECKED_JOB_PROJECT,
    "POST /umrs/models/{model_name}/download": "DEFERRED H-1: " + _UNCHECKED_JOB_PROJECT,
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
    "POST /umrs/projects": (
        "DEFERRED H-1: creates only for the caller (for_user is admin-only), but an id already taken returns"
        " that project's row, which GET /umrs/projects/{project_id} hides as 404 (guard-lane finding)"
    ),
    "PUT /umrs/projects/{project_id}/budget": (
        "OWNER-OR-ADMIN: inline check on the loaded project row (a non-owner gets 403, not 404: see KNOWN_GAPS)"
    ),
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

_HEADER_TRUST = (
    "reads X-User-ID / X-Verified-Project-ID / X-Verified-Is-Admin without the gateway HMAC,"
    " so a direct sidecar caller can claim any project or admin"
)
_RAG = "DEFERRED H-1: rag._require_rag_scope " + _HEADER_TRUST

# Whole modules: every route lacks a resolver (stale once one route has one).
MODULE_ALLOWLIST: dict[str, str] = {
    "umrs.server.http_sidecar": "INFRA: /health, /metrics and /readiness probes, open by design",
    "umrs.api.llm": (
        "DEFERRED H-1: _get_caller_identity (verified_identity, no verify_hmac) and _endpoint_caller "
        + _HEADER_TRUST
    ),
    "umrs.api.chat": (
        "DEFERRED H-1: _get_caller_identity and the rag._require_rag_scope it calls "
        + _HEADER_TRUST
    ),
    "umrs.api.grants": "DEFERRED H-1: _caller " + _HEADER_TRUST,
    "umrs.api.skills": (
        "DEFERRED H-1: _caller_identity (request_identity, no verify_hmac) " + _HEADER_TRUST
    ),
}

# Single routes in modules that otherwise resolve the caller.
ROUTE_IDENTITY_ALLOWLIST: dict[str, str] = {
    "GET /umrs/a2l/ledger/health": "PROBE: static liveness for r4 artifact",
    "GET /umrs/mcp/health": "PROBE: server and tool counts only",
    "GET /umrs/ops/ping": "PROBE: readiness for umrs ping, no data",
    "GET /umrs/web/dashboard/health": "PROBE: dashboard liveness, open by design",
    "GET /umrs/web/dashboard/metrics": "PROBE: Prometheus text, open by design",
    "GET /umrs/zmq/server-key": "PUBLIC: the CURVE server public key is not secret",
    "GET /umrs/components/chunkers": "PUBLIC: static registry of chunking methods",
    "GET /umrs/components/embedders": "PUBLIC: static registry of embedders",
    "GET /umrs/components/rerankers": "PUBLIC: static registry of rerankers",
    "GET /umrs/models/system": "PUBLIC: system-scope models are readable by every user (U-817)",
    "DELETE /umrs/ops/audit/retention": _ADMIN_TOKEN,
    "GET /umrs/ops/governance/analytics": _ADMIN_TOKEN,
    "GET /umrs/ops/governance/orphans": _ADMIN_TOKEN,
    "DELETE /umrs/ops/governance/orphans": _ADMIN_TOKEN,
    "POST /umrs/ops/governance/ownership/transfer": _ADMIN_TOKEN,
    "PUT /umrs/ops/governance/quotas": _ADMIN_TOKEN,
    "GET /umrs/ops/connect-info": (
        "DEFERRED H-1: discovery is open by design, but with UMRS_EMIT_S3CFG=1 a bare X-User-ID returns that"
        " user's s3cfg; resolve_caller before the s3cfg branch (or drop the D-16 emission)"
    ),
    "GET /umrs/ops/timeline": (
        "DEFERRED H-1: the decorator is bound to the helper _enum_value, so the path answers 422 or 500"
        " without auth and ops_timeline (require_ops_admin) is unreachable"
    ),
    "GET /umrs/files/download-status": (
        "DEFERRED H-1: any job's status and readiness by ?job_id with no identity (A.2 class); add"
        " Depends(get_project_id) and answer 404 when job.project_id differs"
    ),
    "GET /umrs/jobs/stats": "DEFERRED H-1: no identity; dead route (see ALLOWLIST)",
    "POST /umrs/knowledge/external/search": (
        "DEFERRED H-1: anonymous server-side egress to the fixed academic sources; add Depends(get_project_id)"
        " so the egress is attributable"
    ),
    "GET /umrs/rag/corpus-access": _RAG,
    "PUT /umrs/rag/corpus-access": _RAG,
    "POST /umrs/rag/query": _RAG,
    "GET /umrs/rag/documents": _RAG,
    "GET /umrs/rag/documents/{document_id}": _RAG,
    "DELETE /umrs/rag/documents/{document_id}": _RAG,
    "GET /umrs/rag/stats": _RAG,
}

# ── rule 3 allowlist: '# auth: verified' handlers without a resolver ──────

ANNOTATION_ALLOWLIST: dict[str, str] = {
    "GET /umrs/grants": "DEFERRED H-1: grants._caller " + _HEADER_TRUST,
    "POST /umrs/grants": "DEFERRED H-1: grants._caller " + _HEADER_TRUST,
    "GET /umrs/grants/mine": "DEFERRED H-1: grants._caller " + _HEADER_TRUST,
    "POST /umrs/grants/{grant_id}/revoke": "DEFERRED H-1: grants._caller " + _HEADER_TRUST,
    "GET /umrs/rag/corpus-access": _RAG,
    "PUT /umrs/rag/corpus-access": _RAG,
}

# ── rule 4 allowlist: identity read without the HMAC ──────────────────────
# Keyed by the reader (module.qualname): the defect lives there, and one fix
# clears every route it serves. A NO_RESOLVER finding is keyed by route.

_MEMBERSHIPS_READER = "DEFERRED H-1: " + _UNSIGNED_MEMBERSHIPS

UNSIGNED_READER_ALLOWLIST: dict[str, str] = {
    "umrs.api.rag._get_caller_identity": "DEFERRED H-1: verified_identity without verify_hmac",
    "umrs.api.rag._require_rag_scope": (
        "DEFERRED H-1: raw X-Verified-Is-Admin; /upload, /retrieve and /passages reach request_caller_scope"
        " only on graph-selection branches, so their project scope is unsigned too"
    ),
    "umrs.api.rag._query_via_spine": "DEFERRED H-1: raw X-Verified-Is-Admin",
    "umrs.api.rag.put_corpus_access": "DEFERRED H-1: raw X-Verified-Is-Admin",
    "umrs.api.rag_retrieve.is_admin": (
        "DEFERRED H-1: raw X-Verified-Is-Admin gates include_ambient and node_visible on /retrieve and /passages"
    ),
    "umrs.api.chat._get_caller_identity": "DEFERRED H-1: verified_identity without verify_hmac",
    "umrs.api.llm._get_caller_identity": "DEFERRED H-1: verified_identity without verify_hmac",
    "umrs.api.llm._endpoint_caller": "DEFERRED H-1: raw X-Verified-Is-Admin",
    "umrs.api.llm._build_upstream_headers": (
        "DEFERRED H-1: forwards raw X-User-ID / X-Project-ID upstream (take them from the verified tuple)"
    ),
    "umrs.api.skills._caller_identity": "DEFERRED H-1: request_identity without verify_hmac",
    "umrs.api.grants._caller": "DEFERRED H-1: " + _HEADER_TRUST,
    "umrs.api.ops.ops_connect_info": (
        "DEFERRED H-1: bare X-User-ID selects the s3cfg returned under UMRS_EMIT_S3CFG=1"
    ),
    "umrs.api.models.get_access_context": _MEMBERSHIPS_READER,
    "umrs.api.workflows.trigger_workflow": (
        "DEFERRED H-1: the declared-project check " + _UNSIGNED_MEMBERSHIPS
    ),
    "umrs.api.workflows.trigger_run_definition": (
        "DEFERRED H-1: the declared-project check " + _UNSIGNED_MEMBERSHIPS
    ),
}

# ── deferred route gaps the rules above cannot see ─────────────────────────
# Object-id routes (Appendix A.2 class: load by id, compare the row's project
# or owner, a foreign id reads as not found) are not proven by any rule; the
# fix lanes' tests cover the ones they fixed, and the known open ones are
# listed here by hand.
# Checked and closed, so not listed: PATCH /umrs/verification/findings/{key}
# (knowledge-objects lane question): get_caller -> resolve_caller never
# returns None in strict mode, so only a verified admin transitions there.

KNOWN_GAPS: dict[str, str] = {
    "GET /umrs/web/dashboard/jobs": (
        "DEFERRED H-1: the /umrs/web static shell (app.js) calls the dashboard API without identity, so strict"
        " mode now answers 401; needs an nginx /gw/web/ block with auth_request and a /gw/ API base (streams lane)"
    ),
    "GET /umrs/web/dashboard/overview": (
        "DEFERRED H-1: job counts are zeros for every caller (no JobStore.count_jobs); not a scope gap (streams lane)"
    ),
    "GET /umrs/web/dashboard/models": (
        "DEFERRED H-1: shows only the active signed project's models (memberships are left out because the"
        " gateway HMAC does not cover them); revisit once the gateway signs memberships"
    ),
    "GET /umrs/projects/{project_id}": (
        "DEFERRED H-1: a project reachable only through the unsigned memberships header reads 404, and the r4"
        " client (_project_data_class) fails open on it; r4 must fail closed on 403/404 (projects-sessions lane)"
    ),
    "GET /umrs/circles/{project_id}": (
        "DEFERRED H-1: an unknown project answers 404 but another existing project 200 with an empty list,"
        " which shows that it exists; answer 404 to unaffiliated callers (projects-sessions lane)"
    ),
    "PUT /umrs/projects/{project_id}/budget": (
        "DEFERRED H-1: a non-owner gets 403 'only the owner or an admin', not 404, while GET hides the project"
        " (projects-sessions lane)"
    ),
    "POST /umrs/buckets": (
        "DEFERRED H-1: any project segment is accepted, and PUT /umrs/buckets/{bucket}/access later emits"
        " bucket.access_changed under that project; bind the segment or the audit project to the caller (files lane)"
    ),
    "POST /umrs/endpoints/provision": (
        "DEFERRED H-1: rule 1 passes on _authorize_model_entry_for_caller, but mounts.project_data/rag_index/"
        "mcp_servers sources are checked only against the site-wide UMRS_ENDPOINT_ALLOWED_MOUNT_PREFIXES"
        " (/home, /projects, ...), not the caller's project; bind non-admin sources to the caller (endpoints)"
    ),
    "POST /umrs/companion/workers": (
        "DEFERRED H-1: an identified caller can still register an ownerless worker (owner_user_id may be empty);"
        " require owner == caller or default it (projects-sessions lane)"
    ),
    "POST /umrs/companion/approvals/{approval_id}/decision": (
        "DEFERRED H-1: a non-owner of an owned approval gets 401 'approval ownership mismatch', not 404,"
        " which discloses that it exists (projects-sessions lane)"
    ),
    "POST /umrs/a2l/resolve": (
        "DEFERRED H-1: knowledge/a2l_bridge.py calls /gw/a2l/{resolve,artifacts} with no identity (nginx sends"
        " them to the ledger); dormant: remove it or give it a service identity (lineage lane)"
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
    "DEFERRED H-1: parent tools/route_inventory.py still whitelists the '/umrs/web' prefix as open; narrow it"
    " to the static mount, /umrs/web/dashboard/health and /metrics (streams lane)",
    "DEFERRED H-1: demos/28_egress_audit/README.md posts without identity; strict mode needs a signed admin"
    " (registries lane)",
    "DEFERRED H-1: experiments status filter errors are a plain 400 with no typed code (no"
    " experiments.bad_request in r4-contracts) (registries lane)",
    "DEFERRED H-1: plan status row, gap analysis and wave ledger for the H-1 item 5/7 and A.2 fixes"
    " (orchestrator commit)",
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


def _constant_keyword(instructions: list[dis.Instruction], index: int, name: str) -> Any:
    """The constant value pushed for keyword ``name`` of the call whose
    keyword-names tuple is ``instructions[index]`` (``None`` if not constant)."""
    kwnames = instructions[index].argval
    values = instructions[max(index - len(kwnames), 0) : index]
    if len(values) != len(kwnames) or any(v.opname != "LOAD_CONST" for v in values):
        return None
    return values[kwnames.index(name)].argval


def _passes_verify_hmac(instructions: list[dis.Instruction]) -> bool:
    """A call in this block passes the constant keyword ``verify_hmac=True``.

    The keyword-names tuple follows the pushed keyword values (3.10
    ``CALL_FUNCTION_KW``, 3.11/3.12 ``KW_NAMES``, 3.13 ``CALL_KW``); only
    constant keyword values are read, so write the call with literals.
    """
    return any(
        ins.opname in ("LOAD_CONST", "KW_NAMES")
        and isinstance(ins.argval, tuple)
        and "verify_hmac" in ins.argval
        and _constant_keyword(instructions, index, "verify_hmac") is True
        for index, ins in enumerate(instructions)
    )


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
        hmac_kw = hmac_kw or _passes_verify_hmac(instructions)
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


def _unsigned_headers(headers: frozenset[str], verifies: bool) -> frozenset[str]:
    """Headers no HMAC covers. The legacy ``X-Project-ID`` input of a
    callable that verifies the tuple is its mvp fallback (strict resolvers
    refuse it), not a read."""
    unsigned = headers & UNSIGNED_HEADERS
    return unsigned - {"x-project-id"} if verifies else unsigned


@dataclass
class _Facts:
    """Everything one route's closure references.

    ``unchecked``: callables that read the signed identity headers (or call
    ``_identity`` without ``verify_hmac=True``) and verify nothing; safe only
    when a ``Depends`` target already verified this request's tuple.
    ``unsigned``: callables reading headers no HMAC covers (memberships,
    role, the legacy ``X-Project-ID``); never safe.
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
        if _unsigned_headers(headers, verifies):
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
        verified = any(_verifies(sub.call) for sub in subs)
        _FACTS[id(route)] = (route, _RouteFacts(_walk(roots), verified))
    return _FACTS[id(route)][1]


def _is_guarded(route: APIRoute) -> bool:
    facts = _route_facts(route).facts
    return bool(facts.refs & SCOPE_GUARDS.keys() or facts.names & METHOD_GUARDS.keys())


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
    keys = [*SCOPE_GUARDS, *IDENTITY_RESOLVERS, *IDENTITY_DEPENDENCIES, *_IDENTITY_CALLS]
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
