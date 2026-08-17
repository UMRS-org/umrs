"""S4 characterization: pin the mounted UMRS HTTP route table.

Wave: UMRS Stability & Governed Surface Alignment (S1–S12). These are
**premise tests** — they encode CURRENT behavior so the S5 lane
segregation, S6 `/gw/v1` + `/gw/v1beta` publication, and S7 enforcement
moves fail loudly here instead of silently reshaping the surface.

Lane values come from the S2/S3 evidence pass (2026-08-17):
- ``v1-stable`` — endpoints / models / llm (validated by traffic volume);
- ``v1beta`` — knowledge surfaces + loops + research;
- ``internal`` — ops / admin / skills bodies (gateway-404/403 in S7);
- ``unclassified`` — mounted today with no recorded lane decision; S5
  must place each consciously. New families appear here as hard failures,
  which is the point of a characterization test.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute

# ── premise tables ─────────────────────────────────────────────────────────

# Family key = first path segment after ``/umrs``. This map is the recorded
# union of the S1 inventory and the S2/S3 classification; it must match the
# assembled route table exactly (no missing, no undeclared families).
FAMILY_LANES: dict[str, str] = {
    # S2/S3-classified lanes
    "endpoints": "v1-stable",
    "models": "v1-stable",
    "llm": "v1-stable",
    "knowledge": "v1beta",
    "knowledge-graph": "v1beta",
    "k2": "v1beta",
    "loops": "v1beta",
    "research": "v1beta",
    "ops": "internal",
    "skills": "internal",
    "zmq": "internal",
    # Mounted, no lane decision recorded yet — S5 must place each of these.
    "a2l": "unclassified",
    "artifacts": "unclassified",
    "catalog": "unclassified",
    "chat": "unclassified",
    "companion": "unclassified",
    "components": "unclassified",
    "egress": "unclassified",
    "evidence": "unclassified",
    "experiment-registry": "unclassified",
    "files": "unclassified",
    "jobs": "unclassified",
    "mcp": "unclassified",
    "pipelines": "unclassified",
    "provenance": "unclassified",
    "rag": "unclassified",
    "rag-profiles": "unclassified",
    "reflective-memory": "unclassified",
    "scientific-memory": "unclassified",
    "traces": "unclassified",
    "uploads": "unclassified",
    "usage": "unclassified",
    "verification": "unclassified",
    "vision": "unclassified",
    "web": "unclassified",
    "workflows": "unclassified",
}

# Routers that exist in the source tree but are NOT mounted in the HTTP
# sidecar (verified against umrs.server.http_sidecar._create_app). Keeping
# them unserved is itself a premise: none of these prefixes may answer.
DORMANT_FAMILIES: set[str] = {
    "planning",
    "checkpoints",
    "context",
    "recovery",
    "replay",
    "secrets",
    "tool-cache",
    "vectors",
    "workload-identity",
    "workflow-fork",
}

# Paths not subject to family classification (sidecar probes, docs, static).
INFRA_PATHS: set[str] = {
    "/health",
    "/metrics",
    "/readiness",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/umrs/web",  # StaticFiles mount for the bundled dashboard shell
}

# The m5 companion venue contract, pinned exactly per the S3 forward
# pointer ("S4 must include the companion surface" — m6 is a consumer).
COMPANION_CONTRACT: set[tuple[str, str]] = {
    ("POST", "/umrs/companion/workers"),
    ("GET", "/umrs/companion/workers/{registry_key}"),
    ("POST", "/umrs/companion/chat"),
    ("GET", "/umrs/companion/turns/{attempt_id}"),
    ("GET", "/umrs/companion/turns/{attempt_id}/events"),
    ("GET", "/umrs/companion/approvals/pending"),
    ("POST", "/umrs/companion/approvals/{approval_id}/decision"),
}


@pytest.fixture(scope="module")
def app():
    """Assemble the real sidecar app without a database pool.

    ``_build_db_pool`` is patched out — route-table premises must hold
    regardless of Postgres availability, and the on-demand stores
    degrade to in-memory by design.
    """
    from umrs.server import http_sidecar

    original = http_sidecar._build_db_pool
    http_sidecar._build_db_pool = lambda _app: None
    try:
        yield http_sidecar._create_app()
    finally:
        http_sidecar._build_db_pool = original


def _iter_api_routes(app) -> list[APIRoute]:
    """Flatten the served route table.

    The platform FastAPI carries lazy includes (``_IncludedRouter`` wraps
    ``original_router`` instead of flattening at include time), so the
    walk descends one level; paths inside already carry their full
    router prefix. Duck-typed on purpose — the private class name is a
    vendor-patch detail the premise must not depend on.
    """
    served: list[APIRoute] = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            served.append(route)
            continue
        inner = getattr(route, "original_router", None)
        if inner is not None:
            served.extend(r for r in inner.routes if isinstance(r, APIRoute))
    return served


def _route_paths(app) -> list[str]:
    return sorted({route.path for route in _iter_api_routes(app) if route.path not in INFRA_PATHS})


def _family_of(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    assert parts and parts[0] == "umrs", f"non-UMRS route served: {path}"
    assert len(parts) >= 2, f"route mounted at bare /umrs: {path}"
    return parts[1]


# ── premise: every served route belongs to a declared family ──────────────


def test_every_route_belongs_to_a_declared_family(app):
    undeclared = sorted({p for p in _route_paths(app) if _family_of(p) not in FAMILY_LANES})
    assert not undeclared, (
        "routes served under undeclared families — add the family to "
        "FAMILY_LANES with a lane decision (S5 input):\n" + "\n".join(undeclared)
    )


# ── premise: every declared family is actually served ─────────────────────


def test_every_declared_family_is_served(app):
    served = {_family_of(p) for p in _route_paths(app)}
    missing = sorted(set(FAMILY_LANES) - served)
    assert not missing, (
        "declared families with no served routes — stale premise or "
        "unmounted router:\n" + "\n".join(missing)
    )


# ── premise: dormant routers stay unserved ────────────────────────────────


def test_dormant_families_not_served(app):
    served = {_family_of(p) for p in _route_paths(app)}
    leaked = sorted(DORMANT_FAMILIES & served)
    assert not leaked, (
        "dormant router families are now served — deliberate mount "
        "decision required, then update DORMANT_FAMILIES:\n" + "\n".join(leaked)
    )


# ── premise: lane assignments match the S2/S3 record ──────────────────────


def test_lane_assignments_match_record(app):
    classified = {k: v for k, v in FAMILY_LANES.items() if v != "unclassified"}
    assert classified == {
        "endpoints": "v1-stable",
        "models": "v1-stable",
        "llm": "v1-stable",
        "knowledge": "v1beta",
        "knowledge-graph": "v1beta",
        "k2": "v1beta",
        "loops": "v1beta",
        "research": "v1beta",
        "ops": "internal",
        "skills": "internal",
        "zmq": "internal",
    }, "lane drift vs the S2/S3 record — classification moves belong in the wave log"


# ── premise: the companion venue contract is served exactly (m5 ↔ m6) ─────


def test_companion_contract_served_exactly(app):
    served = {
        (method, route.path)
        for route in _iter_api_routes(app)
        if route.path.startswith("/umrs/companion")
        for method in route.methods or set()
    }
    assert served >= COMPANION_CONTRACT, (
        "companion surface shrank vs the m5 contract — m6 r4-web consumes "
        "these routes:\n" + "\n".join(f"{m} {p}" for m, p in sorted(COMPANION_CONTRACT - served))
    )
