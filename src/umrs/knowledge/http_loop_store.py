"""HTTP-backed loop store for cluster-venue drivers (WP-05 round 2).

A cluster-side engine/agent driver's ``store.save()`` rides the UMRS gateway:

    GET  /umrs/loops/{loop_id}              → load (status+metadata projection)
    PUT  /umrs/loops/{loop_id}/state        → status projection write

The loop's live truth stays server-side in ``umrs_db``; the adapter only
projects it over the guard-allowed HTTP path. Findings-level iteration data
rides POST /umrs/loops/{loop_id}/iterations (already live).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from urllib.request import Request, urlopen

from .loop import LoopPhase, LoopStatus, ResearchLoop

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterLoopTransport:
    """Request summon surface (unit-tested; stdlib-only — the r4-mcp driver
    image carries no requests)."""

    base_url: str  # guard-allowed ingress, e.g. https://data0.ai.r-ccs.riken.jp/rccs
    owner_headers: dict[str, str]
    timeout: int = 30
    path_prefix: str = "/gw/loopjob"  # HMAC lane; rewrites to server's /umrs/loops/*

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers = dict(self.owner_headers)
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(f"{self.base_url}{self.path_prefix}{path}", data=data,
                      headers=headers, method=method)
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read() or b"{}")

    def get(self, path: str) -> dict[str, Any]:
        return self._call("GET", path)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", path, body)

    def put(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._call("PUT", path, body)


def _loop_from_status(loop_id: str, payload: dict[str, Any]) -> ResearchLoop:
    """Project the status route's wire shape onto a ResearchLoop record."""
    return ResearchLoop(
        loop_id=loop_id,
        project_id=str(payload.get("project_id") or ""),
        research_goal=str(payload.get("research_goal") or ""),
        status=LoopStatus(str(payload.get("status") or "running")),
        phase=LoopPhase(str(payload.get("phase") or "initialized")),
        current_iteration=int(payload.get("current_iteration") or 0),
        max_iterations=int(payload.get("max_iterations") or 0),
        total_hypotheses_tested=int(payload.get("total_hypotheses_tested") or 0),
        total_experiments_run=int(payload.get("total_experiments_run") or 0),
        total_findings=int(payload.get("total_findings") or 0),
        success_rate=float(payload.get("success_rate") or 0.0),
        started_at=str(payload.get("started_at") or ""),
        completed_at=str(payload.get("completed_at") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


class HttpLoopStore:
    """Drop-in ``LoopStore`` for cluster-side drivers (remote durable loop)."""

    def __init__(self, transport: ClusterLoopTransport) -> None:
        self._transport = transport

    @property
    def durability(self) -> str:
        """FindingSink-protocol surface (fold bookkeeping only)."""
        return "remote-loop-record"

    def get(self, loop_id: str) -> ResearchLoop | None:
        """LoopStore.get contract: None when the record does not exist."""
        try:
            return self.load(loop_id)
        except KeyError:
            return None

    def mint_artifact(self, **kwargs: Any) -> None:  # FindingSink protocol
        raise RuntimeError(
            "cluster driver pods must not mint knowledge artifacts server-side "
            "(driver-side findings mint after pull-resume)"
        )

    def link(self, **kwargs: Any) -> None:  # FindingSink protocol
        raise RuntimeError(
            "cluster driver pods must not link knowledge artifacts server-side"
        )

    def load(self, loop_id: str) -> ResearchLoop:
        """Load the durable loop record (404-safe; cluster provisioner only
        hosts loops that already exist)."""
        payload = self._transport.get(f"/{loop_id}")
        if payload.get("detail"):
            raise KeyError(f"loop {loop_id} load failed: {payload.get('detail')}")
        return _loop_from_status(loop_id, payload)

    def save(self, loop: ResearchLoop) -> None:
        """Projection write (only driver-owned fields; the server keeps plan,
        iterations, and timestamps)."""
        body = {
            "status": loop.status.value,
            "phase": loop.phase.value,
            "current_iteration": loop.current_iteration,
            "total_hypotheses_tested": loop.total_hypotheses_tested,
            "total_experiments_run": loop.total_experiments_run,
            "total_findings": loop.total_findings,
            "success_rate": loop.success_rate,
            "completed_at": loop.completed_at or None,
            "metadata": loop.metadata,
        }
        self._transport.put(f"/{loop.loop_id}/state", body)

    def list_iterations(self, loop_id: str) -> list[dict[str, Any]]:
        """Read-side iteration projection for restart checks."""
        payload = self._transport.get(f"/{loop_id}/iterations")
        return list(payload or [])


def _owner_headers_from_env() -> dict[str, str]:
    """Verified-identity headers for a cluster job (HMAC shape — the job
    runs under an owner-scoped verified envelope against the gateway lane)."""
    import os

    from r4_contracts.handshake_hmac import sign_claims

    secret = os.environ.get("UMRS_GATEWAY_HMAC_SECRET", "")
    if not secret:
        raise RuntimeError(
            "UMRS_GATEWAY_HMAC_SECRET must be set on the cluster job "
            "(mount from podman/slurm secrets — never hardcoded)"
        )
    user = os.environ.get("R4_LOOP_OWNER", "")
    project = os.environ.get("R4_LOOP_PROJECT", "")
    admin = os.environ.get("R4_LOOP_ADMIN", "").lower() in ("1", "true", "yes")
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(secret, user, project, admin),
    }


def build_cluster_store(gateway_base_url: str | None = None) -> HttpLoopStore:
    """Assemble the HTTP-backed store from the environment."""
    import os


    base = gateway_base_url or os.environ.get("R4_GATEWAY_BASE", "").rstrip("/")
    if not base:
        raise RuntimeError("R4_GATEWAY_BASE is required for cluster loop hosting")
    return HttpLoopStore(
        ClusterLoopTransport(base_url=base, owner_headers=_owner_headers_from_env())
    )
