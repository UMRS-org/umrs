"""A2L artifact bridge — best-effort cross-reference to A2L identity space.

Communication is exclusively via Gateway HTTP:
  POST /gw/a2l/resolve   — resolve a2l_id from content hash
  GET  /gw/a2l/artifacts — list known a2l artifacts for a project

No Python import of ``a2l`` in ``umrs``.  If the gateway path is unavailable,
all operations degrade silently; knowledge jobs never fail because A2L is absent.

Design invariants (ADR-017 / Invariant 7.x):
  * ``a2l_artifact_refs`` is the single source of truth for UMRS→A2L mapping.
  * All A2L access goes through ``GatewayClient`` (HTTP).
  * ``resolve_a2l_hash()`` and ``sync_a2l_ref()`` catch all exceptions → log + return.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class A2lRef:
    """A row in ``a2l_artifact_refs``."""

    artifact_id: str
    a2l_hash: str
    a2l_id: str
    synced_at: str = ""
    source: str = "auto"


class A2lBridge:
    """Best-effort bridge between UMRS knowledge artifacts and A2L identity space.

    Public API:
        * ``resolve_a2l_hash(content_hash)`` → ``A2lRef | None``
        * ``sync_a2l_ref(ref)`` → ``bool``
        * ``resolve_artifacts(project_id, artifact_ids)`` → ``dict[str, A2lRef | None]``
    """

    def __init__(self, gateway_url: str = "") -> None:
        self.gateway_url = gateway_url

    # ── Public API ──────────────────────────────────────────────────────────

    def resolve_a2l_hash(self, content_hash: str) -> A2lRef | None:
        """Look up an A2L reference by content hash.

        Returns ``None`` if the hash is not known (A2L absent or new).
        """
        if not content_hash or not self.gateway_url:
            return None
        try:
            import httpx  # stdlib-adjacent HTTP client; never imports a2l

            url = f"{self.gateway_url}/gw/a2l/resolve"
            resp = httpx.post(
                url,
                json={"content_hash": content_hash},
                timeout=3.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return A2lRef(
                    artifact_id=data.get("artifact_id", ""),
                    a2l_hash=content_hash,
                    a2l_id=data["a2l_id"],
                    synced_at=data.get("synced_at", ""),
                )
            return None
        except Exception:
            logger.warning("A2L resolve failed for hash %s (best-effort: continuing)", content_hash[:8])
            return None

    def sync_a2l_ref(self, ref: A2lRef) -> bool:
        """Persist an A2L→UMRS mapping in the local store.

        Returns ``True`` on success, ``False`` on any error (best-effort).
        """
        try:
            from umrs.knowledge.store import get_store

            store = get_store()
            store.insert_a2l_ref(
                artifact_id=ref.artifact_id,
                a2l_hash=ref.a2l_hash,
                a2l_id=ref.a2l_id,
                source=ref.source,
            )
            return True
        except Exception:
            logger.warning("A2L ref sync failed for %s (best-effort: continuing)", ref.a2l_id)
            return False

    def _fetch_artifacts(self, project_id: str, artifact_ids: list[str]) -> list[dict]:
        """Fetch A2L artifact list from Gateway."""
        try:
            import httpx

            url = f"{self.gateway_url}/gw/a2l/artifacts"
            resp = httpx.get(
                url,
                params={"project_id": project_id, "artifact_ids": artifact_ids},
                timeout=3.0,
            )
            if resp.status_code == 200:
                return resp.json().get("artifacts", [])
            return []
        except Exception:
            logger.warning("A2L artifact fetch failed (best-effort)", exc_info=True)
            return []

    def resolve_artifacts(
        self, project_id: str, artifact_ids: list[str]
    ) -> dict[str, A2lRef | None]:
        """Resolve A2L refs for a list of UMRS artifact IDs."""
        if not self.gateway_url or not artifact_ids:
            return {aid: None for aid in artifact_ids}
        items = self._fetch_artifacts(project_id, artifact_ids)
        mapping: dict[str, A2lRef | None] = {}
        for item in items:
            aid = item.get("artifact_id", "")
            if aid in artifact_ids and "a2l_id" in item:
                mapping[aid] = A2lRef(**item)
        return mapping
