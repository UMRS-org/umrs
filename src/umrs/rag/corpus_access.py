"""RAG corpus read-sharing (ADR-080).

A RAG corpus is a project namespace in the vector store — it has no durable
row of its own, so migration 0116's ``rag_corpus_access`` registry carries
its access scope list (``user:<id>`` | ``project:<id>`` | ``group:<id>`` |
``system``). The list governs READS (query/documents/search/stats); ingest,
upload and delete stay with the owning project (``project_writable``), and
the existing three-tier namespaces (``system`` corpus, ``user:<uid>``) are
unchanged.

Group membership resolves in SQL against the one authority (migration 0097;
retired groups never resolve) — a user may hold many memberships, any match
admits. Absent rows and error paths are fail-closed (no sharing).
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _default_pool() -> Any:
    from umrs.shared.pg import get_default_sync_pool, pg_configured_from_env

    return get_default_sync_pool() if pg_configured_from_env() else None


def _borrowed(pool: Any, work: Callable[[Any], Any], *, absent: Any) -> Any:
    """Run *work* on a borrowed connection; *absent* without the registry
    (no Postgres, or migration 0116 not applied yet). Fail-closed."""
    if pool is None:
        return absent
    conn = pool.getconn()
    try:
        return work(conn)
    except Exception as exc:
        if getattr(exc, "pgcode", None) in ("42P01", "42703"):  # no table/column yet
            return absent
        raise
    finally:
        try:
            conn.rollback()
        except Exception:
            logger.debug("rollback after corpus-access read failed", exc_info=True)
        pool.putconn(conn)


class CorpusAccessStore:
    """``rag_corpus_access`` rows (per-corpus ADR-080 read-sharing)."""

    def __init__(self, pool_factory: Callable[[], Any] = _default_pool) -> None:
        self._pool_factory = pool_factory

    def _pool(self) -> Any:
        return self._pool_factory()

    def get_access(self, project_id: str) -> list[str]:
        """The corpus's stored access list ([] when unshared or absent)."""

        def work(conn: Any) -> list[str]:
            cur = conn.cursor()
            cur.execute(
                "SELECT access FROM rag_corpus_access WHERE project_id = %s",
                (project_id,),
            )
            row = cur.fetchone()
            if not row:
                return []
            raw = row[0]
            if isinstance(raw, str):
                raw = json.loads(raw)
            return [str(entry) for entry in raw] if isinstance(raw, list) else []

        return _borrowed(self._pool(), work, absent=[])

    def set_access(self, project_id: str, access: list[str], *, set_by: str) -> None:
        """Replace the corpus's access list (validated upstream); upsert."""

        def work(conn: Any) -> None:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO rag_corpus_access (project_id, access, set_by, updated_at) "
                "VALUES (%s, %s::jsonb, %s, now()) "
                "ON CONFLICT (project_id) DO UPDATE "
                "SET access = EXCLUDED.access, set_by = EXCLUDED.set_by, "
                "    updated_at = now()",
                (project_id, json.dumps(list(access)), set_by),
            )
            conn.commit()

        pool = self._pool()
        if pool is None:
            raise RuntimeError("corpus access needs the UMRS Postgres store")
        _borrowed(pool, work, absent=None)

    def shared_readable(self, project_id: str, user_id: str) -> bool:
        """May *user_id* read the corpus through its access list?

        One query: ``system`` admits everyone; ``user:<id>`` the named user;
        ``group:<id>`` any ACTIVE member (multi-membership allowed).
        """
        if not user_id:
            return False

        def work(conn: Any) -> bool:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM rag_corpus_access c "
                "WHERE c.project_id = %s AND ("
                "  c.access @> '[\"system\"]'::jsonb"
                "  OR c.access @> jsonb_build_array(%s)"
                "  OR EXISTS ("
                "    SELECT 1 FROM group_members gm"
                "    JOIN groups g ON g.group_id = gm.group_id"
                "    WHERE gm.user_id = %s AND g.status = 'active'"
                "      AND c.access @> jsonb_build_array('group:' || gm.group_id)"
                "  )) LIMIT 1",
                (project_id, f"user:{user_id}", user_id),
            )
            return cur.fetchone() is not None

        return bool(_borrowed(self._pool(), work, absent=False))


_store: CorpusAccessStore | None = None
_store_lock = threading.Lock()


def get_corpus_access_store() -> CorpusAccessStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = CorpusAccessStore()
        return _store


def set_corpus_access_store(store: CorpusAccessStore | None) -> None:
    """Tests / boot code replace the process store."""
    global _store
    with _store_lock:
        _store = store


def shared_corpus_readable(project_id: str, user_id: str) -> bool:
    """Process-wide :meth:`CorpusAccessStore.shared_readable`; False on any
    failure (a lookup error never grants a read)."""
    try:
        return get_corpus_access_store().shared_readable(project_id, user_id)
    except Exception:
        logger.warning("corpus access lookup failed for %s", project_id, exc_info=True)
        return False
