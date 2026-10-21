"""A bundle carries every artifact and embedding row, not the first page.

Found 2026-10-11 while building ADR-093's S8 probe: the export read ONE page
of 500 artifacts and one cursor page of 500 ``memory_vectors`` rows, so the
stand-alone project's 5,183 SciFact artifacts exported as 500 while the
manifest counted 5,183, and an import could only restore those 500
(INV-13: the bundle is self-contained).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from umrs.knowledge import bundle_export
from umrs.knowledge.bundle_export import KnowledgeExportEngine


class _PagedStore:
    """``list_artifacts`` honours page/page_size, as the real store does."""

    def __init__(self, n: int) -> None:
        self.rows = [SimpleNamespace(artifact_id=f"art-{i:05d}") for i in range(n)]
        self.pages: list[int] = []

    async def list_artifacts(self, *, project_id, status, page, page_size):
        assert (project_id, status) == ("p", "active")
        self.pages.append(page)
        start = (page - 1) * page_size
        return self.rows[start : start + page_size], len(self.rows)


def _engine(store) -> KnowledgeExportEngine:
    return KnowledgeExportEngine(
        store=store, output_dir=Path("/tmp/unused"), secret_key=""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("n", "pages"), [(0, 1), (7, 1), (10, 2), (23, 3)])
async def test_every_artifact_page_is_read(monkeypatch, n, pages):
    monkeypatch.setattr(bundle_export, "_PAGE_SIZE", 10)
    store = _PagedStore(n)
    rows, total = await _engine(store)._all_artifacts("p")
    assert [r.artifact_id for r in rows] == [r.artifact_id for r in store.rows]
    assert total == n and len(store.pages) == pages


@pytest.mark.asyncio
async def test_a_store_whose_count_disagrees_still_ends_and_counts_what_it_read(
    monkeypatch,
):
    monkeypatch.setattr(bundle_export, "_PAGE_SIZE", 10)
    store = _PagedStore(15)

    async def undercount(**kw):
        rows, _ = await _PagedStore.list_artifacts(store, **kw)
        return rows, 3

    store.list_artifacts = undercount
    rows, total = await _engine(store)._all_artifacts("p")
    assert len(rows) == 15 and total == 15


def test_artifact_pages_have_a_total_order():
    """OFFSET paging over ``created_at`` alone can skip or repeat rows that
    share a timestamp (a bulk upload); the artifact id breaks the tie."""
    import inspect

    from umrs.knowledge.store import KnowledgeStore

    assert "ORDER BY created_at DESC, artifact_id" in inspect.getsource(
        KnowledgeStore.list_artifacts
    )
