"""In-memory catalog store for unit tests."""

from __future__ import annotations

from copy import deepcopy

from umrs.data_catalog.pg_store import derive_project_id
from umrs.data_catalog.query import CatalogHit
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState
from umrs.data_catalog.scoring import allowed, score_meta, tokenize


class FakeCatalogMetaStore:
    """Dict-backed ``CatalogMetaStore`` stand-in for unit tests."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], CatalogObjectMeta] = {}

    def upsert(self, meta: CatalogObjectMeta, project_id: str) -> None:
        canonical = (project_id or derive_project_id(meta)).strip()
        self.rows[(canonical, meta.s3_key)] = deepcopy(meta)

    def get(self, project_id: str, s3_key: str) -> CatalogObjectMeta | None:
        meta = self.rows.get((project_id, s3_key))
        if meta is None:
            return None
        if not allowed(meta, project_id):
            return None
        return deepcopy(meta)

    def get_by_s3_key(self, s3_key: str) -> CatalogObjectMeta | None:
        for (_, key), meta in self.rows.items():
            if key == s3_key:
                return deepcopy(meta)
        return None

    def list_by_project(
        self,
        project_id: str,
        *,
        prefix: str | None = None,
        include_pending: bool = True,
        include_error: bool = False,
    ) -> list[CatalogObjectMeta]:
        normalized_prefix = prefix or f"{project_id}/"
        if not normalized_prefix.endswith("/"):
            normalized_prefix = f"{normalized_prefix}/"

        metas: list[CatalogObjectMeta] = []
        for (_, key), meta in self.rows.items():
            if not allowed(meta, project_id):
                continue
            if not key.startswith(normalized_prefix):
                continue
            if not include_pending and meta.state == CatalogState.PENDING:
                continue
            if not include_error and meta.state == CatalogState.ERROR:
                continue
            metas.append(deepcopy(meta))
        return metas

    def search(
        self,
        project_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[CatalogHit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        hits: list[CatalogHit] = []
        for meta in self.list_by_project(project_id, include_pending=True, include_error=False):
            score = score_meta(meta, tokens)
            if score <= 0:
                continue
            hits.append(CatalogHit(meta=meta, score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def delete(self, project_id: str, s3_key: str) -> bool:
        return self.rows.pop((project_id, s3_key), None) is not None

    def project_stats(self, project_id: str) -> dict:
        by_state: dict[str, int] = {}
        by_modality: dict[str, int] = {}
        total = 0
        for meta in self.list_by_project(
            project_id,
            prefix=f"{project_id}/",
            include_pending=True,
            include_error=True,
        ):
            total += 1
            by_state[meta.state.value] = by_state.get(meta.state.value, 0) + 1
            by_modality[meta.detected_modality] = (
                by_modality.get(meta.detected_modality, 0) + 1
            )
        return {
            "project_id": project_id,
            "object_count": total,
            "states": by_state,
            "modalities": by_modality,
        }

    def list_s3_keys_for_project(self, project_id: str) -> list[str]:
        return [meta.s3_key for meta in self.list_by_project(project_id, prefix=f"{project_id}/")]
