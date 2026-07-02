"""Data catalog schema v1 (MVP)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class CatalogState(str, Enum):
    """Per-object catalog lifecycle state."""

    PENDING = "pending"
    READY = "ready"
    ERROR = "error"


@dataclass
class CatalogObjectMeta:
    """Per-object metadata written to ``_data_catalog/objects/<key>.meta.json``."""

    s3_key: str
    object_etag: str
    state: CatalogState
    owner: str
    acl_scope: str
    updated_at: str
    summary: str = ""
    deterministic_summary: str = ""
    perception_summary: str | None = None
    perception_model: str | None = None
    tags: list[str] = field(default_factory=list)
    filename_signals: list[str] = field(default_factory=list)
    detected_modality: str = "unknown"
    extracted_text_ref: str | None = None
    extracted_knowledge_ref: str | None = None
    error_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogObjectMeta:
        state_raw = data.get("state", CatalogState.PENDING.value)
        state = CatalogState(state_raw) if isinstance(state_raw, str) else state_raw
        return cls(
            s3_key=str(data["s3_key"]),
            object_etag=str(data.get("object_etag", "")),
            state=state,
            owner=str(data.get("owner", "")),
            acl_scope=str(data.get("acl_scope", "")),
            updated_at=str(data.get("updated_at", "")),
            summary=str(data.get("summary", "")),
            deterministic_summary=str(
                data.get("deterministic_summary") or data.get("summary", "")
            ),
            perception_summary=data.get("perception_summary"),
            perception_model=data.get("perception_model"),
            tags=list(data.get("tags") or []),
            filename_signals=list(data.get("filename_signals") or []),
            detected_modality=str(data.get("detected_modality", "unknown")),
            extracted_text_ref=data.get("extracted_text_ref"),
            extracted_knowledge_ref=data.get("extracted_knowledge_ref"),
            error_note=data.get("error_note"),
        )

    @classmethod
    def pending(
        cls,
        *,
        s3_key: str,
        object_etag: str,
        owner: str,
        acl_scope: str,
        filename: str | None = None,
    ) -> CatalogObjectMeta:
        signals: list[str] = []
        if filename:
            signals.append(filename)
        return cls(
            s3_key=s3_key,
            object_etag=object_etag,
            state=CatalogState.PENDING,
            owner=owner,
            acl_scope=acl_scope,
            updated_at=_utc_now(),
            filename_signals=signals,
            detected_modality="unknown",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
