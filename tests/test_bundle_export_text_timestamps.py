"""A bundle exports rows whose timestamps are text, not only datetimes.

``knowledge_artifacts.created_at`` is TEXT on the stand-alone schema, and
every KNOWLEDGE_EXPORT there failed with ``'str' object has no attribute
'isoformat'`` (found 2026-10-12 running ADR-093 S8).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from umrs.knowledge import bundle_export
from umrs.knowledge.bundle_export import KnowledgeExportEngine

WHEN = datetime(2026, 10, 12, 3, 4, 5, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (WHEN, WHEN.isoformat()),
        ("2026-10-12T03:04:05+00:00", "2026-10-12T03:04:05+00:00"),
        (None, None),
        ("", None),
    ],
)
def test_iso_takes_a_datetime_or_text(value, expected):
    assert bundle_export._iso(value) == expected


def _engine() -> KnowledgeExportEngine:
    return KnowledgeExportEngine(
        store=None, output_dir=Path("/tmp/unused"), secret_key=""
    )


def test_an_artifact_with_text_timestamps_serializes():
    artifact = SimpleNamespace(
        artifact_id="a",
        content_hash="h",
        kind="document",
        project_id="p",
        source_ref="s",
        size_bytes=1,
        mime_type="text/plain",
        status="active",
        metadata={},
        created_at="2026-10-12T03:04:05+00:00",
        updated_at=WHEN,
    )
    row = _engine()._serialize_artifact(artifact)
    assert row["created_at"] == "2026-10-12T03:04:05+00:00"
    assert row["updated_at"] == WHEN.isoformat()


def test_nodes_and_edges_with_text_timestamps_serialize():
    node = SimpleNamespace(
        node_id="n",
        project_id="p",
        label="l",
        node_type="chunk",
        content={},
        content_hash="h",
        source_ir=None,
        created_at="2026-10-12",
        updated_at=None,
    )
    edge = SimpleNamespace(
        edge_id="e",
        project_id="p",
        source_id="n",
        target_id="m",
        relation_type="r",
        content={},
        created_at="t",
    )
    engine = _engine()
    assert engine._serialize_node(node)["created_at"] == "2026-10-12"
    assert engine._serialize_edge(edge)["created_at"] == "t"
