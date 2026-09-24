"""ADR-032 #5: the exporter's exclusions derived from a reconcile report —
eligible = COMPLETE / EMBEDDING_PENDING; PARTIAL assets and orphaned chunks out."""

from __future__ import annotations

from umrs.knowledge.plane_reconciler import Asset, ReconcileReport
from umrs.knowledge.reconcile_snapshot import (
    EXPORT_ELIGIBLE_STATES,
    export_snapshot_from_report,
    take_export_snapshot,
)


def _asset(doc, state, docs=(), chunks=(), reasons=()):
    return Asset(
        document_id=doc,
        doc_node_ids=list(docs),
        chunk_node_ids=list(chunks),
        state=state,
        reasons=list(reasons),
    )


def _report():
    return ReconcileReport(
        project_id="p",
        assets=[
            _asset("d-complete", "COMPLETE", ["n1"], ["c1"]),
            _asset("d-pending", "EMBEDDING_PENDING", ["n2"], ["c2"]),
            _asset("d-partial", "PARTIAL", ["n3"], ["c3", "c4"], ["no catalogue row"]),
        ],
        orphaned_vectors=["v-orphan"],
        orphaned_chunks=["c-orphan"],
    )


def test_ineligible_assets_and_orphan_chunks_are_excluded():
    snap = export_snapshot_from_report(_report())
    assert snap.excluded_node_ids == frozenset({"n3", "c3", "c4", "c-orphan"})
    assert {"n1", "c1", "n2", "c2"}.isdisjoint(snap.excluded_node_ids)  # eligible kept


def test_summary_states_what_was_omitted_and_why():
    summary = export_snapshot_from_report(_report()).summary
    assert summary["consumed"] is True
    assert summary["export_eligible_states"] == list(EXPORT_ELIGIBLE_STATES)
    assert summary["excluded_assets"] == [
        {"document_id": "d-partial", "state": "PARTIAL", "reasons": ["no catalogue row"]}
    ]
    assert (summary["orphaned_chunks_excluded"], summary["excluded_node_count"]) == (1, 4)
    assert summary["assets_total"] == 3 and len(summary["digest"]) == 64


def test_no_dsn_means_no_snapshot(monkeypatch):
    monkeypatch.delenv("UMRS_DB_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert take_export_snapshot("p") is None
