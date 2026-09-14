"""ADE-031 /reconcile payload shaping (unit level — no FastAPI roundtrip).

Pins the read-side projection decisions in `knowledge._reconcile_payload`
that the preflight/health surfaces consume: counts passthrough, sample cap,
sorted by (state, document_id), orphaned-row tallies, and catalogue
availability carried through so badges can render 'unknown' honestly.
"""

from __future__ import annotations

from umrs.api.knowledge import _reconcile_payload
from umrs.knowledge.plane_reconciler import (
    STATE_COMPLETE,
    STATE_EMBEDDING_PENDING,
    STATE_PARTIAL,
    Asset,
    ReconcileReport,
)


def _asset(doc_id: str, state: str = STATE_COMPLETE, **kw) -> Asset:
    """Build a scan Asset; `has_*` are properties — seed the raw fields instead."""
    asset = Asset(document_id=doc_id, state=state, reasons=kw.pop("reasons", []))
    asset.doc_node_ids = kw.pop("doc_node_ids", ["n-doc-0"]) or []
    asset.chunk_node_ids = kw.pop("chunk_node_ids", []) or []
    asset.vector_rows_by_chunk = kw.pop("vector_rows_by_chunk", {})
    asset.catalogue_keys = kw.pop("catalogue_keys", set()) or set()
    if kw.pop("has_embeddings", False):
        asset.vector_rows_by_chunk = dict.fromkeys(asset.chunk_node_ids, 1)
    if kw.pop("has_catalogue", False):
        asset.catalogue_keys = {"proj/file.txt"}
    for k, v in kw.items():
        setattr(asset, k, v)
    asset.state = state
    return asset


def _report(assets: list[Asset], orphaned_vectors=None, orphaned_chunks=None) -> ReconcileReport:
    return ReconcileReport(
        project_id="proj-x",
        assets=assets,
        orphaned_vectors=orphaned_vectors or [],
        orphaned_chunks=orphaned_chunks or [],
    )


def test_counts_and_orphans_carried():
    payload = _reconcile_payload(
        _report([_asset("doc-1")], orphaned_vectors=["n-cv"], orphaned_chunks=["n-cc"]),
        sample_limit=10,
    )
    assert payload["counts"][STATE_COMPLETE] == 1
    assert payload["orphaned_vector_rows"] == 1
    assert payload["orphaned_chunk_nodes"] == 1
    assert payload["assets_sampled"] == 1


def test_catalogue_availability_flag_passthrough():
    report = _report([_asset("doc-1")])
    report.catalogue_available = False
    payload = _reconcile_payload(report, sample_limit=10)
    assert payload["catalogue_available"] is False


def test_sample_cap_is_applied_for_ordering():
    assets = [_asset(f"doc-{i}") for i in range(30)]
    payload = _reconcile_payload(_report(assets), sample_limit=5)
    assert payload["assets_sampled"] == 5
    assert len(payload["assets"]) == 5


def test_sort_orders_state_then_document_id():
    assets = [
        _asset("doc-z", state=STATE_COMPLETE),
        _asset("doc-a", state=STATE_PARTIAL),
        _asset("doc-m", state=STATE_EMBEDDING_PENDING),
        _asset("doc-b", state=STATE_PARTIAL),
    ]
    payload = _reconcile_payload(_report(assets), sample_limit=10)
    states = [a["state"] for a in payload["assets"]]
    docs = [a["document_id"] for a in payload["assets"]]
    pairs = list(zip(states, docs))
    assert pairs == sorted(pairs)


def test_asset_shape_has_badge_semantics():
    asset = _asset(
        "doc-1",
        state=STATE_EMBEDDING_PENDING,
        reasons=["no embedding rows"],
        doc_node_ids=["n-doc-0"],
        chunk_node_ids=["n-ch-0", "n-ch-1"],
        has_catalogue=True,
    )
    payload = _reconcile_payload(_report([asset]), sample_limit=10)
    row = payload["assets"][0]
    assert row["state"] == STATE_EMBEDDING_PENDING
    assert row["reasons"] == ["no embedding rows"]
    assert row["chunk_nodes"] == 2
    assert row["has_catalogue"] is True
