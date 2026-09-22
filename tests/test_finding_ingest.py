"""Findings → knowledge assets with provenance (ADR-036 D4, WP-05; G-P1-1)."""

from __future__ import annotations

from pathlib import Path

from umrs.knowledge import finding_ingest
from umrs.knowledge.finding_ingest import InMemoryFindingSink, mint_iteration
from umrs.knowledge.loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop

from r4_contracts.research import EvidenceRef, EvidenceSource, Finding


def _loop() -> ResearchLoop:
    return ResearchLoop(
        loop_id="loop-f1",
        project_id="proj-a",
        research_goal="Does X raise yield?",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.PUBLISHING,
        max_iterations=3,
        current_iteration=1,
        metadata={"durable_job_id": "job-42"},
    )


def _finding(fid: str = "finding-1", *, simulated: bool = False, evidence=("art-1",)) -> Finding:
    return Finding(
        finding_id=fid,
        statement="Batch-3 yield 85%",
        hypothesis_id="hyp-0",
        evidence_refs=[
            EvidenceRef(source=EvidenceSource.CATALOG, excerpt="yield 85%", artifact_id=a)
            for a in evidence
        ],
        workflow_run_id="wf-run-9",
        durable_job_id="job-wf-9",
        confidence=0.7,
        simulated=simulated,
    )


def _iteration(findings, *, status: str = "completed", results=None) -> LoopIteration:
    return LoopIteration(
        iteration_id="it-1",
        loop_id="loop-f1",
        iteration_number=1,
        phase=LoopPhase.MEMORY_UPDATE,
        hypothesis="X raises yield",
        analysis="a",
        evaluation="e",
        findings=list(findings),
        results=results or {},
        status=status,
    )


def _minted():
    """One real finding minted into an in-memory sink → (loop, sink, iteration, outcome)."""
    loop, sink = _loop(), InMemoryFindingSink()
    iteration = _iteration([_finding()])
    outcome = mint_iteration(loop, iteration, sink=sink)
    return loop, sink, iteration, outcome


def test_real_finding_minted_as_knowledge_artifact_with_provenance():
    _, sink, _, outcome = _minted()
    assert outcome.minted == ["finding-1"]
    art = sink.artifacts["finding-1"]
    assert art["project_id"] == "proj-a"
    assert art["source_ref"] == "loop:loop-f1#iteration-1"
    meta = art["metadata"]
    assert meta["asset_kind"] == "finding"
    assert meta["source_loop"] == "loop-f1"
    assert meta["hosting_job_id"] == "job-42"
    assert meta["evidence_artifact_ids"] == ["art-1"]
    assert meta["finding"]["statement"] == "Batch-3 yield 85%"


def test_real_finding_gets_graph_lineage_nodes_and_edges():
    _, sink, _, _ = _minted()
    assert set(sink.nodes) == {"finding-1", "loop:loop-f1", "art-1"}
    assert sink.nodes["finding-1"]["node_type"] == "concept"
    assert sink.nodes["loop:loop-f1"]["node_type"] == "entity"
    relations = {(e["source_id"], e["relation_type"], e["target_id"]) for e in sink.edges.values()}
    assert relations == {
        ("finding-1", "produced_by", "loop:loop-f1"),
        ("finding-1", "derived_from", "art-1"),
    }


def test_minting_records_artifact_ids_and_folds_outcome_into_loop_metadata():
    loop, _, iteration, _ = _minted()
    assert iteration.artifact_ids == ["finding-1"]
    assert loop.metadata["finding_ingest"] == {
        "sink": "process",
        "minted": 1,
        "skipped_simulated": 0,
        "errors": 0,
        "last_error": "",
    }


def test_simulated_findings_are_never_minted():
    loop, sink = _loop(), InMemoryFindingSink()
    iteration = _iteration([_finding(simulated=True)])
    outcome = mint_iteration(loop, iteration, sink=sink)
    assert outcome.minted == [] and outcome.skipped_simulated == 1
    assert sink.artifacts == {} and sink.edges == {}
    assert iteration.artifact_ids == []
    assert loop.metadata["finding_ingest"]["skipped_simulated"] == 1


def test_rerun_is_idempotent():
    loop, sink = _loop(), InMemoryFindingSink()
    iteration = _iteration([_finding()])
    mint_iteration(loop, iteration, sink=sink)
    second = mint_iteration(loop, iteration, sink=sink)
    assert second.minted == [] and second.skipped_existing == 1
    assert iteration.artifact_ids == ["finding-1"]
    assert len(sink.artifacts) == 1
    assert loop.metadata["finding_ingest"]["minted"] == 1


def test_incomplete_iteration_is_untouched():
    loop, sink = _loop(), InMemoryFindingSink()
    iteration = _iteration([_finding()], status="running")
    assert mint_iteration(loop, iteration, sink=sink).minted == []
    assert sink.artifacts == {} and "finding_ingest" not in loop.metadata


def test_workflow_artifacts_recorded_alongside_findings():
    loop, sink = _loop(), InMemoryFindingSink()
    iteration = _iteration(
        [_finding()],
        results={"artifacts": ["wf-art-1", {"artifact_id": "wf-art-2"}, {"id": "wf-art-3"}]},
    )
    mint_iteration(loop, iteration, sink=sink)
    assert iteration.artifact_ids == ["wf-art-1", "wf-art-2", "wf-art-3", "finding-1"]


def test_dict_findings_from_reloaded_rows_are_handled():
    loop, sink = _loop(), InMemoryFindingSink()
    iteration = _iteration([_finding("finding-d").to_dict()])
    assert mint_iteration(loop, iteration, sink=sink).minted == ["finding-d"]
    assert ("finding-d", "derived_from", "art-1") in {
        (e["source_id"], e["relation_type"], e["target_id"]) for e in sink.edges.values()
    }


def test_missing_sink_is_reported_not_silent():
    loop = _loop()
    iteration = _iteration([_finding()], results={"artifacts": ["wf-art-1"]})
    outcome = mint_iteration(loop, iteration, sink=None)
    assert outcome.minted == []
    assert iteration.artifact_ids == ["wf-art-1"]  # workflow ids still recorded
    assert loop.metadata["finding_ingest"]["sink"] == "none"


def test_sink_failure_is_folded_into_loop_metadata():
    class _Broken(InMemoryFindingSink):
        def mint_artifact(self, **fields):  # noqa: ARG002
            raise RuntimeError("knowledge pool unavailable")

    loop = _loop()
    iteration = _iteration([_finding(), _finding("finding-2")])
    outcome = mint_iteration(loop, iteration, sink=_Broken())
    assert outcome.minted == [] and len(outcome.errors) == 2
    assert iteration.artifact_ids == []
    fi = loop.metadata["finding_ingest"]
    assert fi["errors"] == 2 and "knowledge pool unavailable" in fi["last_error"]


def test_workflow_artifact_id_helper_dedupes():
    iteration = _iteration([], results={"artifacts": ["a", "a", {"artifact_id": "b"}, None]})
    assert finding_ingest.workflow_artifact_ids(iteration) == ["a", "b"]


def test_migration_0082_adds_artifact_ids_column():
    root = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
    sql = (root / "0082_research_loop_iteration_artifacts.sql").read_text(encoding="utf-8")
    assert "ALTER TABLE research_loop_iterations" in sql
    assert "ADD COLUMN IF NOT EXISTS artifact_ids JSONB NOT NULL DEFAULT '[]'" in sql


def test_iteration_record_carries_artifact_ids():
    iteration = _iteration([_finding()])
    iteration.artifact_ids.append("finding-1")
    record = iteration.to_record()
    assert record.artifact_ids == ["finding-1"]
    assert record.to_dict()["artifact_ids"] == ["finding-1"]


def test_engine_runner_invokes_hosting_hook_per_completed_iteration():
    from umrs.knowledge.engine_loop_runner import EngineLoopRunner
    from umrs.knowledge.loop import AutonomousResearchLoop

    loop = ResearchLoop(
        loop_id="loop-hook",
        project_id="p",
        research_goal="g",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.PLANNING,
        max_iterations=3,
        metadata={"simulated": True, "mode": "auto"},
    )
    seen: list[tuple[str, int, str]] = []

    class _Store:
        def get(self, loop_id):  # noqa: ARG002
            return loop

        def save(self, _loop):
            pass

    runner = EngineLoopRunner(
        loop,
        engine_factory=lambda: AutonomousResearchLoop(simulated=True),
        store=_Store(),
        on_iteration=lambda lp, it: seen.append((lp.loop_id, it.iteration_number, it.status)),
    )
    out = runner.drive(loop)
    assert out.status == LoopStatus.STOPPED
    assert [n for _, n, s in seen if s == "completed"] == [1, 2, 3]


def test_finding_metadata_carries_trace_id():
    """D-52: minted findings carry the loop's trace_id (may be empty pre-stamp)."""
    from umrs.knowledge.finding_ingest import _finding_payload
    from umrs.knowledge.loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop

    loop = ResearchLoop(
        loop_id="L-t",
        project_id="proj",
        research_goal="g",
        status=LoopStatus.RUNNING,
    )
    loop.metadata["trace_id"] = "L-t"
    iteration = LoopIteration(
        iteration_id="it-1",
        loop_id="L-t",
        iteration_number=1,
        phase=LoopPhase.EXPERIMENT_EXECUTION,
        started_at="",
        completed_at="",
        hypothesis="",
        experiment_id="",
        results={},
        analysis="",
        evaluation="",
        findings=[],
        status="completed",
    )
    _, metadata = _finding_payload(loop, iteration, {"statement": "s"})
    assert metadata["trace_id"] == "L-t"
    assert metadata["source_loop"] == "L-t"
