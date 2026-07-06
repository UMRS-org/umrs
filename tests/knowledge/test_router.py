"""Tests for the knowledge compilation router (Phase 1)."""

from __future__ import annotations

import json

from umrs.knowledge.ir import EXTRACTOR_VERSION, build_knowledge_ir
from umrs.knowledge.persistence import (
    knowledge_graph_index_path,
    knowledge_ir_dir,
    read_local_knowledge_graph,
    write_local_knowledge_ir,
)
from umrs.knowledge.router import (
    CompileResult,
    KnowledgeRouter,
    LocalGraphSink,
    default_local_router,
    ir_to_artifact_dict,
    regenerate_local_indexes,
)


def _ir(artifact_id: str, *, filename: str, content: bytes, extracted: dict | None = None):
    payload = {"artifact_id": artifact_id, "artifact_type": "code", "filename": filename}
    payload.update(extracted or {})
    return build_knowledge_ir(
        project_id="proj-1",
        artifact_id=artifact_id,
        artifact_type="code",
        content=content,
        extracted=payload,
        extractor_version=EXTRACTOR_VERSION,
    )


class _RecordingSink:
    name = "recording"

    def __init__(self):
        self.compiled: list[str] = []

    def accepts(self, ir):
        return ir.artifact_type == "code"

    def compile(self, irs):
        self.compiled = [ir.artifact_id for ir in irs]
        return CompileResult(sink=self.name, accepted=len(irs))


class _BoomSink:
    name = "boom"

    def accepts(self, ir):
        return ir is not None

    def compile(self, irs):
        raise RuntimeError(f"kaboom on {len(irs)}")


class TestRouterDispatch:
    def test_only_accepted_irs_reach_sink(self):
        sink = _RecordingSink()
        router = KnowledgeRouter([sink])
        code = _ir("c1", filename="a.py", content=b"x")
        text = build_knowledge_ir(
            project_id="p",
            artifact_id="t1",
            artifact_type="text",
            content=b"y",
            extracted={"artifact_id": "t1", "artifact_type": "text", "filename": "n.txt"},
            extractor_version=EXTRACTOR_VERSION,
        )
        results = router.regenerate([code, text])
        assert sink.compiled == ["c1"]
        assert results[0].accepted == 1

    def test_failing_sink_is_isolated(self):
        good = _RecordingSink()
        router = KnowledgeRouter([_BoomSink(), good])
        results = router.regenerate([_ir("c1", filename="a.py", content=b"x")])
        # Boom sink recorded as error, good sink still ran.
        assert results[0].sink == "boom"
        assert results[0].error is not None and "kaboom" in results[0].error
        assert good.compiled == ["c1"]

    def test_register_chaining(self):
        router = KnowledgeRouter()
        assert router.register(_RecordingSink()) is router
        assert len(router.sinks) == 1


class TestIrToArtifactDict:
    def test_recovers_linker_fields_from_ir(self):
        ir = _ir("c1", filename="solver.py", content=b"x", extracted={"apis": ["solve"]})
        d = ir_to_artifact_dict(ir)
        assert d["artifact_id"] == "c1"
        assert d["artifact_type"] == "code"
        assert d["filename"] == "solver.py"
        assert d["extracted_knowledge"]["apis"] == ["solve"]
        assert d["metadata"]["ir_id"] == ir.ir_id
        assert d["metadata"]["content_hash"] == ir.content_hash


class TestLocalGraphSink:
    def test_writes_graph_index(self, tmp_path):
        sink = LocalGraphSink(str(tmp_path))
        irs = [
            _ir("c1", filename="a.py", content=b"a"),
            _ir("c2", filename="b.py", content=b"b"),
        ]
        result = sink.compile(irs)
        assert result.sink == "local_graph"
        assert result.accepted == 2
        assert result.detail["nodes"] == 2
        assert knowledge_graph_index_path(tmp_path).is_file()

    def test_graph_index_json_is_sorted(self, tmp_path):
        sink = LocalGraphSink(str(tmp_path))
        sink.compile([_ir("c1", filename="a.py", content=b"a")])
        raw = knowledge_graph_index_path(tmp_path).read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert raw == json.dumps(parsed, indent=2, sort_keys=True)

    def test_overwrite_latest(self, tmp_path):
        sink = LocalGraphSink(str(tmp_path))
        sink.compile([_ir("c1", filename="a.py", content=b"a")])
        first = read_local_knowledge_graph(tmp_path)
        assert len(first["nodes"]) == 1
        sink.compile([
            _ir("c1", filename="a.py", content=b"a"),
            _ir("c2", filename="b.py", content=b"b"),
        ])
        second = read_local_knowledge_graph(tmp_path)
        assert len(second["nodes"]) == 2


class TestRegenerateFromDisk:
    def test_regenerate_local_indexes_reads_canonical_irs(self, tmp_path):
        # Persist canonical IRs write-once, then regenerate the derived graph
        # from disk alone (no in-memory state).
        ir_dir = knowledge_ir_dir(tmp_path)
        for aid, fn, c in [("c1", "a.py", b"a"), ("c2", "b.py", b"b")]:
            write_local_knowledge_ir(ir_dir, _ir(aid, filename=fn, content=c))

        results = regenerate_local_indexes(str(tmp_path))
        assert results[0].sink == "local_graph"
        assert results[0].detail["nodes"] == 2
        graph = read_local_knowledge_graph(tmp_path)
        assert len(graph["nodes"]) == 2

    def test_regenerate_with_no_irs_is_empty(self, tmp_path):
        results = regenerate_local_indexes(str(tmp_path))
        assert results[0].accepted == 0

    def test_default_local_router_has_graph_sink(self, tmp_path):
        router = default_local_router(str(tmp_path))
        assert [s.name for s in router.sinks] == ["local_graph"]
