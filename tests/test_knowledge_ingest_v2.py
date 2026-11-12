"""ADR-116 ingest lane (D-1, D-2, D-3, D-6 write side, D-7 hook) in the
understand job: windowed extraction through the patch seams, extract modes,
the extractor-version probe, the alias-aware endpoint, slug collisions, the
parser and pre-tagger fallbacks, and the entity-resolution hook.

Hermetic: fake chat, fake stores, fake ontology and entity-resolution modules.
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.jobs import knowledge_adapters as ka

_EMPTY = {k: [] for k in ("modules", "classes", "functions", "algorithms", "dependencies")}


class FakeVocab:
    ontology_id = "t"
    version = "1"

    def prompt_block(self) -> str:
        return "Classes: material, method."

    def match_class(self, raw: str) -> str | None:
        return raw.lower() if raw.lower() in ("material", "method") else None

    def match_relation(self, raw: str) -> str | None:
        return "uses" if raw.lower() == "uses" else None

    def bucket_for(self, cls: str) -> str:
        return "entity" if cls == "material" else "concept"

    def gliner_labels(self) -> list[str]:
        return ["material", "method"]

    def class_names(self) -> list[str]:
        return ["material", "method"]

    def relation_names(self) -> list[str]:
        return ["uses"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "UMRS_AGENT_LITELLM_URL",
        "UMRS_KNOWLEDGE_EXTRACT_MODEL",
        "UMRS_KNOWLEDGE_LLM_EXTRACT",
        "UMRS_KNOWLEDGE_EXTRACT_AUTO_MODE",
        "UMRS_KNOWLEDGE_DOC_PARSER",
        "UMRS_KNOWLEDGE_DOCLING_ARTIFACTS",
        "UMRS_KNOWLEDGE_GLINER_MODEL",
        "UMRS_EMBEDDING_SANDBOX",
        "UMRS_KNOWLEDGE_EXTRACT_WINDOW_CHARS",
        "UMRS_KNOWLEDGE_EXTRACT_WINDOWS_MAX",
    ):
        monkeypatch.delenv(var, raising=False)


def _chat_recorder(monkeypatch, replies: dict[str, dict[str, Any]]) -> list[str]:
    """Patch the late-imported chat seam; answer by markers in the window."""
    monkeypatch.setenv("UMRS_AGENT_LITELLM_URL", "http://litellm:4000")
    users: list[str] = []

    def chat(messages: list[dict[str, Any]], **_kw: Any) -> str:
        user = messages[-1]["content"]
        users.append(user)
        entities = [e for marker, r in replies.items() if marker in user for e in r["entities"]]
        return json.dumps({"entities": entities, "relations": []})

    monkeypatch.setattr("umrs.knowledge.agent_executor._default_litellm_chat", chat)
    return users


def _prose(count: int) -> str:
    return " ".join(f"Background sentence {i} adds nothing new." for i in range(count))


# ── D-2: the whole document is read ─────────────────────────────────────────


def test_text_past_9000_characters_reaches_the_model(monkeypatch):
    """The old extractor sent text[:9000]; a fact on page 3 was never read."""
    text = _prose(300) + " The ZIRCONIA marker sits here. " + _prose(50)
    assert text.index("ZIRCONIA") > 9000
    users = _chat_recorder(
        monkeypatch, {"ZIRCONIA": {"entities": [{"name": "Zirconia", "type": "material"}]}}
    )
    notes: list[str] = []
    payload = ka._llm_extract_document(text, project_id="p", notes=notes, vocab=FakeVocab())
    assert payload is not None
    assert any("ZIRCONIA" in u for u in users)
    assert [n["name"] for n in payload["nodes"]] == ["Zirconia"]
    assert notes == []


def test_lite_mode_reads_one_window(monkeypatch):
    users = _chat_recorder(monkeypatch, {})
    notes: list[str] = []
    ka._llm_extract_document(_prose(600), project_id="p", notes=notes, extract_mode="lite")
    assert len(users) == 1
    assert not any("skipped" in n for n in notes)


def test_off_mode_makes_no_call(monkeypatch):
    users = _chat_recorder(monkeypatch, {})
    assert ka._llm_extract_document(_prose(10), project_id="p", extract_mode="off") is None
    assert users == []


def test_off_mode_never_reaches_the_seam(monkeypatch):
    def _no_llm(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("extract_mode off must not call the extractor")

    monkeypatch.setattr(ka, "_llm_extract_document", _no_llm)
    nodes, _ = ka._nodes_edges_for_artifact(
        "art-1",
        "ir-1",
        "document",
        b"plain prose",
        _EMPTY,
        "p",
        "",
        inputs=ka._ExtractInputs(mode="off"),
    )
    assert [n.node_type for n in nodes] == ["file"]


def test_session_history_is_windowed_over_its_turns_in_lite(monkeypatch):
    seen: list[tuple[str, dict[str, Any]]] = []

    def fake(text: str, **kw: Any) -> None:
        seen.append((text, kw))

    monkeypatch.setattr(ka, "_llm_extract_document", fake)
    package = json.dumps(
        {"lane": "session-history", "lines": [{"role": "user", "text": "Zirconia phases?"}]}
    ).encode()
    ka._nodes_edges_for_artifact("art-1", "ir-1", "document", package, _EMPTY, "p", "")
    ((text, kw),) = seen
    assert text == "user: Zirconia phases?"
    assert kw["extract_mode"] == "lite"
    assert kw["parser_id"] == "session-history"


# ── [C-ING] rows ────────────────────────────────────────────────────────────


def _windowed_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {
                "name": "Zirconia",
                "type": "material",
                "bucket": "entity",
                "aliases": ["ZrO2"],
                "evidence": [{"window": 0, "quote": "ZrO2 powder", "parser": "pypdf"}],
            },
            {
                "name": "Dr. X",
                "type": "unclassified",
                "type_raw": "person",
                "bucket": "concept",
                "aliases": [],
                "evidence": [],
            },
        ],
        "edges": [
            {"source": "Dr. X", "target": "Zirconia", "relation": "related_to", "rel_raw": "grinds"}
        ],
        "extractor": "llm-windowed@2",
        "parser": "pypdf",
    }


def test_semantic_nodes_carry_aliases_evidence_and_raw_types():
    nodes, _edges = ka._semantic_nodes_edges("art-1", "ir-1", "p", "", _windowed_payload())
    rows = {n.label: (n.node_type, n.content) for n in nodes}
    assert rows == {
        "Zirconia": (
            "entity",
            {
                "artifact_id": "art-1",
                "semantic_type": "material",
                "source": "Zirconia",
                "aliases": ["ZrO2"],
                "evidence": [{"window": 0, "quote": "ZrO2 powder", "parser": "pypdf"}],
                "extractor": "llm-windowed@2",
            },
        ),
        "Dr. X": (
            "concept",
            {
                "artifact_id": "art-1",
                "semantic_type": "unclassified",
                "source": "Dr. X",
                "aliases": [],
                "evidence": [],
                "extractor": "llm-windowed@2",
                "type_raw": "person",
            },
        ),
    }


def test_an_unmatched_relation_is_related_to_with_its_raw_phrase():
    _nodes, edges = ka._semantic_nodes_edges("art-1", "ir-1", "p", "", _windowed_payload())
    relations = [
        (e.relation_type, e.content.get("rel_raw"), e.content.get("extractor"))
        for e in edges
        if e.relation_type != "mentions"
    ]
    assert relations == [("related_to", "grinds", "llm-windowed@2")]


def test_two_long_names_sharing_40_characters_get_two_nodes():
    """Slugs were cut at 40 characters: the second entity overwrote the first."""
    stem = "Temperature dependence of the ionic conductivity in"
    payload = {
        "nodes": [
            {"name": f"{stem} tetragonal zirconia", "type": "finding"},
            {"name": f"{stem} cubic zirconia", "type": "finding"},
        ],
        "edges": [
            {
                "source": f"{stem} tetragonal zirconia",
                "target": f"{stem} cubic zirconia",
                "relation": "contrasts",
            }
        ],
    }
    nodes, edges = ka._semantic_nodes_edges("art-1", "ir-1", "p", "", payload)
    ids = [n.node_id for n in nodes]
    assert len(set(ids)) == 2
    assert all(len(i.removeprefix("node-sem-").removesuffix("-art-1")) <= 40 for i in ids)
    (relation,) = [e for e in edges if e.relation_type != "mentions"]
    assert {relation.source_id, relation.target_id} == set(ids)


def test_short_names_keep_their_ids():
    nodes, _ = ka._semantic_nodes_edges(
        "art-1", "ir-1", "p", "", {"nodes": [{"name": "Nek5000", "type": "software"}]}
    )
    assert nodes[0].node_id == "node-sem-nek5000-art-1"


# ── D-2: alias-aware endpoint ───────────────────────────────────────────────


def _ep(model: str, served: str, url: str) -> Any:
    return SimpleNamespace(
        project_id="p", model=model, metadata={"served_model_name": served}, litellm_url=url
    )


def test_a_role_alias_selects_the_endpoint_serving_its_model(monkeypatch):
    class _Registry:
        def list_all(self, project_id=None, **_kw):
            eps = [_ep("big-model", "big", "http://a:1"), _ep("Qwen3-30B", "qwen3", "http://b:1")]
            return [e for e in eps if project_id in (None, e.project_id)]

    monkeypatch.setattr("umrs.orchestrator.endpoint_registry.get_registry", _Registry)
    monkeypatch.setattr(
        "umrs.orchestrator.provision_roster.role_alias_models",
        lambda alias: ("Qwen3-30B",) if alias == "svc-fast" else (),
    )
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_MODEL", "svc-fast")
    assert ka._extract_chat_endpoint("p") == ("http://b:1", "qwen3")


def test_a_plain_model_name_still_matches_by_name(monkeypatch):
    class _Registry:
        def list_all(self, **_kw):
            return [_ep("m1", "m1", "http://a:1"), _ep("m2", "m2", "http://b:1")]

    monkeypatch.setattr("umrs.orchestrator.endpoint_registry.get_registry", _Registry)
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_MODEL", "m2")
    assert ka._extract_chat_endpoint("p") == ("http://b:1", "m2")


# ── D-2: extractor version ──────────────────────────────────────────────────


class _Pool:
    def __init__(self, fetchval: Any = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._fetchval = fetchval

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        return self._fetchval

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        return "UPDATE 1"


def test_the_extractor_version_is_2():
    assert ka._KNOWLEDGE_EXTRACTOR_VERSION == "2.0.0"


def test_the_understood_probe_asks_for_this_extractor_version():
    """It matched any current IR: a version bump never re-extracted."""
    pool = _Pool(fetchval=None)
    store = SimpleNamespace(_pool=pool)
    assert asyncio.run(ka._existing_current_ir(store, "art-1")) is None
    ((sql, args),) = pool.calls
    assert "extractor_version" in sql
    assert args == ("art-1", "2.0.0")


def test_a_new_extractor_ir_is_a_new_row_version_and_retires_the_old(monkeypatch):
    inserted: list[dict[str, Any]] = []

    class _Store:
        _pool = _Pool(fetchval=2)

        async def insert_ir(self, **kw: Any) -> bool:
            inserted.append(kw)
            return True

    store = _Store()
    ir = SimpleNamespace(ir_id="ir-new", to_dict=lambda: {}, content_hash="h")
    result = ka.KnowledgeUnderstandResult()
    assert asyncio.run(ka._store_ir(store, ir, "art-1", "document", result)) is True
    assert inserted[0]["version"] == 2
    retire_sql, retire_args = store._pool.calls[-1]
    assert "superseded" in retire_sql and retire_args == ("art-1", "ir-new")


def test_an_ir_already_there_retires_nothing():
    class _Store:
        _pool = _Pool(fetchval=3)

        async def insert_ir(self, **_kw: Any) -> bool:
            return False

    store = _Store()
    ir = SimpleNamespace(ir_id="ir-x", to_dict=lambda: {}, content_hash="h")
    result = ka.KnowledgeUnderstandResult()
    assert asyncio.run(ka._store_ir(store, ir, "art-1", "document", result)) is False
    assert not any("superseded" in sql for sql, _ in store._pool.calls)


# ── D-2: extract modes on the job specs ─────────────────────────────────────


class _Svc:
    def __init__(self) -> None:
        self.submitted: list[Any] = []

    def submit_job(self, submission: Any) -> str:
        self.submitted.append(submission)
        return f"job-{len(self.submitted)}"


def _job(spec: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(job_id="parent", spec=spec or {}, priority=5, project_id="p")


def test_a_user_upload_understands_in_full(monkeypatch):
    svc = _Svc()
    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: svc)
    ka._enqueue_understand_job("p", "u", _job(), [{"artifact_id": "a", "content_hash": "h"}])
    assert svc.submitted[0].spec["extract_mode"] == "full"


def test_the_auto_ingest_chain_defaults_to_lite(monkeypatch):
    svc = _Svc()
    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: svc)
    monkeypatch.setattr(ka, "_store_artifact_provenance", lambda *_a: None)
    monkeypatch.setattr(ka, "_create_produced_by_edges", lambda *_a: None)
    deduped = [{"artifact_id": "a", "content_hash": "h", "source_ref": "s3://b/k"}]
    ka._enqueue_ingest_chain(deduped, "b", "p", "u", _job(), None, "wf", "*")
    ingest, understand = (s.spec for s in svc.submitted)
    assert ingest["extract_mode"] == "lite"
    assert understand["extract_mode"] == "lite"


def test_the_auto_mode_is_configurable(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_AUTO_MODE", "off")
    assert ka.auto_extract_mode() == "off"
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_AUTO_MODE", "bogus")
    assert ka.auto_extract_mode() == "lite"


def test_an_ingest_job_passes_its_mode_to_the_understand_job_it_chains(monkeypatch):
    svc = _Svc()
    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: svc)
    ka._enqueue_understand_job("p", "u", _job({"extract_mode": "lite"}), [{"artifact_id": "a"}])
    assert svc.submitted[0].spec["extract_mode"] == "lite"


def test_a_run_may_ask_for_a_mode_for_its_outputs(monkeypatch):
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "umrs.jobs.knowledge_adapters.enqueue_knowledge_ingest",
        lambda **kw: calls.append(kw),
    )
    from umrs.jobs.adapters import enqueue_auto_ingest

    job = SimpleNamespace(
        job_id="j",
        job_type="workflow_run",
        spec={"ingest_results": True, "extract_mode": "full"},
    )
    enqueue_auto_ingest(job, SimpleNamespace())
    assert calls[0]["extract_mode"] == "full"


# ── D-1 parser seam ─────────────────────────────────────────────────────────


def test_docling_missing_falls_back_to_the_existing_text(monkeypatch, tmp_path):
    from umrs.knowledge import doc_parse

    monkeypatch.setenv("UMRS_KNOWLEDGE_DOC_PARSER", "docling")
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", str(tmp_path))
    monkeypatch.setitem(sys.modules, "docling", None)  # import fails
    monkeypatch.setitem(sys.modules, "docling.datamodel", None)
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", None)
    parsed = doc_parse.parse_for_extraction(
        b"%PDF-1.7 ...", "paper.pdf", "document", fallback_text="pypdf text"
    )
    assert parsed.text == "pypdf text"
    assert parsed.parser == "pypdf"
    assert parsed.notes == ("doc_parser_fallback:docling_not_installed",)


def test_docling_without_staged_artifacts_falls_back_by_name(monkeypatch, tmp_path):
    from umrs.knowledge import doc_parse

    monkeypatch.setenv("UMRS_KNOWLEDGE_DOC_PARSER", "docling")
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", str(tmp_path / "absent"))
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert parsed.notes == ("doc_parser_fallback:artifacts_missing",)


def test_the_default_parser_is_the_existing_path():
    from umrs.knowledge import doc_parse

    parsed = doc_parse.parse_for_extraction(b"# Note\nZirconia.", "note.md", "document")
    assert (parsed.text, parsed.parser, parsed.notes) == ("# Note\nZirconia.", "raw", ())


def test_docling_runs_in_the_sandbox_worker_when_the_site_opted_in(monkeypatch, tmp_path):
    from umrs.knowledge import doc_parse, embed_sandbox

    monkeypatch.setenv("UMRS_KNOWLEDGE_DOC_PARSER", "docling")
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", str(tmp_path))
    monkeypatch.setenv("UMRS_EMBEDDING_SANDBOX", "podman")
    monkeypatch.setenv("UMRS_EMBEDDING_SANDBOX_IMAGE", "umrs-server:test")
    argvs: list[list[str]] = []

    def runner(argv: list[str], *, input_text: str, timeout: float) -> Any:
        argvs.append(argv)
        request = json.loads(input_text)
        assert request["op"] == "parse" and request["filename"] == "a.pdf"
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"text": "# Title", "version": "2.9"}), stderr=""
        )

    monkeypatch.setattr(embed_sandbox, "_default_runner", runner)
    parsed = doc_parse.parse_for_extraction(b"%PDF-1.7", "a.pdf", "document", fallback_text="t")
    assert (parsed.text, parsed.parser) == ("# Title", "docling@2.9")
    argv = argvs[0]
    assert argv[-3:] == ["python", "-m", "umrs.knowledge.ingest_worker"]
    assert "--network=none" in argv
    assert f"{tmp_path}:/models-ingest/docling:ro" in argv


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "/models/docling"),
        ({"UMRS_KNOWLEDGE_MODELS_HOST_DIR": "/srv/r4/models"}, "/srv/r4/models/docling"),
        (
            {"UMRS_KNOWLEDGE_MODELS_HOST_DIR": "/srv/m", "X_HOST_DIR": "/explicit"},
            "/explicit",
        ),
    ],
)
def test_the_sandbox_mounts_the_host_path_of_a_model_under_models(monkeypatch, env, expected):
    """On umrs-front the server sees the host's ./models as /models; the
    sandbox mounted the server path, which does not exist on the host."""
    from umrs.knowledge.doc_parse import sandbox_host_path

    monkeypatch.delenv("UMRS_KNOWLEDGE_MODELS_HOST_DIR", raising=False)
    monkeypatch.delenv("X_HOST_DIR", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert sandbox_host_path("/models/docling", "X_HOST_DIR") == expected
    assert sandbox_host_path("/elsewhere/m", "Y_UNSET") == "/elsewhere/m"


def test_gliner_in_the_sandbox_mounts_the_host_model_dir(monkeypatch):
    from umrs.knowledge import embed_sandbox, pretag

    monkeypatch.setenv("UMRS_KNOWLEDGE_MODELS_HOST_DIR", "/srv/r4/models")
    monkeypatch.delenv("UMRS_KNOWLEDGE_GLINER_MODEL_HOST_DIR", raising=False)
    seen: list[Any] = []

    def run_worker(_config: Any, request: dict[str, Any], **kw: Any) -> dict[str, Any]:
        seen.append(kw["mounts"])
        return {"candidates": [[] for _ in request["texts"]]}

    monkeypatch.setattr(embed_sandbox, "run_worker", run_worker)
    pretag._tag_in_sandbox(object(), "/models/gliner_medium-v2.1", ["t"], ["material"])
    assert seen == [[("/srv/r4/models/gliner_medium-v2.1", "/models-ingest/gliner")]]


def test_the_embed_sandbox_argv_is_unchanged():
    from umrs.knowledge.embed_sandbox import SandboxConfig, build_argv

    argv = build_argv(SandboxConfig(image="img", model_host_dir="/m"))
    assert argv[-1] == "umrs.knowledge.embed_worker"
    assert argv.count("-v") == 1


def test_the_understand_job_hands_the_original_bytes_to_the_parser(monkeypatch):
    seen: list[tuple[bytes, str]] = []

    def parse(data: bytes, filename: str, kind: str, *, fallback_text: str | None = None):
        from umrs.knowledge.doc_parse import ParsedText

        seen.append((data, filename))
        return ParsedText(text="parsed", parser="docling@x", notes=("doc_parser_fallback:t",))

    monkeypatch.setattr("umrs.knowledge.doc_parse.parse_for_extraction", parse)
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(ka, "_llm_extract_document", lambda t, **kw: calls.append((t, kw)))
    notes: list[str] = []
    inputs = ka._ExtractInputs(raw=b"%PDF-raw", source_name="p.pdf")
    ka._nodes_edges_for_artifact(
        "a", "i", "document", b"pypdf text", _EMPTY, "p", "", notes=notes, inputs=inputs
    )
    assert seen == [(b"%PDF-raw", "p.pdf")]
    assert calls[0][0] == "parsed" and calls[0][1]["parser_id"] == "docling@x"
    assert notes == ["doc_parser_fallback:t"]


# ── D-3 pre-tagging ─────────────────────────────────────────────────────────


def test_pretag_unset_is_silent():
    from umrs.knowledge import pretag

    notes: list[str] = []
    assert pretag.window_tagger(FakeVocab(), notes) is None
    assert pretag.pretag_windows(["text"], ["material"], notes) is None
    assert notes == []


def test_pretag_with_a_missing_dir_is_skipped_by_name(monkeypatch, tmp_path):
    from umrs.knowledge import pretag

    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", str(tmp_path / "absent"))
    notes: list[str] = []
    assert pretag.pretag_windows(["text"], ["material"], notes) is None
    assert notes == ["pretag_unavailable:model_dir_missing"]


def test_pretag_without_the_library_is_skipped_by_name(monkeypatch, tmp_path):
    from umrs.knowledge import pretag

    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", str(tmp_path))
    monkeypatch.setitem(sys.modules, "gliner", None)
    monkeypatch.setattr(pretag, "_MODELS", {})
    notes: list[str] = []
    tagger = pretag.window_tagger(FakeVocab(), notes)
    assert tagger is not None
    assert tagger(["a", "b"]) == [[], []]
    assert notes == ["pretag_unavailable:gliner_not_installed"]


def test_pretag_tags_each_window_with_the_vocabulary_labels(monkeypatch, tmp_path):
    from umrs.knowledge import pretag

    class _Model:
        def predict_entities(self, _text: str, labels: list[str], threshold: float) -> list:
            assert labels == ["material", "method"] and threshold == 0.5
            return [
                {"text": "Zirconia", "label": "material", "score": 0.7},
                {"text": "zirconia", "label": "material", "score": 0.9},
            ]

    monkeypatch.setenv("UMRS_KNOWLEDGE_GLINER_MODEL", str(tmp_path))
    monkeypatch.setattr(pretag, "_MODELS", {str(tmp_path): _Model()})
    notes: list[str] = []
    tagged = pretag.window_tagger(FakeVocab(), notes)(["w1", "w2"])
    assert tagged == [[{"text": "zirconia", "label": "material", "score": 0.9}]] * 2
    assert notes == []


# ── the understand job: off the loop, vocabulary once, ER hook ──────────────


class _Store:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    async def insert_ir(self, **_kw: Any) -> bool:
        return True

    async def upsert_node(self, row: Any) -> None:
        self.rows.append(row)

    async def upsert_edge(self, row: Any) -> None:
        self.rows.append(row)


def _run_job(monkeypatch, *, er_module: Any, spec: dict[str, Any] | None = None, llm: Any = None):
    """One understand job over two artifacts. ``llm``: a stand-in for
    ``_llm_extract_document`` (default: a recorder), or ``"real"`` to keep
    the real one (then patch the chat seam)."""
    store = _Store()
    sources = [
        {"artifact_id": f"art-{i}", "content_hash": f"h{i}", "source_ref": f"k{i}"}
        for i in range(2)
    ]
    resolved: list[tuple[str, str]] = []
    extract_calls: list[dict[str, Any]] = []
    vocab = FakeVocab()

    def resolve(project_id: str, graph_id: str) -> Any:
        resolved.append((project_id, graph_id))
        return vocab, ["ontology_fallback:test"]

    def fake_llm(text: str, **kw: Any) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
            kw["on_loop"] = True
        except RuntimeError:
            kw["on_loop"] = False
        extract_calls.append(kw)
        return _windowed_payload()

    async def to_process(srcs: Any, _store: Any, _p: str) -> Any:
        return srcs

    async def prepared(_job, _att, _store, _s3, _b, src, _result, raw_sink=None):
        if raw_sink is not None:
            raw_sink["raw"] = b"raw bytes"
        return src["artifact_id"], "document", b"plain prose", dict(_EMPTY)

    async def no(*_a: Any, **_k: Any) -> Any:
        return None

    async def not_yet(*_a: Any, **_k: Any) -> bool:
        return False

    monkeypatch.setattr("umrs.knowledge.llm_extract.resolve_vocabulary", resolve)
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", lambda _pool: store)
    monkeypatch.setattr(ka, "_artifacts_to_process", to_process)
    monkeypatch.setattr(ka, "_understand_input", prepared)
    monkeypatch.setattr(ka, "_next_ir_version", lambda *_a: asyncio.sleep(0, 1))
    monkeypatch.setattr(ka, "_retire_older_irs", no)
    monkeypatch.setattr(ka, "_already_understood", not_yet)
    if llm != "real":
        monkeypatch.setattr(ka, "_llm_extract_document", llm or fake_llm)
    monkeypatch.setattr(ka, "_derive_a2l_link", lambda *_a, **_k: None)
    monkeypatch.setattr(ka, "_compile_skill_proposals", lambda *_a, **_k: asyncio.sleep(0, 0))
    monkeypatch.setattr(ka, "_label_embedder", lambda: (None, "e5"))
    monkeypatch.setattr(ka.KnowledgeEvent, "emit", lambda self: self)
    monkeypatch.setitem(sys.modules, "umrs.knowledge.entity_resolution", er_module)
    job = SimpleNamespace(
        job_id="job-u", spec={"graph_id": "", **(spec or {})}, project_id="p", user_id="u"
    )
    result = asyncio.run(
        ka._run_understand(job, SimpleNamespace(attempt_id="a"), "p", sources, object(), None, "b")
    )
    return result, store, resolved, extract_calls, vocab


def _er_module(calls: list[dict[str, Any]], proposed: int = 2) -> Any:
    async def propose_same_as(store: Any, **kw: Any) -> Any:
        calls.append({"store": store, **kw})
        return SimpleNamespace(proposed=proposed, candidates=5, notes=["er_note"])

    return SimpleNamespace(propose_same_as=propose_same_as)


def test_extraction_runs_off_the_event_loop(monkeypatch):
    """_nodes_edges_for_artifact ran inline: N windows x 60 s froze the API loop."""
    _result, _store, _resolved, calls, _vocab = _run_job(monkeypatch, er_module=_er_module([]))
    assert calls and all(c["on_loop"] is False for c in calls)


def test_the_vocabulary_is_resolved_once_per_job_and_passed_down(monkeypatch):
    result, _store, resolved, calls, vocab = _run_job(monkeypatch, er_module=_er_module([]))
    assert resolved == [("p", "")]
    assert len(calls) == 2 and all(c["vocab"] is vocab for c in calls)
    assert all(c["extract_mode"] == "full" and c["parser_id"] == "raw" for c in calls)
    assert "ontology_fallback:test" in result.extract_notes


def test_the_job_spec_mode_reaches_the_extractor(monkeypatch):
    _result, _store, _resolved, calls, _vocab = _run_job(
        monkeypatch, er_module=_er_module([]), spec={"extract_mode": "lite"}
    )
    assert {c["extract_mode"] for c in calls} == {"lite"}


def test_entity_resolution_gets_the_semantic_node_ids_of_the_job(monkeypatch):
    er_calls: list[dict[str, Any]] = []
    result, store, _resolved, _calls, _vocab = _run_job(monkeypatch, er_module=_er_module(er_calls))
    (call,) = er_calls
    node_ids = [getattr(r, "node_id", "") for r in store.rows]
    written = [i for i in node_ids if i.startswith("node-sem-")]
    assert len(written) == 4
    assert call == {
        "store": store,
        "project_id": "p",
        "graph_id": "",
        "node_ids": written,
        "job_id": "job-u",
        "embed_labels": None,
        "embedder_name": "e5",
    }
    assert (result.same_as_proposed, result.degraded) == (2, False)
    assert {"same_as:proposed=2,candidates=5", "er_note"} <= set(result.extract_notes)


def test_the_attempt_result_and_ref_carry_the_proposals(monkeypatch):
    result, *_rest = _run_job(monkeypatch, er_module=_er_module([]))
    attempt = SimpleNamespace(metrics={})
    ka._write_result(attempt, result)
    stored = json.loads(attempt.metrics["result"])
    assert stored["same_as_proposed"] == 2
    ref = "knowledge_understand|project=p|ir_ids=2|already=0|fail=0|no_object=0|partial=0|same_as=2"
    assert ka._knowledge_ref_counts(ref, "ir_ids") == 2
    assert ka._knowledge_ref_counts(ref, "same_as") == 2


def test_a_missing_resolution_module_is_named_not_fatal(monkeypatch):
    result, *_rest = _run_job(monkeypatch, er_module=None)
    assert "entity_resolution_unavailable:module_missing" in result.extract_notes
    assert result.same_as_proposed == 0 and len(result.ir_ids) == 2


def test_no_semantic_nodes_means_no_resolution_call(monkeypatch):
    er_calls: list[dict[str, Any]] = []
    monkeypatch.setenv("UMRS_KNOWLEDGE_LLM_EXTRACT", "0")
    result, *_rest = _run_job(
        monkeypatch, er_module=_er_module(er_calls), spec={"extract_mode": "off"}
    )
    assert er_calls == []
    assert result.same_as_proposed == 0


def test_the_label_embedder_reads_the_first_vector_of_each_label(monkeypatch):
    class _Lane:
        def chunk_and_embed(self, texts: list[str]) -> list[dict[str, Any]]:
            return [{"chunks": [{"text": t}], "vectors": [[1.0, float(len(t))]]} for t in texts]

    monkeypatch.setattr(ka, "_chunk_embedder_for_site", lambda: (_Lane(), ""))
    embed, name = ka._label_embedder()
    assert embed(["ab", "abc"]) == [[1.0, 2.0], [1.0, 3.0]]
    assert name


def test_no_site_embedder_means_no_label_vectors(monkeypatch):
    monkeypatch.setattr(ka, "_chunk_embedder_for_site", lambda: (None, "no embedder"))
    assert ka._label_embedder() == (None, "")


# ── agent_executor: JSON mode and reasoning replies ─────────────────────────


def test_chat_sends_response_format_and_reads_reasoning_content(monkeypatch):
    import httpx
    from umrs.knowledge import agent_executor

    bodies: list[dict[str, Any]] = []

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": None, "reasoning_content": "{}"}}]}

    def post(url: str, **kw: Any) -> _Resp:
        bodies.append(kw["json"])
        return _Resp()

    monkeypatch.setattr(httpx, "post", post)
    reply = agent_executor._default_litellm_chat(
        [{"role": "user", "content": "x"}],
        model="m",
        base_url="http://l:1",
        api_key="k",
        response_format={"type": "json_object"},
        reasoning_fallback=True,
    )
    assert reply == "{}"
    assert bodies[0]["response_format"] == {"type": "json_object"}
    # existing callers: no field sent, content as before
    agent_executor._default_litellm_chat(
        [{"role": "user", "content": "x"}], model="m", base_url="http://l:1", api_key="k"
    )
    assert "response_format" not in bodies[1]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"content": "hi"}, "hi"),
        ({"content": None, "reasoning_content": "r"}, "r"),
        ({"content": "", "reasoning": "r2"}, "r2"),
        ({"content": None}, ""),
    ],
)
def test_message_text(message, expected):
    from umrs.knowledge.agent_executor import message_text

    assert message_text(message) == expected


# ── review fixes: timeout, job budget, parse order, degraded notes ──────────


def test_the_call_timeout_covers_a_reply_at_the_token_cap(monkeypatch):
    """4000 tokens under a 60 s timeout timed out at ~57 tok/s per stream;
    the cap is 2500 now and the timeout covers it at 25 tok/s."""
    monkeypatch.delenv("UMRS_KNOWLEDGE_EXTRACT_TIMEOUT", raising=False)
    monkeypatch.setenv("UMRS_AGENT_LITELLM_URL", "http://litellm:4000")
    seen: list[dict[str, Any]] = []

    def chat(_messages: list[dict[str, Any]], **kw: Any) -> str:
        seen.append(kw)
        return json.dumps({"entities": [], "relations": []})

    monkeypatch.setattr("umrs.knowledge.agent_executor._default_litellm_chat", chat)
    ka._llm_extract_document(_prose(5), project_id="p", vocab=FakeVocab())
    timeout, max_tokens = seen[0]["timeout"], seen[0]["max_tokens"]
    assert max_tokens == 2500
    assert timeout >= max_tokens / ka._EXTRACT_MIN_DECODE_TOK_S >= 100


@pytest.mark.parametrize(("raw", "expected"), [("90", 90.0), ("x", 100.0)])
def test_an_explicit_timeout_still_wins(monkeypatch, raw, expected):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_TIMEOUT", raw)
    assert ka._extract_timeout(2500) == expected


def test_the_extraction_budget_is_per_job_not_per_document(monkeypatch):
    """Per-document budgets let N documents hold the scheduler N x 900 s:
    the job budget cuts the first document, and the second one, started
    after it is spent, reads one window (lite) instead of a full budget."""
    import time

    from umrs.knowledge import llm_extract

    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_BUDGET_S", "30")
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_JOB_BUDGET_S", "1")
    monkeypatch.setattr(llm_extract, "LITE_FLOOR_S", 0.5)
    monkeypatch.setenv("UMRS_AGENT_LITELLM_URL", "http://litellm:4000")

    def slow_chat(_messages: list[dict[str, Any]], **_kw: Any) -> str:
        time.sleep(1.5)
        return json.dumps({"entities": [{"name": "Zirconia", "type": "material"}]})

    monkeypatch.setattr("umrs.knowledge.agent_executor._default_litellm_chat", slow_chat)
    started = time.monotonic()
    result, store, *_rest = _run_job(monkeypatch, er_module=_er_module([]), llm="real")
    assert time.monotonic() - started < 3.4  # 1 s job budget + one lite call (+ slack)
    assert result.degraded and "budget exhausted" in result.degraded_reason
    assert "extract_job_budget_exhausted" in result.extract_notes
    sem = [i for r in store.rows if (i := str(getattr(r, "node_id", ""))).startswith("node-sem-")]
    assert len(sem) == 1  # the second document still got its one window


def test_a_docling_parse_past_its_time_falls_back_by_name(monkeypatch, tmp_path):
    import time

    from umrs.knowledge import doc_parse

    monkeypatch.setenv("UMRS_KNOWLEDGE_DOC_PARSER", "docling")
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCLING_ARTIFACTS", str(tmp_path))
    monkeypatch.setattr(doc_parse, "docling_convert", lambda *_a: time.sleep(2))
    started = time.monotonic()
    parsed = doc_parse.parse_for_extraction(
        b"%PDF-1", "p.pdf", "document", fallback_text="pypdf text", timeout_s=0.1
    )
    assert time.monotonic() - started < 1.0
    assert (parsed.text, parsed.notes) == ("pypdf text", ("doc_parser_fallback:timeout",))
    spent = doc_parse.parse_for_extraction(
        b"%PDF-1", "p.pdf", "document", fallback_text="t", timeout_s=0
    )
    assert spent.notes == ("doc_parser_fallback:budget_exhausted",)


def _recording_parser(monkeypatch) -> list[str]:
    calls: list[str] = []

    def parse(_data: bytes, filename: str, _kind: str, **_kw: Any) -> Any:
        from umrs.knowledge.doc_parse import ParsedText

        calls.append(filename)
        return ParsedText(text="parsed", parser="docling@x")

    monkeypatch.setattr("umrs.knowledge.doc_parse.parse_for_extraction", parse)
    return calls


def _pdf_artifact(notes: list[str]) -> None:
    inputs = ka._ExtractInputs(raw=b"%PDF-raw", source_name="p.pdf")
    ka._nodes_edges_for_artifact(
        "a", "i", "document", b"pypdf text", _EMPTY, "p", "", notes=notes, inputs=inputs
    )


def test_extraction_switched_off_parses_nothing(monkeypatch):
    calls = _recording_parser(monkeypatch)
    monkeypatch.setenv("UMRS_KNOWLEDGE_LLM_EXTRACT", "0")
    notes: list[str] = []
    _pdf_artifact(notes)
    assert calls == [] and notes == []


def test_no_chat_endpoint_means_no_docling_parse(monkeypatch):
    calls = _recording_parser(monkeypatch)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOC_PARSER", "docling")
    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _p: None)
    notes: list[str] = []
    _pdf_artifact(notes)
    assert calls == []
    assert notes and notes[0].startswith("semantic extraction unavailable: no READY")


def test_the_endpoint_resolved_before_the_parse_is_reused(monkeypatch):
    calls = _recording_parser(monkeypatch)
    monkeypatch.setenv("UMRS_KNOWLEDGE_DOC_PARSER", "docling")
    lookups: list[str] = []

    def endpoint(project_id: str) -> tuple[str, str]:
        lookups.append(project_id)
        return "http://l:1", "m"

    monkeypatch.setattr(ka, "_extract_chat_endpoint", endpoint)
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(ka, "_llm_extract_document", lambda _t, **kw: seen.append(kw))
    _pdf_artifact([])
    assert calls == ["p.pdf"] and lookups == ["p"]
    assert seen[0]["target"] == ("http://l:1", "m")


def test_parser_and_window_notes_do_not_degrade_a_run_that_extracted(monkeypatch):
    def llm(_text: str, **kw: Any) -> dict[str, Any]:
        kw["notes"].extend(["doc_parser_fallback:artifacts_missing", "extract_windows_failed:1"])
        return _windowed_payload()

    result, *_rest = _run_job(monkeypatch, er_module=_er_module([]), llm=llm)
    assert result.degraded is False
    notes = result.extract_notes
    assert notes.count("doc_parser_fallback:artifacts_missing") == 1
    assert "extract_windows_failed:1" in notes


def test_no_semantic_payload_still_degrades_the_run(monkeypatch):
    def llm(_text: str, **kw: Any) -> None:
        kw["notes"].append("semantic extraction unavailable: x")

    result, *_rest = _run_job(monkeypatch, er_module=_er_module([]), llm=llm)
    assert result.degraded
    assert result.degraded_reason.startswith("semantic extraction unavailable: x")
