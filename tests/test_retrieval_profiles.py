"""The retrieval profile UMRS reads with (ADR-093 D-14, C18).

The stand-alone reader is ``llama3.2:3b``: the small profile, which must
behave exactly as before D-14. The rikyu deployment reads with ``kimi-k3``
and ``glm-5.3``, frontier MoE models with 1M-token windows: the large
profile (k_max 10, partial 6, pool 100, merge on, no explicit answer
limit). Pinned here: the resolution order (request > UMRS_RAG_PROFILE > the
size rule), the route's defaults per profile with explicit parameters
winning, the env site defaults, the compose path sized by the registry's
answer model, the loop store sized by the loop's model, and the partial cap
and ``lanes.profile`` report of the core.

Lead decision for the maintainer (2026-10-10), which re-pins C18's answers:
ONE size rule (the EFFECTIVE size, shared with R4 through
``r4_contracts.models.profile_for``), so a mixture-of-experts reader such as
``qwen3-30b-a3b`` or ``gpt-oss-20b`` reads with medium; and a NAMED model of
unknown scale reads with medium. A request that names no reader keeps the
site's own defaults (small), as before D-14.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import rag as rag_api
from umrs.api import rag_retrieve, rag_spine
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import agent_executor
from umrs.knowledge.retrieval import calibrate, pipeline, profiles
from umrs.knowledge.retrieval import hydrate as hydrate_mod
from umrs.knowledge.retrieval.calibrate import NullStats
from umrs.knowledge.retrieval.types import Candidate, LaneResult, RetrieveSpec

from r4_contracts.models import RETRIEVAL_PROFILES
from r4_contracts.retrieval import RetrievalSufficiency, RetrievedPassage, RetrieveResult

_ENV = (
    "UMRS_RAG_PROFILE",
    "UMRS_RAG_FINAL_K_MAX",
    "UMRS_RAG_POOL",
    "UMRS_RAG_PER_ARTIFACT",
    "UMRS_RAG_ANSWER_MAX_TOKENS",
    "UMRS_RAG_GRAPH_LANE",
    "UMRS_RAG_MERGE",
    "UMRS_AGENT_MODEL",
    "UMRS_AGENT_LITELLM_URL",
    "UMRS_KNOWLEDGE_EXTRACT_MODEL",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", None)
    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _project: None)
    calibrate.clear_cache()


# ── resolution ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "name"),
    [
        ("llama3.2:3b", "small"),
        ("kimi-k3", "large"),
        ("glm-5.3", "large"),
        ("GLM-5.3-FP8@rikyu", "large"),
        ("qwen2.5:7b", "medium"),
        # Lead decision 2026-10-10: the effective size (C18 read these large)
        # and a named unknown reads with medium (C18 read it small).
        ("qwen3-30b-a3b", "medium"),
        ("gpt-oss-20b", "medium"),
        ("gpt-oss-120b", "large"),
        ("mystery-model", "medium"),
        # Review of ADR-093 (profiles-graph-2): ollama's MoE tags carry no
        # -a<n>b tag; the shared family table sizes them (R4 and UMRS alike).
        ("qwen3:30b", "medium"),
        ("qwen3-coder:30b", "medium"),
        ("deepseek-v2:16b", "medium"),
        ("mixtral", "large"),
        ("qwen3:32b", "large"),  # the dense sibling keeps its stated size
        # No reader named: the site's own defaults, as before D-14.
        (None, "small"),
        ("", "small"),
        ("  ", "small"),
    ],
)
def test_auto_reads_the_profile_from_the_reading_models_name(model, name):
    assert profiles.resolve_profile(model) == (name, RETRIEVAL_PROFILES[name])


def test_the_request_wins_over_the_site_which_wins_over_the_model(monkeypatch):
    monkeypatch.setenv("UMRS_RAG_PROFILE", "large")
    assert profiles.resolve_profile("llama3.2:3b")[0] == "large"
    assert profiles.resolve_profile("kimi-k3", " Small ")[0] == "small"
    assert profiles.resolve_profile("kimi-k3", "auto")[0] == "large"


@pytest.mark.parametrize(("value", "site"), [("auto", "auto"), ("MEDIUM", "medium"), ("x", "auto")])
def test_the_site_setting_is_read_and_a_bad_value_is_auto(monkeypatch, value, site):
    monkeypatch.setenv("UMRS_RAG_PROFILE", value)
    assert profiles.site_profile() == site


def test_small_keeps_the_site_value_and_larger_never_goes_below_it():
    assert profiles.sized("small", 3, 4) == 3  # a site's own small value stands
    assert profiles.sized("large", 4, 10) == 10
    assert profiles.sized("large", 12, 10) == 12


@pytest.mark.parametrize(
    ("asked", "name", "mode"),
    [
        ("auto", "small", "auto"),
        (None, "large", "on"),
        ("off", "large", "off"),
        ("on", "small", "on"),
    ],
)
def test_merge_defaults_to_the_profile_and_an_explicit_value_wins(asked, name, mode):
    assert profiles.merge_mode(asked, RETRIEVAL_PROFILES[name]) == mode


@pytest.mark.parametrize(
    ("name", "site", "limit"), [("small", 250, 250), ("medium", 400, 600), ("large", 400, None)]
)
def test_the_answer_limit_per_profile(name, site, limit):
    assert profiles.answer_limit(name, RETRIEVAL_PROFILES[name], site) == limit


# ── the retrieve verb ───────────────────────────────────────────────────────


@pytest.fixture
def route(monkeypatch):
    specs: list[RetrieveSpec] = []

    def retrieve(_pool, spec, **_kwargs):
        specs.append(spec)
        return RetrieveResult(query="q")

    monkeypatch.setattr(rag_api, "_require_rag_scope", lambda *_a, **_k: ("u1", "p1"))
    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_retrieve, "hidden_graphs", lambda *_a: ())
    monkeypatch.setattr(rag_retrieve, "batch_encoder", lambda: "ENCODER")
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: "FETCH")
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = object()
    client = TestClient(app)

    def get(**params):
        response = client.get("/umrs/rag/retrieve", params={"q": "q", **params})
        assert response.status_code == 200, response.text
        return specs[-1]

    return get


def _sizes(spec):
    return (spec.k_max, spec.pool, spec.merge, spec.partial_k, spec.profile, spec.reader_model)


@pytest.mark.parametrize(
    ("params", "sizes"),
    [
        ({}, (4, 50, "auto", 2, "small", "")),  # an older R4: unchanged
        ({"reader_model": "llama3.2:3b"}, (4, 50, "auto", 2, "small", "llama3.2:3b")),
        ({"reader_model": "kimi-k3"}, (10, 100, "on", 6, "large", "kimi-k3")),
        ({"reader_model": "glm-5.3"}, (10, 100, "on", 6, "large", "glm-5.3")),
        ({"reader_model": "qwen2.5:7b"}, (6, 50, "on", 3, "medium", "qwen2.5:7b")),
        # Lead decision 2026-10-10: one effective-size rule; unknown -> medium.
        ({"reader_model": "qwen3-30b-a3b"}, (6, 50, "on", 3, "medium", "qwen3-30b-a3b")),
        ({"reader_model": "house-model"}, (6, 50, "on", 3, "medium", "house-model")),
        ({"profile": "large"}, (10, 100, "on", 6, "large", "")),
    ],
)
def test_the_route_defaults_per_profile(route, params, sizes):
    assert _sizes(route(**params)) == sizes


def test_explicit_parameters_win_over_the_profile(route):
    spec = route(reader_model="kimi-k3", k_max=3, pool=20, merge="off")
    assert _sizes(spec) == (3, 20, "off", 6, "large", "kimi-k3")


def test_a_request_profile_wins_over_the_reader_and_the_site(route, monkeypatch):
    monkeypatch.setenv("UMRS_RAG_PROFILE", "large")
    assert route(reader_model="llama3.2:3b").profile == "large"
    assert route(reader_model="kimi-k3", profile="small").profile == "small"


def test_site_defaults_apply_to_small_and_only_widen_a_larger_profile(route, monkeypatch):
    monkeypatch.setenv("UMRS_RAG_FINAL_K_MAX", "3")
    monkeypatch.setenv("UMRS_RAG_POOL", "150")
    assert (route().k_max, route().pool) == (3, 150)
    assert (route(reader_model="kimi-k3").k_max, route(reader_model="kimi-k3").pool) == (10, 150)


def test_the_legacy_threshold_mode_keeps_k_five_for_every_profile(route):
    assert route(reader_model="kimi-k3", threshold=0.7).k_max == 5


@pytest.mark.parametrize("bad", [{"profile": "huge"}, {"reader_model": "m" * 201}])
def test_a_bad_profile_or_reader_model_is_refused(monkeypatch, bad):
    monkeypatch.setattr(rag_api, "_require_rag_scope", lambda *_a, **_k: ("u1", "p1"))
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = object()
    response = TestClient(app).get("/umrs/rag/retrieve", params={"q": "q", **bad})
    assert response.status_code == 422


# ── the compose path: the registry's answer model ──────────────────────────


def _passage(rank: int) -> RetrievedPassage:
    return RetrievedPassage(
        rank=rank,
        source_id=f"node-chunk-a{rank}-0",
        artifact_id=f"a{rank}",
        source_name=f"note-{rank}.md",
        text=f"Passage {rank} about alumina encapsulation.",
    )


@pytest.fixture
def compose(monkeypatch):
    seen: dict = {"chat": []}

    def retrieve(_pool, spec, **_kwargs):
        seen["spec"] = spec
        return RetrieveResult(
            query="q",
            passages=tuple(_passage(i) for i in range(1, 9)),
            sufficiency=RetrievalSufficiency(status=seen.get("status", "sufficient")),
        )

    def chat(messages, **kwargs):
        seen["chat"].append(kwargs)
        return "Alumina slows it [1]."

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    monkeypatch.setattr(agent_executor, "_default_litellm_chat", chat)

    def answer(model: str, status: str = "sufficient"):
        seen["status"] = status
        monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _p: ("http://llm", model))
        spine = rag_spine.answer_from_spine(
            object(), "q", "p1", encode=lambda _t: ("m", None, "x"), fetch_bytes=lambda _r: None
        )
        return spine, seen["spec"], seen["chat"][-1]

    return answer


def test_the_compose_path_sizes_by_the_registry_answer_model(compose):
    spine, spec, call = compose("kimi-k3")
    assert _sizes(spec) == (10, 100, "on", 6, "large", "kimi-k3")
    assert call["model"] == "kimi-k3" and call["max_tokens"] is None  # the family default
    assert spine.model == "kimi-k3"


def test_the_compose_path_keeps_a_small_model_exactly_as_before(compose):
    _spine, spec, call = compose("llama3.2:3b")
    assert _sizes(spec) == (4, 50, "auto", 2, "small", "llama3.2:3b")
    assert (call["max_tokens"], call["temperature"]) == (400, 0.0)


@pytest.mark.parametrize("model", ["qwen3-30b-a3b", "house-model"])
def test_the_compose_path_reads_a_moe_or_unknown_answer_model_as_medium(compose, model):
    """Lead decision 2026-10-10: the registry's answer model goes through the
    same effective-size rule as R4's reader, and unknown reads with medium."""
    _spine, spec, call = compose(model)
    assert _sizes(spec) == (6, 50, "on", 3, "medium", model)
    assert call["max_tokens"] == 600


@pytest.mark.parametrize(("model", "shown"), [("llama3.2:3b", 2), ("glm-5.3", 6)])
def test_a_partial_verdict_shows_the_profiles_partial_cap(compose, model, shown):
    spine, _spec, _call = compose(model, status="partial")
    assert len(spine.passages) == shown


def test_the_answer_metadata_reports_the_profile(monkeypatch):
    result = RetrieveResult(
        query="q", lanes={"profile": profiles.report("large", "kimi-k3")}, searched=1
    )
    metadata = rag_spine.SpineAnswer(retrieval=result).metadata()
    assert metadata["lanes"]["profile"] == {
        "status": "ok",
        "name": "large",
        "reader_model": "kimi-k3",
    }


# ── the loop store: the loop's model ────────────────────────────────────────


@pytest.fixture
def loop_spec(monkeypatch):
    seen = {}

    def retrieve(_pool, spec, **_kwargs):
        seen["spec"] = spec
        return RetrieveResult(query="q")

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: lambda _ref: None)
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)

    def search(**store_kwargs):
        store = rag_spine.SpineKnowledgeStore(
            pool=lambda: object(), encode=lambda _t: ("m", None, "x"), **store_kwargs
        )
        store.for_project("p").search_artifacts("goal")
        return seen["spec"]

    return search


def test_the_loop_store_sizes_by_a_named_loop_model(loop_spec):
    assert _sizes(loop_spec(reader_model="kimi-k3"))[:5] == (10, 100, "on", 6, "large")


def test_the_loop_store_reads_the_sites_agent_model_then_the_registry(loop_spec, monkeypatch):
    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _p: ("http://llm", "glm-5.3"))
    assert loop_spec().profile == "large"
    monkeypatch.setenv("UMRS_AGENT_MODEL", "llama3.2:3b")
    assert _sizes(loop_spec())[:5] == (4, 50, "auto", 2, "small")


def test_the_loop_store_reads_a_moe_loop_model_as_medium(loop_spec):
    """Lead decision 2026-10-10: the loop's model, the same rule."""
    assert _sizes(loop_spec(reader_model="qwen3-30b-a3b"))[:5] == (6, 50, "on", 3, "medium")


def test_an_explicit_loop_k_max_still_wins(loop_spec):
    assert loop_spec(reader_model="kimi-k3", k_max=3).k_max == 3


# ── the core: partial cap from the spec, the profile report ────────────────


def _cand(node_id: str, cosine: float, artifact: str) -> Candidate:
    return Candidate(
        node_id=node_id,
        artifact_id=artifact,
        content={"kind": "chunk", "artifact_id": artifact, "chunk_start": 0, "chunk_end": 50},
        cosine=cosine,
        lane_scores={"dense": cosine},
        lane_ranks={"dense": 1},
    )


def _core_lanes(found):
    def dense(_pool, _vectors, _model, _spec, *, limit):
        for rank, candidate in enumerate(found, 1):
            candidate.lane_ranks["dense"] = rank
        return LaneResult(name="dense", candidates=list(found)[:limit], info={"in_scope": 100})

    def hydrate(_pool, candidates, _fetch, *, documents=None):
        for candidate in candidates:
            candidate.text, candidate.text_source = f"text of {candidate.node_id}", "artifact"

    null = NullStats(mu=0.70, sigma=0.02, n=30, model="e5-base-v2")
    return {"dense": dense, "hydrate": hydrate, "null": lambda *_a, **_k: null}


def _core(**spec_fields):
    found = [_cand(f"n{i}", 0.745 - i * 0.001, f"a{i}") for i in range(5)]
    spec = RetrieveSpec(
        queries=("perovskite encapsulation",),
        project_id="p1",
        lexical="off",
        rerank="off",
        **spec_fields,
    )
    encode = lambda texts: ("e5-base-v2", [[1.0, 0.0] for _ in texts], "")  # noqa: E731
    return pipeline.retrieve(object(), spec, encode=encode, lanes=_core_lanes(found))


@pytest.mark.parametrize(("partial_k", "kept"), [(2, 2), (3, 3), (6, 5)])
def test_the_core_caps_a_partial_verdict_at_the_specs_partial_k(partial_k, kept):
    result = _core(partial_k=partial_k, k_max=10)
    assert result.sufficiency.status == "partial" and len(result.passages) == kept


def test_the_core_reports_the_profile_only_when_one_was_resolved():
    assert "profile" not in _core().lanes
    sized = _core(profile="large", reader_model="kimi-k3")
    assert sized.lanes["profile"] == {"status": "ok", "name": "large", "reader_model": "kimi-k3"}


# ── lexical inclusion follows the reader (measured 2026-10-10, ADR-093) ──────


def test_a_small_reader_reads_without_lexical_inclusion_by_default():
    from umrs.knowledge.retrieval import profiles

    from r4_contracts.models import RETRIEVAL_PROFILES

    assert profiles.lexical_mode(None, RETRIEVAL_PROFILES["small"]) == "off"
    assert profiles.lexical_mode("auto", RETRIEVAL_PROFILES["small"]) == "off"


def test_larger_readers_keep_the_site_lexical_default():
    from umrs.knowledge.retrieval import profiles

    from r4_contracts.models import RETRIEVAL_PROFILES

    for name in ("medium", "large"):
        assert profiles.lexical_mode("auto", RETRIEVAL_PROFILES[name]) == "auto"


def test_an_explicit_lexical_request_wins_over_the_profile():
    from umrs.knowledge.retrieval import profiles

    from r4_contracts.models import RETRIEVAL_PROFILES

    assert profiles.lexical_mode("on", RETRIEVAL_PROFILES["small"]) == "on"
    assert profiles.lexical_mode("off", RETRIEVAL_PROFILES["large"]) == "off"


def test_the_retrieve_spec_takes_lexical_from_the_reader_profile():
    from umrs.api.rag_retrieve import build_spec

    small = build_spec(("q",), "p", {"reader_model": "llama3.2:3b"})
    large = build_spec(("q",), "p", {"reader_model": "kimi-k3"})
    asked = build_spec(("q",), "p", {"reader_model": "llama3.2:3b", "lexical": "on"})
    assert (small.lexical, large.lexical, asked.lexical) == ("off", "auto", "on")


def test_the_dense_gap_default_is_the_tuned_value():
    from umrs.knowledge.retrieval import cutoff

    assert cutoff.constant("UMRS_RAG_GAMMA") == 0.02


def test_a_request_naming_no_reader_keeps_the_site_lexical_default():
    from umrs.api.rag_retrieve import build_spec

    assert build_spec(("q",), "p", {}).lexical == "auto"
