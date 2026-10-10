"""Route-level tests for the K3 unified knowledge query facade."""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import knowledge_query as kq
from umrs.api.knowledge_query import router
from umrs.data_catalog.schema import CatalogObjectMeta
from umrs.data_catalog.scoring import CatalogHit
from umrs.knowledge.retrieval.types import Candidate, LaneResult, RetrieveSpec

from r4_contracts.retrieval import RetrievalSufficiency


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class _FakeMetaStore:
    """Minimal CatalogMetaStore stand-in for facade tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def search(self, project_id: str, query: str, *, limit: int = 20, **_filters):
        self.calls.append((project_id, query, limit))
        if project_id != "proj-a":
            return []
        meta = CatalogObjectMeta.pending(
            s3_key="proj-a/docs/ct.tif",
            object_etag="etag-1",
            owner="alice",
            acl_scope="proj-a",
            filename="ct.tif",
        )
        meta.detected_modality = "image"
        meta.summary = "CT scan of sample S12"
        return [CatalogHit(meta=meta, score=0.92)]


def _headers(user: str = "alice", project: str = "proj-a") -> dict[str, str]:
    return {"X-User-ID": user, "X-Verified-Project-ID": project}


@pytest.fixture
def meta_store(monkeypatch) -> _FakeMetaStore:
    store = _FakeMetaStore()
    monkeypatch.setattr(
        "umrs.api.knowledge_query.get_optional_catalog_meta_store",
        lambda: store,
    )
    return store


@pytest.mark.usefixtures("meta_store")
class TestKnowledgeQueryFacade:
    def test_503_when_catalog_store_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(
            "umrs.api.knowledge_query.get_optional_catalog_meta_store",
            lambda: None,
        )
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct", "project_id": "proj-a"},
            headers=_headers(),
        )
        assert resp.status_code == 503

    def test_happy_path_returns_normalized_results(self, client):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct scan", "project_id": "proj-a"},
            headers=_headers(),
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["query"] == "ct scan"
        assert payload["total_results"] == 1

        _assert_hit_shape(payload["results"][0])

        # Coverage truth: catalog wired, others honestly unavailable.
        assert payload["wired_backends"] == ["catalog"]
        assert set(payload["unavailable_backends"].keys()) == {
            "knowledge_graph",
            "embeddings",
            "rag",
        }


def _assert_hit_shape(hit: dict) -> None:
    """Normalized search_api shape for a catalog hit.

    ``scope=all`` is fused (ADR-093 C7): rank 1 of the only lane that ran
    scores 1.0; the catalogue's own score is kept per lane.
    """
    assert hit["title"] == "ct.tif"
    assert hit["source_id"] == "proj-a/docs/ct.tif"
    assert hit["result_type"] == "catalog_hit"
    assert hit["score"] == 1.0
    assert hit["metadata"]["lanes"] == ["catalog"]
    assert hit["metadata"]["lane_score"] == {"catalog": 0.92}


@pytest.mark.usefixtures("meta_store")
class TestKnowledgeQueryFacadeEdge:
    def test_unknown_scope_is_400(self, client):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct", "project_id": "proj-a", "scope": "nope"},
            headers=_headers(),
        )
        assert resp.status_code == 400

    def test_missing_project_is_401_via_identity_gate(self, client):
        # Identity gate (get_caller_identity) fires before the route's own
        # project check: with no verified project header and no legacy query
        # params the caller identity itself is incomplete → 401, never 400.
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct"},
            headers={"X-User-ID": "alice"},
        )
        assert resp.status_code == 401

    def test_verified_project_header_overrides_param(self, client, meta_store):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct", "project_id": "other-project"},
            headers=_headers(project="proj-a"),
        )
        assert resp.status_code == 200
        assert meta_store.calls[-1][0] == "proj-a"

    def test_catalog_scope_matches(self, client):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct", "project_id": "proj-a", "scope": "catalog"},
            headers=_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["query"]["scope"] == "catalog"


class TestOntologyBlocks:
    def test_ontologies_blocks_per_selected_graph(self, client, meta_store, monkeypatch):
        """ADR-049 Leg B: response carries the bound ontology per selected
        graph as its own block; anunknown reference is reported via
        ontology_error, never smudged."""
        from umrs.knowledge.graph_registry import KnowledgeGraphMeta

        metas = [
            KnowledgeGraphMeta(
                graph_id="g-default",
                name="default",
                project_id="proj-a",
                ontology_id="r4-default",
            ),
            KnowledgeGraphMeta(
                graph_id="g-bogus",
                name="bogus",
                project_id="proj-a",
                ontology_id="never-registered",
            ),
        ]

        class _FakeRegistry:
            def ensure_default_graph(self, project_id):
                return metas[0]

        monkeypatch.setattr(
            "umrs.api.knowledge_query._resolve_selected_graphs",
            lambda *_args, **_kwargs: (metas, []),
        )
        # Pool slot occupied so the graph backend wires — psycopg shape
        # (getconn/cursor) returning zero rows; registry resolution is
        # monkeypatched above.
        from unittest.mock import MagicMock

        pool = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        cursor.fetchone.return_value = None  # ontology registry: no live row
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        pool.getconn.return_value = conn
        client.app.state.umrs_db_pool = pool
        try:
            resp = client.get(
                "/umrs/knowledge/query",
                params={"q": "ct", "project": "proj-a"},
                headers={"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"},
            )
        finally:
            client.app.state.umrs_db_pool = None

        assert resp.status_code == 200
        blocks = resp.json()["query"]["ontologies"]
        by_id = {b["graph_id"]: b for b in blocks}
        assert set(by_id) == {"g-default", "g-bogus"}
        assert by_id["g-default"]["ontology"]["ontology_id"] == "r4-default"
        assert "transitive_relations" in by_id["g-default"]["ontology"]
        assert "ontology_error" in by_id["g-bogus"]


# ── ADR-093 C7: the K3 response carries the retrieval core's sufficiency ────

_CORE = "umrs.knowledge.retrieval"


def _fake_module(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(f"{_CORE}.{name}")
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _fake_cutoff(monkeypatch, verdict=None, *, raises=None, hit=None) -> list[dict]:
    """A cutoff module recording its calls; ``lexical_evidence`` records under
    "rule"; the cut keeps every candidate (the real cut is pinned elsewhere)."""
    calls: list[dict] = []

    def lexical_evidence(queries, kept, *, strength=None):
        calls.append({"rule": (queries, [c.node_id for c in kept], strength)})
        return hit if hit is not None else (strength or 0.0) >= 3.0

    def cut(candidates, spec, *, mode):
        calls.append({"cut": (mode, spec.k_max)})
        return list(candidates)

    def sufficiency(kept, *, mode, null, lexical_hit, searched):
        calls.append(
            {
                "kept": kept,
                "mode": mode,
                "null": null,
                "lexical_hit": lexical_hit,
                "searched": searched,
            }
        )
        if raises is not None:
            raise raises
        return verdict or RetrievalSufficiency("insufficient", "z_null", 1.2, 3.0)

    _fake_module(
        monkeypatch, "cutoff", sufficiency=sufficiency, lexical_evidence=lexical_evidence, cut=cut
    )
    return calls


def _verdicts(calls: list[dict]) -> list[dict]:
    return [c for c in calls if "rule" not in c and "cut" not in c]


def _rules(calls: list[dict]) -> list[tuple]:
    return [c["rule"] for c in calls if "rule" in c]


def _fake_calibrate(monkeypatch, null="NULL") -> list[tuple]:
    calls: list[tuple] = []

    def null_for(pool, spec, encode, model):
        calls.append((pool, spec, encode, model))
        return null

    _fake_module(monkeypatch, "calibrate", null_for=null_for)
    return calls


def _dense_backend(scores=(0.81, 0.93)) -> kq._EmbeddingsSearchBackend:
    backend = kq._EmbeddingsSearchBackend(pool=None)
    backend.dense_model = "/models/e5-base-v2"
    backend.dense_hits = [{"source_id": f"n{i}", "score": s} for i, s in enumerate(scores)]
    return backend


def _lexical_backend(candidates) -> kq._RAGSearchBackend:
    backend = kq._RAGSearchBackend(pool=None)
    backend.lexical_candidates = list(candidates)
    return backend


_SPEC = RetrieveSpec(queries=("q",), project_id="proj-a", graph_ids=("g-1",))


def _block(emb=None, rag=None, spec=_SPEC, pool="POOL") -> dict:
    return kq._sufficiency_block(kq._SufficiencyInputs(pool, spec, emb, rag))


class TestSufficiencyBlock:
    def test_no_dense_or_lexical_lane_is_unknown_never_a_guess(self):
        block = _block()
        assert block["status"] == "unknown"
        assert block["detail"] == "no dense or lexical lane ran for this query"

    def test_without_the_retrieval_core_it_is_unknown(self):
        assert _block(_dense_backend(), spec=None)["detail"] == "retrieval core not installed"

    def test_a_missing_cutoff_module_is_reported(self, monkeypatch):
        monkeypatch.setitem(sys.modules, f"{_CORE}.cutoff", None)
        block = _block(_dense_backend())
        assert block["status"] == "unknown"
        assert block["detail"].startswith("sufficiency unavailable")

    def test_the_cores_verdict_is_the_block(self, monkeypatch):
        verdict = RetrievalSufficiency("sufficient", "z_null", 4.2, 3.0, "")
        _fake_cutoff(monkeypatch, verdict)
        _fake_calibrate(monkeypatch)
        assert _block(_dense_backend()) == verdict.to_dict()

    def test_dense_hits_reach_the_cut_best_first(self, monkeypatch):
        calls = _fake_cutoff(monkeypatch)
        _fake_calibrate(monkeypatch)
        _block(_dense_backend())
        kept = _verdicts(calls)[0]["kept"]
        assert [(c.node_id, c.cosine) for c in kept] == [("n1", 0.93), ("n0", 0.81)]
        assert [c.lane_ranks for c in kept] == [{"dense": 1}, {"dense": 2}]

    def test_a_dense_only_verdict_is_dense_mode_with_the_null(self, monkeypatch):
        calls = _fake_cutoff(monkeypatch)
        _fake_calibrate(monkeypatch)
        _block(_dense_backend())
        call = _verdicts(calls)[0]
        assert (call["mode"], call["null"]) == ("dense", "NULL")
        assert (call["lexical_hit"], call["searched"]) == (False, 2)

    def test_the_null_is_calibrated_in_the_dense_space_and_scope(self, monkeypatch):
        _fake_cutoff(monkeypatch)
        null_calls = _fake_calibrate(monkeypatch)
        _block(_dense_backend())
        pool, spec, _encode, model = null_calls[0]
        assert (pool, spec, model) == ("POOL", _SPEC, "/models/e5-base-v2")

    def test_a_dense_lane_that_found_nothing_is_searched_zero(self, monkeypatch):
        calls = _fake_cutoff(monkeypatch)
        null_calls = _fake_calibrate(monkeypatch)
        _block(_dense_backend(scores=()))
        verdict = _verdicts(calls)[0]
        assert verdict["kept"] == [] and verdict["searched"] == 0
        assert null_calls == []  # no vectors: no calibration

    def test_the_lex_lanes_strength_of_a_kept_passage_feeds_the_cores_rule(self, monkeypatch):
        calls = _fake_cutoff(monkeypatch)
        _fake_calibrate(monkeypatch)
        # The per-term basis (C2: one matched lexeme with idf >= 3), as the
        # retrieve pipeline reads it; the IDF-sum basis would report 0 here.
        seen = []

        def strength(cands, *, per_term=False):
            seen.append([c.node_id for c in cands])
            return 3.5 if per_term else 0.0

        _fake_module(monkeypatch, "lexical", lexical_hit_strength=strength)
        lex = [Candidate(node_id="n1", lane_scores={"lex": 3.5}), Candidate(node_id="n9")]
        _block(_dense_backend(), _lexical_backend(lex))
        assert seen == [["n1"]]  # only the kept passage the lex lane also found
        assert _rules(calls) == [(("q",), ["n1", "n0"], 3.5)]  # the top dense passage first
        verdict = _verdicts(calls)[0]
        assert (verdict["mode"], verdict["lexical_hit"], verdict["searched"]) == (
            "fused",
            True,
            2,
        )

    def test_a_lexical_find_outside_the_kept_passages_lends_no_strength(self, monkeypatch):
        """Review core-1 (2026-10-10): K3 took the strength from every lexical
        candidate, so any rare word of a nonsense query made it 'sufficient'."""
        calls = _fake_cutoff(monkeypatch)
        _fake_calibrate(monkeypatch)
        _fake_module(monkeypatch, "lexical", lexical_hit_strength=lambda _c, **_k: 8.1)
        rare = [Candidate(node_id="n-restaurant", lane_scores={"lex": 8.1})]
        _block(_dense_backend(), _lexical_backend(rare))
        assert _rules(calls) == [(("q",), ["n1", "n0"], None)]
        assert _verdicts(calls)[0]["lexical_hit"] is False

    def test_the_dense_ranking_is_cut_as_the_verb_cuts_it(self, monkeypatch):
        calls = _fake_cutoff(monkeypatch)
        _fake_calibrate(monkeypatch)
        _block(_dense_backend())
        assert [c["cut"] for c in calls if "cut" in c] == [("dense", _SPEC.k_max)]

    def test_without_the_lex_lane_only_term_coverage_can_hit(self, monkeypatch):
        calls = _fake_cutoff(monkeypatch)
        _fake_calibrate(monkeypatch)
        _block(_dense_backend())
        assert _rules(calls) == [(("q",), ["n1", "n0"], None)]

    def test_lexical_rows_alone_are_unknown_not_judged(self, monkeypatch):
        """No query vector: the core judges nothing, and never from lexical
        rows alone (review core-4/core-5) — K3 says so instead of guessing."""
        calls = _fake_cutoff(monkeypatch, hit=True)
        _fake_module(monkeypatch, "lexical", lexical_hit_strength=lambda _cands, **_k: 9.0)
        block = _block(None, _lexical_backend([Candidate(node_id="n9")]))
        assert block["status"] == "unknown" and "no query vector" in block["detail"]
        assert _verdicts(calls) == []

    def test_the_dense_passage_text_reaches_the_coverage_rule(self):
        backend = _dense_backend()
        backend.dense_hits[1]["content"] = "IL-6 rises in sepsis"  # n1, the best
        kept = kq._dense_candidates(backend)
        assert [c.text for c in kept] == ["IL-6 rises in sepsis", ""]

    def test_a_failing_verdict_is_unknown_not_an_error(self, monkeypatch):
        _fake_cutoff(monkeypatch, raises=RuntimeError("boom"))
        _fake_calibrate(monkeypatch)
        block = _block(_dense_backend())
        assert block["status"] == "unknown"
        assert block["detail"] == "sufficiency failed (RuntimeError)"


class TestSufficiencyWithTheCore:
    """The seam against the retrieval core's own cutoff module (C2): K3 hands
    it candidates and a null, and reports its verdict unchanged."""

    @pytest.fixture(autouse=True)
    def _null(self, monkeypatch):
        _fake_calibrate(monkeypatch, null=types.SimpleNamespace(mu=0.70, sigma=0.02))

    def test_a_neighbour_far_above_the_null_is_sufficient(self):
        block = _block(_dense_backend(scores=(0.80, 0.74)))
        assert (block["status"], block["basis"]) == ("sufficient", "z_null")

    def test_a_neighbour_at_the_null_is_insufficient(self):
        block = _block(_dense_backend(scores=(0.71,)))
        assert (block["status"], block["basis"]) == ("insufficient", "z_null")

    def test_a_space_with_no_vectors_in_scope_is_insufficient(self):
        block = _block(_dense_backend(scores=()))
        assert (block["status"], block["basis"]) == ("insufficient", "no_vectors")

    def test_a_rare_word_outside_the_kept_passages_never_lifts_the_verdict(self):
        """Review core-1: 'restaurant' (idf 8.1) matched one abstract the dense
        cut did not keep; its IDF lifted a partial z_null to sufficient."""
        rare = Candidate(node_id="n-restaurant", lane_scores={"lex": 8.1, "lex:idf_max": 8.1})
        block = _block(_dense_backend(scores=(0.75, 0.74)), _lexical_backend([rare]))
        assert (block["status"], block["basis"]) == ("partial", "z_null")  # z = 2.5

    def test_a_rare_word_in_a_kept_passage_lifts_a_partial_verdict(self):
        found = Candidate(node_id="n0", lane_scores={"lex": 8.1, "lex:idf_max": 8.1})
        block = _block(_dense_backend(scores=(0.75, 0.74)), _lexical_backend([found]))
        assert (block["status"], block["basis"]) == ("sufficient", "lexical")


class TestNullEncoder:
    def test_the_retrieve_verbs_batch_encoder_is_preferred(self, monkeypatch):
        def batch_encode(texts):
            return "m", [[0.0] for _ in texts], ""

        fake = types.ModuleType("umrs.api.rag_retrieve")
        fake.batch_encode = batch_encode
        monkeypatch.setitem(sys.modules, "umrs.api.rag_retrieve", fake)
        assert kq._null_encoder() is batch_encode

    def test_without_it_each_text_is_encoded_once(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "umrs.api.rag_retrieve", None)
        seen: list[str] = []
        monkeypatch.setattr(kq, "_encode_query", lambda t: seen.append(t) or ("m", [1.0], ""))
        encode = kq._null_encoder()
        assert encode(["a", "b"]) == ("m", [[1.0], [1.0]], "")
        assert seen == ["a", "b"]

    def test_an_encoder_outage_is_the_reason(self, monkeypatch):
        monkeypatch.setattr(kq, "_encode_query", lambda _t: ("", None, "encoder down"))
        assert kq._encode_each(["a"]) == ("", None, "encoder down")


@pytest.mark.usefixtures("meta_store")
class TestSufficiencyInTheResponse:
    def test_a_catalogue_only_answer_says_unknown(self, client):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct scan", "project_id": "proj-a"},
            headers=_headers(),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["sufficiency"]["status"] == "unknown"

    def test_the_dense_and_lexical_verdict_is_in_the_k3_response(self, client, monkeypatch):
        """Nonsense finds nearest neighbours but no answer: K3 says so, so R4's
        knowledge_insufficient can fire although total_results is not zero."""
        sim_row = ("n-1", "/models/e5-base-v2", 768, None, "chunk 0 of a", "proj-a", 0.31, "{}")
        emb = kq._EmbeddingsSearchBackend(
            _RowsPool([sim_row]), encoder=lambda _q: ("/models/e5-base-v2", [0.1], "")
        )
        monkeypatch.setattr(kq, "_embeddings_backend_for", lambda *_a, **_k: (emb, None))
        monkeypatch.setattr(
            kq, "_rag_backend_for", lambda *_a, **_k: (kq._RAGSearchBackend(None), None)
        )
        _fake_module(
            monkeypatch,
            "lexical",
            lexical_lane=lambda *_a, **_k: LaneResult(name="lex"),
            lexical_hit_strength=lambda _cands, **_k: 0.0,
        )
        calls = _fake_cutoff(monkeypatch)
        _fake_calibrate(monkeypatch)
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "best pizza in Naples", "project_id": "proj-a"},
            headers=_headers(),
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["total_results"] == 2  # the catalogue row and the neighbour
        assert payload["sufficiency"]["status"] == "insufficient"
        assert payload["rag_lane"] == {"lane": "lexical", "status": "ok", "n": 0}
        assert payload["embeddings_lane"]["lane"] == "similarity"
        verdict = _verdicts(calls)[0]
        assert verdict["mode"] == "dense" and verdict["searched"] == 1


class _RowsPool:
    """A pool whose every query returns ``rows``."""

    def __init__(self, rows) -> None:
        self.rows = rows

    def getconn(self):
        rows = self.rows

        class _Cur:
            def execute(self, *_a):
                return None

            def fetchall(self):
                return rows

            def __enter__(self):
                return self

            def __exit__(self, *_e):
                return False

        class _Conn:
            def cursor(self):
                return _Cur()

            def __enter__(self):
                return self

            def __exit__(self, *_e):
                return False

        return _Conn()

    def putconn(self, _conn):
        return None
