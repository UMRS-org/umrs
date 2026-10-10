"""K3 facade — graph backend wiring, selection, and provenance (0069)."""

from __future__ import annotations

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.knowledge_query import router
from umrs.knowledge.graph_registry import default_graph_id


class _FakeMetaStore:
    def search(self, project_id: str, query: str, *, limit: int = 20, **_f):
        del project_id, query, limit  # metadata backend not exercised in these tests
        return []


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setattr(
        "umrs.api.knowledge_query.get_optional_catalog_meta_store",
        lambda: _FakeMetaStore(),
    )


def _client(script) -> tuple[TestClient, FakePool]:
    app = FastAPI()
    app.include_router(router)
    pool = FakePool(script)
    app.state.umrs_db_pool = pool
    return TestClient(app), pool


def _graph_row(**overrides):
    row = {
        "graph_id": "g-exp",
        "name": "experiment-history",
        "description": "Historical experiments",
        "owner_id": "alice",
        "project_id": "proj-a",
        "kind": "research",
        "status": "active",
        "ontology_id": None,
        "ontology_version": None,
        "provenance": {},
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


# (node_id, project_id, label, node_type, content, graph_id, matched tokens)
_NODE = (
    "node:ir-1",
    "proj-a",
    "FWI inversion solver",
    "class",
    {"filename": "solver.py"},
    "g-exp",
    1,
)


def _query(client: TestClient, **params) -> object:
    params.setdefault("q", "FWI")
    return client.get(
        "/umrs/knowledge/query",
        params=params,
        headers={"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"},
    )


class TestGraphBackendWiring:
    def test_default_project_graph_wires_backend(self):
        default_id = default_graph_id("proj-a")
        client, pool = _client(
            [
                None,  # ensure_default insert
                _graph_row(graph_id=default_id, name="proj-a"),  # ensure re-read
                None,  # 0075 regclass probe (embeddings unwired here)
                None,  # WP-15: knowledge_nodes regclass probe (rag unwired here)
                [("node:ir-9", "proj-a", "FWI notes", "concept", {}, "g-legacy", 1)],
            ]
        )
        resp = _query(client, project_id="proj-a")
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert "knowledge_graph" in payload["wired_backends"]
        assert payload["selected_graphs"] == [default_id]
        assert payload["unavailable_backends"] == {
            # P2 wave 2: 0075 probe drives two honest states for the
            # embeddings backend (missing migration here via FakePool).
            "embeddings": "embeddings search unavailable (migration 0075 not applied)",
            # WP-15: the rag scope is wired over the central corpus; absent
            # knowledge_nodes it names the corpus honestly (never the
            # endpoint-coupled corpus as the default truth).
            "rag": "central knowledge corpus unavailable (knowledge nodes not present)",
        }

    def test_explicit_graph_selection_carries_provenance(self):
        client, pool = _client(
            [
                None,  # resolve 'experiment-history': by id
                _graph_row(),  # resolve: by name
                None,  # 0075 regclass probe (embeddings unwired here)
                None,  # WP-15: knowledge_nodes regclass probe (rag unwired here)
                [_NODE],  # graph search rows
            ]
        )
        resp = _query(
            client, project_id="proj-a", graphs="experiment-history", scope="knowledge_graph"
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["selected_graphs"] == ["g-exp"]
        hit = payload["results"][0]
        assert hit["metadata"]["graph_id"] == "g-exp"
        assert hit["metadata"]["graph_name"] == "experiment-history"
        assert hit["metadata"]["node_type"] == "class"
        assert hit["source_type"] == "knowledge_graph_node"
        # SQL scoped to the resolved graph id
        search_sql, params = pool.record[4]
        assert "graph_id = ANY" in search_sql
        assert params[0] == ["g-exp"]

    def test_legacy_rows_visible_through_default_graph(self):
        default_id = default_graph_id("proj-a")
        client, pool = _client(
            [
                None,
                _graph_row(graph_id=default_id, name="proj-a"),
                None,  # 0075 regclass probe (embeddings unwired here)
                None,  # WP-15: knowledge_nodes regclass probe (rag unwired here)
                [("node:ir-2", "proj-a", "FWI paper", "concept", {}, "", 1)],
            ]
        )
        resp = _query(client, project_id="proj-a", scope="knowledge_graph")
        assert resp.status_code == 200, resp.text
        hit = resp.json()["query"]["results"][0]
        # Legacy '' row attributed to the default graph
        assert hit["metadata"]["graph_id"] == default_id
        search_sql, params = pool.record[4]
        assert "graph_id = '' AND project_id = ANY" in search_sql
        assert params[1] == ["proj-a"]

    def test_exclusion_wins_and_causes_honest_unavailability(self):
        client, _ = _client([_graph_row()])
        resp = _query(
            client,
            project_id="proj-a",
            graphs="experiment-history",
            exclude_graphs="experiment-history",
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["selected_graphs"] == []
        assert payload["excluded_graphs"] == ["experiment-history"]
        assert "knowledge_graph" in payload["unavailable_backends"]

    def test_unknown_include_is_skipped(self):
        client, _ = _client([None, None])
        resp = _query(client, project_id="proj-a", graphs="g-missing")
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["selected_graphs"] == []

    def test_no_pool_keeps_honest_reason(self):
        app = FastAPI()
        app.include_router(router)
        resp = _query(TestClient(app), project_id="proj-a")
        assert resp.status_code == 200
        payload = resp.json()["query"]
        assert payload["wired_backends"] == ["catalog"]
        assert "requires the database" in payload["unavailable_backends"]["knowledge_graph"]


class TestGraphReadAccess:
    """KG-05 (2026-10-09): `graphs=` resolves an id, or a name across every
    project (default graphs are named after their project), and the selection
    was never access-checked: any caller could read another project's nodes
    by naming its graph. A graph the caller may not read is now skipped like
    an unknown one, so its existence is not disclosed either."""

    def test_another_projects_graph_is_skipped_like_an_unknown_one(self):
        foreign = _graph_row(graph_id="g-b", name="proj-b", project_id="proj-b", owner_id="bob")
        client, _ = _client([foreign])
        resp = _query(client, project_id="proj-a", graphs="g-b")
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["selected_graphs"] == [] and payload["ontologies"] == []
        assert "g-b" not in resp.text

    def test_a_graph_shared_with_the_callers_project_is_selected(self):
        shared = _graph_row(
            graph_id="g-b",
            name="proj-b",
            project_id="proj-b",
            owner_id="bob",
            access=["project:proj-a"],
        )
        client, _ = _client([shared])
        resp = _query(client, project_id="proj-a", graphs="g-b")
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["selected_graphs"] == ["g-b"]

    def test_a_shared_name_resolves_to_the_callers_own_graph(self):
        """Names are not unique across projects; another project's older graph
        of the same name must neither leak nor hide the caller's own."""
        foreign = _graph_row(graph_id="g-b", name="batteries", project_id="proj-b", owner_id="bob")
        own = _graph_row(graph_id="g-a", name="batteries")
        client, _ = _client([None, [foreign, own]])  # no id match, then every name match
        resp = _query(client, project_id="proj-a", graphs="batteries")
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["selected_graphs"] == ["g-a"]

    def test_a_shared_graph_exposes_none_of_its_owners_legacy_rows(self):
        """Legacy rows (graph_id '') belong to the owner's DEFAULT graph only.
        Sharing a non-default graph exposed every one of them (review, 2026-10-09)."""
        shared = _graph_row(graph_id="g-shared", name="shared", access=["project:proj-b"])
        client, pool = _client([shared])
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "secret", "graphs": "g-shared", "scope": "knowledge_graph"},
            headers={"X-User-ID": "bob", "X-Verified-Project-ID": "proj-b"},
        )
        assert resp.status_code == 200, resp.text
        node_query = next(p for sql, p in pool.record if "graph_id = ANY" in sql)
        assert node_query[0] == ["g-shared"] and node_query[1] == []

    def test_an_excluded_name_stays_excluded_whatever_project_holds_it(self):
        foreign = _graph_row(graph_id="g-b", name="batteries", project_id="proj-b", owner_id="bob")
        own = _graph_row(graph_id="g-a", name="batteries")
        # exclusions resolve first (id miss, every name match), then includes
        client, _ = _client([None, [foreign, own], None, [foreign, own]])
        resp = _query(client, project_id="proj-a", graphs="batteries", exclude_graphs="batteries")
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["selected_graphs"] == []


class TestGraphLaneTokens:
    """ADR-093 C7: the graph lane matched the whole query as one ILIKE phrase
    (a claim is almost never a substring of a node) with a fixed 0.9/0.6
    score. It now matches query tokens OR-wise and scores the matched fraction."""

    def _search(self, q: str, rows: list) -> tuple[dict, list]:
        client, pool = _client(
            [
                None,  # resolve 'experiment-history': by id
                _graph_row(),  # resolve: by name
                None,  # 0075 regclass probe (embeddings unwired here)
                None,  # knowledge_nodes regclass probe (rag unwired here)
                rows,  # graph search rows
            ]
        )
        resp = _query(
            client, q=q, project_id="proj-a", graphs="experiment-history", scope="knowledge_graph"
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["query"], pool.record

    def test_tokens_match_or_wise_and_score_the_matched_fraction(self):
        row = ("node:ir-1", "proj-a", "FWI solver", "class", {}, "g-exp", 2)
        payload, record = self._search("FWI inversion residuals", [row])
        hit = payload["results"][0]
        assert hit["score"] == pytest.approx(2 / 3, abs=1e-6)
        assert hit["metadata"]["matched_tokens"] == 2
        assert hit["metadata"]["query_tokens"] == 3
        sql, params = record[4]
        assert sql.count("(s.body LIKE %s)::int") == 3
        assert sql.count("(s.lab LIKE %s)::int") == 3
        patterns = ["%fwi%", "%inversion%", "%residuals%"]
        assert list(params) == [["g-exp"], [], *patterns, *patterns, 20]

    def test_nodes_rank_by_matched_tokens_then_label_matches(self):
        _, record = self._search("FWI inversion", [])
        sql = record[4][0]
        assert "WHERE hits > 0" in sql
        assert "ORDER BY hits DESC, label_hits DESC, node_id" in sql

    def test_json_keys_are_not_matched_only_string_values(self):
        _, record = self._search("chunk kind", [])
        sql = record[4][0]
        assert "jsonb_path_query_array(content, '$.** ? (@.type() == \"string\")')" in sql
        assert "content::text" not in sql

    def test_the_selection_gate_still_binds_first(self):
        _, record = self._search("FWI", [])
        sql, params = record[4]
        assert "graph_id = ANY(%s) OR (graph_id = '' AND project_id = ANY(%s))" in sql
        assert params[0] == ["g-exp"] and params[1] == []


class TestGraphTokens:
    def test_stop_words_short_words_and_repeats_are_dropped(self):
        from umrs.api.knowledge_query import _graph_tokens

        assert _graph_tokens("The IL-6 levels of IL-6 in sepsis") == ["il-6", "levels", "sepsis"]

    def test_edge_punctuation_is_stripped_identifiers_kept(self):
        from umrs.api.knowledge_query import _graph_tokens

        assert _graph_tokens('"ZSM-5" (zeolite), TiO2.') == ["zsm-5", "zeolite", "tio2"]

    def test_a_query_with_no_content_word_falls_back_to_the_phrase(self):
        from umrs.api.knowledge_query import _graph_tokens

        assert _graph_tokens("is it ok") == ["is it ok"]

    def test_tokens_are_capped(self):
        from umrs.api.knowledge_query import _GRAPH_TOKEN_MAX, _graph_tokens

        words = " ".join(f"word{i}" for i in range(20))
        assert _graph_tokens(words) == [f"word{i}" for i in range(_GRAPH_TOKEN_MAX)]

    def test_like_metacharacters_in_a_token_are_literal(self):
        from umrs.api.knowledge_query import _like_contains

        assert _like_contains("50%_a\\b") == "%50\\%\\_a\\\\b%"


class TestEveryLaneHonoursGraphs:
    """ADR-093 C7: ``graphs=`` bound only the graph lane; the embeddings and
    rag lanes searched the whole project. A named selection (or an exclusion)
    now binds them too, under the legacy ``graph_id ''`` rule."""

    @pytest.fixture
    def captured(self, monkeypatch) -> dict:
        seen: dict = {}

        def emb(pool, **kwargs):
            seen["embeddings"] = kwargs.get("graph_scope", "missing")
            return None, "captured"

        def rag(pool, **kwargs):
            seen["rag"] = kwargs.get("graph_scope", "missing")
            return None, "captured"

        monkeypatch.setattr("umrs.api.knowledge_query._embeddings_backend_for", emb)
        monkeypatch.setattr("umrs.api.knowledge_query._rag_backend_for", rag)
        return seen

    def test_named_graphs_bind_the_embeddings_and_rag_lanes(self, captured):
        client, _ = _client([None, _graph_row(), []])
        resp = _query(client, project_id="proj-a", graphs="experiment-history")
        assert resp.status_code == 200, resp.text
        for lane in ("embeddings", "rag"):
            scope = captured[lane]
            assert scope.graph_ids == ("g-exp",)
            assert scope.default_projects == ()  # not a default graph: no legacy rows

    def test_without_a_selection_the_lanes_keep_the_project_rule(self, captured):
        default_id = default_graph_id("proj-a")
        client, _ = _client([None, _graph_row(graph_id=default_id, name="proj-a"), []])
        resp = _query(client, project_id="proj-a")
        assert resp.status_code == 200, resp.text
        assert captured == {"embeddings": None, "rag": None}

    def test_an_exclusion_alone_keeps_the_project_rule_for_the_passage_lanes(self, captured):
        """Review of ADR-093 (2026-10-10): an exclusion alone bound both lanes
        to the default graph, dropping every other graph of the project (and
        everything when the default graph was the one excluded), unlike
        /rag/retrieve's "whole project minus the exclusions"."""
        default_id = default_graph_id("proj-a")
        client, _ = _client(
            [
                None,  # exclusion 'other': no id match
                [],  # ... and no name match
                None,  # ensure default insert
                _graph_row(graph_id=default_id, name="proj-a"),  # ensure re-read
                [],  # graph search rows
            ]
        )
        resp = _query(client, project_id="proj-a", exclude_graphs="other")
        assert resp.status_code == 200, resp.text
        assert captured == {"embeddings": None, "rag": None}  # no default-graph binding

    def test_an_unreadable_selection_binds_the_lanes_to_nothing(self, captured):
        foreign = _graph_row(graph_id="g-b", name="proj-b", project_id="proj-b", owner_id="bob")
        client, _ = _client([foreign])
        resp = _query(client, project_id="proj-a", graphs="g-b")
        assert resp.status_code == 200, resp.text
        assert captured["embeddings"].graph_ids == () and captured["rag"].graph_ids == ()


class TestWholeProjectExclusions:
    """Without ``graphs=`` the embeddings and rag lanes read the whole project
    minus the excluded graphs and the graphs the caller may not read — the
    retrieve verb's scope (``rag_retrieve.resolve_graphs``)."""

    @staticmethod
    def _backends():
        from umrs.api.knowledge_query import _EmbeddingsSearchBackend, _RAGSearchBackend

        return _EmbeddingsSearchBackend(None), _RAGSearchBackend(None)

    @staticmethod
    def _scope(graphs=None, exclude=None):
        from umrs.api.knowledge_query import _LaneScope

        return _LaneScope(object(), object(), "proj-a", graphs, exclude)

    @pytest.fixture
    def resolved(self, monkeypatch):
        from types import SimpleNamespace

        from umrs.api import knowledge_query as kq
        from umrs.api import rag_retrieve

        class Registry:
            def __init__(self, _pool):
                pass

            def graph_candidates(self, token):
                return [SimpleNamespace(graph_id=f"g-{token}")]

        monkeypatch.setattr(kq, "GraphRegistryStore", Registry)
        monkeypatch.setattr(rag_retrieve, "hidden_graphs", lambda *_a: ("g-u2-personal",))

    @pytest.mark.usefixtures("resolved")
    def test_exclusions_and_unreadable_graphs_bind_both_lanes(self):
        from umrs.api.knowledge_query import _bind_lane_exclusions

        emb, rag = self._backends()
        ids = _bind_lane_exclusions(self._scope(exclude="other"), emb, rag)
        assert ids == ("g-other", "g-u2-personal")
        assert emb.exclude_graph_ids == ids and rag.exclude_graph_ids == ids

    @pytest.mark.usefixtures("resolved")
    def test_a_named_selection_needs_no_exclusion(self):
        from umrs.api.knowledge_query import _bind_lane_exclusions

        emb, rag = self._backends()
        assert _bind_lane_exclusions(self._scope(graphs="g-exp"), emb, rag) == ()
        assert emb.exclude_graph_ids == ()

    def test_the_vector_lane_renders_the_exclusion(self):
        from umrs.api.knowledge_query import _EmbeddingsSearchBackend

        backend = _EmbeddingsSearchBackend(None)
        backend.exclude_graph_ids = ("g-other",)
        gate, params = backend._gate("proj-a")
        assert gate.endswith(" AND n.graph_id <> ALL(%s)")
        assert params == ("proj-a", ["g-other"])

    def test_excluding_the_default_graph_drops_its_legacy_rows(self):
        from umrs.api.knowledge_query import _EmbeddingsSearchBackend

        backend = _EmbeddingsSearchBackend(None)
        backend.exclude_graph_ids = (default_graph_id("proj-a"),)
        gate, _params = backend._gate("proj-a")
        assert gate.endswith(" AND n.graph_id <> ALL(%s) AND n.graph_id <> ''")

    def test_the_lexical_lane_reads_the_exclusion(self, monkeypatch):
        import sys
        from types import ModuleType

        from umrs.api.knowledge_query import _RAGSearchBackend
        from umrs.knowledge.retrieval.types import LaneResult

        seen = {}
        module = ModuleType("umrs.knowledge.retrieval.lexical")

        def lane(_pool, _query, spec, **_kwargs):
            seen["spec"] = spec
            return LaneResult(name="lex")

        module.lexical_lane = lane
        monkeypatch.setitem(sys.modules, "umrs.knowledge.retrieval.lexical", module)
        backend = _RAGSearchBackend(None)
        backend.exclude_graph_ids = ("g-other",)
        backend.search_rag("q", project_id="proj-a")
        assert seen["spec"].exclude_graph_ids == ("g-other",) and seen["spec"].graph_ids is None
