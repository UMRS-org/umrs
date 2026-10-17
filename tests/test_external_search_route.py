"""D5/D6 — ``/umrs/knowledge/external/search`` route tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")


_SECRET = "external-search-route-secret"


@pytest.fixture()
def client(monkeypatch):
    # H-1 (2026-10-18): the route was anonymous server-side egress; it now
    # needs the verified project, so the client carries a signed caller.
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import knowledge as knowledge_module

    from r4_contracts.handshake_hmac import sign_claims

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", _SECRET)
    app = FastAPI()
    app.include_router(knowledge_module.router)
    return TestClient(
        app,
        headers={
            "X-User-ID": "alice",
            "X-Verified-Project-ID": "proj-a",
            "X-Verified-Is-Admin": "false",
            "X-Verified-HMAC": sign_claims(_SECRET, "alice", "proj-a", False),
        },
    )


def _ok(source_type_name: str):
    from umrs.knowledge.external import (
        ExternalReference,
        ExternalSearchResult,
        ExternalSourceType,
        SearchStatus,
    )

    r = ExternalSearchResult(
        query="q",
        source_type=ExternalSourceType(source_type_name),
        searched_at="2026-07-30T00:00:00Z",
    )
    r.status = SearchStatus.SUCCESS
    r.references = [ExternalReference(source_type=ExternalSourceType(source_type_name), title="T1")]
    r.total_results = 1
    r.search_duration_sec = 0.01
    return r


def _one_result_per_type(*names):
    return [_ok(n) for n in names]


class TestExternalSearchRoute:
    def test_default_includes_ja_sources(self, client):
        with patch("umrs.knowledge.external.ExternalKnowledgeSearcher") as MockSearcher:
            instance = MockSearcher.return_value
            instance.search_all.return_value = _one_result_per_type(
                "arxiv", "semantic_scholar", "crossref", "cinii", "jstage"
            )
            r = client.post("/umrs/knowledge/external/search", json={"query": "深層学習"})

        assert r.status_code == 200
        body = r.json()
        assert body["sources"] == ["arxiv", "semantic_scholar", "crossref", "cinii", "jstage"]
        source_types = {item["source_type"] for item in body["results"]}
        assert "cinii" in source_types
        assert "jstage" in source_types
        # searcher was called with the JA sources among others
        called_sources = {s.value for s in instance.search_all.call_args.kwargs["sources"]}
        assert {"cinii", "jstage"} <= called_sources

    def test_explicit_sources_override(self, client):
        with patch("umrs.knowledge.external.ExternalKnowledgeSearcher") as MockSearcher:
            instance = MockSearcher.return_value
            instance.search_all.return_value = _one_result_per_type("cinii", "jstage")
            r = client.post(
                "/umrs/knowledge/external/search",
                json={"query": "透過率", "sources": ["cinii", "jstage"]},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["sources"] == ["cinii", "jstage"]
        called_sources = {s.value for s in instance.search_all.call_args.kwargs["sources"]}
        assert called_sources == {"cinii", "jstage"}

    def test_unknown_source_rejected(self, client):
        r = client.post(
            "/umrs/knowledge/external/search",
            json={"query": "x", "sources": ["definitely-not-a-source"]},
        )
        assert r.status_code == 400
        assert "unknown source" in r.json()["detail"]

    def test_limit_per_source_passed(self, client):
        with patch("umrs.knowledge.external.ExternalKnowledgeSearcher") as MockSearcher:
            instance = MockSearcher.return_value
            instance.search_all.return_value = _one_result_per_type("arxiv")
            r = client.post(
                "/umrs/knowledge/external/search",
                json={"query": "x", "sources": ["arxiv"], "limit_per_source": 3},
            )
        assert r.status_code == 200
        assert instance.search_all.call_args.kwargs["max_results_per_source"] == 3

    def test_query_too_short_rejected(self, client):
        r = client.post("/umrs/knowledge/external/search", json={"query": ""})
        assert r.status_code == 422

    def test_result_shape_matches_client_contract(self, client):
        with patch("umrs.knowledge.external.ExternalKnowledgeSearcher") as MockSearcher:
            MockSearcher.return_value.search_all.return_value = _one_result_per_type("arxiv")
            r = client.post(
                "/umrs/knowledge/external/search",
                json={"query": "attention", "sources": ["arxiv"]},
            )
        assert r.status_code == 200
        body = r.json()
        item = body["results"][0]
        assert set(item) == {
            "source_type",
            "status",
            "total_results",
            "error_message",
            "search_duration_sec",
            "references",
        }
        assert body["query"] == "attention"
