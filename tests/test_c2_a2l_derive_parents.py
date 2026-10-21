"""Cycle-2 umrs-api lane (2026-10-21): a derive names only parents the
caller can read.

Found live (gateway, two projects): bob@pb got 404 on GET of an artifact in
project pa, yet POST /gw/a2l/artifacts/derive with that artifact as a
parent answered 201 and wrote a lineage edge from pa's artifact to his; a
parent id that does not exist answered 500 (the edge's foreign key), so
201-vs-500 told him which artifact ids exist in other projects.
"""

from __future__ import annotations

from test_a2l_ledger_access import _ANA, _DERIVE, _client

PATH = "/umrs/a2l/ledger/artifacts/derive"


def test_a_parent_in_another_project_reads_as_not_found(monkeypatch):
    client, store = _client(monkeypatch, identity=_ANA)
    r = client.post(PATH, json={**_DERIVE, "parents": ["a-mine", "a-theirs"]})
    assert r.status_code == 404, r.text
    assert "derive" not in store.calls


def test_a_missing_parent_is_not_found_not_500(monkeypatch):
    client, store = _client(monkeypatch, identity=_ANA)
    r = client.post(PATH, json={**_DERIVE, "parents": ["no-such-artifact"]})
    assert r.status_code == 404, r.text
    assert "derive" not in store.calls


def test_readable_parents_still_derive(monkeypatch):
    client, store = _client(monkeypatch, identity=_ANA)
    assert client.post(PATH, json=_DERIVE).status_code == 201
    assert store.calls["derive"].parents == ["a-mine"]
