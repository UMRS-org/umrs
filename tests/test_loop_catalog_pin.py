"""S-D39b (DEC-D39, decisions-2026-11-24 §5 wave 1): a loop may pin
``catalog_id@version``; "exactly one active catalog" is only the fallback.

Acceptance: two active catalogs + a named one → the loop's findings pin
there (the old rule pinned nothing: several catalogs were ambiguous).
Same behaviour on a hosted front and a self-hosted stand-alone UMRS.

Real Postgres leg runs with ``UMRS_TEST_DB_URL`` (or the CI job's
``POSTGRES_*``) in a scratch schema dropped afterwards.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
from umrs.scientific_catalogs import pinner
from umrs.scientific_catalogs.store import CatalogNotFound

# ── in-memory store with the four calls the pinner uses ────────────────────


class _Store:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = {r["catalog_id"]: _complete(r) for r in rows}

    def get(self, catalog_id: str) -> dict[str, Any]:
        if catalog_id not in self.rows:
            raise CatalogNotFound(catalog_id)
        return self.rows[catalog_id]

    def list(self, *, project_id: str) -> list[dict[str, Any]]:
        return [
            r
            for r in self.rows.values()
            if r["project_id"] == project_id and r["status"] != "retired"
        ]

    def versions(self, *, project_id: str, name: str) -> list[dict[str, Any]]:
        return [r for r in self.rows.values() if (r["project_id"], r["name"]) == (project_id, name)]

    def add_pin(self, catalog_id: str, *, kind: str, ref: str, note: str = "") -> None:
        prov = self.rows[catalog_id]["provenance"]
        prov.setdefault("pins", []).append({"kind": kind, "ref": ref, "note": note})


def _complete(row: dict[str, Any]) -> dict[str, Any]:
    base = {"project_id": "p1", "name": row["catalog_id"], "version": "0.1.0"}
    base.update({"status": "active", "provenance": {}}, **row)
    return base


def _refs(store: _Store, catalog_id: str) -> list[str]:
    return [p["ref"] for p in store.rows[catalog_id]["provenance"].get("pins", [])]


def _two_active() -> _Store:
    return _Store([{"catalog_id": "sc-a"}, {"catalog_id": "sc-b"}])


# ── the pinner ──────────────────────────────────────────────────────────────


def test_two_active_catalogs_and_a_named_one_pin_there() -> None:
    store = _two_active()
    target = pinner.pin_findings(store, "p1", [("f-1", "n")], catalog_id="sc-b")
    assert target == "sc-b"
    assert _refs(store, "sc-b") == ["f-1"]
    assert _refs(store, "sc-a") == []


def test_unnamed_loop_keeps_the_single_active_fallback() -> None:
    one = _Store([{"catalog_id": "sc-a"}])
    assert pinner.pin_findings(one, "p1", [("f-1", "n")]) == "sc-a"
    several = _two_active()
    assert pinner.pin_findings(several, "p1", [("f-1", "n")]) == ""
    assert _refs(several, "sc-a") == _refs(several, "sc-b") == []


def test_named_pin_dedupes_and_stays_append_only() -> None:
    store = _Store([{"catalog_id": "sc-a", "provenance": {"pins": [{"ref": "f-1"}]}}])
    pinner.pin_findings(store, "p1", [("f-1", "x"), ("f-2", "y"), ("f-2", "y")], "sc-a")
    assert _refs(store, "sc-a") == ["f-1", "f-2"]


def test_parse_catalog_ref() -> None:
    assert pinner.parse_catalog_ref(" sc-a@0.2.0 ") == ("sc-a", "0.2.0")
    assert pinner.parse_catalog_ref("sc-a") == ("sc-a", "")


def _chain() -> _Store:
    return _Store(
        [
            {"catalog_id": "sc-v1", "name": "mice", "version": "0.1.0"},
            {"catalog_id": "sc-v2", "name": "mice", "version": "0.2.0"},
            {"catalog_id": "sc-old", "name": "old", "status": "retired"},
            {"catalog_id": "sc-other", "project_id": "p2"},
        ]
    )


def test_catalog_id_at_version_picks_that_version_of_the_catalog() -> None:
    store = _chain()
    assert pinner.resolve_catalog_pin(store, "p1", "sc-v1@0.2.0")["catalog_id"] == "sc-v2"
    assert pinner.resolve_catalog_pin(store, "p1", "sc-v2")["catalog_id"] == "sc-v2"
    assert pinner.resolve_catalog_pin(store, "p1", "sc-v1@0.1.0")["catalog_id"] == "sc-v1"


@pytest.mark.parametrize(
    ("ref", "reason"),
    [
        ("sc-v1@9.9.9", pinner.CATALOG_VERSION_NOT_FOUND),
        ("sc-nope", pinner.CATALOG_NOT_FOUND),
        ("", pinner.CATALOG_NOT_FOUND),
        ("sc-other", pinner.CATALOG_NOT_FOUND),  # another project: no disclosure
        ("sc-old", pinner.CATALOG_RETIRED),
    ],
)
def test_unpinnable_catalogs_are_refused_by_name(ref: str, reason: str) -> None:
    with pytest.raises(pinner.CatalogPinRefused) as err:
        pinner.resolve_catalog_pin(_chain(), "p1", ref)
    assert err.value.reason == reason


@pytest.mark.parametrize(
    ("named", "reason"),
    [
        ("sc-other", pinner.CATALOG_NOT_FOUND),  # another project's catalog
        ("sc-old", pinner.CATALOG_RETIRED),  # retired after the loop started
        ("sc-gone", pinner.CATALOG_NOT_FOUND),
    ],
)
def test_a_named_pin_never_writes_outside_the_project_or_onto_retired(
    named: str, reason: str
) -> None:
    store = _chain()
    with pytest.raises(pinner.CatalogPinRefused) as err:
        pinner.pin_findings(store, "p1", [("f-1", "n")], catalog_id=named)
    assert err.value.reason == reason
    assert all(not r["provenance"].get("pins") for r in store.rows.values())


# ── loop hosting passes the loop's pin to the pinner ───────────────────────


def _loop_with_finding(metadata: dict[str, Any]):
    from umrs.knowledge.loop import Finding, LoopIteration, ResearchLoop

    loop = ResearchLoop(loop_id="l1", project_id="p1")
    loop.metadata.update(metadata)
    finding = Finding(
        finding_id="f-1",
        hypothesis_id="h-1",
        statement="s",
        evidence_refs=["e-1"],
        workflow_run_id="wf-1",
        durable_job_id="j-1",
        confidence=0.7,
    )
    iteration = LoopIteration(iteration_number=1)
    iteration.findings = [finding]
    return loop, iteration


@pytest.fixture()
def _hosting_calls():
    from umrs.knowledge.loop_hosting import HostingWiring, configure

    calls: list[tuple[Any, ...]] = []
    configure(
        HostingWiring(
            store=lambda: None,
            engine_factory=lambda: lambda: None,
            agent_executor=lambda _pid: None,
            checkpoint_store=lambda: None,
            finding_sink=lambda: None,
            catalog_pinner=lambda: lambda *args: calls.append(args),
        )
    )
    yield calls
    configure(None)


def test_hosting_hands_the_loops_named_catalog_to_the_pinner(_hosting_calls) -> None:
    from umrs.knowledge.loop_hosting import _pin_real_findings

    loop, it = _loop_with_finding({"catalog_pin": {"catalog_id": "sc-b", "version": "1"}})
    _pin_real_findings(loop, it)
    assert _hosting_calls == [("p1", [("f-1", "h-1 wf=wf-1")], "sc-b")]


def test_hosting_without_a_pin_calls_the_fallback(_hosting_calls) -> None:
    from umrs.knowledge.loop_hosting import _pin_real_findings

    loop, it = _loop_with_finding({})
    _pin_real_findings(loop, it)
    assert _hosting_calls == [("p1", [("f-1", "h-1 wf=wf-1")])]


# ── POST /umrs/loops `catalog` field ──────────────────────────────────────


class _Pool:
    def __init__(self) -> None:
        self.leased = 0

    def getconn(self) -> object:
        self.leased += 1
        return object()

    def putconn(self, _conn: object) -> None:
        self.leased -= 1


@pytest.fixture()
def _api(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import loop as loop_api
    from umrs.knowledge.loop import AutonomousResearchLoop
    from umrs.knowledge.loop_store import LoopStore
    from umrs.scientific_catalogs import store as store_mod

    store, pool = _two_active(), _Pool()
    monkeypatch.setattr(loop_api, "_loop_store", LoopStore.memory())
    monkeypatch.setattr(loop_api, "_loop_engine", AutonomousResearchLoop(simulated=True))
    monkeypatch.setattr(loop_api, "_jobs_pool", lambda: pool)
    monkeypatch.setattr(store_mod, "ScientificCatalogStore", lambda _conn: store)
    app = FastAPI()
    app.include_router(loop_api.router)
    yield TestClient(app), loop_api, store, pool
    loop_api.clear_loop_store()


def _start(client, **extra):
    body = {"project_id": "p1", "research_goal": "g", "max_iterations": 1, **extra}
    return client.post("/umrs/loops", json=body)


def test_start_records_the_named_catalog_on_the_loop(_api) -> None:
    client, loop_api, _store, pool = _api
    resp = _start(client, catalog="sc-b@0.1.0")
    assert resp.status_code == 201, resp.text
    loop = loop_api.get_loop(resp.json()["loop_id"])
    assert loop.metadata["catalog_pin"] == {
        "catalog_id": "sc-b",
        "version": "0.1.0",
        "requested": "sc-b@0.1.0",
    }
    assert pool.leased == 0


def test_start_refuses_an_unknown_catalog_by_name_and_makes_no_loop(_api) -> None:
    client, loop_api, _store, _pool = _api
    resp = _start(client, catalog="sc-nope")
    assert resp.status_code == 404, resp.text
    assert "catalog_not_found" in resp.text
    assert loop_api.get_loop_store().list_all() == []


def test_start_without_a_catalog_store_refuses_a_named_catalog(_api, monkeypatch) -> None:
    client, loop_api, _store, _pool = _api
    monkeypatch.setattr(loop_api, "_jobs_pool", lambda: None)
    resp = _start(client, catalog="sc-b")
    assert resp.status_code == 503, resp.text
    assert "catalog_store_unavailable" in resp.text
    assert _start(client).status_code == 201  # no catalog named: unchanged


def test_pool_pinner_pins_onto_the_named_catalog(_api) -> None:
    _client, loop_api, store, pool = _api
    loop_api._pin_findings_onto_catalog(pool, "p1", [("f-9", "n")], "sc-a")
    assert _refs(store, "sc-a") == ["f-9"]
    assert pool.leased == 0


# ── real Postgres (scratch schema) ─────────────────────────────────────────

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit or not os.environ.get("POSTGRES_HOST"):
        return explicit
    env = os.environ.get
    return (
        f"postgresql://{env('POSTGRES_USER', 'test')}:{env('POSTGRES_PASSWORD', 'test')}"
        f"@{env('POSTGRES_HOST')}:{env('POSTGRES_PORT', '5432')}/{env('POSTGRES_DB', 'umrs_test')}"
    )


def _catalog_ddl() -> list[str]:
    """0101/0102/0103/0108 plus 0116's scientific_catalogs statement."""
    names = (
        "0101_scientific_catalogs.sql",
        "0102_scientific_catalogs_changelog.sql",
        "0103_scientific_catalogs_redaction.sql",
        "0108_catalog_mcp_trust_classes.sql",
    )
    ddl = [(_MIGRATIONS / n).read_text() for n in names]
    access = (_MIGRATIONS / "0116_resource_access.sql").read_text()
    ddl += re.findall(r"ALTER TABLE scientific_catalogs\s+ADD COLUMN[^;]*;", access)
    return ddl


@pytest.fixture()
def pg_store():
    url = _db_url()
    if not url:
        pytest.skip("no Postgres (UMRS_TEST_DB_URL / POSTGRES_HOST)")
    psycopg2 = pytest.importorskip("psycopg2")
    from umrs.scientific_catalogs.store import ScientificCatalogStore

    schema = f"sd39b_{uuid.uuid4().hex[:10]}"
    conn = psycopg2.connect(url, options=f"-c search_path={schema}")
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        for sql in _catalog_ddl():
            cur.execute(sql)
    conn.commit()
    yield ScientificCatalogStore(conn)
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


def test_pg_two_active_catalogs_and_a_named_version_pin_there(pg_store) -> None:
    from umrs.scientific_catalogs.store import CatalogSpec

    def mk(name: str, version: str) -> str:
        spec = CatalogSpec(project_id="p1", name=name, version=version)
        return pg_store.create(spec=spec, owner_id="u")["catalog_id"]

    v1, v2, other = mk("mice", "0.1.0"), mk("mice", "0.2.0"), mk("rats", "0.1.0")
    assert len(pg_store.list(project_id="p1")) == 3
    row = pinner.resolve_catalog_pin(pg_store, "p1", f"{v1}@0.2.0")
    assert row["catalog_id"] == v2
    pinner.pin_findings(pg_store, "p1", [("f-1", "n")], row["catalog_id"])
    assert pinner.pin_findings(pg_store, "p1", [("f-2", "n")]) == ""  # fallback: ambiguous
    pins = {c: pg_store.get(c)["provenance"].get("pins", []) for c in (v1, v2, other)}
    assert [p["ref"] for p in pins[v2]] == ["f-1"]
    assert pins[v1] == pins[other] == []


def test_driver_state_write_cannot_repoint_the_catalog_pin(_api) -> None:
    """2026-11-24 review: ``catalog_pin`` is server-owned; a driver's
    ``PUT /state`` cannot repoint it, nor add one to an unpinned loop."""
    client, loop_api, _store, _pool = _api
    pinned = _start(client, catalog="sc-b").json()["loop_id"]
    unpinned = _start(client).json()["loop_id"]
    foreign = {"catalog_pin": {"catalog_id": "sc-foreign"}, "note": "kept"}
    for loop_id in (pinned, unpinned):
        resp = client.put(f"/umrs/loops/{loop_id}/state", json={"metadata": foreign})
        assert resp.status_code == 200, resp.text
    assert loop_api.get_loop(pinned).metadata["catalog_pin"]["catalog_id"] == "sc-b"
    assert loop_api.get_loop(pinned).metadata["note"] == "kept"
    assert "catalog_pin" not in loop_api.get_loop(unpinned).metadata


def test_hosting_logs_a_refused_named_pin_and_does_not_fall_back(caplog) -> None:
    from umrs.knowledge.loop_hosting import HostingWiring, _pin_real_findings, configure

    store = _chain()
    configure(
        HostingWiring(
            store=lambda: None,
            engine_factory=lambda: lambda: None,
            agent_executor=lambda _pid: None,
            checkpoint_store=lambda: None,
            finding_sink=lambda: None,
            catalog_pinner=lambda: lambda pid, pins, cid="": pinner.pin_findings(
                store, pid, pins, cid
            ),
        )
    )
    try:
        loop, it = _loop_with_finding({"catalog_pin": {"catalog_id": "sc-other"}})
        _pin_real_findings(loop, it)
    finally:
        configure(None)
    assert "finding pin skipped" in caplog.text
    assert all(not r["provenance"].get("pins") for r in store.rows.values())
