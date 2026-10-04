"""ADR-060 U-1 rollout helper pins: per-cluster table → hub registry copy.

One-shot per venue (U-1 follow-up 2026-10-03, migration 0118): the marker,
the rows and the counts commit in one transaction; a venue with a marker is
skipped without opening its source, so rows the servers removed after the
flip never come back. Venue-stamped, positional column contract (ADR-058)
with venue 17th, JSONB metadata re-serialised, errors exit 1 masked.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path

import pytest
from umrs.migrations import hub_endpoint_registry_copy as hub_copy
from umrs.migrations.hub_endpoint_registry_copy import (
    SOURCE_COLUMNS,
    CopyResult,
    copy_endpoints,
    mask_dsn,
)

_MARKER_TIME = datetime(2026, 10, 3, 1, 30)


def _sql(text) -> str:
    return " ".join(str(text).split())


# ── fakes: a hub DB with transaction semantics + a per-cluster source ──────


class _HubCursor:
    def __init__(self, db):
        self._db = db
        self.rowcount = 0
        self._one = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        sql = _sql(sql)
        self._db.log.append((sql, params))
        if self._db.fail_on and self._db.fail_on in sql:
            self._db.fail_hits += 1
            if self._db.fail_hits >= self._db.fail_after:
                raise RuntimeError("boom at postgres://umrs:s3cret@127.0.0.1:5432/umrs_db")
        handler = self._db.route(sql)
        self.rowcount, self._one = handler(params)

    def fetchone(self):
        return self._one


class _HubDB:
    """Committed state + one pending transaction, like a psycopg2 session."""

    def __init__(self, markers=None):
        self.rows: dict[str, tuple] = {}
        self.markers: dict[str, dict] = dict(markers or {})
        self._rows_tx: dict[str, tuple] = {}
        self._markers_tx: dict[str, dict] = {}
        self.log: list = []
        self.commits = 0
        self.rollbacks = 0
        self.fail_on = ""
        self.fail_after = 1
        self.fail_hits = 0

    # psycopg2 connection surface
    def cursor(self):
        return _HubCursor(self)

    def commit(self):
        self.rows.update(self._rows_tx)
        self.markers.update(self._markers_tx)
        self._rows_tx, self._markers_tx = {}, {}
        self.commits += 1

    def rollback(self):
        self._rows_tx, self._markers_tx = {}, {}
        self.rollbacks += 1

    # SQL routing
    def route(self, sql):
        if sql.startswith("INSERT INTO hub_registry_copy_markers"):
            return self._claim
        if sql.startswith("SELECT copied_at FROM hub_registry_copy_markers"):
            return self._marker_at
        if sql.startswith("UPDATE hub_registry_copy_markers"):
            return self._record
        if sql.startswith("INSERT INTO endpoint_registry"):
            return self._insert
        raise AssertionError(f"unexpected hub SQL: {sql}")

    def _claim(self, params):
        venue = params[0]
        if venue in self.markers or venue in self._markers_tx:
            return 0, None
        self._markers_tx[venue] = {"copied_at": _MARKER_TIME, "note": params[1]}
        return 1, (venue,)

    def _marker_at(self, params):
        marker = self.markers.get(params[0])
        return (1, (marker["copied_at"],)) if marker else (0, None)

    def _record(self, params):
        source_rows, inserted, venue = params
        self._markers_tx[venue].update(source_rows=source_rows, inserted=inserted)
        return 1, None

    def _insert(self, params):
        job_id = params[0]
        if job_id in self.rows or job_id in self._rows_tx:
            return 0, None
        self._rows_tx[job_id] = params
        return 1, None


class _SrcCursor:
    def __init__(self, src):
        self._src = src
        self._one = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        sql = _sql(sql)
        self._src.log.append((sql, params))
        if "to_regclass" in sql:
            self._one = ("endpoint_registry",) if self._src.exists else (None,)

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._src.rows)


class _Source:
    def __init__(self, rows=(), exists=True):
        self.rows = list(rows)
        self.exists = exists
        self.log: list = []
        self.opened = 0

    def cursor(self):
        return _SrcCursor(self)

    def opener(self):
        def _open():
            self.opened += 1
            return contextlib.nullcontext(self)

        return _open


def _row(job_id, metadata=None):
    return (
        job_id,
        "default",
        "m",
        "node",
        1,
        "ready",
        "slurm-remote",
        None,
        "t",
        "t",
        "t",
        None,
        None,
        0,
        {} if metadata is None else metadata,
        "bj1",
    )


# ── one-shot semantics ─────────────────────────────────────────────────────


def test_first_run_copies_rows_venue_stamped_and_records_the_marker():
    hub, src = _HubDB(), _Source([_row("j1"), _row("j2")])
    result = copy_endpoints(hub, venue="rccs-front", open_source=src.opener())
    assert result == CopyResult("rccs-front", "copied", source_rows=2, inserted=2)
    assert set(hub.rows) == {"j1", "j2"}
    for params in hub.rows.values():
        assert params[-1] == "rccs-front"
        assert len(params) == len(SOURCE_COLUMNS) + 1
    marker = hub.markers["rccs-front"]
    assert (marker["source_rows"], marker["inserted"]) == (2, 2)
    assert hub.commits == 1


def test_second_run_copies_nothing_even_after_the_servers_removed_the_rows():
    hub, src = _HubDB(), _Source([_row("j1"), _row("j2")])
    copy_endpoints(hub, venue="rccs-front", open_source=src.opener())
    hub.rows.clear()  # remove()/replace on the owning server since the flip
    opened_before = src.opened

    again = copy_endpoints(hub, venue="rccs-front", open_source=src.opener())

    assert again.status == "skipped" and again.inserted == 0
    assert again.marker_at == _MARKER_TIME.isoformat()
    assert hub.rows == {}  # nothing resurrected
    assert src.opened == opened_before  # the frozen source is never opened
    assert not any("INSERT INTO endpoint_registry" in sql for sql, _ in hub.log[-2:])


def test_marker_claim_rows_and_counts_share_one_transaction():
    hub, src = _HubDB(), _Source([_row("j1")])
    copy_endpoints(hub, venue="ai4s", open_source=src.opener())
    sqls = [sql for sql, _ in hub.log]
    assert sqls[0].startswith("INSERT INTO hub_registry_copy_markers")
    assert "ON CONFLICT (venue) DO NOTHING RETURNING venue" in sqls[0]
    assert sqls[-1].startswith("UPDATE hub_registry_copy_markers")
    assert hub.commits == 1 and hub.rollbacks == 0  # one commit covers all


def test_a_failure_mid_copy_leaves_neither_rows_nor_marker_and_propagates():
    hub, src = _HubDB(), _Source([_row("j1"), _row("j2")])
    hub.fail_on, hub.fail_after = "INSERT INTO endpoint_registry", 2
    with pytest.raises(RuntimeError):
        copy_endpoints(hub, venue="bdrt-front", open_source=src.opener())
    assert hub.rows == {} and "bdrt-front" not in hub.markers
    assert hub.rollbacks == 1 and hub.commits == 0
    hub.fail_on = ""  # fixed: the next converge copies the venue whole
    retry = copy_endpoints(hub, venue="bdrt-front", open_source=src.opener())
    assert retry.status == "copied" and retry.inserted == 2


def test_a_sealed_venue_is_skipped_without_touching_its_source():
    hub = _HubDB(markers={"octopus-front": {"copied_at": _MARKER_TIME}})
    src = _Source([_row("j1")])
    result = copy_endpoints(hub, venue="octopus-front", open_source=src.opener())
    assert result.status == "skipped" and src.opened == 0
    assert hub.rows == {} and hub.commits == 0


def test_missing_source_table_is_an_honest_zero_and_closes_the_venue():
    hub, src = _HubDB(), _Source(exists=False)
    result = copy_endpoints(hub, venue="ai4s", open_source=src.opener())
    assert result == CopyResult("ai4s", "no-source-table", source_rows=0, inserted=0)
    assert hub.rows == {}
    assert hub.markers["ai4s"]["source_rows"] == 0


def test_jsonb_metadata_is_reserialised_for_the_hub_insert():
    hub = _HubDB()
    src = _Source([_row("j1", metadata={"litellm_port": 4000})])
    copy_endpoints(hub, venue="rccs-front", open_source=src.opener())
    metadata = hub.rows["j1"][SOURCE_COLUMNS.index("metadata")]
    assert metadata == '{"litellm_port": 4000}'  # a dict cannot be adapted


def test_source_select_uses_the_positional_contract():
    assert SOURCE_COLUMNS[0] == "job_id"
    assert SOURCE_COLUMNS[-1] == "backend_job_id"
    assert len(SOURCE_COLUMNS) == 16
    hub, src = _HubDB(), _Source([_row("j1")])
    copy_endpoints(hub, venue="octopus-front", open_source=src.opener())
    sql, _ = src.log[1]  # log[0] is the table-exists probe
    for name in SOURCE_COLUMNS:
        assert name in sql
    assert "venue" not in sql  # venue exists only in the hub's INSERT


# ── CLI: honest exit status, one line per venue, secrets masked ────────────


def test_cli_prints_the_result_line(monkeypatch, capsys):
    monkeypatch.setenv("SRC_DSN", "postgres://u:p@h/front_rccs")
    monkeypatch.setenv("DST_DSN", "postgres://u:p@h/umrs_db")
    monkeypatch.setattr(
        hub_copy,
        "_run",
        lambda *_: CopyResult("rccs-front", "copied", source_rows=3, inserted=1),
    )
    assert hub_copy.main(["--venue", "rccs-front"]) == 0
    assert capsys.readouterr().out.strip() == (
        "hub-copy venue=rccs-front status=copied source_rows=3 inserted=1"
    )


def test_cli_error_exits_nonzero_with_the_dsn_masked(monkeypatch, capsys):
    monkeypatch.setenv("SRC_DSN", "postgres://umrs:s3cret@h/front_rccs")
    monkeypatch.setenv("DST_DSN", "postgres://umrs:s3cret@h/umrs_db")

    def _boom(*_):
        raise RuntimeError("could not connect\nto postgres://umrs:s3cret@h:5432/x")

    monkeypatch.setattr(hub_copy, "_run", _boom)
    assert hub_copy.main(["--venue", "rccs-front"]) == 1
    err = capsys.readouterr().err
    assert "hub-copy venue=rccs-front status=error" in err
    assert "s3cret" not in err and "postgres://***@" in err
    assert err.count("\n") == 1  # one line for the converge's masked print


def test_mask_dsn_covers_both_url_schemes():
    assert mask_dsn("postgresql://a:b@h/d and postgres://c:d@h/e") == (
        "postgres://***@h/d and postgres://***@h/e"
    )


def test_mask_dsn_hides_a_password_that_contains_an_at_sign():
    # The converge interpolates the password unencoded into the DSN.
    masked = mask_dsn('to "postgres://umrs:p@ss@w0rd@127.0.0.1:5432/umrs_db" failed')
    assert masked == 'to "postgres://***@127.0.0.1:5432/umrs_db" failed'


def test_migration_0118_creates_the_marker_table_and_seals_the_four_fronts():
    sql = (Path(hub_copy.__file__).parent / "0118_hub_registry_copy_markers.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS hub_registry_copy_markers" in sql
    assert "venue        VARCHAR(32) PRIMARY KEY" in sql
    for venue in ("rccs-front", "ai4s", "bdrt-front", "octopus-front"):
        assert f"('{venue}'," in sql
    assert "ON CONFLICT (venue) DO NOTHING" in sql
