"""Regressions for scalable ledger reports and canonical K4 pricing (ADR-112 D).

The store's current-row reads must not materialize the global append-only
``hpc_usage_current`` view. Report totals cover every matching current row,
not only a returned page. A published partition that cannot be priced must
stop the submit rather than lowering the estimate.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from hpc_ledger_pg import delete_ledger_rows, ledger_pool_or_skip
from umrs.cli.admin import site_accounts
from umrs.hpc_accounts import site_profile, usage_reports
from umrs.hpc_accounts import usage_ledger as ul
from umrs.server import gated_step_lane
from umrs.server.step_executor import StepExecutionError

from r4_contracts import hpc_gate

SINCE = datetime(2026, 11, 9, tzinfo=timezone.utc)


class _RecordingPg(ul.PgUsageLedger):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        self.calls.append((" ".join(sql.split()), params))
        return []


def _unit(venue: str, attempt: str) -> ul.LedgerUnit:
    return ul.LedgerUnit(
        venue=venue,
        attempt_id=attempt,
        person="alice",
        lane="staged_step",
        tag="t",
        run_id=attempt.split("/")[0],
        step="s1",
    )


def _reserved_pair(ledger: ul.UsageLedger, venue: str) -> None:
    for attempt, resources, job in (
        ("r1/s1/1", ul.Resources(1, 1, 60), "101"),
        ("r2/s1/1", ul.Resources(1, 2, 120), "102"),
    ):
        unit = _unit(venue, attempt)
        ledger.pending(unit, resources)
        ledger.submitted(unit, job, None)


def test_current_of_is_a_linear_latest_row_reduction(monkeypatch):
    rows = [
        {
            "ledger_id": ledger_id,
            "venue": "v",
            "run_id": "r1",
            "attempt_id": attempt,
            "step": "s1",
            "state": state,
        }
        for ledger_id, attempt, state in (
            (2, "r1/s1/1", "submitted"),
            (1, "r1/s1/1", "pending"),
            (3, "r2/s1/1", "pending"),
        )
    ]

    def no_sort(*_args: Any) -> None:
        raise AssertionError("current-row reduction sorted the history")

    monkeypatch.setattr(ul, "sorted", no_sort, raising=False)
    current = ul._current_of(rows)
    assert {row["attempt_id"]: row["state"] for row in current} == {
        "r1/s1/1": "submitted",
        "r2/s1/1": "pending",
    }


def test_pg_current_pushes_latest_filters_order_and_limit_to_the_base_table():
    ledger = _RecordingPg()
    ledger.current("v", person="alice", project="p1", since=SINCE, limit=1)
    ((sql, params),) = ledger.calls
    assert "FROM hpc_usage_ledger AS cur" in sql
    assert "NOT EXISTS" in sql
    assert "hpc_usage_current" not in sql
    assert "ORDER BY cur.ledger_id DESC LIMIT %s" in sql
    assert params == ("v", "alice", "p1", SINCE, 1)


def test_pg_open_rows_filter_current_rows_in_sql():
    ledger = _RecordingPg()
    ledger.open_rows("v")
    ((sql, params),) = ledger.calls
    assert "FROM hpc_usage_ledger AS cur" in sql
    assert "NOT EXISTS" in sql
    assert "hpc_usage_current" not in sql
    assert "cur.state IN (%s, %s)" in sql
    assert params == ("v", "pending", "submitted")


def test_pg_project_rows_filter_current_rows_in_sql():
    ledger = _RecordingPg()
    ledger.project_rows("p1")
    ((sql, params),) = ledger.calls
    assert "FROM hpc_usage_ledger AS cur" in sql
    assert "NOT EXISTS" in sql
    assert "hpc_usage_current" not in sql
    assert "cur.project_id = %s" in sql
    assert "ORDER BY cur.ledger_id DESC" in sql
    assert params == ("p1",)


def test_binding_budget_sums_do_not_read_the_global_current_view():
    sql = " ".join(ul._SUMS.split())
    assert "FROM hpc_usage_ledger AS cur" in sql
    assert "NOT EXISTS" in sql
    assert "hpc_usage_current" not in sql


def test_my_usage_totals_are_independent_of_the_returned_page():
    ledger = ul.InMemoryUsageLedger()
    _reserved_pair(ledger, "v")
    answer = usage_reports.my_usage(ledger, "v", "alice", limit=1)
    assert len(answer["rows"]) == 1
    assert set(answer["totals"]) == {
        "jobs",
        "gpu_hours",
        "core_hours",
        "reserved_gpu_hours",
        "reserved_core_hours",
    }
    assert answer["totals"]["jobs"] == 2
    assert answer["totals"]["reserved_gpu_hours"] == pytest.approx(300 / 3600)
    assert answer["totals"]["reserved_core_hours"] == pytest.approx(60 / 3600 + 120 / 3600)


def test_admin_json_totals_are_independent_of_the_row_cap(monkeypatch, capsys):
    ledger = ul.InMemoryUsageLedger()
    _reserved_pair(ledger, "v")
    monkeypatch.setattr(site_accounts, "_usage_ledger", lambda _args: (ledger, "v"))
    monkeypatch.setattr(usage_reports, "LIMIT_MAX", 1)
    args = SimpleNamespace(person=None, project=None, since=None, json=True)
    assert site_accounts._cmd_usage(args, None) == 0
    answer = json.loads(capsys.readouterr().out)
    assert len(answer["rows"]) == 1
    assert answer["totals"]["jobs"] == 2
    assert answer["totals"]["reserved_gpu_hours"] == pytest.approx(300 / 3600)


@pytest.mark.integration
def test_pg_current_totals_cover_all_matching_rows_beyond_the_page():
    pool = ledger_pool_or_skip()
    venue = f"report-{uuid.uuid4().hex[:8]}"
    ledger = ul.PgUsageLedger(pool)
    try:
        _reserved_pair(ledger, venue)
        assert len(ledger.current(venue, limit=1)) == 1
        sums = ledger.current_totals(venue)
        assert sums["jobs"] == 2
        assert sums["gpu_seconds"] == 0
        assert sums["reserved_gpu_seconds"] == 300
        assert sums["reserved_core_hours"] == pytest.approx(180 / 3600)
    finally:
        delete_ledger_rows(pool, [venue])


def test_estimate_delegates_to_the_canonical_grant_cost(monkeypatch):
    cost = Mock(wraps=hpc_gate.grant_cost)
    monkeypatch.setattr(hpc_gate, "grant_cost", cost)
    partition = {"exclusive": True, "max_cpus": 64, "max_gpus": 8}
    resources = ul.estimate(4, 1, 61, partition)
    assert resources == ul.Resources(64, 8, 120)
    cost.assert_called_once_with({"cpus": 4, "gpus": 1, "time": 61}, partition)


def test_gated_pricing_fails_closed_when_profile_publication_fails(monkeypatch):
    def unavailable(_site: Any) -> Any:
        raise site_profile.SiteProfileUnavailable("missing profile")

    monkeypatch.setattr(site_profile, "published", unavailable)
    with pytest.raises(StepExecutionError, match="hpcaccess.ledger_unavailable"):
        gated_step_lane._profile_partition("gpu")


@pytest.mark.parametrize(
    "profile",
    [{"partitions": {}}, {"partitions": {"gpu": None}}, {"partitions": {"gpu": []}}],
)
def test_gated_pricing_fails_closed_on_a_missing_or_invalid_partition(monkeypatch, profile):
    monkeypatch.setattr(site_profile, "published", lambda _site: (profile, b"", "sha"))
    with pytest.raises(StepExecutionError, match="hpcaccess.ledger_unavailable"):
        gated_step_lane._profile_partition("gpu")


def test_run_estimate_fails_closed_when_its_partition_is_unavailable():
    from umrs.server.grant_admission import run_estimate_seconds

    body = {"steps": [{"partition": "gpu", "cpus": 1, "gpus": 1, "time": 60}]}
    with pytest.raises(ul.LedgerUnavailable, match="partition"):
        run_estimate_seconds(body, {"partitions": {}})
