"""D-44 pins: proposal acceptance rate as the loop-proposal quality signal."""

from __future__ import annotations

import pytest
from fake_pg_pool import FakePool
from umrs.knowledge.graph_registry import ProposalStore


class TestAcceptanceStats:
    def test_accepted_counts_approved_and_executed(self):
        # aggregate row: accepted=2 (approved+executed), declined=1, pending=1
        pool = FakePool(script=[(2, 1, 1)])
        stats = ProposalStore(pool).acceptance_stats()
        assert stats["accepted"] == 2
        assert stats["declined"] == 1
        assert stats["pending"] == 1
        assert stats["decided"] == 3
        assert stats["acceptance_rate"] == pytest.approx(2 / 3)

    def test_no_decisions_reports_null_rate_never_fake_hundred(self):
        pool = FakePool(script=[(0, 0, 4)])
        stats = ProposalStore(pool).acceptance_stats()
        assert stats["decided"] == 0
        assert stats["acceptance_rate"] is None

    def test_empty_past_reports_null_rate(self):
        pool = FakePool(script=[None])
        stats = ProposalStore(pool).acceptance_stats()
        assert stats["acceptance_rate"] is None
        assert stats["decided"] == 0

    def test_graph_filter_travels_in_both_params(self):
        pool = FakePool(script=[(1, 0, 0)])
        ProposalStore(pool).acceptance_stats(target_graph_id="kg-A")
        _, params = pool.record[0]
        assert params == ("kg-A", "kg-A")
        assert pool.record[0][0].count("target_graph_id") >= 1
