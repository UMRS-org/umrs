"""Tests for Slurm nodelist parsing (multinode head-hostname resolution)."""

from __future__ import annotations

import pytest
from umrs.orchestrator.nodelist import first_node_hostname


@pytest.mark.parametrize(
    ("nodelist", "expected"),
    [
        ("ng-dgx-m3-14", "ng-dgx-m3-14"),
        ("ng-dgx-m3-[14-15]", "ng-dgx-m3-14"),
        ("ng-dgx-m3-[14,17-19]", "ng-dgx-m3-14"),
        ("ng-dgx-m3-[014-015]", "ng-dgx-m3-014"),
        ("nodeA,nodeB", "nodeA"),
        ("ng-dgx-m3-[14-15],ng-dgx-m4-[1-2]", "ng-dgx-m3-14"),
        ("  ng-dgx-m0-3  ", "ng-dgx-m0-3"),
    ],
)
def test_first_node_hostname(nodelist: str, expected: str) -> None:
    assert first_node_hostname(nodelist) == expected


@pytest.mark.parametrize("value", ["", "(null)", "None", "n/a", "pending", "unknown"])
def test_first_node_hostname_passthrough_sentinels(value: str) -> None:
    # Null-ish / sentinel values are returned unchanged so callers keep their
    # existing pending/unknown handling.
    assert first_node_hostname(value) == value


def test_first_node_hostname_none() -> None:
    assert first_node_hostname(None) == ""
