"""U3 S1 item 9: live session counts and caps are core.

The quota status route read them from ``session_cap``, whose start-lock half
moves to ``umrs_r4`` with the venue cluster in U3 S5. They now live in
``umrs.quotas.session_count``; ``session_cap`` uses the same objects.
"""

from __future__ import annotations

import ast
from pathlib import Path

from umrs.quotas import session_count
from umrs_r4.quotas import session_cap

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs"


def test_session_cap_uses_the_core_objects():
    assert session_cap.count_live_sessions is session_count.count_live_sessions
    assert session_cap.session_limits is session_count.session_limits
    assert session_cap.SESSION_JOB_TYPE == session_count.SESSION_JOB_TYPE == "frontend_session"


def test_quota_status_reads_counts_from_core():
    tree = ast.parse((SRC / "api" / "quota_status.py").read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "umrs.quotas.session_count" in modules
    assert "umrs_r4.quotas.session_cap" not in modules


def test_no_pool_counts_zero():
    assert session_count.count_live_sessions(None, venue="v", user_id="u") == 0
