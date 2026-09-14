"""validate_key namespacing + foreign-key visibility pins.

validate_key silently re-namespaces foreign keys into the caller's project
(fail-closed: the operation then targets the caller's own namespace and 404s).
That hides client misconfiguration — the review asked for visibility on
client-supplied read paths, which is what ``warn_on_rename`` provides.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException
from umrs.api.files import validate_key


def test_bare_key_is_namespaced() -> None:
    assert validate_key("data.csv", "proj-1") == "proj-1/data.csv"


def test_own_namespaced_key_passes_through() -> None:
    assert validate_key("proj-1/uploads/data.csv", "proj-1") == "proj-1/uploads/data.csv"


def test_path_traversal_rejected() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_key("../etc/passwd", "proj-1")
    assert exc_info.value.status_code == 400


def test_foreign_key_rename_warns_when_opted_in(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="umrs.api.files"):
        key = validate_key("other-proj/data.csv", "proj-1", warn_on_rename=True)
    assert key == "proj-1/other-proj/data.csv"
    assert any("re-namespacing" in record.getMessage() for record in caplog.records)


def test_foreign_key_rename_silent_by_default(caplog) -> None:
    # Upload paths pass generated purpose-relative keys — they must not warn.
    with caplog.at_level(logging.WARNING, logger="umrs.api.files"):
        validate_key("uploads/uuid_data.csv", "proj-1")
    assert not caplog.records
