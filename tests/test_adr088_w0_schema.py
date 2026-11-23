"""ADR-088 W0: the security-family schema, its redaction, and the admin.* rows.

Each family test asserts both halves: the row is written through the real
``emit_audit`` (captured at its persistent-store seam), and nothing the
call site handed in beyond the schema (tokens, addresses, free text)
reaches the serialized row.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from umrs.api import audit_events
from umrs.audit import security_events
from umrs.audit.security_events import (
    REDACTED,
    REDACTED_IP,
    SECURITY_FAMILIES,
    emit_security,
    redact_details,
)

# A JWT shape (header.payload.signature) joined at run time, so the security
# workflow's secret scan (gitleaks jwt rule) does not flag the source line.
JWT = ".".join(("eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiJhbGljZSJ9", "c2lnbmF0dXJl"))


class _Capture:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.rows.append(record)

    def of(self, event: str) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["event"] == event]

    def text(self) -> str:
        return json.dumps(self.rows, default=str)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> _Capture:
    sink = _Capture()
    monkeypatch.setattr(audit_events, "_audit_store", sink)
    return sink


def test_every_family_is_named_and_reserved_security_names_exist() -> None:
    for event in ("auth.failed", "zmq.session_minted", "zmq.session_revoked"):
        assert event in SECURITY_FAMILIES
    for event in ("egress.denied", "mcp.invoke_refused", "scanner.finding"):
        assert event in SECURITY_FAMILIES
    assert {e for e in SECURITY_FAMILIES if e.startswith("security.")} == {
        "security.incident",
        "security.action",
        "security.watch_degraded",
    }


def test_unknown_fields_are_dropped_and_counted() -> None:
    clean = redact_details(
        "egress.denied",
        {"host": "example.org", "reason": "not_listed", "url": "https://x/?t=1", "body": "x"},
    )
    assert clean == {
        "host": "example.org",
        "reason": "not_listed",
        "security_schema": "1",
        "dropped_keys": 2,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (JWT, REDACTED),
        ("umrs-admin-abcdef", REDACTED),
        ("Bearer abc", REDACTED),
        ("alice@10.1.2.3", REDACTED),
        ("::1", REDACTED),
        ("a b c", REDACTED),
        ("x" * 200, REDACTED),
        ("alice", "alice"),
        ("8f0c2c1e9a7b4d3e", "8f0c2c1e9a7b4d3e"),
    ],
)
def test_id_fields_refuse_tokens_addresses_and_text(value: str, expected: str) -> None:
    assert redact_details("export.requested", {"job_id": value})["job_id"] == expected


def test_kinds_reject_wrong_types() -> None:
    clean = redact_details(
        "zmq.session_minted",
        {"ttl_seconds": True, "max_gpus": "8", "kind": "Not A Code!"},
    )
    assert clean["ttl_seconds"] is None
    assert clean["max_gpus"] is None
    assert clean["kind"] == "other"


@pytest.mark.parametrize(
    ("host", "expected"),
    [("10.0.0.1", REDACTED_IP), ("[fe80::1]", REDACTED_IP), ("Example.ORG.", "example.org")],
)
def test_host_kind_never_keeps_an_address(host: str, expected: str) -> None:
    assert redact_details("egress.denied", {"host": host})["host"] == expected


def test_route_kind_keeps_two_segments_only() -> None:
    clean = redact_details("auth.failed", {"route": "/gw/files/download/secret.txt?sig=abc"})
    assert clean["route"] == "/gw/files"


def test_unknown_family_is_not_written(captured: _Capture) -> None:
    assert emit_security("made.up", token=JWT) is None
    assert captured.rows == []


def test_emitter_failure_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("store down")

    monkeypatch.setattr(audit_events, "emit_audit", boom)
    assert emit_security("auth.failed", reason="x") is not None


def test_actor_and_project_are_checked_like_ids(captured: _Capture) -> None:
    emit_security("auth.failed", actor=JWT, project_id="10.0.0.9", reason="bad_token")
    (row,) = captured.of("auth.failed")
    assert row["actor"] == REDACTED
    assert row["project_id"] == REDACTED
    assert row["outcome"] == "denied"
    assert JWT not in captured.text() and "10.0.0.9" not in captured.text()


# -- admin.access_system: derived from every *access_changed row ------------


def test_access_to_system_derives_admin_access_system(captured: _Capture) -> None:
    audit_events.emit_audit(
        "endpoint.access_changed",
        project_id="proj-1",
        details={"job_id": "job-7", "access": ["system"], "by": "alice", "note": JWT},
    )
    (row,) = captured.of("admin.access_system")
    assert row["actor"] == "alice"
    assert row["details"]["resource_kind"] == "endpoint"
    assert row["details"]["resource_id"] == "job-7"
    assert row["details"]["source_event"] == "endpoint.access_changed"
    assert JWT not in json.dumps(row)


def test_access_without_system_derives_nothing(captured: _Capture) -> None:
    audit_events.emit_audit(
        "workflow.access_changed",
        details={"workflow_name": "wf", "access": ["project:p1"], "by": "bob"},
    )
    assert captured.of("admin.access_system") == []


# -- admin.role_changed / account_state_changed / quota_changed -------------


def test_user_store_role_disable_and_quota_changes_are_audited(
    tmp_path: Any, captured: _Capture
) -> None:
    from umrs.admin.user_store import UserStore

    store = UserStore(tmp_path)
    store.create("carol", email="carol@example.org", quota={"gpus": 1})
    store.set_role("carol", "admin")
    store.set_disabled("carol", True)
    store.update("carol", quota={"gpus": 4, "storage_gb": 10})
    store.update("carol", fullname="Carol C")  # no audited field: no row

    (role,) = captured.of("admin.role_changed")
    assert role["details"]["old_role"] == "user"
    assert role["details"]["new_role"] == "admin"
    (state,) = captured.of("admin.account_state_changed")
    assert state["details"] == {"username": "carol", "disabled": True, "security_schema": "1"}
    (quota,) = captured.of("admin.quota_changed")
    assert quota["details"]["fields"] == ["gpus", "storage_gb"]
    text = captured.text()
    assert "carol@example.org" not in text and "Carol C" not in text
    assert '"gpus": 4' not in text  # field names, never values


def test_project_budget_write_is_audited(captured: _Capture) -> None:
    from umrs.api._identity import CallerIdentity
    from umrs.api.projects import BudgetBody, put_project_budget

    class _Store:
        def get(self, project_id: str) -> dict[str, Any]:
            return {"project_id": project_id, "owner_id": "dave", "members": []}

        def get_budget(self, _project_id: str) -> dict[str, Any]:
            return {}

        def set_budget(self, _project_id: str, **caps: Any) -> dict[str, Any]:
            return dict(caps)

    body = BudgetBody(max_gpu_hours=5.0)
    identity = CallerIdentity(user_id="root", project_id="p1", is_admin=True)
    put_project_budget("p1", body, store=_Store(), identity=identity)
    (row,) = captured.of("admin.quota_changed")
    assert row["details"]["subject_kind"] == "project"
    assert row["details"]["fields"] == ["max_gpu_hours"]
    assert row["details"]["by_admin"] is True
    assert "5.0" not in json.dumps(row["details"])


def test_module_names_every_family_in_its_docstring() -> None:
    doc = security_events.__doc__ or ""
    assert all(event in doc for event in SECURITY_FAMILIES)
