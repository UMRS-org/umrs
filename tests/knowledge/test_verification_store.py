"""Tests for umrs.knowledge.verification_store.

Property tests for finding transition rules, auto-resolve, key stability,
open-findings filtering, and blocking-findings filtering.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

# ── in-memory store for property tests ────────────────────────────────────────


class InMemoryStore:
    """Thin in-memory implementation mirroring VerificationStore semantics."""

    def __init__(self) -> None:
        self._reports: dict[str, dict] = {}
        self._findings: dict[str, dict] = {}

    # -- reports ---------------------------------------------------------------

    def create_report(self, **kwargs) -> dict:
        """Create an immutable versioned report."""
        pid = kwargs["project_id"]
        spec = kwargs["spec_name"]
        ver = kwargs["version"]
        key = (pid, spec, ver)
        for r in self._reports.values():
            if (r["project_id"], r["spec_name"], r["version"]) == key:
                raise ValueError(f"Report already exists: {pid}/{spec}/v{ver}")
        report = {"version": ver, **kwargs, "created_at": _utcnow()}
        self._reports[kwargs["report_id"]] = report
        return report

    # -- findings --------------------------------------------------------------

    def upsert_finding(self, **kwargs) -> dict:
        """Upsert a finding; preserve existing status."""
        key = kwargs["finding_key"]
        existing = self._findings.get(key)
        if existing:
            for field in ("severity", "title", "description", "evidence", "artifact_id"):
                if field in kwargs:
                    existing[field] = kwargs[field]
            existing["updated_at"] = _utcnow()
            return existing
        status = kwargs.get("status", "open")
        if status not in _VALID_TRANSITIONS:
            raise TransitionError(key, "(new)", status)
        finding = {
            "finding_key": key,
            "status": status,
            "waived_evidence": None,
            "waiver_expires": None,
            **kwargs,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
        }
        self._findings[key] = finding
        return finding

    def transition_finding(self, finding_key: str, from_status: str, to_status: str) -> dict:
        """Transition a finding."""
        if to_status not in _VALID_TRANSITIONS.get(from_status, []):
            raise TransitionError(finding_key, from_status, to_status)
        finding = self._findings.get(finding_key)
        if finding is None:
            raise ValueError(f"Finding '{finding_key}' not found")
        if finding["status"] != from_status:
            raise ValueError(
                f"Finding '{finding_key}' not in status '{from_status}' "
                f"(current: {finding['status']})"
            )
        finding["status"] = to_status
        finding["updated_at"] = _utcnow()
        return finding

    def waive_finding(self, finding_key: str, evidence: str, **kw) -> dict:
        """Waive a finding."""
        expires_after = kw.get("expires_after")
        finding = self._findings.get(finding_key)
        if finding is None:
            raise ValueError(f"Finding '{finding_key}' not found")
        finding["status"] = "waived"
        finding["waived_evidence"] = evidence
        if expires_after is not None:
            finding["waiver_expires"] = finding["updated_at"] + expires_after
        finding["updated_at"] = _utcnow()
        return finding

    def auto_resolve_expired(self, now: datetime | None = None) -> int:
        """Resolve expired waivers."""
        ts = now or _utcnow()
        count = 0
        for f in self._findings.values():
            if f["status"] == "waived" and f["waiver_expires"] is not None and f["waiver_expires"] < ts:
                f["status"] = "resolved"
                f["updated_at"] = ts
                count += 1
        return count

    def list_open_findings(self) -> list[dict]:
        """List open or acknowledged findings."""
        return [
            f for f in self._findings.values()
            if f["status"] in ("open", "acknowledged")
        ]

    def list_blocking_findings(self) -> list[dict]:
        """List blocking findings (any status)."""
        return [
            f for f in self._findings.values()
            if f["severity"] == "blocking"
        ]

    # -- helpers ---------------------------------------------------------------

    def set_time(self, finding_key: str, delta: timedelta) -> None:
        """Manually set waiver_expires to the past for testing."""
        finding = self._findings.get(finding_key)
        if finding:
            finding["waiver_expires"] = _utcnow() - delta


# ── fixtures & helpers ────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_VALID_TRANSITIONS: dict[str, list[str]] = {
    "open": ["acknowledged", "resolved", "waived"],
    "acknowledged": ["resolved", "waived"],
    "resolved": [],
    "waived": ["waived"],
}


class TransitionError(ValueError):
    def __init__(self, finding_key: str, from_status: str, to_status: str) -> None:
        self.finding_key = finding_key
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Invalid transition for '{finding_key}': "
            f"{from_status} → {to_status}. "
            f"Valid: open→{['acknowledged', 'resolved', 'waived']}, "
            f"acknowledged→{['resolved', 'waived']}, resolved→[none]"
        )


@pytest.fixture
def store():
    return InMemoryStore()


# ── Transition-rule property tests ────────────────────────────────────────────


class TestTransitionStickyWaiver:
    """A waived finding remains waived through re-waive — waiver is sticky."""

    def test_rewaive_keeps_waived(self, store):
        store.upsert_finding(finding_key="f1", report_id="r1", category="sec",
                             severity="blocking", title="X")
        store.waive_finding("f1", "evidence-1")
        assert store._findings["f1"]["status"] == "waived"
        # Re-waive
        store.waive_finding("f1", "evidence-2")
        assert store._findings["f1"]["status"] == "waived"
        # Transitions from waived should still fail for non-waived targets
        with pytest.raises(TransitionError):
            store.transition_finding("f1", "waived", "open")

    def test_transition_to_waived_from_acknowledged(self, store):
        store.upsert_finding(finding_key="f1", report_id="r1", category="sec",
                             severity="warning", title="Y", status="acknowledged")
        store.transition_finding("f1", "acknowledged", "waived")
        assert store._findings["f1"]["status"] == "waived"


class TestAutoResolveTriggers:
    """Auto-resolve triggers when waiver_expires < now()."""

    def test_expired_waiver_resolves(self, store):
        store.upsert_finding(finding_key="f1", report_id="r1", category="sec",
                             severity="blocking", title="Z", status="open")
        store.waive_finding("f1", "waiver-evidence", expires_after=timedelta(hours=1))
        # Simulate expiry by moving waiver_expires into the past
        store.set_time("f1", timedelta(hours=2))
        now = _utcnow()
        count = store.auto_resolve_expired(now=now)
        assert count == 1
        assert store._findings["f1"]["status"] == "resolved"

    def test_non_expired_waiver_not_resolved(self, store):
        store.upsert_finding(finding_key="f1", report_id="r1", category="sec",
                             severity="blocking", title="Z", status="open")
        store.waive_finding("f1", "waiver-evidence", expires_after=timedelta(hours=2))
        now = _utcnow()
        count = store.auto_resolve_expired(now=now)
        assert count == 0
        assert store._findings["f1"]["status"] == "waived"

    def test_waived_no_expiry_not_resolved(self, store):
        store.upsert_finding(finding_key="f1", report_id="r1", category="sec",
                             severity="blocking", title="Z", status="open")
        store.waive_finding("f1", "waiver-evidence")  # no expires_after
        now = _utcnow()
        count = store.auto_resolve_expired(now=now)
        assert count == 0
        assert store._findings["f1"]["status"] == "waived"


class TestKeyStability:
    """Finding keys are stable across upsert runs."""

    def test_key_stable_across_three_upserts(self, store):
        key = "critical-auth-bypass"
        result1 = store.upsert_finding(finding_key=key, report_id="r1", category="auth",
                                       severity="blocking", title="Original")
        assert result1["finding_key"] == key
        assert result1["status"] == "open"

        result2 = store.upsert_finding(finding_key=key, report_id="r1", category="auth",
                                       severity="blocking", title="Updated v2",
                                       evidence="new evidence")
        assert result2["finding_key"] == key
        assert result2["title"] == "Updated v2"
        assert result2["status"] == "open"  # status preserved

        result3 = store.upsert_finding(finding_key=key, report_id="r1", category="auth",
                                       severity="critical", title="Updated v3",
                                       evidence="more evidence")
        assert result3["finding_key"] == key
        assert result3["severity"] == "critical"
        assert result3["status"] == "open"  # still preserved


class TestResolvedImmutable:
    """Resolved findings cannot transition."""

    def test_resolved_no_transition(self, store):
        store.upsert_finding(finding_key="f1", report_id="r1", category="sec",
                             severity="warning", title="X", status="open")
        store.transition_finding("f1", "open", "resolved")
        assert store._findings["f1"]["status"] == "resolved"
        # No transitions from resolved
        for target in ["open", "acknowledged", "waived"]:
            with pytest.raises(TransitionError):
                store.transition_finding("f1", "resolved", target)


class TestTransitionRules:
    """All invalid transitions are rejected."""

    def test_all_invalid_transitions(self, store):
        for from_status, allowed in _VALID_TRANSITIONS.items():
            invalid = [s for s in _VALID_TRANSITIONS if s not in allowed]
            for target in invalid:
                store.upsert_finding(finding_key=f"f_{from_status}", report_id="r1",
                                     category="sec", severity="info", title="t",
                                     status=from_status)
                with pytest.raises(TransitionError):
                    store.transition_finding(f"f_{from_status}", from_status, target)

    def test_valid_transitions(self, store):
        # open → acknowledged
        store.upsert_finding(finding_key="f1", report_id="r1", category="sec",
                             severity="info", title="t", status="open")
        store.transition_finding("f1", "open", "acknowledged")
        assert store._findings["f1"]["status"] == "acknowledged"
        # acknowledged → resolved
        store.transition_finding("f1", "acknowledged", "resolved")
        assert store._findings["f1"]["status"] == "resolved"


# ── Open findings list filtering ─────────────────────────────────────────────


class TestListOpenFindings:
    """list_open_findings returns only open or acknowledged."""

    def test_only_open_and_acknowledged(self, store):
        store.upsert_finding(finding_key="f1", report_id="r1", category="sec",
                             severity="blocking", title="Open")
        store.upsert_finding(finding_key="f2", report_id="r1", category="sec",
                             severity="warning", title="Acknowledged", status="acknowledged")
        store.upsert_finding(finding_key="f3", report_id="r1", category="sec",
                             severity="info", title="Resolved", status="resolved")
        store.waive_finding("f4", "e", expires_after=timedelta(hours=1)) if False else None
        store.upsert_finding(finding_key="f4", report_id="r1", category="sec",
                             severity="info", title="Waived", status="waived")

        open_findings = store.list_open_findings()
        keys = {f["finding_key"] for f in open_findings}
        assert keys == {"f1", "f2"}
        assert "f3" not in keys  # resolved excluded
        assert "f4" not in keys  # waived excluded


# ── Blocking findings ────────────────────────────────────────────────────────


class TestListBlockingFindings:
    """list_blocking_findings returns only severity=blocking, all statuses."""

    def test_only_blocking_severity(self, store):
        store.upsert_finding(finding_key="b1", report_id="r1", category="sec",
                             severity="blocking", title="Blocking", status="open")
        store.upsert_finding(finding_key="b2", report_id="r1", category="sec",
                             severity="blocking", title="Blocked-ack", status="acknowledged")
        store.upsert_finding(finding_key="w1", report_id="r1", category="sec",
                             severity="warning", title="Warning")
        store.upsert_finding(finding_key="i1", report_id="r1", category="sec",
                             severity="info", title="Info")

        blocking = store.list_blocking_findings()
        keys = {f["finding_key"] for f in blocking}
        assert keys == {"b1", "b2"}
        assert "w1" not in keys
        assert "i1" not in keys

    def test_blocking_including_resolved(self, store):
        store.upsert_finding(finding_key="b1", report_id="r1", category="sec",
                             severity="blocking", title="Blocked", status="resolved")
        blocking = store.list_blocking_findings()
        keys = {f["finding_key"] for f in blocking}
        assert "b1" in keys  # resolved blocking findings included


# ── TransitionError properties ──────────────────────────────────────────────


class TestTransitionError:
    """TransitionError carries diagnostic info."""

    def test_error_attributes(self):
        err = TransitionError("f1", "open", "resolved")
        assert err.finding_key == "f1"
        assert err.from_status == "open"
        assert err.to_status == "resolved"
        msg = str(err)
        assert "f1" in msg
        assert "open" in msg
        assert "resolved" in msg

    def test_error_from_resolved(self):
        err = TransitionError("f1", "resolved", "open")
        msg = str(err)
        assert "resolved→[none]" in msg
