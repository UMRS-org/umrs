"""Lease-token principal resolution for the worker rail."""

from __future__ import annotations

import pytest
from umrs.api._rail_identity import RailAuthError, bearer_token, resolve_rail_principal
from umrs.jobs.secret_broker import SecretBroker


def _lease(broker: SecretBroker, sid: str = "sess-1", **kw):
    args = {
        "job_id": f"session-{sid}",
        "secret_name": "inference-token",
        "scope": "inference",
        "owner_id": "alice",
        "session_id": sid,
        "source_ids": ("11111111-1111-4111-8111-111111111111",),
        "lease_seconds": 600,
    }
    args.update(kw)
    return broker.issue_lease(**args)


class TestBearer:
    def test_parses_bearer_only(self):
        assert bearer_token({"authorization": "Bearer tok"}) == "tok"
        assert bearer_token({"authorization": "bearer tok"}) == "tok"
        assert bearer_token({"authorization": "Basic abc"}) is None
        assert bearer_token({"x-api-key": "tok"}) is None
        assert bearer_token({}) is None


class TestResolve:
    def test_ok(self):
        broker = SecretBroker()
        lease, token = _lease(broker)
        p = resolve_rail_principal(token=token, session_id="sess-1", client_ip="", broker=broker)
        assert p.owner == "alice" and p.lease_id == lease.lease_id

    @pytest.mark.parametrize(
        "token,sid,reason",
        [
            (None, "sess-1", "token_invalid"),
            ("nope", "sess-1", "token_invalid"),
        ],
    )
    def test_invalid_tokens(self, token, sid, reason):
        broker = SecretBroker()
        _lease(broker)
        with pytest.raises(RailAuthError) as exc:
            resolve_rail_principal(token=token, session_id=sid, client_ip="", broker=broker)
        assert exc.value.reason == reason

    def test_sid_mismatch(self):
        broker = SecretBroker()
        _, token = _lease(broker)
        with pytest.raises(RailAuthError) as exc:
            resolve_rail_principal(token=token, session_id="other", client_ip="", broker=broker)
        assert exc.value.reason == "sid_mismatch"

    def test_revoked(self):
        broker = SecretBroker()
        lease, token = _lease(broker)
        broker.revoke_lease(lease.lease_id)
        with pytest.raises(RailAuthError) as exc:
            resolve_rail_principal(token=token, session_id="sess-1", client_ip="", broker=broker)
        assert exc.value.reason == "lease_revoked"

    def test_scope_mismatch(self):
        broker = SecretBroker()
        _, token = _lease(broker, scope="download")
        with pytest.raises(RailAuthError) as exc:
            resolve_rail_principal(token=token, session_id="sess-1", client_ip="", broker=broker)
        assert exc.value.reason == "scope_mismatch"

    def test_ip_binding(self):
        broker = SecretBroker()
        lease, token = _lease(broker)
        broker.bind_egress_ip(lease.lease_id, "172.20.8.2")
        with pytest.raises(RailAuthError) as exc:
            resolve_rail_principal(
                token=token, session_id="sess-1", client_ip="10.0.0.9", broker=broker
            )
        assert exc.value.reason == "ip_mismatch"
        ok = resolve_rail_principal(
            token=token, session_id="sess-1", client_ip="172.20.8.2", broker=broker
        )
        assert ok.owner == "alice"
        relaxed = resolve_rail_principal(
            token=token, session_id="sess-1", client_ip="10.0.0.9", broker=broker, ip_check=False
        )
        assert relaxed.owner == "alice"
