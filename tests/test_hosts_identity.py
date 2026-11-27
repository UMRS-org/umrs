"""Host enrollment, host tokens and host-bound leases (ADR-098 D-5/D-6, SH-H1b).

Acceptance (decisions-2026-11-24 §5, SH-H1b): an unenrolled key gets 401;
an expired token replayed gets 401; a lease presented from another host gets
403 ``rail.host_mismatch``; a reused enrollment token is refused; no secret
reaches the repository or the logs.

Every key here is generated per test run (ephemeral); nothing is a real
credential. The Postgres half is env-gated like the other ``*_pg`` suites
and runs in a scratch schema dropped afterwards.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from umrs.hosts.enroll import (
    EnrollmentError,
    enrollment_request,
    generate_host_key,
    issue_enrollment,
    redeem_enrollment,
)
from umrs.hosts.identity import (
    HOST_MISMATCH,
    HOST_TOKEN_HEADER,
    MAX_TOKEN_SECONDS,
    HostAuthError,
    HostKeyStore,
    b64url,
    lease_host_refusal,
    presented_host,
    sign_host_token,
    verify_host_token,
)
from umrs.infer_proxy.leases import check_lease
from umrs.jobs.secret_broker import SecretBroker
from umrs_r4.api._rail_identity import RailAuthError, resolve_rail_principal

pytest.importorskip("cryptography")

VENUE = "site-a"
SRC = "11111111-1111-4111-8111-111111111111"


def _enrolled(store: HostKeyStore, host_id: str) -> Any:
    """A fresh private key for ``host_id``, enrolled through a one-time token."""
    issued = issue_enrollment(store, host_id)
    key = generate_host_key()
    body = enrollment_request(issued.token, key)
    redeem_enrollment(store, body["token"], body["public_key"])
    return key


def _token(key: Any, host_id: str, *, venue: str = VENUE, **kw: Any) -> str:
    return sign_host_token(key, venue=venue, host_id=host_id, **kw)


def _lease(broker: SecretBroker, *, host_id: str = "", sid: str = "sess-1") -> Any:
    return broker.issue_lease(
        job_id=f"session-{sid}",
        secret_name="inference-token",
        scope="inference",
        owner_id="alice",
        session_id=sid,
        source_ids=(SRC,),
        default_source_id=SRC,
        lease_seconds=600,
        egress_ip="10.0.0.5",
        host_id=host_id,
    )


@pytest.fixture()
def store() -> HostKeyStore:
    return HostKeyStore(venue=VENUE)


# ── enrollment ──────────────────────────────────────────────────────────────


class TestEnrollment:
    def test_redeem_enrolls_public_key_only(self, store):
        key = _enrolled(store, "hostd-1")
        enrolled = store.public_key("hostd-1")
        assert enrolled is not None and len(enrolled.public_key) == 32
        assert enrolled.key_id.startswith("sha256:")
        verify_host_token(_token(key, "hostd-1"), store)

    def test_reused_token_refused(self, store):
        issued = issue_enrollment(store, "hostd-1")
        body = enrollment_request(issued.token, generate_host_key())
        redeem_enrollment(store, body["token"], body["public_key"])
        other = enrollment_request(issued.token, generate_host_key())
        with pytest.raises(EnrollmentError) as exc:
            redeem_enrollment(store, other["token"], other["public_key"])
        assert exc.value.code == "enroll.token_used"

    def test_expired_token_refused(self, store, monkeypatch):
        issued = issue_enrollment(store, "hostd-1", ttl_s=60)
        later = enroll_mod_now() + timedelta(seconds=120)
        monkeypatch.setattr("umrs.hosts.identity._now", lambda: later)
        body = enrollment_request(issued.token, generate_host_key())
        with pytest.raises(EnrollmentError) as exc:
            redeem_enrollment(store, body["token"], body["public_key"])
        assert exc.value.code == "enroll.token_expired"
        assert store.public_key("hostd-1") is None

    @pytest.mark.parametrize("token", ["", "umrs-enroll-unknown", "bogus"])
    def test_unknown_token_refused(self, store, token):
        body = enrollment_request("x", generate_host_key())
        with pytest.raises(EnrollmentError) as exc:
            redeem_enrollment(store, token, body["public_key"])
        assert exc.value.code == "enroll.token_invalid"

    @pytest.mark.parametrize(
        "host_id,code",
        [("local", "enroll.host_reserved"), ("Bad Host", "enroll.host_invalid")],
    )
    def test_host_id_checked(self, store, host_id, code):
        with pytest.raises(EnrollmentError) as exc:
            issue_enrollment(store, host_id)
        assert exc.value.code == code

    def test_ttl_bounded(self, store):
        with pytest.raises(EnrollmentError) as exc:
            issue_enrollment(store, "hostd-1", ttl_s=10**6)
        assert exc.value.code == "enroll.ttl_invalid"

    def test_bad_public_key_refused_before_token_spent(self, store):
        issued = issue_enrollment(store, "hostd-1")
        with pytest.raises(EnrollmentError) as exc:
            redeem_enrollment(store, issued.token, b64url(b"short"))
        assert exc.value.code == "enroll.key_invalid"
        body = enrollment_request(issued.token, generate_host_key())
        redeem_enrollment(store, body["token"], body["public_key"])

    def test_enrolled_host_needs_rekey(self, store):
        old = _enrolled(store, "hostd-1")
        with pytest.raises(EnrollmentError) as exc:
            issue_enrollment(store, "hostd-1")
        assert exc.value.code == "enroll.already_enrolled"
        new = generate_host_key()
        issued = issue_enrollment(store, "hostd-1", rekey=True)
        body = enrollment_request(issued.token, new)
        redeem_enrollment(store, body["token"], body["public_key"])
        verify_host_token(_token(new, "hostd-1"), store)
        with pytest.raises(HostAuthError) as refused:
            verify_host_token(_token(old, "hostd-1"), store)
        assert refused.value.reason == "host_token_invalid"

    def test_unenroll_refuses_next_token(self, store):
        key = _enrolled(store, "hostd-1")
        assert store.unenroll("hostd-1")
        with pytest.raises(HostAuthError) as exc:
            verify_host_token(_token(key, "hostd-1"), store)
        assert exc.value.reason == "host_unenrolled"

    def test_token_never_stored_in_plaintext(self, store):
        issued = issue_enrollment(store, "hostd-1")
        backend = store._backend
        assert issued.token not in repr(backend.enrollments)


def enroll_mod_now():
    from umrs.hosts.identity import _now

    return _now()


# ── host tokens ─────────────────────────────────────────────────────────────


class TestHostToken:
    def test_valid_token(self, store):
        key = _enrolled(store, "hostd-1")
        claims = verify_host_token(_token(key, "hostd-1"), store)
        assert claims.host_id == "hostd-1" and claims.venue == VENUE

    def test_unenrolled_key_refused(self, store):
        stranger = generate_host_key()
        with pytest.raises(HostAuthError) as exc:
            verify_host_token(_token(stranger, "hostd-9"), store)
        assert exc.value.reason == "host_unenrolled" and exc.value.status == 401

    def test_other_key_for_enrolled_host_refused(self, store):
        _enrolled(store, "hostd-1")
        forged = _token(generate_host_key(), "hostd-1")
        with pytest.raises(HostAuthError) as exc:
            verify_host_token(forged, store)
        assert exc.value.reason == "host_token_invalid"

    def test_expired_replay_refused(self, store):
        key = _enrolled(store, "hostd-1")
        old = _token(key, "hostd-1", now=time.time() - 2 * MAX_TOKEN_SECONDS)
        with pytest.raises(HostAuthError) as exc:
            verify_host_token(old, store)
        assert exc.value.reason == "host_token_expired" and exc.value.status == 401

    def test_future_token_refused(self, store):
        key = _enrolled(store, "hostd-1")
        ahead = _token(key, "hostd-1", now=time.time() + 600)
        with pytest.raises(HostAuthError) as exc:
            verify_host_token(ahead, store)
        assert exc.value.reason == "host_token_invalid"

    def test_other_site_token_refused(self, store):
        key = _enrolled(store, "hostd-1")
        with pytest.raises(HostAuthError) as exc:
            verify_host_token(_token(key, "hostd-1", venue="site-b"), store)
        assert exc.value.reason == "host_token_invalid"

    def test_lifetime_capped_at_signing(self, store):
        key = _enrolled(store, "hostd-1")
        claims = verify_host_token(_token(key, "hostd-1", ttl_s=10**6), store)
        assert claims.expires_at - claims.issued_at == MAX_TOKEN_SECONDS

    def test_tampered_payload_refused(self, store):
        key = _enrolled(store, "hostd-1")
        _enrolled(store, "hostd-2")
        good = _token(key, "hostd-1")
        prefix, rest = good[: len("umrs-host.v1.")], good[len("umrs-host.v1.") :]
        body, sig = rest.split(".")
        swapped = _token(key, "hostd-2").split(".")[2]
        with pytest.raises(HostAuthError) as exc:
            verify_host_token(f"{prefix}{swapped}.{sig}", store)
        assert exc.value.reason == "host_token_invalid"
        assert body != swapped

    @pytest.mark.parametrize("junk", ["x", "umrs-host.v1.", "umrs-host.v1.a.b.c"])
    def test_malformed_refused(self, store, junk):
        with pytest.raises(HostAuthError) as exc:
            verify_host_token(junk, store)
        assert exc.value.reason == "host_token_invalid"

    def test_presented_host(self, store):
        key = _enrolled(store, "hostd-1")
        assert presented_host({}, store) == ""
        headers = {HOST_TOKEN_HEADER: _token(key, "hostd-1")}
        assert presented_host(headers, store) == "hostd-1"
        with pytest.raises(HostAuthError) as exc:
            presented_host(headers, None)
        assert exc.value.reason == "host_unenrolled"


# ── host-bound leases ───────────────────────────────────────────────────────


class TestLeaseBinding:
    @pytest.mark.parametrize(
        "bound,presented,reason",
        [
            ("", "", ""),
            ("local", "", ""),
            ("", "hostd-1", HOST_MISMATCH),
            ("hostd-1", "", "host_token_required"),
            ("hostd-1", "hostd-1", ""),
            ("hostd-1", "hostd-2", HOST_MISMATCH),
        ],
    )
    def test_table(self, bound, presented, reason):
        lease = SimpleNamespace(host_id=bound)
        assert lease_host_refusal(lease, presented) == reason

    def test_broker_keeps_host(self):
        broker = SecretBroker()
        lease, _token_value = _lease(broker, host_id="hostd-1")
        assert lease.host_id == "hostd-1"
        assert lease.to_dict()["host_id"] == "hostd-1"


def _resolve(broker, store, secret, headers, *, client_ip="10.9.9.9", ip_check=True):
    return resolve_rail_principal(
        token=secret,
        session_id="sess-1",
        client_ip=client_ip,
        broker=broker,
        ip_check=ip_check,
        headers=headers,
        host_keys=store,
    )


class TestRail:
    def test_agent_lease_with_its_host_token(self, store):
        broker = SecretBroker()
        key = _enrolled(store, "hostd-1")
        lease, secret = _lease(broker, host_id="hostd-1")
        headers = {HOST_TOKEN_HEADER: _token(key, "hostd-1")}
        # Different client IP than lease.egress_ip: the host token replaces it.
        principal = _resolve(broker, store, secret, headers)
        assert principal.lease_id == lease.lease_id

    def test_lease_from_another_host_is_403(self, store):
        broker = SecretBroker()
        _enrolled(store, "hostd-1")
        other = _enrolled(store, "hostd-2")
        _lease_obj, secret = _lease(broker, host_id="hostd-1")
        headers = {HOST_TOKEN_HEADER: _token(other, "hostd-2")}
        with pytest.raises(RailAuthError) as exc:
            _resolve(broker, store, secret, headers)
        assert exc.value.reason == HOST_MISMATCH and exc.value.host_mismatch

    def test_agent_lease_without_host_token(self, store):
        broker = SecretBroker()
        _lease_obj, secret = _lease(broker, host_id="hostd-1")
        with pytest.raises(RailAuthError) as exc:
            _resolve(broker, store, secret, {}, ip_check=False)
        assert exc.value.reason == "host_token_required"

    def test_unenrolled_key_refused_before_lease(self, store):
        broker = SecretBroker()
        headers = {HOST_TOKEN_HEADER: _token(generate_host_key(), "hostd-7")}
        with pytest.raises(RailAuthError) as exc:
            _resolve(broker, store, None, headers)
        assert exc.value.reason == "host_unenrolled"

    def test_expired_host_token_replay(self, store):
        broker = SecretBroker()
        key = _enrolled(store, "hostd-1")
        _lease_obj, secret = _lease(broker, host_id="hostd-1")
        stale = _token(key, "hostd-1", now=time.time() - 2 * MAX_TOKEN_SECONDS)
        with pytest.raises(RailAuthError) as exc:
            _resolve(broker, store, secret, {HOST_TOKEN_HEADER: stale})
        assert exc.value.reason == "host_token_expired"

    def test_inline_lease_unchanged(self, store):
        broker = SecretBroker()
        _lease_obj, secret = _lease(broker)
        assert _resolve(broker, store, secret, {}, client_ip="10.0.0.5").owner
        with pytest.raises(RailAuthError) as exc:
            _resolve(broker, store, secret, {}, client_ip="10.9.9.9")
        assert exc.value.reason == "ip_mismatch"

    def test_inline_lease_with_agent_token_is_403(self, store):
        broker = SecretBroker()
        key = _enrolled(store, "hostd-1")
        _lease_obj, secret = _lease(broker)
        headers = {HOST_TOKEN_HEADER: _token(key, "hostd-1")}
        with pytest.raises(RailAuthError) as exc:
            _resolve(broker, store, secret, headers, client_ip="10.0.0.5")
        assert exc.value.reason == HOST_MISMATCH


def _rail_app(broker: SecretBroker, store: HostKeyStore) -> TestClient:
    from umrs_r4.api.companion_venue import rail_principal

    app = FastAPI()
    app.state.companion_venue_service = SimpleNamespace(
        broker=broker, config=SimpleNamespace(rail_ip_check=True)
    )
    app.state.host_key_store = store

    @app.get("/rail/{session_id}")
    async def _probe(session_id: str, request: Request, p=Depends(rail_principal)):
        return {"owner": p.owner}

    return TestClient(app)


class TestRailRoute:
    def test_typed_403_and_401(self, store):
        broker = SecretBroker()
        _enrolled(store, "hostd-1")
        other = _enrolled(store, "hostd-2")
        _lease_obj, secret = _lease(broker, sid="s1", host_id="hostd-1")
        client = _rail_app(broker, store)
        auth = {"authorization": f"Bearer {secret}"}
        r = client.get("/rail/s1", headers={**auth, HOST_TOKEN_HEADER: _token(other, "hostd-2")})
        assert r.status_code == 403
        assert r.headers["x-platform-error-code"] == "rail.host_mismatch"
        stranger = _token(generate_host_key(), "hostd-5")
        r = client.get("/rail/s1", headers={**auth, HOST_TOKEN_HEADER: stranger})
        assert r.status_code == 401 and r.json()["detail"] == "host_unenrolled"

    def test_rail_gw_drops_host_token(self):
        from umrs_r4.api import companion_venue

        assert HOST_TOKEN_HEADER in companion_venue._RAIL_GW_DROP_HEADERS


# ── inference proxy ─────────────────────────────────────────────────────────


class TestInferProxy:
    def test_check_lease_host_binding(self):
        broker = SecretBroker()
        lease, _secret = _lease(broker, host_id="hostd-1")
        kw = {"sid": "sess-1", "client_ip": "10.9.9.9", "source_override": None}
        assert check_lease(lease, host="hostd-1", **kw).ok
        assert check_lease(lease, host="hostd-2", **kw).reason == HOST_MISMATCH
        assert check_lease(lease, **kw).reason == "host_token_required"

    def test_inline_lease_ip_check_kept(self):
        broker = SecretBroker()
        lease, _secret = _lease(broker)
        kw = {"sid": "sess-1", "source_override": None}
        assert check_lease(lease, client_ip="10.0.0.5", **kw).ok
        assert check_lease(lease, client_ip="10.9.9.9", **kw).reason == "ip_mismatch"

    def test_statuses(self):
        from umrs.infer_proxy.app import _verbatim_error_status

        assert _verbatim_error_status("host_token_required") == 401
        assert _verbatim_error_status(HOST_MISMATCH) == 403

    def test_host_token_not_forwarded_upstream(self):
        from umrs.infer_proxy.policies import strip_credential_headers

        out = strip_credential_headers({HOST_TOKEN_HEADER: "t", "accept": "x"})
        assert out == {"accept": "x"}

    def test_log_redaction(self, store):
        from umrs.infer_proxy.app import SecretRedactingFilter

        key = _enrolled(store, "hostd-1")
        token = _token(key, "hostd-1")
        issued = issue_enrollment(store, "hostd-3")
        record = logging.LogRecord(
            "x", logging.INFO, __file__, 1, "h=%s e=%s", (token, issued.token), None
        )
        SecretRedactingFilter().filter(record)
        assert token not in record.msg and issued.token not in record.msg


# ── admin CLI and Postgres ──────────────────────────────────────────────────


_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_PG_MIGRATIONS = (
    "0085_secret_leases.sql",
    "0124_secret_leases_venue.sql",
    "*_hosts.sql",
    "*_host_identity.sql",
)


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


class _SchemaPool:
    def __init__(self, url: str, schema: str) -> None:
        self.url, self.schema = url, schema

    def getconn(self) -> Any:
        import psycopg2

        return psycopg2.connect(self.url, options=f"-c search_path={self.schema}")

    def putconn(self, conn: Any) -> None:
        conn.close()


@pytest.fixture()
def pg_pool():
    url = _db_url()
    if url is None:
        pytest.skip("no CI-shape Postgres")
    import psycopg2

    schema = f"shh1b_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(url)
    with admin.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
        # 0149 alters the venue sessions table; only its existence matters here.
        cur.execute("CREATE TABLE companion_venue_sessions (state TEXT)")
        for pattern in _PG_MIGRATIONS:
            cur.execute(next(_MIGRATIONS.glob(pattern)).read_text())
    admin.commit()
    yield _SchemaPool(url, schema)
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.commit()


class TestPostgres:
    def test_enroll_verify_reuse_unenroll(self, pg_pool):
        store = HostKeyStore(pg_pool, venue=VENUE)
        key = _enrolled(store, "hostd-1")
        assert verify_host_token(_token(key, "hostd-1"), store).host_id == "hostd-1"
        other_site = HostKeyStore(pg_pool, venue="site-b")
        with pytest.raises(HostAuthError):
            verify_host_token(_token(key, "hostd-1", venue="site-b"), other_site)
        issued = issue_enrollment(store, "hostd-2", ttl_s=60)
        body = enrollment_request(issued.token, generate_host_key())
        redeem_enrollment(store, body["token"], body["public_key"])
        with pytest.raises(EnrollmentError) as exc:
            redeem_enrollment(store, body["token"], body["public_key"])
        assert exc.value.code == "enroll.token_used"
        assert store.unenroll("hostd-1") and store.public_key("hostd-1") is None

    def test_expired_and_unknown_on_db_clock(self, pg_pool):
        store = HostKeyStore(pg_pool, venue=VENUE)
        issued = issue_enrollment(store, "hostd-1", ttl_s=1)
        conn = pg_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("UPDATE host_enrollments SET expires_at = now() - interval '1 s'")
        conn.commit()
        pg_pool.putconn(conn)
        body = enrollment_request(issued.token, generate_host_key())
        with pytest.raises(EnrollmentError) as exc:
            redeem_enrollment(store, body["token"], body["public_key"])
        assert exc.value.code == "enroll.token_expired"
        with pytest.raises(EnrollmentError) as unknown:
            redeem_enrollment(store, "umrs-enroll-nope", body["public_key"])
        assert unknown.value.code == "enroll.token_invalid"

    def test_lease_host_round_trip(self, pg_pool):
        broker = SecretBroker(db_pool=pg_pool, venue=VENUE)
        _lease_obj, secret = _lease(broker, host_id="hostd-1")
        fresh = SecretBroker(db_pool=pg_pool, venue=VENUE)
        assert fresh.lease_by_plaintext_token(secret).host_id == "hostd-1"

    def test_cli_enroll_prints_token_once(self, pg_pool, monkeypatch, capsys):
        from umrs.cli.admin import hosts as cli

        audits = _cli_harness(monkeypatch, pg_pool)
        args = SimpleNamespace(host_id="hostd-1", ttl=600, rekey=False, venue=VENUE)
        assert cli._cmd_enroll(args, None) == 0
        out = capsys.readouterr()
        token = out.out.strip()
        assert token.startswith("umrs-enroll-")
        assert token not in out.err
        assert token not in repr(audits)
        assert [a["event"] for a in audits] == ["host.enroll"]

    def test_cli_enrolled_host_and_unenroll(self, pg_pool, monkeypatch, capsys):
        from umrs.cli.admin import hosts as cli

        _cli_harness(monkeypatch, pg_pool)
        _enrolled(HostKeyStore(pg_pool, venue=VENUE), "hostd-1")
        args = SimpleNamespace(host_id="hostd-1", ttl=600, rekey=False, venue=VENUE)
        assert cli._cmd_enroll(args, None) == 1
        assert "enroll.already_enrolled" in capsys.readouterr().err
        unenroll = SimpleNamespace(host_id="hostd-1", reason="", venue=VENUE)
        assert cli._cmd_unenroll(unenroll, None) == 0
        assert cli._cmd_unenroll(unenroll, None) == 1
        assert cli._cmd_enroll(args, None) == 0


def _cli_harness(monkeypatch: Any, pool: Any) -> list[dict]:
    """The CLI against the scratch schema; audit lines captured, not sent."""
    audits: list[dict] = []
    monkeypatch.setattr("umrs.cli.admin.cascade.cli_db_pool", lambda: pool)
    monkeypatch.setattr(
        "umrs.venues.session_audit.audit_venue",
        lambda event, **kw: audits.append({"event": event, **kw}),
    )
    return audits
