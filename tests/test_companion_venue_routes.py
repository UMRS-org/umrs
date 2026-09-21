"""Frontend-venue routes: owner start/status/stop + the lease-token worker rail.

Fakes: in-memory broker + registry, FakePodman-driven provisioner, a
snapshot pool that answers exactly the SQL shapes the shared row helpers
issue, and the sqlite T4 approvals store. No live podman, no Postgres.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.companion_venue import router
from umrs.companion_trust.chat_turn_approvals import ChatTurnApprovalsStore
from umrs.jobs.secret_broker import SecretBroker
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.frontend_podman import FrontendPodmanProvisioner
from umrs.venues.frontend_service import FrontendVenueService
from umrs.venues.venue_sessions import VenueSessionStore

SRC_ID = "11111111-1111-4111-8111-111111111111"
READY_ENV = {
    "UMRS_VENUE_ENABLED": "1",
    "UMRS_VENUE_INFER_HOST": "umrs-infer",
    "UMRS_VENUE_API_UPSTREAM": "169.254.1.2:8000",
    "UMRS_VENUE_WORKER_IMAGE": "localhost/r4-companion@sha256:abc",
    "UMRS_VENUE_NGINX_IMAGE": "docker.io/library/nginx@sha256:def",
    "UMRS_VENUE_SQUID_IMAGE": "docker.io/ubuntu/squid@sha256:123",
    "UMRS_VENUE_RAIL_IP_CHECK": "0",
}


# ── snapshot pool fake (only the helper SQL shapes) ─────────────────────────


class _Cursor:
    def __init__(self, table: dict, recorded: list):
        self._table = table
        self._recorded = recorded
        self._row = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        sql_l = " ".join(sql.split()).lower()
        self._recorded.append(sql_l)
        params = params or ()
        if sql_l.startswith("select 1 from companion_session_snapshots"):
            sid, uid = params
            self._row = (1,) if (uid, sid) in self._table else None
        elif "select s.session_id, s.name, s.payload" in sql_l:
            uid, sid = params
            row = self._table.get((uid, sid))
            self._row = (
                (sid, row["name"], row["payload"], row["marker"], row["venue"], row["state"], None)
                if row
                else None
            )
        elif "select payload from companion_session_snapshots" in sql_l:
            uid, sid = params
            row = self._table.get((uid, sid))
            self._row = (row["payload"],) if row else None
        elif sql_l.startswith("update companion_session_snapshots set venue"):
            venue, state, uid, sid = params
            row = self._table.get((uid, sid))
            if row:
                row["venue"], row["state"] = venue, state
            self.rowcount = 1 if row else 0
        elif sql_l.startswith("update companion_session_snapshots set payload"):
            payload, uid, sid = params
            row = self._table.get((uid, sid))
            if row:
                row["payload"] = payload
            self.rowcount = 1 if row else 0
        elif sql_l.startswith("insert into companion_session_snapshots"):
            uid, sid, name, payload, marker, venue, state = params
            existing = self._table.get((uid, sid))
            if existing is None or existing["marker"] <= marker:
                self._table[(uid, sid)] = {
                    "name": name,
                    "payload": payload,
                    "marker": marker,
                    "venue": venue or (existing or {}).get("venue", "local"),
                    "state": state or (existing or {}).get("state", ""),
                }
                self.rowcount = 1
            else:
                self.rowcount = 0
        else:
            raise AssertionError(f"unexpected SQL in venue route test: {sql_l[:80]}")

    def fetchone(self):
        return self._row


class _Conn:
    def __init__(self, table, recorded):
        self._table = table
        self._recorded = recorded
        self.commits = 0

    def cursor(self):
        return _Cursor(self._table, self._recorded)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class _Pool:
    def __init__(self):
        self.table: dict = {}
        self.recorded: list = []
        self._conn = _Conn(self.table, self.recorded)

    def getconn(self):
        return self._conn

    def putconn(self, _):
        pass

    def seed(self, uid: str, sid: str, *, queue=None, messages=None):
        payload = {"id": sid, "messages": messages or [], "metadata": {"chat_queue": queue or []}}
        self.table[(uid, sid)] = {
            "name": "S",
            "payload": json.dumps(payload),
            "marker": "2026-09-21T00:00:00+00:00",
            "venue": "local",
            "state": "",
        }


class FakePodman:
    def __init__(self, fail_at: str | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.fail_at = fail_at

    def __call__(self, *args, input_text=None, **_):
        del input_text  # token never inspected here; the rail fixture captures it
        self.calls.append(tuple(args))
        if self.fail_at and any(self.fail_at in a for a in args):
            return subprocess.CompletedProcess(args, 1, "", f"boom {self.fail_at}")
        if args[0] == "inspect":
            body = [{"NetworkSettings": {"Networks": {"egress-net": {"IPAddress": "172.20.8.2"}}}}]
            return subprocess.CompletedProcess(args, 0, json.dumps(body), "")
        return subprocess.CompletedProcess(args, 0, "", "")


class FakeSealed:
    def __init__(self, rows: dict[tuple[str, str], object]):
        self.rows = rows

    def get_meta(self, owner, name):
        return self.rows.get((owner, name))


def _meta(name: str, *, tombstoned=None):
    return SimpleNamespace(id=SRC_ID, name=name, tombstoned_at=tombstoned)


@pytest.fixture
def harness(monkeypatch):
    approvals = ChatTurnApprovalsStore()
    pool = _Pool()
    broker = SecretBroker()
    podman = FakePodman()
    provisioner = FrontendPodmanProvisioner(
        broker, podman=podman, session_root=Path(tempfile.mkdtemp(prefix="venue-route-"))
    )
    service = FrontendVenueService(
        config=FrontendVenueConfig.from_env(READY_ENV),
        broker=broker,
        store=VenueSessionStore(),
        provisioner=provisioner,
        pool=pool,
        sealed_store=FakeSealed({("alice", "openai-main"): _meta("openai-main")}),
    )
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = pool
    app.state.companion_venue_service = service
    app.state.companion_venue_approvals_factory = lambda _conn: approvals
    client = TestClient(app)
    return SimpleNamespace(
        client=client, pool=pool, broker=broker, podman=podman, service=service, approvals=approvals
    )


H = {"x-user-id": "alice"}


def _start(h, sid="sess-1", **over):
    body = {"source_refs": ["openai-main"], **over}
    return h.client.post(f"/umrs/companion/sessions/{sid}/venue/start", json=body, headers=H)


class TestOwnerRoutes:
    def test_start_requires_identity(self, harness):
        r = harness.client.post(
            "/umrs/companion/sessions/s/venue/start", json={"source_refs": ["x"]}
        )
        assert r.status_code == 401

    def test_start_404_when_not_mirrored(self, harness):
        r = _start(harness)
        assert r.status_code == 404
        assert r.headers["x-platform-error-code"] == "companionvenue.not_found"

    def test_start_409_source_not_pushed(self, harness):
        harness.pool.seed("alice", "sess-1")
        r = _start(harness, source_refs=["openai-main", "missing-one"])
        assert r.status_code == 409
        assert r.headers["x-platform-error-code"] == "companionvenue.source_not_pushed"
        assert "missing-one" in r.json()["detail"]

    def test_start_400_bad_policy(self, harness):
        harness.pool.seed("alice", "sess-1")
        r = _start(harness, web_policy="wide-open")
        assert r.status_code == 400

    def test_start_503_when_not_configured(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.service.config = FrontendVenueConfig.from_env({})
        r = _start(harness)
        assert r.status_code == 503
        assert "UMRS_VENUE_ENABLED" in r.json()["detail"]

    def test_start_provisions_and_projects(self, harness):
        harness.pool.seed("alice", "sess-1")
        r = _start(harness)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["state"] == "running"
        assert body["egress_ip"] == "172.20.8.2"
        assert body["source_names"] == ["openai-main"]
        assert body["lease"]["is_valid"] is True
        # registry projection: the snapshot row now says frontend/ongoing
        row = harness.pool.table[("alice", "sess-1")]
        assert row["venue"] == "frontend" and row["state"] == "ongoing"
        # worker argv carries the loop command + rail env, never the token
        worker = next(c for c in harness.podman.calls if c[0] == "run" and "worker-sess-1" in c)
        assert worker[-2:] == ("venue", "worker")
        assert "R4_VENUE_RAIL_URL=http://egress:8081" in worker
        assert "R4_VENUE_SESSION_ID=sess-1" in worker

    def test_second_start_is_409_already_running(self, harness):
        harness.pool.seed("alice", "sess-1")
        assert _start(harness).status_code == 200
        r = _start(harness)
        assert r.status_code == 409
        assert r.headers["x-platform-error-code"] == "companionvenue.already_running"

    def test_provision_failure_is_502_and_row_failed(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.podman.fail_at = "egress-sess-1"
        r = _start(harness)
        assert r.status_code == 502
        assert r.headers["x-platform-error-code"] == "companionvenue.provision_failed"
        status = harness.client.get("/umrs/companion/sessions/sess-1/venue", headers=H).json()
        assert status["state"] == "failed" and "egress" in status["reason"]

    def test_status_owner_scoped(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        other = harness.client.get(
            "/umrs/companion/sessions/sess-1/venue", headers={"x-user-id": "mallory"}
        )
        assert other.status_code == 404
        mine = harness.client.get("/umrs/companion/sessions/sess-1/venue", headers=H)
        assert mine.status_code == 200 and mine.json()["state"] == "running"

    def test_stop_tears_down_and_revokes(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        r = harness.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        assert r.status_code == 200
        assert r.json()["state"] == "stopped"
        flattened = [" ".join(c) for c in harness.podman.calls]
        assert any("rm -f worker-sess-1" in c for c in flattened)
        assert all(not lease.is_valid for lease in harness.broker.list_leases(scope="inference"))
        assert harness.pool.table[("alice", "sess-1")]["state"] == "stopped"
        # idempotent
        again = harness.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        assert again.status_code == 200 and again.json()["state"] == "stopped"


class TestRail:
    @pytest.fixture
    def rail(self, harness):
        captured = {}

        class _Podman(FakePodman):
            def __call__(self, *args, input_text=None, **kw):
                if args[:2] == ("secret", "create"):
                    captured["token"] = input_text
                return super().__call__(*args, input_text=input_text, **kw)

        podman = _Podman()
        harness.podman = podman
        harness.service.provisioner = FrontendPodmanProvisioner(
            harness.broker, podman=podman, session_root=Path(tempfile.mkdtemp(prefix="venue-rail-"))
        )
        harness.pool.seed(
            "alice",
            "sess-1",
            queue=[{"id": "q-1", "content": "hello remote", "client_id": "tui", "added_at": "t"}],
        )
        assert _start(harness).status_code == 200
        harness.token = captured["token"]
        harness.rail_headers = {"authorization": f"Bearer {harness.token}"}
        return harness

    def test_rail_rejects_without_token(self, rail):
        r = rail.client.get("/umrs/companion/sessions/sess-1/rail/snapshot")
        assert r.status_code == 401
        assert r.json()["detail"] == "token_invalid"

    def test_rail_rejects_user_identity_header_alone(self, rail):
        r = rail.client.get("/umrs/companion/sessions/sess-1/rail/snapshot", headers=H)
        assert r.status_code == 401

    def test_rail_rejects_other_session(self, rail):
        r = rail.client.get(
            "/umrs/companion/sessions/other/rail/snapshot", headers=rail.rail_headers
        )
        assert r.status_code == 401 and r.json()["detail"] == "sid_mismatch"

    def test_rail_snapshot_roundtrip(self, rail):
        r = rail.client.get(
            "/umrs/companion/sessions/sess-1/rail/snapshot", headers=rail.rail_headers
        )
        assert r.status_code == 200
        assert r.json()["session_id"] == "sess-1"
        put = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/snapshot",
            headers=rail.rail_headers,
            json={
                "name": "S",
                "updated_at": "2026-09-21T01:00:00+00:00",
                "payload": json.dumps(
                    {"id": "sess-1", "messages": [{"role": "user", "content": "x"}]}
                ),
                "venue": "local",  # ignored: rail pins frontend
                "state": "ongoing",
            },
        )
        assert put.status_code == 200 and put.json()["applied"] is True
        row = rail.pool.table[("alice", "sess-1")]
        assert row["venue"] == "frontend" and row["state"] == "ongoing"

    def test_rail_queue_pop(self, rail):
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/queue/pop", headers=rail.rail_headers
        )
        assert r.status_code == 200
        body = r.json()
        assert body["applied"] is True
        assert body["front"]["content"] == "hello remote"
        assert body["queue"] == []
        empty = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/queue/pop", headers=rail.rail_headers
        )
        assert empty.json()["front"] is None

    def test_rail_heartbeat_renews_and_records_runtime(self, rail):
        lease_before = next(iter(rail.broker.list_leases(scope="inference")))
        expires_before = lease_before.expires_at
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/heartbeat",
            headers=rail.rail_headers,
            json={"phase": "running", "turn_seq": 2, "partial": "thinking…", "extend_seconds": 300},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["renewed"] is True and body["stop_requested"] is False
        assert rail.broker.get_lease(lease_before.lease_id).expires_at > expires_before
        status = rail.client.get("/umrs/companion/sessions/sess-1/venue", headers=H).json()
        assert status["runtime"]["phase"] == "running"
        assert status["runtime"]["turn_seq"] == 2
        assert status["runtime"]["partial"] == "thinking…"

    def test_rail_heartbeat_after_stop_requests_exit(self, rail):
        rail.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/heartbeat", headers=rail.rail_headers, json={}
        )
        # lease revoked by stop → the rail refuses the principal outright
        assert r.status_code == 401 and r.json()["detail"] == "lease_revoked"

    def test_rail_approvals_cycle_with_owner_decision(self, rail):
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/approvals",
            headers=rail.rail_headers,
            json={"turn_id": "t3", "action": {"tool": "shell_exec", "description": "ls"}},
        )
        assert r.status_code == 200
        approval = r.json()["approval"]
        assert approval["status"] == "pending" and approval["user_id"] == "alice"
        aid = approval["approval_id"]
        pending = rail.client.get(
            f"/umrs/companion/sessions/sess-1/rail/approvals/{aid}", headers=rail.rail_headers
        )
        assert pending.status_code == 200 and pending.json()["approval"]["status"] == "pending"
        rail.approvals.decide(aid, "once", decided_by="alice")
        decided = rail.client.get(
            f"/umrs/companion/sessions/sess-1/rail/approvals/{aid}", headers=rail.rail_headers
        ).json()["approval"]
        assert decided["status"] == "approved" and decided["decision"] == "once"
        cross = rail.client.get(
            f"/umrs/companion/sessions/other/rail/approvals/{aid}", headers=rail.rail_headers
        )
        assert cross.status_code == 401  # sid_mismatch before any lookup
