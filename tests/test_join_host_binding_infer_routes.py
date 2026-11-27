"""Wave 2/3 join: SH-H1b host binding on S-D37-3's merged lease routes.

SH-H1b bound leases to hosts in ``check_lease``; S-D37-3 replaced the
proxy's ``check_lease`` calls with ``check_route_lease`` (``/s/`` vs
``/w/``). The join resolved the conflict by threading ``host`` through
``check_route_lease``. Each lane's own tests call one of the two functions
alone, so a resolution that drops ``host`` passes both suites. These tests
drive the merged proxy app with real host tokens on both routes.
"""

from __future__ import annotations

from typing import Any

from test_hosts_identity import _enrolled, _token
from test_infer_proxy import _lease
from test_infer_workstation_lease import _Harness, _ws_lease
from umrs.hosts.identity import HOST_MISMATCH, HOST_TOKEN_HEADER, HostKeyStore
from umrs.infer_proxy.leases import check_route_lease

VENUE = "site-a"
SANDBOX_CHAT = "/s/sess-1/v1/chat/completions"
TOKEN = {"authorization": "Bearer tok-session"}


def _store_with(*hosts: str) -> tuple[HostKeyStore, dict[str, Any]]:
    store = HostKeyStore(venue=VENUE)
    return store, {host: _enrolled(store, host) for host in hosts}


def _harness(monkeypatch, lease, store: HostKeyStore) -> _Harness:
    harness = _Harness(monkeypatch, lease)
    harness.app.state.host_key_store = store
    return harness


def _host_headers(key: Any, host: str) -> dict[str, str]:
    return {**TOKEN, HOST_TOKEN_HEADER: _token(key, host, venue=VENUE)}


def test_route_verdict_keeps_the_host_binding():
    bound = _lease(host_id="hostd-1", egress_ip="10.0.0.5")
    common = {"workstation": False, "sid": "sess-1", "source_override": None}
    missing = check_route_lease(bound, client_ip="10.0.0.5", **common)
    other = check_route_lease(bound, client_ip="10.0.0.5", host="hostd-2", **common)
    own = check_route_lease(bound, client_ip="192.0.2.9", host="hostd-1", **common)
    assert missing.reason == "host_token_required"
    assert other.reason == HOST_MISMATCH
    assert own.ok


def test_sandbox_route_refuses_another_hosts_token(monkeypatch):
    store, keys = _store_with("hostd-1", "hostd-2")
    lease = _lease(host_id="hostd-1", egress_ip="10.0.0.5", owner_id="alice")
    harness = _harness(monkeypatch, lease, store)
    resp = harness.call(SANDBOX_CHAT, headers=_host_headers(keys["hostd-2"], "hostd-2"))
    assert (resp.status_code, resp.json()["error"]) == (403, HOST_MISMATCH)
    assert harness.upstream == []


def test_sandbox_route_admits_the_bound_host_without_the_ip(monkeypatch):
    store, keys = _store_with("hostd-1")
    lease = _lease(host_id="hostd-1", egress_ip="10.0.0.5", owner_id="alice")
    harness = _harness(monkeypatch, lease, store)
    resp = harness.call(SANDBOX_CHAT, headers=_host_headers(keys["hostd-1"], "hostd-1"))
    assert resp.status_code == 200
    assert HOST_TOKEN_HEADER not in harness.upstream[0].headers


def test_workstation_route_refuses_a_host_token(monkeypatch):
    """A workstation lease is inline-bound: an enrolled host's token never
    upgrades it, and the edge route answers the host refusal by name."""
    store, keys = _store_with("hostd-1")
    harness = _harness(monkeypatch, _ws_lease(), store)
    resp = harness.call(headers=_host_headers(keys["hostd-1"], "hostd-1"))
    assert (resp.status_code, resp.json()["error"]) == (403, HOST_MISMATCH)
    assert harness.upstream == []


def test_workstation_route_refuses_a_forged_host_token(monkeypatch):
    store, _keys = _store_with("hostd-1")
    harness = _harness(monkeypatch, _ws_lease(), store)
    resp = harness.call(headers={**TOKEN, HOST_TOKEN_HEADER: "umrs-host.v1.forged.sig"})
    assert resp.status_code == 401
    assert resp.json()["error"].startswith("host_")
    assert harness.upstream == []
