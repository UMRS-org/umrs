"""What a session-mode ROUTER can know about the sender of a message.

Real CURVE + the production ZAP authenticator (``_SessionThreadAuthenticator``)
and ``SessionCredentialsProvider``: the ZAP reply's user-id (the session id) is
readable from every received frame as the ``User-Id`` property. The ROUTER's
routing id is NOT that identity (libzmq assigns it), so the ``client_id`` field
inside the message is the only identity the server used before 2026-10-04.
"""

from __future__ import annotations

import pytest

zmq = pytest.importorskip("zmq")

from umrs.core.session_curve_provider import SessionCredentialsProvider  # noqa: E402
from umrs.core.zmq_security import ZMQSecurityConfig  # noqa: E402


def _keypair() -> tuple[str, str]:
    public, secret = zmq.curve_keypair()
    return public.decode("ascii"), secret.decode("ascii")


@pytest.fixture()
def session_router():
    ctx = zmq.Context()
    provider = SessionCredentialsProvider()
    server_pub, server_sec = _keypair()
    config = ZMQSecurityConfig(
        auth_type="session",
        server_public_key=server_pub,
        server_secret_key=server_sec,
        credentials_provider=provider,
    )
    router = ctx.socket(zmq.ROUTER)
    authenticator = config.apply_to_socket(router, server=True, context=ctx)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    clients = []

    def connect(user_id: str):
        pub, sec = _keypair()
        entry = provider.register(pub, user_id, ttl_seconds=600)
        dealer = ctx.socket(zmq.DEALER)
        dealer.curve_serverkey = server_pub.encode()
        dealer.curve_publickey = pub.encode()
        dealer.curve_secretkey = sec.encode()
        dealer.connect(f"tcp://127.0.0.1:{port}")
        clients.append(dealer)
        return dealer, entry

    yield router, connect, provider
    for dealer in clients:
        dealer.close(linger=0)
    router.close(linger=0)
    if authenticator is not None:
        authenticator.stop()
    from umrs.core import zmq_security

    zmq_security._CONTEXT_AUTHENTICATORS.pop(id(ctx), None)
    ctx.term()


def _recv(router):
    assert router.poll(5000), "no message reached the ROUTER"
    return router.recv_multipart(copy=False)


def test_user_id_property_is_the_senders_session(session_router):
    router, connect, provider = session_router
    alice, alice_entry = connect("alice")
    alice.send_string("JOB_LIST:bob:{}")
    frames = _recv(router)
    routing_id, message = frames[0], frames[-1]
    assert message.bytes.decode() == "JOB_LIST:bob:{}"
    # The claim inside the message says bob; the transport says alice.
    session_id = message.get("User-Id")
    assert session_id == alice_entry.session_id
    assert provider.lookup_session(session_id).user_id == "alice"
    # The routing id carries no identity: libzmq generated it.
    assert routing_id.bytes != alice_entry.session_id.encode()


def test_each_connection_carries_its_own_session(session_router):
    router, connect, provider = session_router
    alice, alice_entry = connect("alice")
    bob, bob_entry = connect("bob")
    alice.send_string("STATUS:alice:all")
    bob.send_string("STATUS:bob:all")
    seen = {}
    for _ in range(2):
        message = _recv(router)[-1]
        seen[message.bytes.decode().split(":")[1]] = message.get("User-Id")
    assert seen == {"alice": alice_entry.session_id, "bob": bob_entry.session_id}


# ── the control loop binds client_id to the session (2026-10-04) ──────────


def _free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture()
def session_control_server(monkeypatch):
    from umrs.core import session_curve_provider
    from umrs.core.MessageQueue import MessageQueue

    provider = SessionCredentialsProvider()
    server_pub, server_sec = _keypair()
    monkeypatch.setenv("UMRS_ZMQ_AUTH_TYPE", "session")
    monkeypatch.setenv("UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY", server_pub)
    monkeypatch.setenv("UMRS_ZMQ_CURVE_SERVER_SECRET_KEY", server_sec)
    monkeypatch.delenv("UMRS_ZMQ_SESSION_IDENTITY", raising=False)
    session_curve_provider.set_shared_session_provider(provider)
    port = _free_port()
    mq = MessageQueue(role="server", queue_type="control", port=port, timeout_ms=5000)
    ctx = zmq.Context()
    clients = []

    def connect(user_id: str, *, roles=("user",), kind=None, to_port=None):
        from umrs.core.session_curve_provider import SessionGrants

        pub, sec = _keypair()
        provider.register(pub, user_id, ttl_seconds=600, grants=SessionGrants(roles=roles))
        sock = ctx.socket(zmq.DEALER if kind is None else kind)
        sock.curve_serverkey = server_pub.encode()
        sock.curve_publickey = pub.encode()
        sock.curve_secretkey = sec.encode()
        sock.connect(f"tcp://127.0.0.1:{to_port or port}")
        clients.append(sock)
        return sock

    yield mq, connect
    for dealer in clients:
        dealer.close(linger=0)
    ctx.term()
    mq.close()
    session_curve_provider.set_shared_session_provider(None)


def test_a_claimed_client_id_is_rebound_to_the_sessions_user(session_control_server):
    mq, connect = session_control_server
    alice = connect("alice")
    alice.send_string("JOB_LIST:bob:{}")
    cmd, (client_id, payload, sender) = mq.server_control_queue_loop()
    assert (cmd, client_id, payload) == ("JOB_LIST", "alice", "{}")
    assert mq.verified_session(sender).user_id == "alice"


def test_a_matching_client_id_passes_unchanged(session_control_server):
    mq, connect = session_control_server
    alice = connect("alice")
    alice.send_string('JOB_CANCEL:alice:{"job_id": "j1"}')
    cmd, (client_id, payload, _sender) = mq.server_control_queue_loop()
    assert (cmd, client_id) == ("JOB_CANCEL", "alice")
    assert payload == '{"job_id": "j1"}'


def test_an_admin_session_may_act_as_another_client_id(session_control_server):
    mq, connect = session_control_server
    root = connect("maint", roles=("admin",))
    root.send_string("JOB_LIST:bob:{}")
    cmd, (client_id, _payload, sender) = mq.server_control_queue_loop()
    assert (cmd, client_id) == ("JOB_LIST", "bob")
    assert mq.verified_session(sender).grants.is_admin


# ── the binding rules, against a fake provider ──────────────────────────────


class _Grants:
    def __init__(self, admin=False):
        self.is_admin = admin


class _Entry:
    def __init__(self, user_id, admin=False, *, session_id=None, expires_in=600.0):
        import time

        self.user_id = user_id
        self.grants = _Grants(admin)
        self.session_id = session_id or f"s-{user_id}"
        self.expires_at = time.time() + expires_in


class _Provider:
    def __init__(self, entries):
        self.entries = dict(entries)

    def lookup_session(self, session_id):
        return self.entries.get(session_id)


def _bare_mq(provider):
    from umrs.core.MessageQueue import MessageQueue

    mq = MessageQueue.__new__(MessageQueue)
    import collections

    mq._session_entries = collections.OrderedDict()
    mq._sender_entries = collections.OrderedDict()
    mq._identity_provider_cache = (provider,)
    mq.sent = []
    mq.send_string = lambda text, sender: mq.sent.append((text, sender))
    return mq


def test_unknown_session_is_refused_with_auth_error():
    mq = _bare_mq(_Provider({}))
    assert mq._bind_session_identity("STATUS", ["STATUS", "bob", "all"], b"r", "s-x") is None
    assert mq.sent == [("AUTH_ERROR:no authenticated ZMQ session", b"r")]


def test_a_connection_without_a_session_id_is_refused():
    mq = _bare_mq(_Provider({}))
    assert mq._bind_session_identity("FILE", ["FILE", "bob", "1"], b"r", None) is None


def test_a_session_expiring_mid_connection_keeps_its_identity():
    provider = _Provider({"s-a": _Entry("alice", expires_in=-1.0)})
    mq = _bare_mq(provider)
    assert mq._bind_session_identity("STATUS", ["STATUS", "alice", "all"], b"r", "s-a")
    provider.entries.clear()  # swept as expired
    bound = mq._bind_session_identity("STATUS", ["STATUS", "bob", "all"], b"r", "s-a")
    assert bound == ["STATUS", "alice", "all"]


def test_a_session_revoked_before_expiry_ends_its_open_connection():
    provider = _Provider({"s-a": _Entry("alice")})
    mq = _bare_mq(provider)
    assert mq._bind_session_identity("STATUS", ["STATUS", "alice", "all"], b"r", "s-a")
    provider.entries.clear()  # revoked / disabled / evicted by the cap
    assert mq._bind_session_identity("STATUS", ["STATUS", "alice", "all"], b"r", "s-a") is None
    assert mq.sent[-1] == ("AUTH_ERROR:no authenticated ZMQ session", b"r")


def test_json_commands_get_a_json_refusal():
    import json

    mq = _bare_mq(_Provider({}))
    assert mq._bind_session_identity("JOB_LIST", ["JOB_LIST", "bob", "{}"], b"r", "s-x") is None
    reply = json.loads(mq.sent[-1][0])
    assert reply == {"status": "error", "message": "no authenticated ZMQ session"}


def test_an_admin_token_left_under_carol_never_makes_carols_session_admin():
    """An admin acting as carol presented an admin token: the context is kept
    per session, so carol's own session still reads USER (review finding 1)."""
    from umrs.shared.auth import AuthContext, UserRole

    admin = _Entry("maint", admin=True, session_id="s-admin")
    carol = _Entry("carol", session_id="s-carol")
    mq = _bare_mq(_Provider({"s-admin": admin, "s-carol": carol}))
    mq._auth_sessions = {}
    assert mq._bind_session_identity(
        "CREATE_DATA_QUEUE", ["CREATE_DATA_QUEUE", "carol"], b"a", "s-admin"
    )
    mq._store_auth("carol", AuthContext(user_id="carol", username="carol", role=UserRole.ADMIN))
    assert mq.get_auth_context("carol").is_admin  # the admin's own session
    assert mq._bind_session_identity("JOB_LIST", ["JOB_LIST", "carol", "{}"], b"c", "s-carol")
    assert not mq.get_auth_context("carol").is_admin


def test_session_grants_make_an_admin_without_a_token():
    mq = _bare_mq(_Provider({"s-admin": _Entry("maint", admin=True, session_id="s-admin")}))
    mq._auth_sessions = {}
    assert mq._bind_session_identity("JOB_LIST", ["JOB_LIST", "maint", "{}"], b"a", "s-admin")
    assert mq.get_auth_context("maint").is_admin


def test_a_drained_message_keeps_its_own_auth_frame():
    import collections

    mq = _bare_mq(None)
    mq._lookahead = collections.deque([(b"r2", "ENROLL_USER:bob:1", "s-b", {"admin_token": "own"})])
    mq._last_control_auth = {"admin_token": "the-drainers"}
    sender, msg, session_id = mq._next_control_message()
    assert (sender, msg, session_id) == (b"r2", "ENROLL_USER:bob:1", "s-b")
    assert mq._last_control_auth == {"admin_token": "own"}


def test_identity_free_and_unknown_commands_pass_untouched():
    mq = _bare_mq(_Provider({}))
    for fields in (["HAND_CHECK"], ["STATUS_ALL"], ["NOT_A_CMD", "bob"]):
        assert mq._bind_session_identity(fields[0], fields, b"r", None) == fields
    assert mq.sent == []


def test_no_binding_outside_session_mode():
    mq = _bare_mq(None)
    fields = ["JOB_LIST", "bob", "{}"]
    assert mq._bind_session_identity("JOB_LIST", fields, b"r", None) == fields


def test_colon_user_ids_compare_in_wire_form():
    mq = _bare_mq(_Provider({"s": _Entry("org:alice")}))
    fields = ["JOB_LIST", "org_alice", "{}"]
    assert mq._bind_session_identity("JOB_LIST", fields, b"r", "s") == fields


# ── the data plane takes frames only from its owner's session ──────────────


def test_a_data_queue_drops_frames_from_another_users_session(session_control_server):
    from umrs.core.MessageQueue import MessageQueue

    mq, connect = session_control_server
    port = _free_port()
    data = MessageQueue(
        role="server", queue_type="data", port=port, timeout_ms=3000, context=mq.context
    )
    data.owner_client_id = "dave"
    try:
        mallory = connect("mallory", kind=zmq.PUSH, to_port=port)
        dave = connect("dave", kind=zmq.PUSH, to_port=port)
        mallory.send_string("s3://dave-bucket/evil.txt")
        dave.send_string("s3://dave-bucket/real.txt")
        _client, message = data.recv_string()
        assert message == "s3://dave-bucket/real.txt"
    finally:
        data.close()
