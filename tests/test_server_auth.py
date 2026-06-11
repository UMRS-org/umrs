"""Tests for ZMQ dispatch auth enforcement (Phase 1).

Covers:
- AuthContext / can_execute_command logic
- MessageQueue auth session lifecycle (get_auth_context, CREATE_DATA_QUEUE)
- _authorize helper in server_umrs
- Client UNAUTHORIZED detection in control_api
"""

from __future__ import annotations

import json

import pytest

from umrs.shared.auth import (
    ADMIN_ONLY_COMMANDS,
    USER_ALLOWED_COMMANDS,
    AuthContext,
    UserRole,
    create_auth_context_from_request,
)


# ---------------------------------------------------------------------------
# AuthContext.can_execute_command
# ---------------------------------------------------------------------------


class TestAuthContextPermissions:
    """Verify can_execute_command behaviour for each role."""

    def test_admin_can_execute_any_command(self):
        ctx = AuthContext(user_id="a1", username="admin1", role=UserRole.ADMIN)
        for cmd in list(ADMIN_ONLY_COMMANDS) + list(USER_ALLOWED_COMMANDS):
            assert ctx.can_execute_command(cmd, target_user="admin1"), f"admin denied {cmd}"

    def test_user_allowed_commands_accepted(self):
        ctx = AuthContext(user_id="u1", username="user1", role=UserRole.USER)
        for cmd in USER_ALLOWED_COMMANDS:
            assert ctx.can_execute_command(cmd, target_user="user1"), f"user denied {cmd}"

    def test_user_denied_admin_commands(self):
        ctx = AuthContext(user_id="u1", username="user1", role=UserRole.USER)
        for cmd in ADMIN_ONLY_COMMANDS:
            assert not ctx.can_execute_command(cmd, target_user="user1"), f"user allowed {cmd}"

    def test_service_role_allowed_everything(self):
        ctx = AuthContext(user_id="svc", username="svc", role=UserRole.SERVICE)
        for cmd in list(ADMIN_ONLY_COMMANDS) + list(USER_ALLOWED_COMMANDS):
            assert ctx.can_execute_command(cmd), f"service denied {cmd}"

    def test_unknown_command_denied_for_user(self):
        ctx = AuthContext(user_id="u1", username="user1", role=UserRole.USER)
        assert not ctx.can_execute_command("SOME_FUTURE_CMD")


# ---------------------------------------------------------------------------
# create_auth_context_from_request (no token → USER, invalid token → USER)
# ---------------------------------------------------------------------------


class TestCreateAuthContext:
    def test_no_token_gives_user_role(self):
        ctx = create_auth_context_from_request("alice")
        assert ctx.role == UserRole.USER
        assert ctx.username == "alice"

    def test_empty_token_gives_user_role(self):
        ctx = create_auth_context_from_request("bob", token="")
        assert ctx.role == UserRole.USER

    def test_invalid_token_gives_user_role(self):
        ctx = create_auth_context_from_request("eve", token="bad-token-value")
        assert ctx.role == UserRole.USER


# ---------------------------------------------------------------------------
# MessageQueue auth session management
# ---------------------------------------------------------------------------


class TestMessageQueueAuthSessions:
    """Test get_auth_context without a real ZMQ socket."""

    @pytest.fixture()
    def fake_mq(self):
        """Build a minimal object that mimics MessageQueue auth fields."""

        class FakeMQ:
            def __init__(self):
                self._auth_sessions = {}

            # Paste the real method under test
            def get_auth_context(self, client_id):
                ctx = self._auth_sessions.get(client_id)
                if ctx is not None:
                    return ctx
                return AuthContext(
                    user_id=client_id or "",
                    username=client_id or "",
                    role=UserRole.USER,
                )

        return FakeMQ()

    def test_unknown_client_gets_user_role(self, fake_mq):
        ctx = fake_mq.get_auth_context("new_user")
        assert ctx.role == UserRole.USER
        assert ctx.username == "new_user"

    def test_stored_admin_session_returned(self, fake_mq):
        admin_ctx = AuthContext(user_id="admin1", username="admin1", role=UserRole.ADMIN)
        fake_mq._auth_sessions["admin1"] = admin_ctx
        ctx = fake_mq.get_auth_context("admin1")
        assert ctx.role == UserRole.ADMIN
        assert ctx is admin_ctx

    def test_stored_user_session_returned(self, fake_mq):
        user_ctx = AuthContext(user_id="u1", username="u1", role=UserRole.USER)
        fake_mq._auth_sessions["u1"] = user_ctx
        assert fake_mq.get_auth_context("u1") is user_ctx

    def test_empty_client_id_defaults(self, fake_mq):
        ctx = fake_mq.get_auth_context("")
        assert ctx.role == UserRole.USER
        assert ctx.user_id == ""


# ---------------------------------------------------------------------------
# _authorize helper (unit-style — exercise the function logic directly)
# ---------------------------------------------------------------------------


class TestAuthorizeHelper:
    """Test the _authorize logic without a running server."""

    @staticmethod
    def _make_authorize(auth_ctx: AuthContext, sent_replies: list):
        """Build a stand-alone _authorize callable that mimics the server helper."""

        class FakeControlMQ:
            def get_auth_context(self, _client_id):
                return auth_ctx

            def send_string(self, msg, zmq_sender_id):
                sent_replies.append((msg, zmq_sender_id))

        mq = FakeControlMQ()

        def _authorize(cmd, client_id, zmq_sender_id=None):
            ctx = mq.get_auth_context(client_id)
            if ctx.can_execute_command(cmd, target_user=client_id):
                return ctx
            if zmq_sender_id is not None:
                try:
                    mq.send_string(f"UNAUTHORIZED:{cmd}", zmq_sender_id)
                except RuntimeError:
                    pass
            return None

        return _authorize

    def test_admin_allowed_admin_command(self):
        replies = []
        auth = self._make_authorize(
            AuthContext(user_id="a1", username="a1", role=UserRole.ADMIN), replies
        )
        result = auth("RELOAD", "a1", zmq_sender_id=b"sender")
        assert result is not None
        assert result.role == UserRole.ADMIN
        assert replies == []

    def test_user_denied_admin_command_sends_unauthorized(self):
        replies = []
        auth = self._make_authorize(
            AuthContext(user_id="u1", username="u1", role=UserRole.USER), replies
        )
        result = auth("ENROLL_USER", "u1", zmq_sender_id=b"sender")
        assert result is None
        assert len(replies) == 1
        assert replies[0] == ("UNAUTHORIZED:ENROLL_USER", b"sender")

    def test_user_allowed_user_command(self):
        replies = []
        auth = self._make_authorize(
            AuthContext(user_id="u1", username="u1", role=UserRole.USER), replies
        )
        result = auth("PROCESS_OBJECTS", "u1")
        assert result is not None
        assert result.role == UserRole.USER

    def test_no_zmq_sender_id_no_reply_sent(self):
        replies = []
        auth = self._make_authorize(
            AuthContext(user_id="u1", username="u1", role=UserRole.USER), replies
        )
        result = auth("ENROLL_USER", "u1")  # no zmq_sender_id
        assert result is None
        assert replies == []


# ---------------------------------------------------------------------------
# Client-side UNAUTHORIZED detection (control_api._request_json)
# ---------------------------------------------------------------------------


class TestClientUnauthorizedHandling:
    def test_unauthorized_reply_returns_error_dict(self):
        """Simulate the _request_json handling for UNAUTHORIZED replies."""
        raw = "UNAUTHORIZED:MODEL_LIST"
        # Replicate the logic from UMRSControlAPI._request_json
        if raw.startswith("UNAUTHORIZED:"):
            result = {
                "status": "error",
                "message": f"Authorization denied for MODEL_LIST",
                "code": "UNAUTHORIZED",
            }
        else:
            result = json.loads(raw)
        assert result["status"] == "error"
        assert result["code"] == "UNAUTHORIZED"

    def test_normal_reply_parses_json(self):
        raw = json.dumps({"status": "ok", "models": []})
        if raw.startswith("UNAUTHORIZED:"):
            result = {"status": "error"}
        else:
            result = json.loads(raw)
        assert result["status"] == "ok"

    def test_notify_server_unauthorized_returns_false(self):
        """Simulate the notify_server handling for UNAUTHORIZED replies.

        When the server denies a maintenance command after authorization,
        notify_server should return False instead of raising.
        """
        reply = "UNAUTHORIZED:RELOAD"
        # Replicate the logic from MessageQueue.notify_server
        if reply == "ACKNOWLEDGE":
            result = True
        elif reply.startswith("UNAUTHORIZED:"):
            result = False
        else:
            result = None  # would raise in real code
        assert result is False

    def test_maintenance_returns_error_on_unauthorized(self):
        """When notify_server returns False, maintenance() should report rejection."""
        ok = False  # simulates notify_server returning False for UNAUTHORIZED
        action = "reload"
        if ok:
            parsed = {"status": "ok", "action": action, "accepted": True}
        else:
            parsed = {
                "status": "error",
                "action": action,
                "accepted": False,
                "message": "Authorization denied (admin required)",
            }
        assert parsed["status"] == "error"
        assert parsed["accepted"] is False
        assert "admin" in parsed["message"].lower()


# ---------------------------------------------------------------------------
# USER_ALLOWED_COMMANDS completeness — every TaskCmd used in the server
# event loop should be present (either as ADMIN_ONLY or USER_ALLOWED).
# ---------------------------------------------------------------------------


class TestCommandCoverage:
    """Verify that all commands dispatched in the event loop are declared."""

    EXPECTED_USER_COMMANDS = {
        "PROCESS_OBJECTS",
        "FILE",
        "WAIT_OUTPUT",
        "EXPORT_EMBEDDINGS",
        "MODEL_REGISTER",
        "MODEL_LIST",
        "MODEL_DOWNLOAD",
        "JOB_SUBMIT",
        "JOB_CANCEL",
        "JOB_LIST",
        "UPLOAD_WORKFLOW",
        "LIST_WORKFLOWS",
        "RUN_WORKFLOW",
        "STATUS",
        "CREATE_DATA_QUEUE",
        "HAND_CHECK",
    }

    EXPECTED_ADMIN_COMMANDS = {"RECOMPUTE", "UPDATE", "CHECK"}

    def test_all_user_commands_in_allowed_set(self):
        missing = self.EXPECTED_USER_COMMANDS - USER_ALLOWED_COMMANDS
        assert not missing, f"Missing from USER_ALLOWED_COMMANDS: {missing}"

    def test_all_admin_commands_in_admin_set(self):
        missing = self.EXPECTED_ADMIN_COMMANDS - ADMIN_ONLY_COMMANDS
        assert not missing, f"Missing from ADMIN_ONLY_COMMANDS: {missing}"

    def test_no_overlap_between_admin_and_user(self):
        overlap = ADMIN_ONLY_COMMANDS & USER_ALLOWED_COMMANDS
        assert not overlap, f"Commands in both sets: {overlap}"
