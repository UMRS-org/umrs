"""ADR-097 D-7 — the ZMQ project check (remote-session plan S-1, wave RS-0).

Before 2026-10-22 every project-scoped ZMQ command took its project from
the payload: a bound CURVE session could submit a job, list jobs or run a
workflow in any project by naming it, and a client could set
``_identity_verified`` to make the workflow-status family treat its project
claim as gateway-verified. These tests drive ``CommandRegistry.dispatch``
(the one ZMQ authorization point) and fail on the code before D-7.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.server.commands import project_check
from umrs.server.commands.base import CommandHandler, HandlerContext
from umrs.server.commands.registry import CommandRegistry
from umrs.server.commands.workflow_commands import WorkflowStatusHandler
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_registry_handler import handle_workflow_command

SENDER = b"sender-1"

# Every registered command whose handler reads a project from the payload,
# with a payload naming ``spoofed``.
PROJECT_SCOPED = [
    (TaskCmd.JOB_SUBMIT, {"job_type": "inference", "spec": {}, "project_id": "spoofed"}),
    (TaskCmd.JOB_LIST, {"project_id": "spoofed"}),
    (TaskCmd.RUN_WORKFLOW, {"workflow_name": "wf", "params": {"project_id": "spoofed"}}),
    (TaskCmd.WORKFLOW_EXECUTIONS, {"project_id": "spoofed"}),
    (TaskCmd.WORKFLOW_STATUS, {"execution_id": "wf-1", "project_id": "spoofed"}),
    (TaskCmd.WORKFLOW_RESULTS, {"execution_id": "wf-1", "params": {"project_id": "spoofed"}}),
    (TaskCmd.WORKFLOW_CANCEL, {"execution_id": "wf-1", "project_id": "spoofed"}),
    (TaskCmd.WORKFLOW_ERRORS, {"execution_id": "wf-1", "project_id": "spoofed"}),
    (
        TaskCmd.MODEL_REGISTER,
        {"name": "m", "path": "/p", "scope": "project", "project_id": "spoofed"},
    ),
    (TaskCmd.MODEL_LIST, {"project_id": "spoofed"}),
]


class _Recording(CommandHandler):
    def __init__(self) -> None:
        self.payloads: list = []

    def execute(self, payload, ctx):  # noqa: ARG002 — the CommandHandler signature
        self.payloads.append(payload)


class _Auth:
    def __init__(self, admin: bool) -> None:
        self.is_admin = admin

    def can_execute_command(self, *_args, **_kwargs):
        return True


class _MQ:
    """control_mq stand-in: role verdict, bound session entry, sent replies."""

    def __init__(self, *, admin=False, user="alice", token_projects=("default",), session=True):
        self.sent: list[tuple[str, object]] = []
        self._admin = admin
        self._entry = (
            SimpleNamespace(user_id=user, grants=SimpleNamespace(projects=tuple(token_projects)))
            if session
            else None
        )

    def send_string(self, message, sender_id=None):
        self.sent.append((message, sender_id))

    def get_auth_context(self, _client_id):
        return _Auth(self._admin)

    def verified_session(self, _zmq_sender_id):
        return self._entry


class _Resolver:
    def __init__(self, memberships):
        self.memberships = memberships
        self.calls: list[str] = []

    def __call__(self, user):
        self.calls.append(user)
        if isinstance(self.memberships, Exception):
            raise self.memberships
        return self.memberships


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("UMRS_ZMQ_PROJECT_MEMBERSHIP", raising=False)
    project_check.set_membership_resolver(None)
    yield
    project_check.set_membership_resolver(None)


def _ctx(mq) -> HandlerContext:
    return HandlerContext(
        task_manager=None,
        control_mq=mq,
        server_cfg=None,
        device="cpu",
        embedding_backend=None,
        configure_file="",
    )


def _dispatch(cmd, body, mq, client_id="alice"):
    registry = CommandRegistry()
    handler = _Recording()
    registry.register(cmd, handler)
    registry.dispatch(cmd, (client_id, json.dumps(body), SENDER), _ctx(mq))
    return handler


def _reply(mq) -> dict:
    assert len(mq.sent) == 1
    message, sender = mq.sent[0]
    assert sender == SENDER
    return json.loads(message)


def _resolve(memberships) -> _Resolver:
    resolver = _Resolver(memberships)
    project_check.set_membership_resolver(resolver)
    return resolver


# ---------------------------------------------------------------------------
# The rule: a named project must be a live membership (admins excepted)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cmd", "body"), PROJECT_SCOPED, ids=[c for c, _ in PROJECT_SCOPED])
def test_a_spoofed_project_is_refused(cmd, body):
    _resolve(["default", "opossum"])
    mq = _MQ()
    handler = _dispatch(cmd, body, mq)
    assert handler.payloads == []
    reply = _reply(mq)
    assert reply["status"] == "error"
    assert reply["code"] == "project_not_member"
    assert "spoofed" in reply["message"]


@pytest.mark.parametrize(("cmd", "body"), PROJECT_SCOPED, ids=[c for c, _ in PROJECT_SCOPED])
def test_a_member_passes(cmd, body):
    _resolve(["default", "spoofed"])
    mq = _MQ()
    handler = _dispatch(cmd, body, mq)
    assert len(handler.payloads) == 1
    assert mq.sent == []


@pytest.mark.parametrize(("cmd", "body"), PROJECT_SCOPED, ids=[c for c, _ in PROJECT_SCOPED])
def test_an_admin_passes_without_a_lookup(cmd, body):
    resolver = _resolve([])
    mq = _MQ(admin=True)
    handler = _dispatch(cmd, body, mq)
    assert len(handler.payloads) == 1
    assert resolver.calls == []


def test_job_submit_without_a_project_acts_in_default():
    """JOB_SUBMIT stores ``payload.get("project_id", "default")``: a caller
    who is not a member of ``default`` cannot reach it by omitting the id."""
    _resolve(["opossum"])
    mq = _MQ(token_projects=("opossum",))
    handler = _dispatch(TaskCmd.JOB_SUBMIT, {"job_type": "inference", "spec": {}}, mq)
    assert handler.payloads == []
    assert _reply(mq)["code"] == "project_not_member"


def test_run_workflow_without_a_project_acts_in_default():
    _resolve(["default"])
    mq = _MQ()
    handler = _dispatch(TaskCmd.RUN_WORKFLOW, {"workflow_name": "wf", "params": {}}, mq)
    assert len(handler.payloads) == 1


def test_every_named_project_is_checked():
    _resolve(["opossum"])
    mq = _MQ(token_projects=())
    body = {"execution_id": "wf-1", "project_id": "opossum", "params": {"project_id": "spoofed"}}
    handler = _dispatch(TaskCmd.WORKFLOW_STATUS, body, mq)
    assert handler.payloads == []
    assert "spoofed" in _reply(mq)["message"]


def test_a_payload_naming_no_project_is_not_looked_up():
    resolver = _resolve(None)
    mq = _MQ()
    handler = _dispatch(TaskCmd.JOB_LIST, {"status": "running"}, mq)
    assert len(handler.payloads) == 1
    assert resolver.calls == []


def test_non_json_payloads_pass_untouched():
    resolver = _resolve(None)
    registry = CommandRegistry()
    handler = _Recording()
    registry.register("FILE", handler)
    payload = ("alice", "not json {", SENDER)
    registry.dispatch("FILE", payload, _ctx(_MQ()))
    assert handler.payloads == [payload]
    assert resolver.calls == []


def test_the_lookup_is_the_bound_sessions_user():
    """In session mode the session's user is looked up, not the wire field."""
    resolver = _resolve(["spoofed"])
    mq = _MQ(user="carol:x")
    _dispatch(TaskCmd.JOB_LIST, {"project_id": "spoofed"}, mq, client_id="carol_x")
    assert resolver.calls == ["carol:x"]


def test_without_a_session_the_client_id_is_looked_up():
    resolver = _resolve(["opossum"])
    mq = _MQ(session=False)
    handler = _dispatch(TaskCmd.JOB_LIST, {"project_id": "opossum"}, mq, client_id="bob")
    assert len(handler.payloads) == 1
    assert resolver.calls == ["bob"]


# ---------------------------------------------------------------------------
# Modes (ADR-097 D-4), the session's token projects in the token's place
# ---------------------------------------------------------------------------


def _mode(monkeypatch, mode):
    monkeypatch.setenv("UMRS_ZMQ_PROJECT_MEMBERSHIP", mode)


def test_token_mode_admits_only_the_tokens_projects(monkeypatch):
    _mode(monkeypatch, "token")
    resolver = _resolve(["spoofed"])
    mq = _MQ(token_projects=("opossum",))
    assert _dispatch(TaskCmd.JOB_LIST, {"project_id": "spoofed"}, mq).payloads == []
    assert _reply(mq)["code"] == "project_not_member"
    passed = _dispatch(
        TaskCmd.JOB_LIST, {"project_id": "opossum"}, _MQ(token_projects=("opossum",))
    )
    assert len(passed.payloads) == 1
    assert resolver.calls == []


def test_live_mode_keeps_a_drifted_token_project(monkeypatch, caplog):
    _mode(monkeypatch, "live")
    _resolve(["default"])
    mq = _MQ(token_projects=("opossum",))
    handler = _dispatch(TaskCmd.JOB_LIST, {"project_id": "opossum"}, mq)
    assert len(handler.payloads) == 1
    assert "membership drift" in caplog.text


def test_enforce_mode_refuses_a_revoked_token_project(monkeypatch):
    _mode(monkeypatch, "enforce")
    _resolve(["default"])
    mq = _MQ(token_projects=("opossum",))
    assert _dispatch(TaskCmd.JOB_LIST, {"project_id": "opossum"}, mq).payloads == []
    reply = _reply(mq)
    assert reply["code"] == "project_membership_revoked"
    assert "no longer a member" in reply["message"]


@pytest.mark.parametrize("failure", [None, RuntimeError("db down")])
def test_live_mode_unreadable_membership_fails_closed_off_the_token(monkeypatch, failure):
    _mode(monkeypatch, "live")
    _resolve(failure)
    mq = _MQ(token_projects=("opossum",))
    assert _dispatch(TaskCmd.JOB_LIST, {"project_id": "spoofed"}, mq).payloads == []
    assert _reply(mq)["code"] == "membership_unavailable"
    token = _dispatch(TaskCmd.JOB_LIST, {"project_id": "opossum"}, _MQ(token_projects=("opossum",)))
    assert len(token.payloads) == 1


def test_enforce_mode_unreadable_membership_refuses_everything(monkeypatch):
    _mode(monkeypatch, "enforce")
    _resolve(None)
    mq = _MQ(token_projects=("opossum",))
    assert _dispatch(TaskCmd.JOB_LIST, {"project_id": "opossum"}, mq).payloads == []
    assert _reply(mq)["code"] == "membership_unavailable"


@pytest.mark.parametrize("value", ["", "off", "bogus"])
def test_an_unknown_mode_is_live_never_unchecked(monkeypatch, value):
    _mode(monkeypatch, value)
    _resolve(["default"])
    mq = _MQ()
    assert _dispatch(TaskCmd.JOB_LIST, {"project_id": "spoofed"}, mq).payloads == []
    assert _reply(mq)["code"] == "project_not_member"


def test_the_default_mode_is_live():
    assert project_check.current_mode() == "live"


# ---------------------------------------------------------------------------
# Cache (ADR-097 D-5)
# ---------------------------------------------------------------------------


def test_memberships_are_cached_per_user_and_failures_are_not():
    resolver = _resolve(["opossum"])
    assert project_check.live_memberships_cached("alice") == ["opossum"]
    assert project_check.live_memberships_cached("alice") == ["opossum"]
    assert resolver.calls == ["alice"]
    failing = _resolve(None)
    assert project_check.live_memberships_cached("bob") is None
    assert project_check.live_memberships_cached("bob") is None
    assert failing.calls == ["bob", "bob"]


def test_the_cache_expires(monkeypatch):
    monkeypatch.setenv("UMRS_ZMQ_MEMBERSHIP_CACHE_TTL_SEC", "0")
    resolver = _resolve(["opossum"])
    project_check.live_memberships_cached("alice")
    project_check.live_memberships_cached("alice")
    assert resolver.calls == ["alice", "alice"]


def test_the_default_resolver_fails_closed_without_postgres(monkeypatch):
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert project_check.live_memberships_cached("alice") is None


# ---------------------------------------------------------------------------
# _identity_verified: never from the wire
# ---------------------------------------------------------------------------


def test_a_client_supplied_identity_claim_never_reaches_a_handler():
    _resolve(["project-a"])
    body = {"execution_id": "wf-1", "project_id": "project-a", "_identity_verified": True}
    handler = _dispatch(TaskCmd.WORKFLOW_STATUS, body, _MQ())
    (payload,) = handler.payloads
    assert payload[0] == "alice"
    assert payload[2] == SENDER
    forwarded = json.loads(payload[1])
    assert "_identity_verified" not in forwarded
    assert forwarded["project_id"] == "project-a"


class _Tasks:
    def create_task(self, *_args):
        return 1

    def update_status(self, task_id, status):
        pass


@pytest.fixture
def _alice_execution(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    reset_execution_registry_for_tests()
    record = get_execution_registry().create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )
    yield record
    reset_execution_registry_for_tests()


def test_a_forged_identity_claim_over_zmq_grants_no_verified_access(_alice_execution):
    """End to end through the registry and the real WORKFLOW_STATUS handler:
    under strict auth the forged flag made the project claim count as
    gateway-verified (status ok); now it is dropped and the caller reads as
    unverified."""
    _resolve(["project-a"])
    registry = CommandRegistry()
    registry.register(TaskCmd.WORKFLOW_STATUS, WorkflowStatusHandler())
    mq = _MQ()
    ctx = _ctx(mq)
    ctx.task_manager = SimpleNamespace(tasks_status=_Tasks())
    body = {
        "execution_id": _alice_execution.execution_id,
        "project_id": "project-a",
        "_identity_verified": True,
    }
    registry.dispatch(TaskCmd.WORKFLOW_STATUS, ("alice", json.dumps(body), SENDER), ctx)
    reply = _reply(mq)
    assert reply["status"] == "error"
    assert reply["code"] == "workflow_access_denied"


def test_the_workflow_handler_ignores_the_flag_unless_trusted(_alice_execution):
    payload = json.dumps(
        {
            "execution_id": _alice_execution.execution_id,
            "project_id": "project-a",
            "_identity_verified": True,
        }
    )
    untrusted = json.loads(handle_workflow_command(TaskCmd.WORKFLOW_STATUS, "alice", payload))
    assert untrusted["code"] == "workflow_access_denied"
    trusted = json.loads(
        handle_workflow_command(TaskCmd.WORKFLOW_STATUS, "alice", payload, trusted_identity=True)
    )
    assert trusted["status"] == "ok"


# ---------------------------------------------------------------------------
# With the real MessageQueue session binding (umrs 4c01cd90)
# ---------------------------------------------------------------------------


def _session_mq(entries):
    import collections
    import time

    from umrs.core.MessageQueue import MessageQueue
    from umrs.core.session_curve_provider import SessionCurveEntry, SessionGrants

    sessions = {
        sid: SessionCurveEntry(
            pubkey_z85="k" * 40,
            session_id=sid,
            user_id=user,
            created_at=time.time(),
            expires_at=time.time() + 600,
            grants=SessionGrants(projects=tuple(projects), roles=tuple(roles)),
        )
        for sid, (user, projects, roles) in entries.items()
    }
    mq = MessageQueue.__new__(MessageQueue)
    mq._session_entries = collections.OrderedDict()
    mq._sender_entries = collections.OrderedDict()
    mq._auth_sessions = {}
    mq._identity_provider_cache = (SimpleNamespace(lookup_session=sessions.get),)
    mq.sent = []
    mq.send_string = lambda text, sender: mq.sent.append((text, sender))
    return mq


def _bound_dispatch(mq, session_id, fields, sender):
    bound = mq._bind_session_identity(fields[0], fields, sender, session_id)
    registry = CommandRegistry()
    handler = _Recording()
    registry.register(fields[0], handler)
    registry.dispatch(fields[0], (bound[1], bound[2], sender), _ctx(mq))
    return handler


def test_a_bound_session_cannot_name_another_project():
    resolver = _resolve(["default"])
    mq = _session_mq({"s-carol": ("carol", ["default"], ["user"])})
    body = json.dumps({"job_type": "inference", "project_id": "opossum"})
    handler = _bound_dispatch(mq, "s-carol", ["JOB_SUBMIT", "bob", body], b"c")
    assert handler.payloads == []
    assert resolver.calls == ["carol"]
    assert json.loads(mq.sent[-1][0])["code"] == "project_not_member"


def test_a_bound_member_and_an_admin_session_pass():
    _resolve(["default", "opossum"])
    mq = _session_mq(
        {
            "s-carol": ("carol", ["default"], ["user"]),
            "s-maint": ("maint", [], ["admin"]),
        }
    )
    member = json.dumps({"project_id": "opossum"})
    assert len(_bound_dispatch(mq, "s-carol", ["JOB_LIST", "carol", member], b"c").payloads) == 1
    elsewhere = json.dumps({"project_id": "elsewhere"})
    admin = _bound_dispatch(mq, "s-maint", ["JOB_LIST", "maint", elsewhere], b"m")
    assert len(admin.payloads) == 1
