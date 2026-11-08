"""The gated step lane: one workflow step under the person's own HPC account
(ADR-112 D-4/D-5/D-6; plan §5 B2.0 §5 and §10.5).

A run whose identity is ``user_account`` reaches the cluster only through
the person's gate (``~/.umrs/gate/umrs_gate.py``, the forced command of the
transport key the server holds for that person). The server sends verbs;
the gate builds the batch script and ``input.json`` from the person's signed
intent and its own site profile, checks every staged byte against the
intent's digests and submits under the person's login. Nothing here is a
pickle, an operator script, ``venue.sh`` or a ``remote_env`` prefix (I-5),
and nothing ever falls back to the platform login (I-1).

Per step:

1. check the step against the intent (in it; the runner's entrypoint is the
   signed one; the routed endpoint is the signed one; an endpoint
   credential is that endpoint's own key, never the platform default);
2. ``lookup <i> <step>``: a nonce with a job is adopted (never submitted
   twice); else ``mkwd`` and record the workdir;
3. ``stage-in``: one tar of ``src/<path>`` (the registered tree, exactly the
   manifest) and ``in/blobs/<sha256>`` (each required digest once); a
   failed stage-in abandons that workdir (``rmwd``) and starts over;
4. ``submit <wd> <step>`` with ``{payload, signature[, endpoint_credential]}``;
   exit 5 or a lost connection: ``lookup`` again (adopt, resubmit with a
   fresh workdir when no nonce, or fail ``submit_unknown``);
5. ``status`` polling with the lane's clocks; stopping is ``cancel``;
6. ``read-reply``, ``collect-out`` into the run's output bucket, the
   reply's keys rewritten to physical keys; ``usage`` measures the job;
7. ``rmwd`` once the job is terminal.

A gate refusal (exit 4) fails the step ``hpcaccess.gate_refused: <name>:
<detail>`` and is never retried under another identity.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO

from umrs.pipeline_sdk.step import StepContext, StepResult

from .slurm_step_executor import (
    TERMINAL_STATES,
    UNKNOWN,
    UNREACHABLE,
    PollClock,
    stamp_slurm_accounting,
)
from .step_executor import EXECUTOR_SOURCE_DIR_METADATA_KEY, StepExecutionError
from .step_reply import result_from_json
from .step_staging import (
    STAGE_REQUEST_METADATA_KEY,
    collect_out,
    rewrite_output_keys,
)

logger = logging.getLogger(__name__)

GATED_PREFIX = "gated:"
#: The gate's exit codes (B1.5a).
EXIT_ABSENT = 3
EXIT_REFUSED = 4
EXIT_SCHEDULER = 5
SSH_LOST = 255
#: Rounds of (fresh workdir, stage-in, submit) one step may take.
_SUBMIT_ROUNDS = 2
_STAGE_ROUNDS = 2
_COPY_CHUNK = 1 << 20
_REPLY_ROOM = 64 * 1024
#: Keys a gated step's ``result.extra`` never carries from the step itself.
_EXECUTOR_EXTRA_KEYS = ("slurm", "slurm_job")


class GateFailure(StepExecutionError):
    """The person's gate refused (``hpcaccess.gate_refused: <name>: <detail>``)."""

    def __init__(self, name: str, detail: str = "", phase: str = "worker") -> None:
        super().__init__(phase, f"hpcaccess.gate_refused: {name}: {detail}".rstrip(": "))
        self.name = name


# ── the run's binding ────────────────────────────────────────────────────────


def transport_tag(venue: str, username: str) -> str:
    """``gated:<venue>:<username>``: the step record's ``transport``."""
    return f"{GATED_PREFIX}{venue}:{username}"


def parse_transport_tag(tag: Any) -> tuple[str, str] | None:
    """``(venue, username)`` of a ``gated:*`` tag, else None."""
    text = str(tag or "")
    if not text.startswith(GATED_PREFIX):
        return None
    venue, sep, username = text[len(GATED_PREFIX) :].partition(":")
    return (venue, username) if sep and venue and username else None


@dataclass(frozen=True)
class GatedRun:
    """A run under the person's own account: who, where, and the signed intent."""

    venue: str
    username: str
    intent_b64: str
    signature: str
    intent_id: str

    @property
    def transport(self) -> str:
        return transport_tag(self.venue, self.username)

    def body(self) -> dict[str, Any]:
        """The intent body (parsed strictly when the run was admitted)."""
        try:
            body = json.loads(base64.b64decode(self.intent_b64, validate=True))
        except ValueError as exc:
            raise StepExecutionError("setup", f"the run's intent is unreadable: {exc}") from exc
        if not isinstance(body, dict):
            raise StepExecutionError("setup", "the run's intent is not an object")
        return body

    def envelope(self, credential: dict[str, Any] | None) -> bytes:
        """``submit``'s stdin: the signed payload and signature (B1.5a)."""
        envelope: dict[str, Any] = {"payload": self.intent_b64, "signature": self.signature}
        if credential is not None:
            envelope["endpoint_credential"] = credential
        return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def gated_run_for(record: Any) -> GatedRun | None:
    """The gated binding of an execution record with an intent (None else)."""
    if not getattr(record, "intent_id", None):
        return None
    from umrs.shared.hpc_user_work_access import current_site

    return GatedRun(
        venue=str(getattr(current_site(), "name", "") or ""),
        username=str(record.user_id),
        intent_b64=str(record.intent_b64 or ""),
        signature=str(record.intent_signature or ""),
        intent_id=str(record.intent_id),
    )


# ── transport: one ssh call per verb, as the person, with their key ──────────

ArgvFor = Callable[[str, tuple[str, ...]], list[str]]


@dataclass
class GateTransport:
    """``argv_for(verb, args)`` builds one call's argv (the contract's ssh
    argv; a test passes a local command that runs a gate double)."""

    argv_for: ArgvFor

    def call(self, verb: str, *args: str, stdin: bytes | None = None) -> Any:
        """One verb: bytes in, ``CompletedProcess`` (rc -1: ssh never ran)."""
        from umrs.hpc_accounts.gate_client import verb_timeout

        try:
            return subprocess.run(
                self.argv_for(verb, tuple(args)),
                input=stdin,
                stdin=None if stdin is not None else subprocess.DEVNULL,
                capture_output=True,
                timeout=verb_timeout(verb),
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            return subprocess.CompletedProcess([verb], -1, b"", str(exc).encode())

    def stream(
        self,
        verb: str,
        *args: str,
        feed: Callable[[BinaryIO], Any] | None = None,
        drain: Callable[[BinaryIO], Any] | None = None,
    ) -> Any:
        from umrs.hpc_accounts.gate_client import verb_timeout

        from .remote_slurm_step_executor import stream_argv

        return stream_argv(
            self.argv_for(verb, tuple(args)),
            timeout=verb_timeout(verb),
            feed=feed,
            drain=drain,
        )


#: ``(venue, username, login) -> GateTransport``; None = the person's
#: transport key over ssh. Tests install a gate double here.
_transport_factory: Callable[[str, str, str], GateTransport] | None = None


def set_transport_factory(factory: Callable[[str, str, str], GateTransport] | None) -> None:
    global _transport_factory
    _transport_factory = factory


def _active_account(venue: str, username: str, login: str) -> dict[str, Any]:
    from umrs.shared.hpc_user_work_access import site_account_store

    account = site_account_store(venue).get_account(username)
    if not account or account.get("status") != "active":
        raise GateFailure("not_enrolled", f"the account of {username} at {venue} is not active", "setup")
    if login and account.get("verified_login") != login:
        raise GateFailure("not_enrolled", "the account's verified login changed since the intent", "setup")
    return account


def transport_for(venue: str, username: str, login: str = "") -> GateTransport:
    """The person's gate transport: their active account's login and current
    transport key, the site's own target (``ssh -G``, read once)."""
    if _transport_factory is not None:
        return _transport_factory(venue, username, login)
    from umrs.hpc_accounts import gate_client, transport_keys
    from umrs.shared.hpc_user_work_access import current_site

    account = _active_account(venue, username, login)
    key = transport_keys.load(venue, username, str(account.get("transport_key_id") or ""))
    if key is None:
        raise GateFailure("not_enrolled", f"no transport key for {username} at {venue}", "setup")
    site = current_site()
    resolved = gate_client.resolve_target_sync(gate_client.site_target(site))
    real_login = str(account.get("verified_login") or "")

    def argv_for(verb: str, args: tuple[str, ...]) -> list[str]:
        return gate_client.gate_argv(site, real_login, key.path, verb, resolved, args)

    return GateTransport(argv_for)


# ── answers ──────────────────────────────────────────────────────────────────


def _answer(proc: Any) -> dict[str, Any]:
    try:
        value = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        value = None
    if not isinstance(value, dict):
        raise GateFailure("bad_answer", "the gate's answer is not one JSON object")
    return value


def _last_json_line(raw: bytes) -> Any:
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    try:
        return json.loads(lines[-1]) if lines else {}
    except ValueError:
        return {}


def _refusal_of(proc: Any) -> tuple[str, str]:
    """The gate's ``{"error", "detail"}`` from the last stderr line."""
    raw = bytes(proc.stderr or b"")
    data = _last_json_line(raw)
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return data["error"], str(data.get("detail") or "")[:300]
    return "failed", raw.decode("utf-8", "replace")[-300:]


def _transient(proc: Any) -> bool:
    return proc.returncode in (EXIT_SCHEDULER, SSH_LOST, -1)


def _raise_refused(verb: str, proc: Any) -> None:
    """Exit 4 is the gate refusing: never retried, never another identity."""
    if proc.returncode == EXIT_REFUSED:
        name, detail = _refusal_of(proc)
        raise GateFailure(name, f"{verb}: {detail}" if detail else verb)


def _checked(verb: str, proc: Any) -> dict[str, Any]:
    if proc.returncode == 0:
        return _answer(proc)
    _raise_refused(verb, proc)
    name, detail = _refusal_of(proc)
    raise StepExecutionError(
        "worker", f"gate {verb} failed (rc={proc.returncode}): {name}: {detail}"
    )


# ── what the intent says about this step ─────────────────────────────────────


def step_name(ctx: StepContext, config: dict[str, Any]) -> str:
    request = ctx.metadata.get(STAGE_REQUEST_METADATA_KEY) or {}
    return str(request.get("step") or config.get("step_name") or "")


def intent_step(intent: dict[str, Any], name: str) -> dict[str, Any]:
    for step in intent.get("steps") or []:
        if isinstance(step, dict) and step.get("name") == name:
            return step
    raise StepExecutionError("setup", f"step {name!r} is not in the run's signed intent; nothing was sent")


def check_entrypoint(intent: dict[str, Any], name: str, entrypoint: str) -> None:
    signed = (intent.get("src_manifest") or {}).get("entrypoints", {}).get(name)
    if signed != entrypoint:
        raise StepExecutionError(
            "setup",
            f"entrypoint_changed: step {name} runs {entrypoint!r}, the intent "
            f"signed {signed!r}; nothing was sent",
        )


def _platform_keys() -> set[str]:
    """Keys a gated step's forwarder may never carry: the platform's own."""
    names = ("OPENAI_API_KEY", "LITELLM_MASTER_KEY", "UMRS_DEFAULT_API_KEY")
    return {os.environ[n] for n in names if os.environ.get(n)}


def _routed_address(routed: tuple[Any, bytes] | None) -> tuple[Any, ...] | None:
    if routed is None:
        return None
    forward = routed[0]
    return (forward.host, forward.port, forward.kind)


def _signed_address(signed: Any) -> tuple[Any, ...] | None:
    if not isinstance(signed, dict):
        return None
    return (signed.get("host"), signed.get("port"), signed.get("kind"))


def _check_scoped_key(handover: Any) -> None:
    key = str((handover or {}).get("api_key") or "")
    if key and key in _platform_keys():
        raise StepExecutionError(
            "setup",
            "endpoint_credential_not_scoped: the routed endpoint's credential is the "
            "platform's default key; nothing was sent",
        )


def check_endpoint(
    signed: Any, routed: tuple[Any, bytes] | None, handover: Any
) -> dict[str, Any] | None:
    """The routed endpoint is the signed one (or both absent); its credential
    is that endpoint's own key. Returns the ``endpoint_credential``."""
    have, want = _routed_address(routed), _signed_address(signed)
    if have != want:
        raise StepExecutionError(
            "setup", f"endpoint_changed: routed {have!r}, signed {want!r}; nothing was sent"
        )
    if routed is None:
        return None
    _check_scoped_key(handover)
    return json.loads(routed[1])


# ── stage-in: src/<path> and in/blobs/<sha256> ───────────────────────────────


class _Hashing:
    """A reader of exactly *size* bytes that hashes what it passes on."""

    def __init__(self, source: BinaryIO, size: int, label: str) -> None:
        self._source, self._left, self._label = source, size, label
        self.digest = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        want = self._left if n is None or n < 0 else min(n, self._left)
        data = self._source.read(want) if want else b""
        if want and not data:
            raise StepExecutionError("setup", f"{self._label} ended early")
        self._left -= len(data)
        self.digest.update(data)
        return data

    def check(self, sha256: str) -> None:
        if self._source.read(1) or self.digest.hexdigest() != sha256:
            raise StepExecutionError(
                "setup", f"input_changed: {self._label} differs from the signed digest"
            )


def _member(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime, info.type = size, 0o600, 0, tarfile.REGTYPE
    return info


def _add_checked(tar: Any, name: str, source: BinaryIO, size: int, sha256: str) -> None:
    reader = _Hashing(source, size, name)
    tar.addfile(_member(name, size), reader)
    reader.check(sha256)


def _src_path(source_dir: str, path: str) -> str:
    root = os.path.realpath(source_dir)
    full = os.path.realpath(os.path.join(root, path))
    if not full.startswith(root + os.sep):
        raise StepExecutionError("setup", f"src path {path!r} leaves the workflow tree")
    return full


def _add_sources(tar: Any, files: list[dict[str, Any]], source_dir: str) -> None:
    for entry in files:
        path = str(entry["path"])
        fd = os.open(_src_path(source_dir, path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            _add_checked(tar, f"src/{path}", handle, int(entry["size"]), str(entry["sha256"]))


def _object_body(s3: Any, bucket: str, key: str) -> BinaryIO:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    if isinstance(body, (bytes, bytearray)):
        import io

        return io.BytesIO(bytes(body))
    return body


def _add_objects(tar: Any, objects: list[dict[str, Any]], s3: Any, seen: set[str]) -> None:
    for obj in objects:
        sha = str(obj["sha256"])
        if sha in seen:
            continue
        body = _object_body(s3, str(obj["bucket"]), str(obj["key"]))
        try:
            _add_checked(tar, f"in/blobs/{sha}", body, int(obj["size"]), sha)
        finally:
            getattr(body, "close", lambda: None)()
        seen.add(sha)


def _spool_into(spool: BinaryIO, s3: Any, bucket: str, key: str) -> tuple[str, int]:
    """A parent output read once into *spool*: its digest names the blob."""
    body = _object_body(s3, bucket, key)
    digest, size = hashlib.sha256(), 0
    try:
        for chunk in iter(lambda: body.read(_COPY_CHUNK), b""):
            spool.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    finally:
        getattr(body, "close", lambda: None)()
    spool.seek(0)
    return digest.hexdigest(), size


def _parent_refs(ctx: StepContext, parent: str) -> list[tuple[str, str]]:
    from umrs.pipeline_sdk.staged_s3 import split_s3_ref

    result = (ctx.parent_results or {}).get(parent)
    keys = list(getattr(result, "output_keys", None) or [])
    refs = [split_s3_ref(raw, ctx.output_bucket) for raw in keys]
    return [(str(bucket or ctx.output_bucket), key) for bucket, key in refs]


def _add_chained(tar: Any, ctx: StepContext, chained: list[str], seen: set[str]) -> None:
    for parent in chained:
        for bucket, key in _parent_refs(ctx, parent):
            with tempfile.TemporaryFile() as spool:
                sha, size = _spool_into(spool, ctx.s3, bucket, key)
                if sha not in seen:
                    _add_checked(tar, f"in/blobs/{sha}", spool, size, sha)
                    seen.add(sha)


def feed_stage_in(
    sink: BinaryIO, ctx: StepContext, intent: dict[str, Any], step: dict[str, Any], source_dir: str
) -> int:
    """Write the step's one stage-in tar; the member count."""
    files = list((intent.get("src_manifest") or {}).get("files") or [])
    inputs = step.get("inputs") or {}
    seen: set[str] = set()
    with tarfile.open(fileobj=sink, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        _add_sources(tar, files, source_dir)
        _add_objects(tar, list(inputs.get("objects") or []), ctx.s3, seen)
        _add_chained(tar, ctx, list(inputs.get("chained") or []), seen)
    return len(files) + len(seen)


# ── the step ─────────────────────────────────────────────────────────────────


class GatedStepLane:
    """One gated step, driven by *executor* (clocks, caps, the run's binding
    and step record) over *transport* (the person's gate)."""

    def __init__(self, executor: Any, run: GatedRun, transport: GateTransport) -> None:
        self.executor = executor
        self.run = run
        self.transport = transport

    # the flow

    def execute(self, entrypoint: str, config: dict[str, Any], ctx: StepContext) -> StepResult:
        intent = self.run.body()
        name = step_name(ctx, config)
        step = intent_step(intent, name)
        check_entrypoint(intent, name, entrypoint)
        credential = check_endpoint(
            step.get("endpoint"), self.executor._endpoint(ctx), self._handover(ctx)
        )
        job = self._begin(ctx, name, intent)
        try:
            self._place(job, ctx, intent, step, credential)
            return self._collect(job, ctx)
        finally:
            self._finish(job)

    @staticmethod
    def _handover(ctx: StepContext) -> Any:
        from .step_executor import EXECUTOR_ENDPOINT_METADATA_KEY

        return ctx.metadata.get(EXECUTOR_ENDPOINT_METADATA_KEY)

    def _begin(self, ctx: StepContext, name: str, intent: dict[str, Any]) -> Any:
        from .remote_slurm_step_executor import _StepJob

        executor = self.executor
        with executor._lock:
            key = (str(ctx.run_id), name)
            attempt = executor._attempts.get(key, 0) + 1
            executor._attempts[key] = attempt
        return _StepJob(
            run_id=str(ctx.run_id),
            step=name,
            attempt=attempt,
            job_name=f"umrs-gated-{self.run.intent_id[:16]}-{name}",
            prefix="",
            site_login=str(intent.get("login") or ""),
            slurm_account=str(intent.get("account") or ""),
            access_mode="user_account",
            transport=self.run.transport,
            intent_id=self.run.intent_id,
        )

    def _place(
        self, job: Any, ctx: StepContext, intent: dict[str, Any], step: dict[str, Any], credential: Any
    ) -> None:
        """Adopt the job of an earlier attempt, or stage and submit (with
        one fresh round when the gate lost a submit and kept no nonce)."""
        if self._adopt(job):
            return
        for _ in range(_SUBMIT_ROUNDS):
            self._fresh_workdir(job, ctx, intent, step)
            reason = self.executor._stop_reason(job, ctx, None)
            if reason:
                raise StepExecutionError("run", f"step {job.step} not submitted: {reason}")
            if self._submit(job, credential):
                return
        raise GateFailure("submit_failed", f"step {job.step}: the gate kept no submit")

    def _lookup(self, job: Any) -> dict[str, Any] | None:
        proc = self._patient("lookup", self.run.intent_id, job.step)
        if proc.returncode == EXIT_ABSENT:
            return None
        return _checked("lookup", proc)

    def _adopt(self, job: Any) -> bool:
        found = self._lookup(job)
        if found is None:
            return False
        job.workdir = str(found.get("wd") or job.workdir)
        if not found.get("job_id"):
            job.keep = True
            raise GateFailure("submit_unknown", f"step {job.step}: the gate holds a submit whose job it cannot find")
        job.job_id, job.state = str(found["job_id"]), str(found.get("state") or "SUBMITTED")
        self.executor._notify(job)
        logger.warning("gated lane: step %s adopts job %s of its intent", job.step, job.job_id)
        return True

    def _fresh_workdir(self, job: Any, ctx: StepContext, intent: dict[str, Any], step: dict[str, Any]) -> None:
        for _ in range(_STAGE_ROUNDS):
            job.workdir = str(_checked("mkwd", self.transport.call("mkwd", self.run.intent_id, job.step)).get("wd") or "")
            self.executor._notify(job)  # the reaper's handle before any byte moves
            if self._stage_in(job, ctx, intent, step):
                return
            self._rmwd(job.workdir)
        raise StepExecutionError("setup", f"stage-in of step {job.step} failed {_STAGE_ROUNDS} times")

    def _stage_in(self, job: Any, ctx: StepContext, intent: dict[str, Any], step: dict[str, Any]) -> bool:
        """True when the gate took the tar; a lost or refused stream poisons
        the workdir (the gate marks it broken), so the caller starts over."""
        source_dir = str(ctx.metadata.get(EXECUTOR_SOURCE_DIR_METADATA_KEY) or "")
        result = self.transport.stream(
            "stage-in",
            job.workdir,
            feed=lambda sink: _guarded_feed(sink, ctx, intent, step, source_dir),
        )
        if result.error is not None and "input_changed" in result.error.message:
            raise StepExecutionError("setup", f"hpcaccess.intent_invalid: {result.error.message}")
        if result.returncode == 0 and result.error is None:
            return True
        if _gate_refused_stage(result):
            _raise_refused("stage-in", result)
        logger.warning(
            "gated lane: stage-in of %s failed (rc=%s); fresh workdir", job.step, result.returncode
        )
        return False

    def _submit(self, job: Any, credential: Any) -> bool:
        proc = self.transport.call("submit", job.workdir, job.step, stdin=self.run.envelope(credential))
        if proc.returncode == 0:
            answer = _answer(proc)
            job.job_id, job.state = str(answer.get("job_id") or ""), "SUBMITTED"
            job.gate_log = answer.get("log_head")
            self.executor._notify(job)
            return True
        _raise_refused("submit", proc)
        if self._adopt(job):
            return True
        self._rmwd(job.workdir)  # sealed and never submitted: the gate dropped the nonce
        return False

    # polling

    def _patient(self, verb: str, *args: str) -> Any:
        """A read-only verb, repeated across a lost connection within the
        lane's unreachable budget."""
        started = time.monotonic()
        while True:
            proc = self.transport.call(verb, *args)
            if not _transient(proc) or time.monotonic() - started >= self.executor.unreachable_budget:
                return proc
            time.sleep(self.executor.poll_interval)

    def _state(self, job_id: str) -> str:
        proc = self.transport.call("status", job_id)
        if _transient(proc):
            return UNREACHABLE
        state = str(_checked("status", proc).get("state") or "").upper()
        return state or UNKNOWN

    def _poll(self, job: Any, ctx: StepContext) -> str:
        executor = self.executor
        clock = PollClock(
            queue_timeout=executor._queue_seconds(),
            wall_seconds=float(_signed_seconds(self.run.body(), job.step)),
            unreachable_budget=executor.unreachable_budget,
        )
        while True:
            job.state = self._state(job.job_id)
            clock.observe(job.state)
            if job.state in TERMINAL_STATES:
                job.terminal = True
                return job.state
            reason = clock.expired() or executor._stop_reason(job, ctx, clock.running_since)
            if reason:
                job.keep = job.keep or job.state == UNREACHABLE
                raise StepExecutionError("run", f"Slurm job {job.job_id} {reason}")
            time.sleep(executor.poll_interval)

    # results

    def _collect(self, job: Any, ctx: StepContext) -> StepResult:
        state = self._poll(job, ctx)
        if state != "COMPLETED":
            raise StepExecutionError(
                "worker", f"Slurm job {job.job_id} failed with status {state}. Error log: {self._tail(job)}"
            )
        blob = self._read_reply(job)
        result = result_from_json(blob)
        mapping = self._collect_out(job, ctx)
        result = rewrite_output_keys(result, mapping, ctx.output_bucket)
        for key in _EXECUTOR_EXTRA_KEYS:
            result.extra.pop(key, None)
        stamp_slurm_accounting(result, self._usage(job.job_id))
        result.extra["slurm_job"] = {"job_id": job.job_id, "workdir": job.workdir}
        return result

    def _read_reply(self, job: Any) -> bytes:
        proc = self._patient("read-reply", job.workdir)
        if _transient(proc):
            job.keep = True
            raise StepExecutionError("worker", f"Slurm job {job.job_id} COMPLETED but its reply could not be read; workdir {job.workdir} is kept")
        _raise_refused("read-reply", proc)
        if proc.returncode == EXIT_ABSENT:
            raise GateFailure(
                "replayed",
                f"step {job.step} of this intent already ran (job {job.job_id}) and its "
                "workdir is gone: sign a new intent to run it again",
            )
        if proc.returncode != 0 or len(proc.stdout) > self.executor.reply_cap + _REPLY_ROOM:
            raise StepExecutionError("worker", f"gate read-reply failed (rc={proc.returncode})")
        return bytes(proc.stdout)

    def _collect_out(self, job: Any, ctx: StepContext) -> dict[str, str]:
        from .remote_slurm_step_executor import _json_object, _raise_stream_failure

        caps = self.executor.caps
        result = self.transport.stream(
            "collect-out",
            job.workdir,
            drain=lambda source: collect_out(source, ctx.s3, ctx.output_bucket, ctx.run_id, job.step, caps),
        )
        if result.returncode != EXIT_REFUSED and (result.returncode != 0 or result.error is not None):
            job.keep = True  # the gate's out/ is the only copy
        _raise_refused("collect-out", result)
        _raise_stream_failure(result, "worker", "gate collect-out failed")
        summary = _json_object((result.stderr.strip().splitlines() or [b""])[-1], "collect-out")
        if summary.get("files") != len(result.value or {}):
            raise StepExecutionError("worker", f"collect-out sent {summary.get('files')!r} files, {len(result.value or {})} arrived")
        return dict(result.value or {})

    def _usage(self, job_id: str) -> dict[str, Any]:
        """The gate's ``usage`` (sacct under the person's login): seconds x
        GPUs, or nothing (never approximated)."""
        from umrs.shared.slurm_accounting import (
            measured_gpu_seconds,
            parse_alloc_tres_gpus,
            parse_slurm_elapsed,
        )

        row = self._usage_row(job_id)
        elapsed = parse_slurm_elapsed(str(row.get("elapsed") or ""))
        if elapsed is None:
            return {}
        gpus = parse_alloc_tres_gpus(str(row.get("alloc_tres") or ""))
        return {
            "job_id": job_id,
            "elapsed_seconds": elapsed,
            "gpus": gpus,
            "gpu_seconds": measured_gpu_seconds(elapsed, gpus),
        }

    def _usage_row(self, job_id: str) -> dict[str, Any]:
        proc = self.transport.call("usage", job_id)
        if proc.returncode != 0:
            return {}
        rows = [r for r in _answer(proc).get("jobs") or [] if isinstance(r, dict)]
        return next((r for r in rows if str(r.get("job_id")) == job_id), {})

    def _tail(self, job: Any) -> str:
        proc = self.transport.call("tail-log", job.workdir)
        if proc.returncode != 0:
            return "<no log>"
        return str(_answer(proc).get("tail") or "")[-1000:]

    # cleanup

    def _cancel(self, job_id: str) -> bool:
        return self.transport.call("cancel", job_id).returncode == 0

    def _rmwd(self, wd: str) -> bool:
        if not wd or self.executor.keep_workdir:
            return False
        proc = self.transport.call("rmwd", wd)
        if proc.returncode != 0:
            logger.warning("gated lane: workdir %s not removed: %s", wd, _refusal_of(proc))
        return proc.returncode == 0

    def _finish(self, job: Any) -> None:
        """Cancel a job that is not terminal; remove the workdir once the
        job is terminal (or was never submitted)."""
        if job.job_id and not job.terminal and not job.keep:
            self._cancel(job.job_id)
            job.state = self._state(job.job_id)
            job.terminal = job.state in TERMINAL_STATES
        if not job.keep and (job.terminal or not job.job_id):
            job.cleaned = self._rmwd(job.workdir)
        self.executor._notify(job)


#: Stage-in refusals that only poison the workdir (a fresh one may pass).
_POISONING = frozenset({"broken", "busy", "sealed"})


class _FeedFailed(StepExecutionError):
    """The server's own side of a stage-in stream failed (an object read or a
    source file read broke off): the gate's answer to the cut tar is a
    consequence, not its decision, so the workdir is abandoned and a fresh
    one staged (plan §5 B2.0 §10.5)."""


def _guarded_feed(
    sink: BinaryIO, ctx: StepContext, intent: dict[str, Any], step: dict[str, Any], source_dir: str
) -> int:
    try:
        return feed_stage_in(sink, ctx, intent, step, source_dir)
    except (BrokenPipeError, ConnectionResetError):
        raise  # the gate ended first: its exit status names the cause
    except OSError as exc:
        raise _FeedFailed("setup", f"the stage-in stream broke off: {exc}") from exc


def _gate_refused_stage(result: Any) -> bool:
    """A stage-in the gate refused by its own decision (never retried): exit
    4 with a non-poisoning name, after the server sent the whole stream."""
    if result.returncode != EXIT_REFUSED or isinstance(result.error, _FeedFailed):
        return False
    return _refusal_of(result)[0] not in _POISONING


def _signed_seconds(intent: dict[str, Any], name: str) -> int:
    step = intent_step(intent, name)
    return int(step.get("time") or 60)


# ── reaper and janitor legs: unsigned verbs on the person's own records ──────


def reap_gated(facts: list[dict[str, Any]], for_retry: bool = False) -> bool:
    """End what a gated step left: ``status``, ``cancel`` and ``rmwd``
    through that person's gate. An account that is not active or a key
    that is gone logs ``gate unreachable`` and returns False: the job ends
    by ``--time`` / ``--deadline`` and its workdir by the gate's ``sweep``."""
    done = [_reap_one(f, for_retry) for f in facts]
    return all(done)


def _reap_one(facts: dict[str, Any], for_retry: bool) -> bool:
    parsed = parse_transport_tag(facts.get("transport"))
    if parsed is None:
        return False
    try:
        transport = transport_for(*parsed)
    except Exception as exc:  # noqa: BLE001 - the reaper reports, never raises
        logger.warning("gated lane: gate unreachable for %s: %s", facts.get("transport"), exc)
        return False
    return _reap_with(transport, facts, for_retry)


#: ``status`` refused ``not_mine``: the gate no longer holds the job's
#: record (it prunes a job's state after the intent's window plus 7 days),
#: so the job ended long ago; nothing is left to cancel.
GONE = "GONE"
_REAPED = frozenset(TERMINAL_STATES) | {GONE}


def _reap_with(transport: GateTransport, facts: dict[str, Any], for_retry: bool) -> bool:
    job_id, wd = str(facts.get("job_id") or ""), str(facts.get("workdir") or "")
    state = _reap_state(transport, job_id)
    if job_id and state not in _REAPED:
        if for_retry:
            return False  # a retry of the run adopts it through lookup
        transport.call("cancel", job_id)
        return False  # ended by the next pass, once terminal
    # rmwd exit 3: the gate's sweep (or an earlier finish) already removed it.
    return not wd or transport.call("rmwd", wd).returncode in (0, EXIT_ABSENT)


def _reap_state(transport: GateTransport, job_id: str) -> str:
    if not job_id:
        return ""
    proc = transport.call("status", job_id)
    if proc.returncode == EXIT_REFUSED and _refusal_of(proc)[0] == "not_mine":
        return GONE
    if proc.returncode != 0:
        return UNREACHABLE
    try:
        return str(_answer(proc).get("state") or "").upper()
    except GateFailure:
        return UNKNOWN


def _live_gated(facts: dict[str, Any], venue: str, username: str) -> bool:
    """A job of this person at this venue that may still be queued or running."""
    if parse_transport_tag(facts.get("transport")) != (venue, username):
        return False
    state = str(facts.get("state") or "").upper()
    return bool(facts.get("job_id")) and not (
        state in TERMINAL_STATES or state.startswith("CANCELLED")
    )


def live_gated_jobs(records: Any, venue: str, username: str) -> list[str]:
    """The job ids of *username*'s gated steps at *venue* that the step
    records do not show as ended."""
    from .step_reaper import step_jobs

    found: set[str] = set()
    for record in records or []:
        for step in list((getattr(record, "steps", None) or {}).values()):
            found |= {
                str(f["job_id"]) for f in step_jobs(step).values() if _live_gated(f, venue, username)
            }
    return sorted(found)


def cancel_person_jobs(venue: str, username: str, records: Any) -> int:
    """The account cascade's leg (plan §5 B2.0 §10.6): before the person's
    transport keys are destroyed, ``cancel`` each of their live gated jobs
    through their gate. Best-effort: the count of jobs the gate cancelled;
    an unreachable gate cancels none (the jobs still end by ``--time`` and
    ``--deadline``)."""
    jobs = live_gated_jobs(records, venue, username)
    if not jobs:
        return 0
    try:
        transport = transport_for(venue, username)
    except Exception as exc:  # noqa: BLE001 - the cascade reports, never raises
        logger.warning("gated lane: gate unreachable for %s at %s: %s", username, venue, exc)
        return 0
    return sum(1 for job_id in jobs if transport.call("cancel", job_id).returncode == 0)


def sweep_person(venue: str, username: str, older_than: int) -> dict[str, Any]:
    """``sweep --older-than <s>`` at one person's gate (the janitor's leg)."""
    transport = transport_for(venue, username)
    return _checked("sweep", transport.call("sweep", "--older-than", str(int(older_than))))


__all__ = [
    "GATED_PREFIX",
    "GateFailure",
    "GateTransport",
    "GatedRun",
    "GatedStepLane",
    "cancel_person_jobs",
    "feed_stage_in",
    "gated_run_for",
    "live_gated_jobs",
    "parse_transport_tag",
    "reap_gated",
    "set_transport_factory",
    "sweep_person",
    "transport_for",
    "transport_tag",
]
