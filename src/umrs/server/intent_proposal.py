"""The server's proposed intent for one gated run (ADR-112 B2; plan §5 B2.0
§1 and §10.3, "The proposal").

Nothing runs and nothing is stored. The proposal is the server's view of
the run the person is about to sign; r4 recomputes everything it can
(``src_manifest``, the step graph, every step config) from the exported
version and refuses to sign on any difference, so the server never decides
what the person's code is.

What the server fills in:

- a ``run_id`` the server issues (HMAC under a server-only key of the
  principal, nonce and ``issued``; the trigger recomputes it), ``issued`` =
  now, ``expires`` = now + 3600, ``max_run_seconds`` = min(604800,
  Σ ``time`` + steps × ``queue_seconds``), a fresh ``nonce``;
- ``params``: the body's params plus the platform-injected ``project_id``
  (the verified project) and ``input_files``; nothing else;
- names, ``after``, ``chained`` and each config from the gate's shared
  ``step_graph`` / ``expected_step_config`` over the registered version's
  ``workflow.json``;
- each step's image (``step_executor.image_digests`` by partition arch),
  ONE partition and its resources. Every step's partition has the trusted
  image's arch (``mixed_arch`` otherwise: the trusted image runs the job's
  seccomp transform, pids probe and forwarder on each node);
- no endpoint: B2 proposes none, so a step the router would send to a model
  endpoint (it declares ``requires`` or ``model_type``) is refused here
  (``endpoint_not_supported``) before anything runs on the person's
  allocation (dated deviation from §10.3 step 3, plan §5 B2.0, 2026-11-08);
- a step named ``intent`` is ``bad_step_name`` (the gate reserves it: its
  nonce file would be the intent's own record);
- inputs: entries the body lists become ``listed`` after the server checks
  their digest; every other object the step's stage plan names becomes
  ``server_listed`` with a digest the server computes by streaming it once
  (deduplicated, 300 s in all); parents become ``chained``;
- ``src_manifest``: ``workflow_export._walk_source`` of the registered tree.

Every refusal is :class:`~umrs.hpc_accounts.intents.IntentInvalid`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from umrs.hpc_accounts.intents import INTENT_MAX_BYTES, IntentInvalid, issued_run_id

INTENT_WINDOW = 3600
RUN_MAX_SECONDS = 604800
HASH_BUDGET_SECONDS = 300.0
_MIN_TIME = 60


@dataclass(frozen=True)
class ProposalRequest:
    """One proposal: the verified run and the site it would run at."""

    workflow: str
    version: str
    source_sha256: str
    project: str
    principal: str
    login: str
    account: str
    params: dict[str, Any]
    workflow_json: dict[str, Any]
    source_path: str
    site: Any
    profile: dict[str, Any]
    profile_sha256: str
    listed: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass
class _Hasher:
    """Streams objects once each, within one deadline for the whole proposal."""

    s3: Any
    deadline: float
    seen: dict[tuple[str, str], tuple[str, int]] = field(default_factory=dict)

    def digest(self, bucket: str, key: str) -> tuple[str, int]:
        if (bucket, key) not in self.seen:
            self.seen[(bucket, key)] = self._stream(bucket, key)
        return self.seen[(bucket, key)]

    def _stream(self, bucket: str, key: str) -> tuple[str, int]:
        try:
            return self._hash(bucket, key)
        except IntentInvalid:
            raise
        except Exception as exc:  # an object gone since listing, S3 down: by name
            raise IntentInvalid(
                "input_unreadable", f"s3://{bucket}/{key}: {type(exc).__name__}: {exc}"[:300]
            ) from exc

    def _hash(self, bucket: str, key: str) -> tuple[str, int]:
        if time.monotonic() > self.deadline:
            raise IntentInvalid("inputs_too_slow", "list the inputs yourself, or split the data")
        body = self.s3.get_object(Bucket=bucket, Key=key)["Body"]
        data = body if isinstance(body, (bytes, bytearray)) else None
        if data is not None:
            return hashlib.sha256(data).hexdigest(), len(data)
        digest, size = hashlib.sha256(), 0
        for chunk in iter(lambda: body.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
            if time.monotonic() > self.deadline:
                raise IntentInvalid("inputs_too_slow", f"s3://{bucket}/{key} did not hash in time")
        return digest.hexdigest(), size


# ── shared code (the gate file) ──────────────────────────────────────────────


def _gate() -> Any:
    from r4_contracts import hpc_gate

    return hpc_gate


def _graph(steps: list[dict[str, Any]]) -> list[tuple[str, list[str], list[str]]]:
    try:
        graph = [(str(n), list(a), list(c)) for n, a, c in _gate().step_graph(steps)]
    except (ValueError, KeyError) as exc:
        raise IntentInvalid("bad_step_name", str(exc)[:200]) from exc
    reserved = _gate().RESERVED_STEP_NAME
    if any(name == reserved for name, _a, _c in graph):
        raise IntentInvalid("bad_step_name", f"the step name {reserved!r} is reserved by the gate")
    return graph


def check_no_endpoints(steps: list[dict[str, Any]]) -> None:
    """B2 signs no endpoint: a step the router would send to a model endpoint
    (capability routing: ``requires`` or ``model_type``) cannot run gated
    yet; refused before any step runs on the person's allocation."""
    for step in steps:
        if step.get("requires") or step.get("model_type"):
            raise IntentInvalid(
                "endpoint_not_supported",
                f"step {step.get('name')!r} reads a model endpoint; runs under your own "
                "account cannot reach one yet",
            )


def check_one_arch(profile: dict[str, Any], steps: list[dict[str, Any]], arch: str) -> None:
    """Every step's partition has the trusted image's *arch*."""
    for step in steps:
        part_arch = profile["partitions"][step["partition"]]["arch"]
        if part_arch != arch:
            raise IntentInvalid(
                "mixed_arch",
                f"step {step['name']} runs on {part_arch}, the run's trusted image on {arch}",
            )


def _config(step: dict[str, Any], name: str, params: dict[str, Any]) -> dict[str, Any]:
    config = _gate().expected_step_config(step, name, params)
    try:
        _gate().canonical_bytes(config, json_values=True)
    except ValueError as exc:
        raise IntentInvalid("non_json_value", f"steps.{name}.config: {exc}") from exc
    return config


# ── lanes, images, resources ─────────────────────────────────────────────────


def check_lanes(req: ProposalRequest) -> None:
    """Every step runs on the gated HPC lane: no manual gate, no step the
    run would keep in-process."""
    from umrs.server.step_trust import is_uploaded_step, workflow_has_uploaded_code

    steps = [s for s in req.workflow_json.get("steps") or [] if isinstance(s, dict)]
    uploaded = workflow_has_uploaded_code(steps, req.source_path)
    for step in steps:
        builtin = uploaded and not is_uploaded_step(
            str(step.get("entrypoint") or ""), req.source_path
        )
        if _is_manual(step) or builtin:
            raise IntentInvalid(
                "mixed_lanes", f"step {step.get('name')!r} would not run on the gated lane"
            )


def _is_manual(step: dict[str, Any]) -> bool:
    return bool(step.get("gate") or step.get("manual_gate"))


def load_image_digests(site: Any) -> dict[str, Any]:
    path = str(getattr(site.step_executor, "image_digests", "") or "")
    if not path:
        raise IntentInvalid("no_image_digests", "this site publishes no step image digests")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise IntentInvalid("no_image_digests", f"image digests unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise IntentInvalid("no_image_digests", "image digests are not a mapping")
    return data


def _image(digests: dict[str, Any], ref: str, arch: str) -> dict[str, str]:
    entry = (digests.get(ref) or {}).get(arch) if isinstance(digests.get(ref), dict) else None
    if not isinstance(entry, dict):
        raise IntentInvalid("no_image_digests", f"no digests for {ref} on {arch}")
    return {"ref": ref, "image_id": str(entry.get("image_id") or ""), "archive_sha256": str(entry.get("archive_sha256") or "")}


def platform_image(site: Any) -> str:
    image = os.environ.get("UMRS_STEP_IMAGE", "").strip() or str(site.step_executor.image or "")
    if not image:
        raise IntentInvalid("no_image_digests", "the lane has no platform step image")
    return image


def _hints(step: dict[str, Any]) -> dict[str, Any]:
    hints = step.get("routing_hints")
    return hints if isinstance(hints, dict) else {}


def _needs_gpu(step: dict[str, Any], profile: dict[str, Any]) -> bool:
    hint = _hints(step).get("gpu")
    wants = hint is True or str(hint).strip().lower() in ("1", "true", "yes")
    return wants and bool(profile["gpu"]["enabled"])


def _partition(profile: dict[str, Any], gpus: int) -> str:
    for name, part in profile["partitions"].items():
        if int(part["max_gpus"]) >= gpus:
            return name
    raise IntentInvalid("resource_not_allowed", f"no partition of the profile gives {gpus} GPUs")


def _time(step: dict[str, Any], part: dict[str, Any]) -> int:
    limit = int(part["max_time"])
    timeout = step.get("timeout_seconds")
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
        limit = min(limit, int(timeout))
    return max(_MIN_TIME, limit)


def _resources(req: ProposalRequest, step: dict[str, Any]) -> dict[str, Any]:
    profile = req.profile
    gpus = int(_hints(step).get("gpu_count") or 1) if _needs_gpu(step, profile) else 0
    name = _partition(profile, gpus)
    part = profile["partitions"][name]
    memory = str(req.site.step_executor.memory or "") if part["mem"] else ""
    return {
        "partition": name,
        "cpus": max(1, min(int(req.site.step_executor.cpus), int(part["max_cpus"]))),
        "mem": memory,
        "gpus": gpus,
        "time": _time(step, part),
    }


# ── inputs ───────────────────────────────────────────────────────────────────


def _buckets(params: dict[str, Any]) -> tuple[str, str]:
    bucket = str(params.get("bucket") or "wf-runner")
    return bucket, bucket + "-output"


def _plan_objects(req: ProposalRequest, step: dict[str, Any], name: str, s3: Any) -> list[Any]:
    """The objects the step's stage plan names now (params and prefixes)."""
    from umrs.pipeline_sdk.step import StepContext
    from umrs.server.step_staging import (
        STAGE_REQUEST_METADATA_KEY,
        build_stage_plan,
        stage_request,
    )

    if not step.get("stage_inputs"):
        return []
    input_bucket, output_bucket = _buckets(req.params)
    request = stage_request({**step, "name": name}, req.params, req.principal)
    ctx = StepContext(
        s3=s3,
        input_bucket=input_bucket,
        input_keys=[],
        output_bucket=output_bucket,
        config={},
        metadata={STAGE_REQUEST_METADATA_KEY: request},
        run_id="proposal",
    )
    return list(build_stage_plan(ctx).objects)


def _listed(req: ProposalRequest, name: str, hasher: _Hasher, planned: set[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for entry in req.listed.get(name) or []:
        bucket, key = str(entry.get("bucket") or ""), str(entry.get("key") or "")
        if (bucket, key) not in planned:
            raise IntentInvalid("input_not_in_plan", f"s3://{bucket}/{key} is not in step {name}'s stage plan")
        sha, size = hasher.digest(bucket, key)
        if (sha, size) != (entry.get("sha256"), entry.get("size")):
            raise IntentInvalid("input_changed", f"s3://{bucket}/{key} differs from the digest you listed")
        out.append({"mode": "listed", "bucket": bucket, "key": key, "sha256": sha, "size": size})
    return out


def _inputs(req: ProposalRequest, step: dict[str, Any], name: str, chained: list[str], hasher: _Hasher) -> dict[str, Any]:
    planned = {(o.bucket, o.key) for o in _plan_objects(req, step, name, hasher.s3)}
    listed = _listed(req, name, hasher, planned)
    named = {(e["bucket"], e["key"]) for e in listed}
    server = [
        {"mode": "server_listed", "bucket": b, "key": k, "sha256": sha, "size": size}
        for b, k in sorted(planned - named)
        for sha, size in [hasher.digest(b, k)]
    ]
    return {
        "objects": sorted(listed + server, key=lambda e: (e["bucket"], e["key"])),
        "prefixes": _prefixes(step, _buckets(req.params)[0]),
        "chained": chained,
    }


def _prefixes(step: dict[str, Any], input_bucket: str) -> list[str]:
    """``<bucket>/<prefix>`` of each declared prefix, as the stager reads it."""
    from umrs.pipeline_sdk.staged_s3 import parse_stage_inputs, split_s3_ref

    declared = parse_stage_inputs(step.get("stage_inputs")).get("prefixes", [])
    refs = [split_s3_ref(p, input_bucket, allow_empty_key=True) for p in declared]
    return sorted({f"{bucket}/{key}" for bucket, key in refs})


# ── source ───────────────────────────────────────────────────────────────────


def _is_utf8(path: str) -> bool:
    """A name the filesystem gave as undecodable bytes (surrogate escapes)
    is not valid UTF-8: it cannot be named in a manifest (plan §5 B2.0 §3)."""
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def src_manifest(source_path: str, steps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Exactly ``workflow_export._walk_source`` of the registered tree."""
    from umrs.api.workflow_export import _walk_source
    from umrs.pipeline_sdk.staged_s3 import normalize_key

    files = []
    for rel, full in _walk_source(source_path).files:
        if not _is_utf8(rel) or normalize_key(rel) != rel:
            raise IntentInvalid("bad_src_path", f"source path {rel!r} cannot be signed")
        with open(full, "rb") as handle:
            data = handle.read()
        files.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    files.sort(key=lambda f: f["path"].encode("utf-8"))
    entrypoints = {name: str(step.get("entrypoint") or "") for name, step in steps.items()}
    return {"files": files, "entrypoints": entrypoints}


# ── the intent ───────────────────────────────────────────────────────────────


def _step_object(req: ProposalRequest, raw: dict[str, Any], graph: tuple[str, list[str], list[str]], images: tuple[dict[str, Any], str], hasher: _Hasher) -> dict[str, Any]:
    name, after, chained = graph
    digests, ref = images
    res = _resources(req, raw)
    arch = req.profile["partitions"][res["partition"]]["arch"]
    image_ref = str(_hints(raw).get("image") or ref)
    return {
        "name": name,
        "after": after,
        "config": _config(raw, name, req.params),
        "image": _image(digests, image_ref, arch),
        **res,
        "inputs": _inputs(req, raw, name, chained, hasher),
    }


def _max_run(steps: list[dict[str, Any]], queue_seconds: int) -> int:
    total = sum(int(s["time"]) for s in steps) + len(steps) * int(queue_seconds)
    return max(_MIN_TIME, min(RUN_MAX_SECONDS, total))


def build_intent(req: ProposalRequest, s3: Any, *, now: int | None = None) -> dict[str, Any]:
    """The intent body for *req* (validated by the gate's own schema)."""
    check_lanes(req)
    raw_steps = [s for s in req.workflow_json.get("steps") or [] if isinstance(s, dict)]
    check_no_endpoints(raw_steps)
    graph = _graph(raw_steps)
    digests, ref = load_image_digests(req.site), platform_image(req.site)
    hasher = _Hasher(s3, time.monotonic() + HASH_BUDGET_SECONDS)
    steps = [_step_object(req, raw, g, (digests, ref), hasher) for raw, g in zip(raw_steps, graph)]
    issued = int(time.time() if now is None else now)
    input_bucket, output_bucket = _buckets(req.params)
    first_arch = req.profile["partitions"][steps[0]["partition"]]["arch"]
    check_one_arch(req.profile, steps, first_arch)
    body = {
        "v": 1, "type": "intent", "site": str(req.site.name), "principal": req.principal,
        "login": req.login, "account": req.account, "project": req.project,
        "workflow": req.workflow, "version": req.version, "source_sha256": req.source_sha256,
        "run_id": "", "input_bucket": input_bucket,
        "output_bucket": output_bucket, "params": dict(req.params),
        "site_profile_sha256": req.profile_sha256,
        "trusted_image": _image(digests, ref, first_arch),
        "src_manifest": src_manifest(req.source_path, {g[0]: r for r, g in zip(raw_steps, graph)}),
        "steps": steps, "nonce": secrets.token_hex(16), "issued": issued,
        "expires": issued + INTENT_WINDOW,
        "max_run_seconds": _max_run(steps, req.profile["slurm"]["queue_seconds"]),
    }
    body["run_id"] = issued_run_id(req.principal, body["nonce"], issued)
    return body


def encode_intent(body: dict[str, Any]) -> tuple[bytes, str]:
    """``(payload bytes, intent_id)``; refuses what the gate would refuse."""
    gate = _gate()
    try:
        payload = gate.canonical_bytes(body, json_values=True)
        gate.validate_intent(gate.parse_strict(payload, json_values=True))
    except (ValueError, gate.GateError) as exc:
        raise IntentInvalid("bad_schema", str(exc)[:300]) from exc
    if len(payload) > INTENT_MAX_BYTES:
        raise IntentInvalid("too_large", f"the intent exceeds {INTENT_MAX_BYTES} bytes")
    return payload, gate.intent_id(payload)


def summary(body: dict[str, Any]) -> dict[str, Any]:
    """Advisory (r4 ignores it): what a person reads first."""
    return {
        "run_id": body["run_id"],
        "site": body["site"],
        "login": body["login"],
        "account": body["account"],
        "steps": [
            {"name": s["name"], "partition": s["partition"], "gpus": s["gpus"], "time": s["time"]}
            for s in body["steps"]
        ],
        "expires": body["expires"],
    }


def proposal_answer(body: dict[str, Any]) -> dict[str, Any]:
    payload, intent_id = encode_intent(body)
    return {
        "intent_b64": base64.b64encode(payload).decode("ascii"),
        "intent_id": intent_id,
        "site_profile_sha256": body["site_profile_sha256"],
        "summary": summary(body),
    }


# ── the route's work (api/workflows.py stays thin) ───────────────────────────


def require_current_gate(identity: Any, profile_sha: str) -> None:
    """A live ``hello``: the installed gate is this server's release and its
    site profile the one this server publishes (``hpcaccess.gate_version``)."""
    from r4_contracts import hpc_gate

    from umrs.server.gated_step_lane import _checked, transport_for

    transport = transport_for(identity.site, identity.principal, identity.login)
    hello = _live_hello(_checked, transport)
    if hello.get("gate_sha256") != hpc_gate.gate_sha256():
        raise GateStale("gate_stale", "the installed gate is not this server's release: run 'r4 hpc add' again")
    if hello.get("site_profile_sha256") != profile_sha:
        raise GateStale("site_profile_stale", "your gate's site profile is not the one this site publishes: run 'r4 hpc profile'")


def _live_hello(checked: Any, transport: Any) -> dict[str, Any]:
    """The gate's ``hello``; a gate that cannot be reached now (SSH 255, a
    timeout, a Slurm-side exit 5) is ``gate_refused`` with gate
    ``unreachable`` (retry later), never a claim that the run is invalid."""
    from umrs.server.gated_step_lane import GateFailure
    from umrs.server.step_executor import StepExecutionError

    try:
        return checked("hello", transport.call("hello"))
    except GateFailure:
        raise
    except StepExecutionError as exc:
        raise GateFailure("unreachable", f"{exc.message} (try again later)") from exc


class GateStale(ValueError):
    """``hpcaccess.gate_version`` with *reason* (gate_stale, site_profile_stale)."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def proposal_s3(params: dict[str, Any], project: str) -> Any:
    """The object store the run would read (as the runner picks it)."""
    import threading
    from types import SimpleNamespace

    from umrs.server.workflow_runner import _resolve_s3_client

    record = SimpleNamespace(params=dict(params), results={}, execution_id="proposal")
    return _resolve_s3_client(record, threading.Lock(), project)[0]


def propose(
    resolved: Any, workflow_json: dict[str, Any], source_path: str, identity: Any, inputs: Any
) -> dict[str, Any]:
    """The whole proposal for a resolved trigger whose identity is the
    person's own account (plan §10.3 steps 2-4)."""
    from umrs.hpc_accounts.site_profile import published
    from umrs.shared.hpc_user_work_access import current_site

    site = current_site()
    profile, _data, sha = published(site)
    require_current_gate(identity, sha)
    req = ProposalRequest(
        workflow=resolved.name,
        version=str(resolved.version or ""),
        source_sha256=str(resolved.entry.get("source_sha256") or ""),
        project=str(resolved.verified_project or ""),
        principal=resolved.user_id,
        login=identity.login,
        account=str(identity.slurm_account or ""),
        params=dict(resolved.params),
        workflow_json=workflow_json,
        source_path=source_path,
        site=site,
        profile=profile,
        profile_sha256=sha,
        listed=dict(inputs or {}),
    )
    body = build_intent(req, proposal_s3(req.params, req.project))
    return proposal_answer(body)


__all__ = [
    "GateStale",
    "ProposalRequest",
    "build_intent",
    "check_lanes",
    "encode_intent",
    "load_image_digests",
    "proposal_answer",
    "propose",
    "src_manifest",
]
