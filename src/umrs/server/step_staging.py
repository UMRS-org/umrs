"""Server-staged step data plane for the connected-HPC lane (ADR-074 D-5).

The step gets files only. The server keeps every S3 operation and moves data
through the step's workdir on the cluster's shared filesystem, over the
existing SSH path (the login-node helper, ``umrs_step_helper.py``):

1. :func:`build_stage_plan` — what one step may read: its ``input_keys``
   (the parents' recorded outputs, physical keys in the parent output
   bucket) plus the trigger inputs its ``stage_inputs`` declares. Every
   bucket must be the run owner's (the owner-scope rule ported from the
   parked ``wip/step-lane-2026-09-25`` ``step_credentials.py``) or the run's
   in-memory store; per-object, per-step and object-count caps apply.
2. :func:`stage_in` — one tar stream built member by member from the S3
   bodies (nothing is buffered whole), written to the helper's ``stage-in``
   stdin. Members: ``in/manifest.json``, ``in/<bucket>/<key>`` per object and,
   when asked, ``src/<rel>`` for the uploaded source tree. No directory,
   link or device member is ever written; the extractor creates parents.
3. :func:`collect_out` — reads the helper's ``collect-out`` tar stream
   (``tarfile`` mode ``r|``), accepts only regular files named
   ``<output_bucket>/<key>`` under caps, and uploads each one to
   ``runs/<run_id>/<step>/<key>`` in the run's output bucket. Outputs bound
   for the run's in-memory store live in server memory, so they share one
   small whole-run budget and are stored in one batch.
4. :func:`rewrite_output_keys` — the reply's ``output_keys`` must each be a
   collected file; they become the physical keys children stage.

The server never follows a step-controlled path and never trusts a tar
header further than its caps.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import re
import stat
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from umrs.pipeline_sdk.staged_s3 import (
    MANIFEST_NAME,
    check_bucket_name,
    normalize_key,
    parse_stage_inputs,
    split_s3_ref,
)
from umrs.pipeline_sdk.step import StepContext, StepResult

from .step_executor import StepExecutionError

logger = logging.getLogger(__name__)

#: ``ctx.metadata`` key under which the runner hands the executor the step's
#: staging request (see :func:`stage_request`).
STAGE_REQUEST_METADATA_KEY = "_umrs_stage_request"

#: Synthetic buckets of the runner's in-memory S3 shim.
IN_MEMORY_BUCKET = "wf-runner"
_IN_MEMORY_BUCKETS = frozenset({IN_MEMORY_BUCKET, IN_MEMORY_BUCKET + "-output"})

GiB = 1024**3
MiB = 1024**2

_HASHED_DIR = ".umrs-hashed"
# A natural path segment starts with [A-Za-z0-9_]; the hashed fallback dir
# starts with "." so no staged key can land on it.
_SEGMENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._@+=,-]{0,254}")
_KEY_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]")
_OWNED_REMAINDER_RE = re.compile(r"[a-z0-9-]+\.[a-z0-9-]+")
_META_TYPES = (
    tarfile.GNUTYPE_LONGNAME,
    tarfile.GNUTYPE_LONGLINK,
    tarfile.XHDTYPE,
    tarfile.XGLTYPE,
    tarfile.SOLARIS_XHDTYPE,
)
_MAX_META_BYTES = 64 * 1024
_COPY_BUFSIZE = MiB
_MEMBER_KINDS = {
    tarfile.SYMTYPE: "a symlink",
    tarfile.LNKTYPE: "a hard link",
    tarfile.CHRTYPE: "a character device",
    tarfile.BLKTYPE: "a block device",
    tarfile.FIFOTYPE: "a FIFO",
    tarfile.CONTTYPE: "a contiguous file",
}


@dataclass(frozen=True)
class StageCaps:
    """Staging limits (spec §5; the defaults are the maintainer defaults)."""

    max_object_bytes: int = 8 * GiB
    max_in_bytes: int = 32 * GiB
    max_in_objects: int = 20_000
    max_src_bytes: int = 1 * GiB
    max_src_files: int = 20_000
    max_out_files: int = 20_000
    max_out_bytes: int = 32 * GiB
    max_out_file_bytes: int = 8 * GiB
    min_free_bytes: int = 200 * GiB
    #: The run's in-memory store lives in server memory: every blob of the
    #: run counts against this whole-run budget, and a step adds at most
    #: ``max_buffered_files`` files to it. Bulk outputs need a bucket param.
    max_buffered_bytes: int = 32 * MiB
    max_buffered_files: int = 64


DEFAULT_CAPS = StageCaps()


class StageStreamError(StepExecutionError):
    """The tar stream itself broke (the other side usually says why)."""


class OutputUploadError(StepExecutionError):
    """Storing a collected output failed; the outputs on the cluster are
    still the only copy."""


@dataclass(frozen=True)
class StagedObject:
    bucket: str
    key: str
    path: str
    size: int
    etag: str = ""
    content_type: str = ""


@dataclass(frozen=True)
class StagePlan:
    """What one step reads, and where each object lands under ``/work/in``."""

    run_id: str
    step: str
    output_bucket: str
    objects: tuple[StagedObject, ...] = ()
    prefixes: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(obj.size for obj in self.objects)

    def manifest(self) -> dict[str, Any]:
        return {
            "objects": {
                f"{obj.bucket}/{obj.key}": {
                    "path": obj.path,
                    "size": obj.size,
                    "etag": obj.etag,
                    "content_type": obj.content_type,
                }
                for obj in self.objects
            },
            "prefixes": list(self.prefixes),
            "output_bucket": self.output_bucket,
        }

    def manifest_bytes(self) -> bytes:
        return json.dumps(self.manifest(), sort_keys=True).encode("utf-8")


# ── owner scope (ported from the parked step_credentials.py) ────────────────

_bucket_registry: Any = None


def register_bucket_registry(buckets_cfg: Any) -> None:
    """Publish (or clear) the server's live provisioned-bucket registry.

    A reference to the one live ``BucketsCfg`` (reloads replace its contents
    in place). Processes that never load user configs have none, and the
    strict name rule is then the only gate.
    """
    global _bucket_registry
    _bucket_registry = buckets_cfg


def _registered_owner(registry: Any, bucket: str) -> str | None:
    """``s3_username`` a provisioned bucket belongs to; an ``-output``
    bucket belongs to the owner of its input bucket."""
    entry = registry.get(bucket)
    if entry is None and bucket.endswith("-output"):
        entry = registry.get(bucket[: -len("-output")])
    if entry is None:
        _, entry = registry.find_by_output_path(bucket)
    owner = getattr(entry, "s3_username", None)
    return owner if isinstance(owner, str) else None


def owner_bucket_prefix(owner: str) -> str:
    """Bucket-name prefix of *owner* (``make_path_to_bucket_string`` rules);
    a prefix match alone never proves ownership."""
    return owner.lower().replace(" ", "-").replace("_", "-") + "."


def _personal_owner(bucket: str) -> str | None:
    """Owner recorded for a self-service personal bucket (ADR-079)."""
    from umrs.shared.storage.user_buckets import owner_of

    return owner_of(bucket)


def _personal_shared(bucket: str, owner: str) -> bool:
    """The bucket's ADR-080 access list admits *owner* (object read/write);
    fail-closed — a lookup error or an old store grants nothing."""
    from umrs.shared.storage.user_buckets import get_user_bucket_store

    can_access = getattr(get_user_bucket_store(), "can_access", None)
    if can_access is None:
        return False
    try:
        return bool(can_access(bucket, owner))
    except Exception:
        logger.warning("bucket access lookup failed for %s", bucket, exc_info=True)
        return False


def bucket_owned_by(bucket: str, owner: str) -> bool:
    """Whether *owner* owns *bucket* — the one rule the HPC lane and the
    files API share: a personal bucket (ADR-079) whose server-written
    ``user_buckets`` row names them, or whose ADR-080 access list admits
    them (shared workspace objects), or a provisioned bucket whose name
    carries their prefix AND whose ``BucketsCfg`` entry names them. An
    ``-output`` name belongs to the owner of its input bucket."""
    if not owner:
        return False
    if _personal_owner(bucket) == owner:
        return True
    if _personal_shared(bucket, owner):
        return True
    prefix = owner_bucket_prefix(owner)
    remainder = bucket[len(prefix) :] if bucket.startswith(prefix) else ""
    if not _OWNED_REMAINDER_RE.fullmatch(remainder):
        return False
    registry = _bucket_registry
    return registry is not None and _registered_owner(registry, bucket) == owner


def _check_registered(bucket: str, owner: str) -> None:
    registry = _bucket_registry
    if registry is None:
        return
    registered = _registered_owner(registry, bucket)
    if registered == owner:
        return
    why = (
        "it is not a provisioned bucket"
        if registered is None
        else "the bucket registry records another owner"
    )
    raise StepExecutionError(
        "setup",
        f"bucket {bucket!r} does not belong to the run owner ({why}); a staged "
        "step reads and writes only its owner's buckets",
    )


def _check_owned(bucket: str, owner: str) -> None:
    if owner and _personal_owner(bucket) == owner:
        return  # ADR-079: the server-written ownership row is the proof
    if owner and _personal_shared(bucket, owner):
        return  # ADR-080: the bucket's access list admits the run owner
    prefix = owner_bucket_prefix(owner)
    remainder = bucket[len(prefix) :] if bucket.startswith(prefix) else ""
    if not _OWNED_REMAINDER_RE.fullmatch(remainder):
        raise StepExecutionError(
            "setup",
            f"bucket {bucket!r} does not belong to the run owner (expected "
            f"'{prefix}<project>.<alias>[-output]'); a staged step reads and "
            "writes only its owner's buckets",
        )
    _check_registered(bucket, owner)


def check_bucket_scope(bucket: str, owner: str, run_buckets: Iterable[str]) -> None:
    """Refuse *bucket* unless the run owner owns it or it is the run's own
    in-memory store (held by the server, invisible to anyone else)."""
    if bucket in _IN_MEMORY_BUCKETS:
        if bucket in set(run_buckets):
            return
        raise StepExecutionError(
            "setup",
            f"bucket {bucket!r} is the in-memory store of another run; a staged "
            "step reads only its own run's store",
        )
    _check_owned(bucket, owner)


# ── stage plan ──────────────────────────────────────────────────────────────


def stage_request(
    spec: Mapping[str, Any], params: Mapping[str, Any] | None, owner: str | None
) -> dict[str, Any]:
    """The staging request the runner puts into ``ctx.metadata`` under
    :data:`STAGE_REQUEST_METADATA_KEY`: the run owner, the step name, its
    declared ``stage_inputs`` and only the trigger params those name."""
    try:
        stage_inputs = parse_stage_inputs(spec.get("stage_inputs"))
    except ValueError as exc:
        raise StepExecutionError("setup", f"stage_inputs: {exc}") from exc
    params = params or {}
    named = {n: params[n] for n in stage_inputs.get("params", []) if n in params}
    return {
        "owner": owner or "",
        "step": str(spec.get("name") or ""),
        "stage_inputs": stage_inputs,
        "params": named,
    }


def _param_refs(name: str, value: Any, input_bucket: str) -> list[tuple[str, str]]:
    values = value if isinstance(value, list) else [value]
    refs = []
    for item in values:
        try:
            bucket, key = split_s3_ref(item, input_bucket)
        except ValueError as exc:
            raise StepExecutionError("setup", f"staged {name!r}: {exc}") from exc
        refs.append((str(bucket), key))
    return refs


def _check_listable(s3_client: Any, bucket: str) -> None:
    if not callable(getattr(s3_client, "list_objects_v2", None)):
        raise StepExecutionError(
            "setup",
            f"stage_inputs.prefixes needs a real bucket to list; {bucket!r} is the "
            "run's in-memory store (give the run a bucket param)",
        )


def _listed_objects(bucket: str, page: Mapping[str, Any]) -> list[StagedObject]:
    listed = []
    for item in page.get("Contents") or []:
        key = str(item.get("Key") or "")
        if key and not key.endswith("/"):
            size, etag = int(item.get("Size") or 0), str(item.get("ETag") or "")
            listed.append(StagedObject(bucket, key, "", size, etag))
    return listed


def _list_prefix(
    s3_client: Any, bucket: str, prefix: str, caps: StageCaps
) -> list[StagedObject]:
    """Every object under ``s3://bucket/prefix`` (folder markers skipped)."""
    _check_listable(s3_client, bucket)
    found: list[StagedObject] = []
    kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
    while True:
        try:
            page = s3_client.list_objects_v2(**kwargs)
        except Exception as exc:  # a missing or denied bucket refuses by name
            raise StepExecutionError(
                "setup",
                f"stage_inputs prefix s3://{bucket}/{prefix} is not listable: {exc}",
            ) from exc
        found.extend(_listed_objects(bucket, page))
        if len(found) > caps.max_in_objects:
            raise StepExecutionError(
                "setup",
                f"s3://{bucket}/{prefix} holds more than {caps.max_in_objects} "
                "objects; declare a narrower stage_inputs prefix",
            )
        if not page.get("IsTruncated"):
            return found
        kwargs["ContinuationToken"] = page["NextContinuationToken"]


def _head(s3_client: Any, bucket: str, key: str) -> StagedObject:
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # any lookup failure refuses the input by name
        raise StepExecutionError(
            "setup", f"staged input s3://{bucket}/{key} is not readable: {exc}"
        ) from exc
    return StagedObject(
        bucket,
        key,
        "",
        int(head.get("ContentLength") or 0),
        str(head.get("ETag") or ""),
        str(head.get("ContentType") or ""),
    )


class _PathAllocator:
    """``<bucket>/<key>`` under ``/work/in`` when the key is a clean path
    that fits; else ``<bucket>/.umrs-hashed/<sha256(key)>``."""

    def __init__(self) -> None:
        self._files: set[str] = {MANIFEST_NAME}
        self._dirs: set[str] = set()

    @staticmethod
    def _parents(path: str) -> list[str]:
        parts = path.split("/")
        return ["/".join(parts[:i]) for i in range(1, len(parts))]

    def _fits(self, path: str) -> bool:
        if path in self._files or path in self._dirs:
            return False
        return not any(parent in self._files for parent in self._parents(path))

    def assign(self, bucket: str, key: str) -> str:
        segments = key.split("/")
        path = "/".join([bucket, *segments])
        if not all(_SEGMENT_RE.fullmatch(seg) for seg in segments) or not self._fits(
            path
        ):
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            path = f"{bucket}/{_HASHED_DIR}/{digest}"
        self._files.add(path)
        self._dirs.update(self._parents(path))
        return path


def _check_count(count: int, caps: StageCaps) -> None:
    if count > caps.max_in_objects:
        raise StepExecutionError(
            "setup", f"the step would stage {count} objects (cap {caps.max_in_objects})"
        )


def _check_totals(objects: list[StagedObject], caps: StageCaps) -> None:
    _check_count(len(objects), caps)
    for obj in objects:
        if obj.size > caps.max_object_bytes:
            raise StepExecutionError(
                "setup",
                f"staged input s3://{obj.bucket}/{obj.key} is {obj.size} bytes; the "
                f"per-object cap is {caps.max_object_bytes}",
            )
    total = sum(obj.size for obj in objects)
    if total > caps.max_in_bytes:
        raise StepExecutionError(
            "setup",
            f"the step would stage {total} bytes; the per-step cap is {caps.max_in_bytes}",
        )


def _request(ctx: StepContext, request: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = (
        request if request is not None else ctx.metadata.get(STAGE_REQUEST_METADATA_KEY)
    )
    req = dict(raw or {})
    if not req.get("owner"):
        raise StepExecutionError(
            "setup",
            "a staged step needs the run owner (none was passed); refusing to "
            "stage without an owner scope",
        )
    if not req.get("step"):
        raise StepExecutionError("setup", "a staged step needs its step name")
    try:
        req["stage_inputs"] = parse_stage_inputs(req.get("stage_inputs"))
    except ValueError as exc:
        raise StepExecutionError("setup", f"stage_inputs: {exc}") from exc
    return req


def _declared_refs(
    ctx: StepContext, req: Mapping[str, Any]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(object refs, prefix refs)`` the step may read."""
    parent_bucket = ctx.parent_output_bucket or ctx.output_bucket
    refs = [_param_refs("input key", key, parent_bucket)[0] for key in ctx.input_keys]
    params = req.get("params") or {}
    for name in req["stage_inputs"].get("params", []):
        if params.get(name) is not None:
            refs.extend(_param_refs(f"param {name}", params[name], ctx.input_bucket))
    prefixes = []
    for prefix in req["stage_inputs"].get("prefixes", []):
        bucket, key = split_s3_ref(prefix, ctx.input_bucket, allow_empty_key=True)
        prefixes.append((str(bucket), key))
    return list(dict.fromkeys(refs)), prefixes


def _check_scopes(ctx: StepContext, owner: str, buckets: list[str]) -> None:
    run_buckets = [
        b for b in (ctx.input_bucket, ctx.output_bucket, ctx.parent_output_bucket) if b
    ]
    for bucket in dict.fromkeys([ctx.output_bucket, *buckets]):
        _bucket_name(bucket)
        check_bucket_scope(bucket, owner, run_buckets)


def _sized_objects(
    s3_client: Any,
    refs: list[tuple[str, str]],
    prefixes: list[tuple[str, str]],
    caps: StageCaps,
) -> list[StagedObject]:
    objects = {(b, k): _head(s3_client, b, k) for b, k in refs}
    for bucket, prefix in prefixes:
        for obj in _list_prefix(s3_client, bucket, prefix, caps):
            objects.setdefault((obj.bucket, obj.key), obj)
    return list(objects.values())


def build_stage_plan(
    ctx: StepContext,
    s3_client: Any = None,
    *,
    request: Mapping[str, Any] | None = None,
    caps: StageCaps = DEFAULT_CAPS,
) -> StagePlan:
    """Scope-check, size and place every object the step may read.

    *request* defaults to ``ctx.metadata[STAGE_REQUEST_METADATA_KEY]``;
    *s3_client* to ``ctx.s3`` (the server's client for this run).
    """
    s3_client = s3_client if s3_client is not None else ctx.s3
    req = _request(ctx, request)
    refs, prefixes = _declared_refs(ctx, req)
    _check_count(len(refs), caps)
    _check_scopes(ctx, str(req["owner"]), [b for b, _ in refs + prefixes])
    staged = _sized_objects(s3_client, refs, prefixes, caps)
    _check_totals(staged, caps)
    paths = _PathAllocator()
    return StagePlan(
        run_id=ctx.run_id,
        step=str(req["step"]),
        output_bucket=ctx.output_bucket,
        objects=tuple(
            dataclasses.replace(obj, path=paths.assign(obj.bucket, obj.key))
            for obj in staged
        ),
        prefixes=tuple(f"{b}/{p}" for b, p in prefixes),
    )


def _bucket_name(bucket: str) -> None:
    try:
        check_bucket_name(bucket)
    except ValueError as exc:
        raise StepExecutionError("setup", f"staging refused: {exc}") from exc


def check_free_space(free_bytes: int, caps: StageCaps = DEFAULT_CAPS) -> None:
    """Refuse to stage when the workdir filesystem is nearly full."""
    if int(free_bytes) < caps.min_free_bytes:
        raise StepExecutionError(
            "setup",
            f"the step workdir filesystem has {int(free_bytes)} bytes free; staging "
            f"needs at least {caps.min_free_bytes}",
        )


def verify_stage_summary(
    sent: Mapping[str, int], summary: Mapping[str, Any], caps: StageCaps = DEFAULT_CAPS
) -> None:
    """The helper's ``stage-in`` JSON summary must match what was sent
    (``files`` / ``bytes`` from :func:`stage_in`) and leave enough room."""
    for field_name in ("files", "bytes"):
        if summary.get(field_name) != sent[field_name]:
            raise StepExecutionError(
                "setup",
                f"stage-in wrote {summary.get(field_name)!r} {field_name}, "
                f"the server sent {sent[field_name]}",
            )
    if summary.get("free_bytes") is not None:
        check_free_space(int(summary["free_bytes"]), caps)


# ── stage-in stream ─────────────────────────────────────────────────────────


class _ExactBody:
    """Reads exactly *size* bytes of an S3 body, in chunks, or fails by name."""

    def __init__(self, body: Any, size: int, label: str) -> None:
        self._body, self._remaining, self._label = body, size, label

    def read(self, n: int = -1) -> bytes:
        want = self._remaining if n is None or n < 0 else min(n, self._remaining)
        chunks, got = [], 0
        while got < want:
            chunk = self._read_body(want - got)
            if not chunk:
                raise StepExecutionError(
                    "setup", f"{self._label} ended early (changed while staging?)"
                )
            chunks.append(chunk)
            got += len(chunk)
        self._remaining -= got
        return b"".join(chunks)

    def _read_body(self, n: int) -> bytes:
        try:
            return self._body.read(n)
        except Exception as exc:  # a mid-stream S3 error is not an OSError
            raise StepExecutionError(
                "setup", f"{self._label} could not be read: {exc}"
            ) from exc

    def assert_drained(self) -> None:
        if self._read_body(1):
            raise StepExecutionError(
                "setup", f"{self._label} grew while staging; refusing a torn copy"
            )


def _open_body(obj: Mapping[str, Any]) -> Any:
    body = obj["Body"]
    if isinstance(body, (bytes, bytearray)):
        return io.BytesIO(bytes(body))
    try:
        body.read(0)
    except TypeError:
        # The runner's in-memory store: the bytes are already in memory.
        return io.BytesIO(body.read())
    return body


def _file_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o600, 0
    return info


def _add_object(tar: tarfile.TarFile, obj: StagedObject, s3_client: Any) -> None:
    label = f"staged input s3://{obj.bucket}/{obj.key}"
    try:
        response = s3_client.get_object(Bucket=obj.bucket, Key=obj.key)
    except Exception as exc:  # any fetch failure fails the step by name
        raise StepExecutionError("setup", f"{label} could not be read: {exc}") from exc
    length = response.get("ContentLength")
    if length is not None and int(length) != obj.size:
        raise StepExecutionError("setup", f"{label} changed size since it was planned")
    body = _open_body(response)
    try:
        reader = _ExactBody(body, obj.size, label)
        tar.addfile(_file_info(f"in/{obj.path}", obj.size), reader)
        reader.assert_drained()
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()


def _source_files(source_dir: str, caps: StageCaps) -> list[tuple[str, str, int]]:
    """``(abs path, member name, size)`` of every regular file; links refused.
    The walk stops at the first file past a cap."""
    files, total = [], 0
    for root, dirs, names in os.walk(source_dir, followlinks=False):
        for name in sorted(dirs + names):
            path = os.path.join(root, name)
            st = os.lstat(path)
            if stat.S_ISDIR(st.st_mode):
                continue
            if not stat.S_ISREG(st.st_mode):
                raise StepExecutionError(
                    "setup",
                    f"workflow source {path!r} is not a regular file (links refused)",
                )
            total += st.st_size
            files.append((path, _source_member(path, source_dir), st.st_size))
            _check_source_caps(len(files), total, caps)
    return files


def _check_source_caps(count: int, total: int, caps: StageCaps) -> None:
    if count > caps.max_src_files or total > caps.max_src_bytes:
        raise StepExecutionError(
            "setup",
            f"workflow source passes its caps ({caps.max_src_files} files / "
            f"{caps.max_src_bytes} bytes) at {count} files / {total} bytes",
        )


def inputs_tag(
    plan: StagePlan,
    payload: bytes,
    source_dir: str | None,
    caps: StageCaps = DEFAULT_CAPS,
    image: str = "",
) -> str:
    """12 hex digits of what a step's job computes from: the staged objects
    with their etags, the payload (entrypoint, config, parent results), the
    uploaded source tree (names, sizes, mtimes) and the step image (its tag
    names the umrs commit). An earlier COMPLETED job is adopted only when it
    ran on the current attempt's inputs (build spec §8): a re-dispatched run
    may have recomputed an upstream output, or run on a newer image."""
    digest = hashlib.sha256(b"umrs-step-inputs-1\0")
    digest.update(f"{image}\0".encode("utf-8", "surrogateescape"))
    digest.update(plan.manifest_bytes())
    digest.update(hashlib.sha256(payload).digest())
    for path, member, size in _source_files(source_dir, caps) if source_dir else ():
        mtime = os.lstat(path).st_mtime_ns
        digest.update(f"{member}\0{size}\0{mtime}\n".encode("utf-8", "surrogateescape"))
    return digest.hexdigest()[:12]


def _source_member(path: str, source_dir: str) -> str:
    try:
        return "src/" + normalize_key(os.path.relpath(path, source_dir))
    except ValueError as exc:
        raise StepExecutionError("setup", f"workflow source {path!r}: {exc}") from exc


def _add_source_tree(
    tar: tarfile.TarFile, source_dir: str, caps: StageCaps
) -> tuple[int, int]:
    count = total = 0
    for path, member, size in _source_files(source_dir, caps):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as fh:
            if os.fstat(fh.fileno()).st_size != size:
                raise StepExecutionError(
                    "setup", f"workflow source {path!r} changed while staging"
                )
            tar.addfile(_file_info(member, size), fh)
        count, total = count + 1, total + size
    return count, total


def stage_in(
    plan: StagePlan,
    s3_client: Any,
    write_stream: BinaryIO,
    *,
    source_dir: str | None = None,
    caps: StageCaps = DEFAULT_CAPS,
) -> dict[str, int]:
    """Stream the plan (and optionally the source tree) as one tar.

    Returns ``{"files", "bytes"}`` as sent — the count of regular-file
    members and the sum of their sizes, which the helper's summary must
    repeat (:func:`verify_stage_summary`). *write_stream* is left open (the
    caller closes the helper's stdin). On a StepExecutionError the stream
    ends without an end-of-archive marker; the caller must discard the
    workdir whatever the helper reports.
    """
    manifest = plan.manifest_bytes()
    files, total = 1 + len(plan.objects), len(manifest) + plan.total_bytes
    try:
        with tarfile.open(
            fileobj=write_stream, mode="w|", format=tarfile.PAX_FORMAT
        ) as tar:
            tar.copybufsize = _COPY_BUFSIZE
            tar.addfile(
                _file_info(f"in/{MANIFEST_NAME}", len(manifest)), io.BytesIO(manifest)
            )
            for obj in plan.objects:
                _add_object(tar, obj, s3_client)
            if source_dir:
                src_files, src_bytes = _add_source_tree(tar, source_dir, caps)
                files, total = files + src_files, total + src_bytes
    except OSError as exc:
        raise StageStreamError("setup", f"the stage-in stream broke: {exc}") from exc
    return {"files": files, "bytes": total}


# ── stage-out stream ────────────────────────────────────────────────────────


class _CappedTarInfo(tarfile.TarInfo):
    """Refuses metadata headers past a small cap before tarfile buffers them,
    and sparse members before tarfile expands their maps."""

    def _proc_member(self, tarfile_: tarfile.TarFile) -> Any:
        if self.type == tarfile.GNUTYPE_SPARSE:
            raise StepExecutionError(
                "worker", f"/work/out member {self.name!r} is sparse"
            )
        if self.type in _META_TYPES and self.size > _MAX_META_BYTES:
            raise StepExecutionError(
                "worker", f"/work/out tar metadata header of {self.size} bytes refused"
            )
        return super()._proc_member(tarfile_)


class _MemberReader:
    """A non-seekable view of one stream member (``upload_fileobj`` probes
    ``seekable()``, which tarfile's stream objects cannot answer)."""

    def __init__(self, fileobj: Any) -> None:
        self._fileobj = fileobj

    def read(self, n: int = -1) -> bytes:
        return self._fileobj.read(n)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def physical_prefix(run_id: str, step: str) -> str:
    """``runs/<run_id>/<step>/`` with both parts made safe key segments."""
    return f"runs/{_key_segment(run_id)}/{_key_segment(step)}/"


def _key_segment(name: str) -> str:
    if _KEY_SEGMENT_RE.fullmatch(name or "") and name not in (".", ".."):
        return name
    safe = _UNSAFE_CHARS_RE.sub("_", name or "")[:40] or "x"
    return f"{safe}-{hashlib.sha256((name or '').encode('utf-8')).hexdigest()[:8]}"


def _member_segments(name: str) -> list[str]:
    if name.startswith("/"):
        raise StepExecutionError(
            "worker", f"/work/out member {name!r} has an absolute name"
        )
    if not [seg for seg in name.split("/") if seg not in ("", ".")]:
        return []  # the archive root ("./")
    try:
        return normalize_key(name).split("/")
    except ValueError as exc:
        raise StepExecutionError("worker", f"/work/out member refused: {exc}") from exc


def out_limits(s3_client: Any, caps: StageCaps = DEFAULT_CAPS) -> tuple[bool, int, int]:
    """``(buffered, max files, max bytes)`` one collection into *s3_client*
    may add. A client without ``upload_fileobj`` is the run's in-memory store:
    buffered, and bounded by what is left of the run's budget."""
    if callable(getattr(s3_client, "upload_fileobj", None)):
        return False, caps.max_out_files, caps.max_out_bytes
    stored = getattr(s3_client, "stored_bytes", None)
    used = int(stored()) if callable(stored) else 0
    return True, caps.max_buffered_files, max(0, caps.max_buffered_bytes - used)


def buffered_hint(caps: StageCaps = DEFAULT_CAPS) -> str:
    return (
        " (the run has no bucket param, so its outputs live in server memory "
        f"under a {caps.max_buffered_bytes}-byte whole-run budget of at most "
        f"{caps.max_buffered_files} files per step; give the run a bucket param "
        "for bulk outputs)"
    )


@dataclass
class _Collected:
    dest_bucket: str
    caps: StageCaps
    buffered: bool
    max_files: int
    max_bytes: int
    files: int = 0
    dirs: int = 0
    bytes: int = 0
    mapping: dict[str, str] = field(default_factory=dict)
    #: Buffered outputs, stored in one batch once the stream is read whole.
    pending: dict[str, bytes] = field(default_factory=dict)

    def hint(self) -> str:
        return buffered_hint(self.caps) if self.buffered else ""

    def accept(self, member: tarfile.TarInfo) -> str | None:
        """The logical key of a file member; None for a directory."""
        if member.isdir():
            _member_segments(member.name)
            self.dirs += 1
            if self.dirs > self.max_files:
                raise StepExecutionError(
                    "worker", f"/work/out has too many directories{self.hint()}"
                )
            return None
        kind = _MEMBER_KINDS.get(member.type)
        if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.sparse:
            raise StepExecutionError(
                "worker",
                f"/work/out member {member.name!r} is {kind or 'not a regular file'}; "
                "only regular files are collected",
            )
        return self._file_key(member)

    def _file_key(self, member: tarfile.TarInfo) -> str:
        segments = _member_segments(member.name)
        if len(segments) < 2 or segments[0] != self.dest_bucket:
            raise StepExecutionError(
                "worker",
                f"/work/out member {member.name!r} is not under the output bucket "
                f"{self.dest_bucket!r}; write outputs through ctx.s3",
            )
        key = "/".join(segments[1:])
        if key in self.mapping:
            raise StepExecutionError(
                "worker", f"/work/out member {key!r} appears twice"
            )
        self._count(member.size, key)
        return key

    def _count(self, size: int, key: str) -> None:
        self.files += 1
        self.bytes += size
        if self.files > self.max_files:
            raise StepExecutionError(
                "worker", f"/work/out has more than {self.max_files} files{self.hint()}"
            )
        if size > self.caps.max_out_file_bytes:
            raise StepExecutionError(
                "worker",
                f"output {key!r} is {size} bytes (cap {self.caps.max_out_file_bytes})",
            )
        if self.bytes > self.max_bytes:
            raise StepExecutionError(
                "worker",
                f"/work/out holds more than {self.max_bytes} bytes; refusing the "
                f"rest{self.hint()}",
            )


def _upload(s3_client: Any, fileobj: Any, bucket: str, physical: str, key: str) -> None:
    try:
        s3_client.upload_fileobj(_MemberReader(fileobj), bucket, physical)
    except StepExecutionError:
        raise
    except Exception as exc:  # an upload failure fails the step by name
        raise OutputUploadError(
            "worker",
            f"uploading output {key!r} to s3://{bucket}/{physical} failed: {exc}",
        ) from exc


def _collect_members(
    tar: tarfile.TarFile, s3_client: Any, state: _Collected, prefix: str
) -> None:
    for member in tar:
        key = state.accept(member)
        if key is None:
            continue
        physical = prefix + key
        fileobj = tar.extractfile(member)
        if state.buffered:
            state.pending[physical] = fileobj.read()
        else:
            _upload(s3_client, fileobj, state.dest_bucket, physical, key)
        state.mapping[key] = physical


def _store_buffered(s3_client: Any, state: _Collected) -> None:
    """One batch into the in-memory store: its budget is re-checked there
    (a parallel step may have claimed it) and the run persists once."""
    if not state.pending:
        return
    put_many = getattr(s3_client, "put_objects", None)
    try:
        if callable(put_many):
            put_many(
                Bucket=state.dest_bucket,
                Objects=state.pending,
                MaxTotalBytes=state.caps.max_buffered_bytes,
            )
        else:
            for physical, body in state.pending.items():
                s3_client.put_object(Bucket=state.dest_bucket, Key=physical, Body=body)
    except ValueError as exc:  # the store's budget refusal
        raise StepExecutionError("worker", f"{exc}{state.hint()}") from exc
    except Exception as exc:  # a store failure fails the step by name
        raise OutputUploadError(
            "worker", f"storing outputs in s3://{state.dest_bucket} failed: {exc}"
        ) from exc


def collect_out(
    read_stream: BinaryIO,
    s3_client: Any,
    dest_bucket: str,
    run_id: str,
    step: str,
    caps: StageCaps = DEFAULT_CAPS,
) -> dict[str, str]:
    """Upload what the step wrote; ``{logical key: physical key}``.

    Only regular files named ``<dest_bucket>/<key>`` are accepted; a link,
    device, FIFO, sparse file, ``..`` or absolute name, a name outside the
    output bucket, a duplicate or anything past the caps fails the step.
    Outputs bound for the run's in-memory store are read whole first and
    stored in one batch, within what is left of the run's budget
    (:func:`out_limits`).
    """
    state = _Collected(dest_bucket, caps, *out_limits(s3_client, caps))
    try:
        with tarfile.open(
            fileobj=read_stream, mode="r|", tarinfo=_CappedTarInfo
        ) as tar:
            _collect_members(tar, s3_client, state, physical_prefix(run_id, step))
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise StageStreamError(
            "worker", f"the stage-out stream is not a valid tar: {exc}"
        ) from exc
    if state.buffered:
        _store_buffered(s3_client, state)
    return state.mapping


# ── key mapping ─────────────────────────────────────────────────────────────


def _physical_output_key(raw: Any, mapping: Mapping[str, str], dest_bucket: str) -> str:
    uri = isinstance(raw, str) and raw.startswith("s3://")
    try:
        bucket, key = split_s3_ref(raw, dest_bucket)
        logical = normalize_key(key)
    except ValueError as exc:
        raise StepExecutionError(
            "worker", f"output key {raw!r} refused: {exc}"
        ) from exc
    physical = mapping.get(logical) if bucket == dest_bucket else None
    if physical is None:
        raise StepExecutionError(
            "worker",
            f"output key {raw!r} names no file the step wrote to its output bucket "
            f"{dest_bucket!r}; a staged step returns only keys it wrote through ctx.s3",
        )
    return f"s3://{dest_bucket}/{physical}" if uri else physical


def rewrite_output_keys(
    result: StepResult, mapping: Mapping[str, str], dest_bucket: str
) -> StepResult:
    """*result* with each logical ``output_key`` replaced by its physical key."""
    return dataclasses.replace(
        result,
        output_keys=[
            _physical_output_key(raw, mapping, dest_bucket)
            for raw in result.output_keys or []
        ],
    )


__all__ = [
    "DEFAULT_CAPS",
    "STAGE_REQUEST_METADATA_KEY",
    "OutputUploadError",
    "StageCaps",
    "StagePlan",
    "StageStreamError",
    "StagedObject",
    "buffered_hint",
    "build_stage_plan",
    "check_bucket_scope",
    "check_free_space",
    "collect_out",
    "out_limits",
    "physical_prefix",
    "register_bucket_registry",
    "rewrite_output_keys",
    "stage_in",
    "stage_request",
    "verify_stage_summary",
]
