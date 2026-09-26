"""File-backed ``ctx.s3`` for steps on the staged HPC lane (ADR-074 D-5).

A step on the connected-HPC lane gets files only: no credential, no network.
The server stages the step's declared inputs into ``/work/in`` (with a
server-authored ``manifest.json``) before the job starts and collects
``/work/out`` after it ends; it keeps every S3 operation. Inside the
container, with ``UMRS_STEP_STAGED=1``, :class:`StagedS3Client` stands in for
the boto3 client the helpers (``download_bytes``, ``upload_bytes``,
``head_object`` …) and most raw callers use:

- ``get_object`` / ``head_object`` / ``download_file`` / ``download_fileobj``
  read ``/work/in`` through the manifest (and the step's own writes back);
- ``put_object`` / ``upload_file`` / ``upload_fileobj`` write
  ``/work/out/<bucket>/<key>`` for the step's output bucket only;
- ``list_objects_v2`` (and its paginator) answers only under a declared
  ``stage_inputs.prefixes`` entry;
- anything else (delete, copy, other buckets, undeclared keys) raises a named
  :class:`StagedS3Error`, so a denial never reads as success.

The shim is not the boundary — the container is; the server re-checks
everything it collects. Stdlib only: the step image carries no boto3.

The same module holds the pure key/bucket/``stage_inputs`` rules the manifest
parser (``workflow.py``) and the server stager (``umrs.server.step_staging``)
share, so the three sides normalise keys identically.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self

#: Env flag the staged job script sets inside the step container.
STAGED_ENV = "UMRS_STEP_STAGED"
STAGED_IN_DIR = "/work/in"
STAGED_OUT_DIR = "/work/out"
MANIFEST_NAME = "manifest.json"
DEFAULT_CONTENT_TYPE = "binary/octet-stream"

MAX_KEY_BYTES = 1024
_MAX_SEGMENT_BYTES = 255
_MAX_STAGE_INPUT_ENTRIES = 64
_MAX_PARAM_NAME = 128
_COPY_CHUNK = 1024 * 1024

_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_STAGE_INPUT_FIELDS = frozenset({"params", "prefixes"})


class StagedS3Error(RuntimeError):
    """A staged step asked for something the staged lane does not provide."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class StagedObjectNotStaged(StagedS3Error, KeyError):
    """The object was not staged for this step (boto3-style ``NoSuchKey``)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.response = {"Error": {"Code": "NoSuchKey", "Message": message}}


class StagedS3Unsupported(StagedS3Error, AttributeError):
    """An S3 operation outside the staged helper set (``hasattr`` is False)."""


# ── shared rules (manifest parser, server stager, shim) ─────────────────────


def is_staged() -> bool:
    """True inside a staged step container."""
    return os.environ.get(STAGED_ENV, "").strip() == "1"


def check_bucket_name(bucket: Any) -> str:
    """*bucket* if it is a plain S3 bucket name; ``ValueError`` otherwise."""
    if (
        not isinstance(bucket, str)
        or not _BUCKET_RE.fullmatch(bucket)
        or ".." in bucket
        or bucket == MANIFEST_NAME
    ):
        raise ValueError(f"not a valid bucket name: {bucket!r}")
    return bucket


def check_key_text(key: Any, *, allow_empty: bool = False) -> str:
    """*key* if it is a string S3 can hold as an object key or prefix."""
    if not isinstance(key, str) or (not key and not allow_empty):
        raise ValueError(f"not an object key: {key!r}")
    if _CONTROL_RE.search(key):
        raise ValueError(f"object key {key!r} contains control characters")
    try:
        encoded = key.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"object key {key!r} is not valid UTF-8") from exc
    if len(encoded) > MAX_KEY_BYTES:
        raise ValueError(f"object key is longer than {MAX_KEY_BYTES} bytes")
    return key


def normalize_key(key: Any) -> str:
    """The canonical form of a key a step writes (and of a collected name).

    Empty and ``.`` segments are dropped (so a leading ``/`` goes too); a
    ``..`` segment, control characters and over-long names are refused.
    """
    check_key_text(key)
    segments = [seg for seg in key.split("/") if seg not in ("", ".")]
    if ".." in segments:
        raise ValueError(f"object key {key!r} has a '..' segment")
    if not segments:
        raise ValueError(f"object key {key!r} names no object")
    if any(len(seg.encode("utf-8")) > _MAX_SEGMENT_BYTES for seg in segments):
        raise ValueError(f"object key {key!r} has a segment over 255 bytes")
    return "/".join(segments)


def split_s3_ref(
    ref: Any, default_bucket: str | None, *, allow_empty_key: bool = False
) -> tuple[str | None, str]:
    """``(bucket, key)`` of ``s3://bucket/key``, or of a bare key/prefix.

    A bare reference lives in *default_bucket* (``None`` while only the
    shape is being checked, as the manifest parser does).
    """
    if isinstance(ref, str) and ref.startswith("s3://"):
        bucket, _, key = ref[len("s3://") :].partition("/")
        check_bucket_name(bucket)
    else:
        bucket, key = default_bucket, ref
    return bucket, check_key_text(key, allow_empty=allow_empty_key)


def _string_list(raw: Any, what: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > _MAX_STAGE_INPUT_ENTRIES:
        raise ValueError(
            f"stage_inputs.{what} must be a list of at most "
            f"{_MAX_STAGE_INPUT_ENTRIES} strings"
        )
    if any(not isinstance(item, str) or not item for item in raw):
        raise ValueError(f"stage_inputs.{what} entries must be non-empty strings")
    return list(dict.fromkeys(raw))


def parse_stage_inputs(raw: Any) -> dict[str, list[str]]:
    """Validate a step's ``stage_inputs`` block; ``{}`` when absent.

    ``{"params": [<trigger param names holding s3:// URIs or keys>],
    "prefixes": [<s3://bucket/prefix, or a prefix in the run's input
    bucket>]}``. Raises ``ValueError`` naming the defect.
    """
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, dict):
        # ValueError, not TypeError: the upload path reports ValueErrors by name.
        raise ValueError("stage_inputs must be an object with params/prefixes")  # noqa: TRY004
    unknown = sorted(set(raw) - _STAGE_INPUT_FIELDS)
    if unknown:
        raise ValueError(f"stage_inputs has unknown field(s): {', '.join(unknown)}")
    parsed = {
        "params": _param_names(raw.get("params")),
        "prefixes": _prefixes(raw.get("prefixes")),
    }
    return {name: value for name, value in parsed.items() if value}


def _param_names(raw: Any) -> list[str]:
    names = _string_list(raw, "params")
    if any(len(name) > _MAX_PARAM_NAME for name in names):
        raise ValueError(
            f"stage_inputs.params names are at most {_MAX_PARAM_NAME} chars"
        )
    return names


def _prefixes(raw: Any) -> list[str]:
    prefixes = _string_list(raw, "prefixes")
    for prefix in prefixes:
        split_s3_ref(prefix, None, allow_empty_key=True)
    return prefixes


# ── the shim ────────────────────────────────────────────────────────────────


class _StagedBody:
    """The ``Body`` of a staged ``get_object`` (the StreamingBody subset)."""

    def __init__(self, fh: io.BufferedReader) -> None:
        self._fh = fh

    def read(self, amt: int | None = None) -> bytes:
        return self._fh.read() if amt is None else self._fh.read(amt)

    def iter_chunks(self, chunk_size: int = _COPY_CHUNK) -> Iterator[bytes]:
        while chunk := self._fh.read(chunk_size):
            yield chunk

    def iter_lines(self, chunk_size: int = _COPY_CHUNK, keepends: bool = False):
        del chunk_size
        for line in self._fh:
            yield line if keepends else line.rstrip(b"\r\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _ListPaginator:
    def __init__(self, client: StagedS3Client) -> None:
        self._client = client

    def paginate(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        kwargs.pop("PaginationConfig", None)
        while True:
            page = self._client.list_objects_v2(**kwargs)
            yield page
            if not page.get("IsTruncated"):
                return
            kwargs["ContinuationToken"] = page["NextContinuationToken"]


def _body_chunks(body: Any) -> Iterator[bytes]:
    """Chunks of a put body; the type is checked before anything is written."""
    if isinstance(body, str):
        body = body.encode("utf-8")
    if isinstance(body, (bytes, bytearray, memoryview)):
        return iter([bytes(body)])
    if not callable(getattr(body, "read", None)):
        raise StagedS3Error(
            f"put_object Body must be bytes, str or a readable file, got {type(body).__name__}"
        )
    return _read_chunks(body)


def _read_chunks(body: Any) -> Iterator[bytes]:
    while chunk := body.read(_COPY_CHUNK):
        yield chunk.encode("utf-8") if isinstance(chunk, str) else chunk


class StagedS3Client:
    """boto3-shaped S3 client backed by the staged ``/work/in`` and ``/work/out``."""

    def __init__(
        self,
        in_dir: str = STAGED_IN_DIR,
        out_dir: str = STAGED_OUT_DIR,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self._in_dir = in_dir
        self._out_dir = out_dir
        if manifest is None:
            manifest = self._load_manifest(in_dir)
        self._objects: dict[str, dict[str, Any]] = dict(manifest.get("objects") or {})
        self._prefixes = [
            (bucket, prefix)
            for bucket, _, prefix in (
                str(entry).partition("/") for entry in manifest.get("prefixes") or []
            )
        ]
        self.output_bucket = str(manifest.get("output_bucket") or "")

    @staticmethod
    def _load_manifest(in_dir: str) -> dict[str, Any]:
        path = os.path.join(in_dir, MANIFEST_NAME)
        try:
            with open(path, "rb") as fh:
                manifest = json.load(fh)
        except (OSError, ValueError) as exc:
            raise StagedS3Error(
                f"staged step: {path} is missing or unreadable ({exc}); the server "
                "writes it with every staged step"
            ) from exc
        if not isinstance(manifest, dict):
            raise StagedS3Error(f"staged step: {path} is not a JSON object")
        return manifest

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise _unsupported(name)

    # ── reads ───────────────────────────────────────────────────────────────

    def _staged_entry(self, bucket: str, key: str) -> dict[str, Any] | None:
        entry = self._objects.get(f"{bucket}/{key}")
        if entry is None:
            return None
        path = str(entry.get("path") or "")
        try:
            clean = normalize_key(path) == path
        except ValueError:
            clean = False
        if not clean:
            raise StagedS3Error(
                f"staged manifest entry for {bucket}/{key} is malformed"
            )
        return {**entry, "file": os.path.join(self._in_dir, path)}

    def _own_output(self, bucket: str, key: str) -> dict[str, Any] | None:
        if not bucket or bucket != self.output_bucket:
            return None
        try:
            path = os.path.join(self._out_dir, bucket, normalize_key(key))
        except ValueError:
            return None
        if not os.path.isfile(path):
            return None
        return {"file": path, "size": os.path.getsize(path), "etag": ""}

    def _locate(self, op: str, bucket: str, key: str) -> dict[str, Any]:
        found = self._staged_entry(bucket, key) or self._own_output(bucket, key)
        if found is None:
            raise StagedObjectNotStaged(
                f"s3.{op}: s3://{bucket}/{key} was not staged for this step; a step "
                "on the HPC lane reads only its parents' outputs and the inputs its "
                "workflow declares in stage_inputs (ADR-074 D-5)"
            )
        return found

    @staticmethod
    def _refuse_extra(op: str, extra: Mapping[str, Any]) -> None:
        if extra:
            raise StagedS3Error(
                f"s3.{op}: argument(s) {', '.join(sorted(extra))} are not supported "
                "on the staged HPC lane"
            )

    @staticmethod
    def _head(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ContentLength": int(entry.get("size") or 0),
            "ETag": str(entry.get("etag") or ""),
            "ContentType": str(entry.get("content_type") or DEFAULT_CONTENT_TYPE),
            "Metadata": {},
        }

    def head_object(self, *, Bucket: str, Key: str, **extra: Any) -> dict[str, Any]:
        self._refuse_extra("head_object", extra)
        return self._head(self._locate("head_object", Bucket, Key))

    def get_object(self, *, Bucket: str, Key: str, **extra: Any) -> dict[str, Any]:
        self._refuse_extra("get_object", extra)
        entry = self._locate("get_object", Bucket, Key)
        return {**self._head(entry), "Body": _StagedBody(open(entry["file"], "rb"))}

    def download_fileobj(
        self,
        Bucket: str,
        Key: str,
        Fileobj: Any,
        ExtraArgs=None,
        Callback=None,
        Config=None,
    ) -> None:
        del Config
        self._refuse_extra("download_fileobj", ExtraArgs or {})
        entry = self._locate("download_fileobj", Bucket, Key)
        with open(entry["file"], "rb") as src:
            shutil.copyfileobj(src, Fileobj, _COPY_CHUNK)
        if Callback is not None:
            Callback(int(entry.get("size") or 0))

    def download_file(
        self,
        Bucket: str,
        Key: str,
        Filename: str,
        ExtraArgs=None,
        Callback=None,
        Config=None,
    ) -> None:
        self._locate("download_file", Bucket, Key)  # refuse before creating Filename
        with open(Filename, "wb") as dst:
            self.download_fileobj(Bucket, Key, dst, ExtraArgs, Callback, Config)

    # ── writes ──────────────────────────────────────────────────────────────

    def _out_path(self, op: str, bucket: str, key: str) -> str:
        if not self.output_bucket or bucket != self.output_bucket:
            raise StagedS3Error(
                f"s3.{op}: a staged step writes only to its output bucket "
                f"{self.output_bucket!r}, not {bucket!r} (ADR-074 D-5)"
            )
        try:
            normalized = normalize_key(key)
        except ValueError as exc:
            raise StagedS3Error(f"s3.{op}: {exc}") from exc
        return os.path.join(self._out_dir, bucket, normalized)

    def _write(self, op: str, bucket: str, key: str, chunks: Iterator[bytes]) -> str:
        path = self._out_path(op, bucket, key)
        digest = hashlib.md5(usedforsecurity=False)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                for chunk in chunks:
                    digest.update(chunk)
                    fh.write(chunk)
        except OSError as exc:
            _discard(path)
            raise StagedS3Error(
                f"s3.{op}: writing s3://{bucket}/{key} failed: {exc}"
            ) from exc
        except BaseException:
            _discard(path)  # never leave a torn object for the server to collect
            raise
        return f'"{digest.hexdigest()}"'

    def put_object(
        self, *, Bucket: str, Key: str, Body: Any = b"", **_metadata: Any
    ) -> dict[str, Any]:
        # ContentType / Metadata / ACL do not survive staging; the server
        # uploads the file itself.
        return {"ETag": self._write("put_object", Bucket, Key, _body_chunks(Body))}

    def upload_fileobj(
        self,
        Fileobj: Any,
        Bucket: str,
        Key: str,
        ExtraArgs=None,
        Callback=None,
        Config=None,
    ) -> None:
        del ExtraArgs, Config
        self._write("upload_fileobj", Bucket, Key, _body_chunks(Fileobj))
        if Callback is not None:
            Callback(os.path.getsize(self._out_path("upload_fileobj", Bucket, Key)))

    def upload_file(
        self,
        Filename: str,
        Bucket: str,
        Key: str,
        ExtraArgs=None,
        Callback=None,
        Config=None,
    ) -> None:
        with open(Filename, "rb") as src:
            self.upload_fileobj(src, Bucket, Key, ExtraArgs, Callback, Config)

    # ── listing ─────────────────────────────────────────────────────────────

    def _require_declared(self, bucket: str, prefix: str) -> None:
        for declared_bucket, declared_prefix in self._prefixes:
            if bucket == declared_bucket and prefix.startswith(declared_prefix):
                return
        raise StagedS3Error(
            f"s3.list_objects_v2: s3://{bucket}/{prefix} is not under a prefix the "
            "workflow declares in stage_inputs.prefixes (ADR-074 D-5)"
        )

    def _listed(self, bucket: str, prefix: str, after: str) -> list[tuple[str, dict]]:
        found = []
        for full, entry in self._objects.items():
            obj_bucket, _, key = full.partition("/")
            if obj_bucket == bucket and key.startswith(prefix) and key > after:
                found.append((key, entry))
        return sorted(found, key=lambda item: item[0])

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        MaxKeys: int = 1000,
        ContinuationToken: str | None = None,
        StartAfter: str = "",
        Delimiter: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        self._refuse_extra("list_objects_v2", extra)
        self._require_declared(Bucket, Prefix)
        after = max(ContinuationToken or "", StartAfter or "")
        page = _ListingPage(Prefix, Delimiter, max(1, int(MaxKeys)), after)
        for key, entry in self._listed(Bucket, Prefix, after):
            if not page.add(key, entry):
                break
        return page.render(Bucket)

    def get_paginator(self, operation_name: str) -> _ListPaginator:
        if operation_name != "list_objects_v2":
            raise _unsupported(f"get_paginator({operation_name!r})")
        return _ListPaginator(self)


def _discard(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def _unsupported(name: str) -> StagedS3Unsupported:
    return StagedS3Unsupported(
        f"s3.{name} is not available to a step on the staged HPC lane: the step "
        "gets files only (get/head/download of staged inputs, put/upload to its "
        "output bucket, list under declared stage_inputs.prefixes) — ADR-074 D-5"
    )


class _ListingPage:
    """One ``list_objects_v2`` page; a delimiter group counts once."""

    def __init__(self, prefix: str, delimiter: str, max_keys: int, after: str) -> None:
        self.prefix, self.delimiter, self.max_keys = prefix, delimiter, max_keys
        self.contents: list[dict[str, Any]] = []
        self.common: list[str] = []
        self.last = after
        self.truncated = False

    def _folded(self, key: str) -> str | None:
        if not self.delimiter:
            return None
        cut = key.find(self.delimiter, len(self.prefix))
        return None if cut < 0 else key[: cut + len(self.delimiter)]

    def add(self, key: str, entry: Mapping[str, Any]) -> bool:
        """Take *key* into the page; False once the page is full."""
        folded = self._folded(key)
        if folded is not None and folded in self.common:
            self.last = key
            return True
        if len(self.contents) + len(self.common) >= self.max_keys:
            self.truncated = True
            return False
        self.last = key
        if folded is None:
            self.contents.append(
                {
                    "Key": key,
                    "Size": int(entry.get("size") or 0),
                    "ETag": str(entry.get("etag") or ""),
                }
            )
        else:
            self.common.append(folded)
        return True

    def render(self, bucket: str) -> dict[str, Any]:
        page: dict[str, Any] = {
            "Name": bucket,
            "Prefix": self.prefix,
            "KeyCount": len(self.contents) + len(self.common),
            "IsTruncated": self.truncated,
            "Contents": self.contents,
        }
        if self.common:
            page["CommonPrefixes"] = [{"Prefix": p} for p in self.common]
        if self.truncated:
            page["NextContinuationToken"] = self.last
        return page
