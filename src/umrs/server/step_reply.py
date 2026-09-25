"""Data-only step replies (ADR-074 D-6).

A step writes its own reply, so the bytes are the step's to choose;
``pickle.loads`` on them runs whatever the step put there inside the UMRS
server. So:

- the worker writes a JSON reply when ``UMRS_STEP_OUTPUT_JSON`` names a
  path (:func:`reply_to_json`): ``status`` / ``phase`` / ``message`` and the
  :class:`StepResult` fields as JSON, numeric ndarray embeddings packed as
  ``{dtype, shape, b64}`` (raw little-endian bytes — a JSON float list is
  about five times the array), everything else JSON cannot carry through
  ``tolist()`` / ``str()``;
- the server rebuilds the result from it (:func:`result_from_json`) with
  every field shape-checked and the packed bytes checked against the shape;
- an image that predates JSON replies still writes only ``output.pkl``: it is
  read through :func:`restricted_loads`, which resolves ``StepResult`` and a
  few builtin containers and nothing else — any other global (``os.system``,
  numpy's constructors) is refused before it can run, so an old image's numpy
  reply is refused by name rather than trusted;
- both are read with a size cap (:func:`reply_max_bytes`): the step chooses
  their size, and the server that reads them holds every other user's work.

Adapted from the parked ``wip/step-lane-2026-09-25`` ``step_reply.py`` (the
JSON half); the numpy-aware unpickler there is deliberately not carried over.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import os
import pickle
from typing import Any

from umrs.pipeline_sdk.step import StepResult

from .step_executor import StepExecutionError

#: Env var naming the JSON reply file inside the step container.
REPLY_JSON_ENV = "UMRS_STEP_OUTPUT_JSON"
REPLY_MAX_BYTES_ENV = "UMRS_STEP_REPLY_MAX_BYTES"
REPLY_MAX_BYTES_DEFAULT = 64 * 1024 * 1024

_PHASES = frozenset({"setup", "run", "teardown", "worker"})
_NUMERIC_DTYPE_KINDS = frozenset("fiu")
_MAX_NDIM = 32
#: Globals a legacy pickle reply may reference.
_LEGACY_GLOBALS = frozenset(
    {
        ("umrs.pipeline_sdk.step", "StepResult"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "complex"),
        ("collections", "OrderedDict"),
    }
)


def json_reply_path(pickle_path: str) -> str:
    """Where the data-only reply sits next to ``output.pkl``."""
    return os.path.splitext(pickle_path)[0] + ".json"


# ── worker side ─────────────────────────────────────────────────────────────


def _primitive(obj: Any) -> Any:
    """``json.dumps`` fallback: numpy via ``tolist()``, sets, bytes, else str."""
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", "replace")
    return str(obj)


def _packed_array(value: Any) -> dict[str, Any] | None:
    """A numeric ndarray as ``{dtype, shape, b64}``, else None."""
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    if getattr(dtype, "kind", "") not in _NUMERIC_DTYPE_KINDS or shape is None:
        return None
    little = dtype.newbyteorder("<")
    raw = value.astype(little, copy=False).tobytes(order="C")
    return {
        "dtype": little.str,
        "shape": [int(n) for n in shape],
        "b64": base64.b64encode(raw).decode("ascii"),
    }


def _json_keys(value: Any) -> Any:
    """*value* with every nested dict key JSON can carry (``str(key)`` else)."""
    if isinstance(value, dict):
        return {
            (
                k if k is None or isinstance(k, (str, int, float)) else str(k)
            ): _json_keys(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_keys(v) for v in value]
    return value


def _result_fields(result: StepResult) -> dict[str, Any]:
    packed = _packed_array(result.embeddings)
    return {
        "output_keys": result.output_keys,
        "checksums": result.checksums,
        "metadata": result.metadata,
        "extra": result.extra,
        "embeddings": result.embeddings if packed is None else packed,
    }


def reply_to_json(reply: dict[str, Any]) -> bytes:
    """The worker's reply dict as data-only JSON (never raises)."""
    data = {k: reply[k] for k in ("status", "phase", "message") if k in reply}
    try:
        result = reply.get("result")
        if isinstance(result, StepResult):
            data["result"] = _json_keys(_result_fields(result))
        return json.dumps(data, default=_primitive).encode("utf-8")
    except Exception as exc:  # noqa: BLE001 — the worker must always reply
        return json.dumps(
            {
                "status": "error",
                "phase": "worker",
                "message": f"step result is not JSON-serialisable: {exc}",
            }
        ).encode("utf-8")


# ── server side ─────────────────────────────────────────────────────────────


def reply_max_bytes() -> int:
    """The reply-size cap: a positive ``UMRS_STEP_REPLY_MAX_BYTES``, else 64 MiB."""
    raw = os.environ.get(REPLY_MAX_BYTES_ENV, "").strip()
    try:
        value = int(raw) if raw else REPLY_MAX_BYTES_DEFAULT
    except ValueError:
        return REPLY_MAX_BYTES_DEFAULT
    return value if value > 0 else REPLY_MAX_BYTES_DEFAULT


def check_reply_size(blob: bytes, path: str, cap: int) -> bytes:
    """*blob* (read with at most ``cap + 1`` bytes) unless it passed *cap*."""
    if len(blob) > cap:
        raise StepExecutionError(
            "worker",
            f"step reply {os.path.basename(path)} exceeds {cap} bytes "
            f"({REPLY_MAX_BYTES_ENV}); return large data through S3 keys",
        )
    return blob


def _field(fields: dict[str, Any], name: str, kind: type) -> Any:
    value = fields.get(name)
    if value is None:
        return kind()
    if not isinstance(value, kind):
        raise StepExecutionError(
            "worker",
            f"step reply field {name!r} must be a {kind.__name__}, got {type(value).__name__}",
        )
    return value


def _bad_array(why: str) -> StepExecutionError:
    return StepExecutionError("worker", f"step reply embeddings refused: {why}")


def _checked_shape(shape: Any) -> list[int]:
    if not (isinstance(shape, list) and len(shape) <= _MAX_NDIM):
        raise _bad_array(f"shape must be a list of at most {_MAX_NDIM} ints")
    if not all(type(n) is int and n >= 0 for n in shape):
        raise _bad_array("shape entries must be non-negative ints")
    return shape


def _unpacked_array(packed: dict[str, Any]) -> Any:
    """``{dtype, shape, b64}`` back to the ndarray: numeric dtype only, and the
    decoded byte count must be exactly what the shape needs."""
    try:
        import numpy as np
    except ImportError as exc:
        raise _bad_array("numpy is not installed on the server") from exc
    try:
        dtype = np.dtype(str(packed.get("dtype")))
        raw = base64.b64decode(str(packed.get("b64", "")), validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise _bad_array(str(exc)) from exc
    if dtype.kind not in _NUMERIC_DTYPE_KINDS:
        raise _bad_array(f"dtype {dtype.str} is not numeric")
    shape = _checked_shape(packed.get("shape"))
    if math.prod(shape) * dtype.itemsize != len(raw):
        raise _bad_array(f"{len(raw)} bytes do not fill shape {shape}")
    return np.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)


def _embeddings(data: Any) -> Any:
    if data is None or isinstance(data, list):
        return data
    if isinstance(data, dict):
        return _unpacked_array(data)
    raise StepExecutionError("worker", "step reply field 'embeddings' must be a list")


def checked_result(fields: dict[str, Any]) -> StepResult:
    """A StepResult from reply fields, every field shape-checked."""
    output_keys = _field(fields, "output_keys", list)
    if not all(isinstance(key, str) for key in output_keys):
        raise StepExecutionError("worker", "step reply output_keys must be strings")
    return StepResult(
        output_keys=output_keys,
        checksums=_field(fields, "checksums", list),
        metadata=_field(fields, "metadata", list),
        extra=_field(fields, "extra", dict),
        embeddings=_embeddings(fields.get("embeddings")),
    )


def result_from_json(blob: bytes) -> StepResult:
    """Rebuild the step's StepResult from its data-only reply."""
    try:
        reply = json.loads(blob)
    except (ValueError, RecursionError) as exc:
        raise StepExecutionError(
            "worker", f"step reply is not valid JSON: {exc}"
        ) from exc
    if not isinstance(reply, dict):
        raise StepExecutionError(
            "worker", f"step reply must be a JSON object, got {type(reply).__name__}"
        )
    if reply.get("status") != "ok":
        phase = reply.get("phase")
        raise StepExecutionError(
            phase if phase in _PHASES else "worker",
            str(reply.get("message") or "step failed without a message"),
        )
    fields = reply.get("result")
    if not isinstance(fields, dict):
        raise StepExecutionError("worker", "ok step reply carries no result object")
    return checked_result(fields)


class _LegacyReplyUnpickler(pickle.Unpickler):
    """Resolves StepResult and a few builtin containers; refuses the rest."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _LEGACY_GLOBALS:
            raise pickle.UnpicklingError(
                f"legacy step reply references {module}.{name}; refused "
                "(rebuild the step image so it writes a JSON reply)"
            )
        return super().find_class(module, name)


def restricted_loads(blob: bytes) -> Any:
    """A legacy pickle reply, loaded without resolving arbitrary globals."""
    try:
        return _LegacyReplyUnpickler(io.BytesIO(blob)).load()
    except pickle.UnpicklingError as exc:
        raise StepExecutionError("worker", str(exc)) from exc
    except Exception as exc:  # malformed bytes, never trusted
        raise StepExecutionError(
            "worker", f"legacy step reply is unreadable: {exc}"
        ) from exc


_TREE_NODE_BUDGET = 1_000_000


def _ensure_tree(value: Any) -> None:
    """Refuse shared or cyclic containers: pickle's memo lets a few bytes
    build a DAG whose later JSON walk (the runner serialises ``extra``) grows
    exponentially; a JSON reply can only ever be a tree."""
    seen: set[int] = set()
    stack = [value]
    nodes = 0
    while stack:
        item = stack.pop()
        nodes += 1
        if nodes > _TREE_NODE_BUDGET:
            raise StepExecutionError(
                "worker", "legacy step reply is too large a structure"
            )
        if isinstance(item, StepResult):
            item = vars(item)
        if isinstance(item, (dict, list, tuple, set, frozenset)):
            if id(item) in seen:
                raise StepExecutionError(
                    "worker", "legacy step reply shares containers; refused"
                )
            seen.add(id(item))
            stack.extend(item.values() if isinstance(item, dict) else item)
            if isinstance(item, dict):
                stack.extend(item.keys())


def result_from_legacy_pickle(blob: bytes) -> StepResult:
    """A pre-JSON image's ``output.pkl``: restricted load, tree check, then the
    same field checks as a JSON reply (``{"status", "result"}`` or a bare
    StepResult)."""
    reply = restricted_loads(blob)
    _ensure_tree(reply)
    if isinstance(reply, dict) and reply.get("status") != "ok":
        phase = reply.get("phase")
        raise StepExecutionError(
            phase if phase in _PHASES else "worker",
            str(reply.get("message") or "step failed without a message"),
        )
    result = reply.get("result") if isinstance(reply, dict) else reply
    if not isinstance(result, StepResult):
        raise StepExecutionError(
            "worker",
            f"legacy step reply carries no StepResult (got {type(result).__name__})",
        )
    return checked_result(vars(result))
