"""Signed run intents at the server (ADR-112 D-4/D-5; plan §5 B2.0 §1, §10.3).

The person signs one intent per workflow run at one site; their gate is the
authority that verifies it at each step's submit. The server never verifies
the signature. It parses the payload strictly (the gate's own parser and
schema, ``r4_contracts.hpc_gate``) and checks that the intent is the run it
is asked to start:

- ``workflow`` and ``version`` are the run's registered version;
- ``project`` is the caller's **verified** project and ``site`` this venue;
- ``principal`` is the caller, and ``login`` / ``account`` the caller's
  active site account and project link (the resolved identity);
- the run's params are the intent's ``params``, byte for byte: a body that
  also carries params must equal them, and never adds or overrides one;
- it has not expired (``now <= expires``; the gate allows 120 s more);
- its ``run_id`` is the one this server issued for the intent's
  ``principal``, ``nonce`` and ``issued`` (:func:`issued_run_id`): the
  caller cannot choose another run's execution id (fixer finding
  2026-11-08).

Each failure is ``hpcaccess.intent_invalid`` (422) with a short reason name.
The run then starts with ``execution_id = intent.run_id`` and keeps the
payload, the signature and ``intent_id`` (migration 0136).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import time
from dataclasses import dataclass
from typing import Any

#: The contract's payload cap (``hpc_gate.INTENT_MAX_BYTES``, 8 MiB).
INTENT_MAX_BYTES = 8 * 1024 * 1024
#: An armored SSHSIG is a few hundred bytes; anything far larger is not one.
SIGNATURE_MAX_BYTES = 16 * 1024
_ARMOR_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_ARMOR_END = "-----END SSH SIGNATURE-----"
_RUN_ID = re.compile(r"wf-[0-9a-f]{12}")
_B64 = re.compile(r"[A-Za-z0-9+/]*={0,2}")


class IntentInvalid(ValueError):
    """``hpcaccess.intent_invalid``: *reason* is a short name, *detail* text."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SignedIntent:
    """A parsed intent: the exact payload bytes, its body and signature."""

    payload: bytes
    body: dict[str, Any]
    signature: str

    @property
    def intent_id(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def b64(self) -> str:
        return base64.b64encode(self.payload).decode("ascii")

    @property
    def run_id(self) -> str:
        return str(self.body.get("run_id") or "")

    def spec(self) -> dict[str, str]:
        """What the durable job spec and the run record keep."""
        return {"b64": self.b64, "signature": self.signature, "id": self.intent_id}


def _decode_b64(text: Any) -> bytes:
    if not isinstance(text, str) or not text or not _B64.fullmatch(text):
        raise IntentInvalid("bad_envelope", "intent is not base64")
    if len(text) > (INTENT_MAX_BYTES * 4) // 3 + 4:
        raise IntentInvalid("too_large", f"the intent exceeds {INTENT_MAX_BYTES} bytes")
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise IntentInvalid("bad_envelope", "intent is not base64") from exc


def _check_signature_text(signature: Any) -> str:
    """Shape only: one armored SSHSIG block (the gate verifies it)."""
    if not isinstance(signature, str) or len(signature) > SIGNATURE_MAX_BYTES:
        raise IntentInvalid("bad_envelope", "signature is not an armored SSHSIG")
    text = signature.strip()
    if not (text.startswith(_ARMOR_BEGIN) and text.endswith(_ARMOR_END)):
        raise IntentInvalid("bad_envelope", "signature is not an armored SSHSIG")
    return signature


def _parse_payload(payload: bytes) -> dict[str, Any]:
    """The gate's strict parser and intent schema (shared code, lane G)."""
    from r4_contracts import hpc_gate

    try:
        body = hpc_gate.parse_strict(payload, json_values=True)
        hpc_gate.validate_intent(body)
    except (ValueError, hpc_gate.GateError) as exc:
        raise IntentInvalid("bad_schema", str(exc)[:300]) from exc
    return body


def decode_signed_intent(intent_b64: Any, signature: Any) -> SignedIntent:
    """Decode and strictly parse an intent; :class:`IntentInvalid` otherwise."""
    payload = _decode_b64(intent_b64)
    if len(payload) > INTENT_MAX_BYTES:
        raise IntentInvalid("too_large", f"the intent exceeds {INTENT_MAX_BYTES} bytes")
    signature = _check_signature_text(signature)
    body = _parse_payload(payload)
    if not _RUN_ID.fullmatch(str(body.get("run_id") or "")):
        raise IntentInvalid("bad_schema", "run_id is not wf-<12 hex>")
    return SignedIntent(payload=payload, body=body, signature=signature)


@dataclass(frozen=True)
class TriggerBinding:
    """What the run is: the values an intent must name exactly."""

    workflow: str
    version: str
    project: str
    site: str
    principal: str
    login: str
    account: str
    #: The trigger body's params after injection; None when it named none.
    params: dict[str, Any] | None


def _mismatch(field: str, signed: Any, actual: Any) -> IntentInvalid:
    return IntentInvalid(
        "intent_mismatch",
        f"{field}: the intent says {str(signed)[:80]!r}, the run is {str(actual)[:80]!r}",
    )


def _canonical(value: Any) -> bytes:
    from r4_contracts import hpc_gate

    try:
        return hpc_gate.canonical_bytes(value, json_values=True)
    except ValueError as exc:
        raise IntentInvalid("non_json_value", str(exc)[:200]) from exc


def check_params(signed: Any, body_params: dict[str, Any] | None) -> None:
    """The run's params are the intent's; a body that carries params must
    equal them exactly (after the server's own injection)."""
    if body_params is None:
        return
    if _canonical(signed) != _canonical(body_params):
        raise IntentInvalid(
            "params_mismatch",
            "the trigger's params differ from the signed intent's params",
        )


_RUN_ID_LABEL = b"umrs-intent-run-id-v1"


def issued_run_id(principal: str, nonce: str, issued: int) -> str:
    """``wf-<12 hex>``: an HMAC under a server-only key (``UMRS_SECRET_KEY``)
    of the intent's principal, nonce and issue time. The proposal sets it;
    the trigger recomputes it, so an execution id is never the caller's
    choice (without ``UMRS_SECRET_KEY`` an intent proposed before a restart
    must be proposed again)."""
    from umrs.workflow_plans import server_key

    message = f"{principal}\n{nonce}\n{int(issued)}".encode()
    mac = hmac.new(server_key(_RUN_ID_LABEL), message, hashlib.sha256).hexdigest()
    return f"wf-{mac[:12]}"


def check_run_id(body: dict[str, Any]) -> None:
    """:class:`IntentInvalid` ``run_id_not_issued`` unless the intent's
    ``run_id`` is :func:`issued_run_id` of its own fields."""
    expected = issued_run_id(
        str(body.get("principal") or ""), str(body.get("nonce") or ""), int(body.get("issued") or 0)
    )
    if not hmac.compare_digest(str(body.get("run_id") or ""), expected):
        raise IntentInvalid("run_id_not_issued", "the intent's run_id was not issued by this server")


def check_trigger_intent(
    intent: SignedIntent, binding: TriggerBinding, *, now: float | None = None
) -> None:
    """Every binding of plan §10.3; :class:`IntentInvalid` on the first miss."""
    body = intent.body
    for field, actual in (
        ("workflow", binding.workflow),
        ("version", binding.version),
        ("project", binding.project),
        ("site", binding.site),
        ("principal", binding.principal),
        ("login", binding.login),
        ("account", binding.account),
    ):
        if body.get(field) != actual:
            raise _mismatch(field, body.get(field), actual)
    check_params(body.get("params"), binding.params)
    current = time.time() if now is None else now
    if current > int(body.get("expires") or 0):
        raise IntentInvalid("expired", "the intent has expired: sign a new one")
    check_run_id(body)


def intent_from_spec(spec: Any) -> dict[str, str] | None:
    """The ``{"b64", "signature", "id"}`` a durable job spec carries."""
    raw = spec.get("intent") if isinstance(spec, dict) else None
    if not isinstance(raw, dict) or not raw.get("id"):
        return None
    return {k: str(raw.get(k) or "") for k in ("b64", "signature", "id")}


__all__ = [
    "INTENT_MAX_BYTES",
    "IntentInvalid",
    "SignedIntent",
    "TriggerBinding",
    "check_params",
    "check_run_id",
    "check_trigger_intent",
    "decode_signed_intent",
    "intent_from_spec",
    "issued_run_id",
]
