"""R4–UMRS wire protocol version (gap row D-04, 2026-09-22).

One public, versioned protocol over both lanes (ADR-059):

- **ZeroMQ data plane**: string commands (see docs/R4-UMRS-PROTOCOL.md);
  the liveness handshake is ``READY!:v<MAJOR.MINOR>`` since protocol 1.0 —
  older servers answer bare ``READY!`` (implicit v0).
- **HTTPS control plane**: lane-classified families under ``/umrs`` and the
  gateway; errors ride the shared typed-taxonomy header
  ``x-platform-error-code``.

Version rules: MAJOR bumps break framing/command sets (both sides must
agree or refuse); MINOR bumps are additive (older peers keep working and
simply cannot use the new commands). Both sides expose the constant; this
package is stdlib-only (ADR-005).
"""

from __future__ import annotations

PROTOCOL_VERSION = "1.0"


def handshake_reply(version: str = PROTOCOL_VERSION) -> str:
    """Server-side liveness reply carrying the protocol version."""
    return f"READY!:v{version}"


def parse_handshake_reply(text: str) -> tuple[bool, str]:
    """(ready, version) for a liveness reply.

    Bare ``READY!`` parses as ready with version "" (pre-1.0 server);
    anything else is not ready.
    """
    if text == "READY!":
        return True, ""
    if text.startswith("READY!:v"):
        return True, text[len("READY!:v") :]
    return False, ""


def is_protocol_compatible(server_version: str, client_version: str = PROTOCOL_VERSION) -> bool:
    """MAJOR-version equality check; empty server version = pre-1.0 (v0 peer)."""
    if not server_version:
        return True  # pre-1.0 peer: v0 lane stays compatible until retired
    return server_version.split(".", 1)[0] == client_version.split(".", 1)[0]
