"""CLI helpers for the unified UMRS warning framework."""

from __future__ import annotations

from typing import Any

from umrs.shared.warnings import UMRSWarning, print_warning_structured


def emit_cli_warning(
    warning: UMRSWarning,
    *,
    output_format: str,
    **payload: Any,
) -> dict[str, Any]:
    """Emit a structured warning for CLI consumers.

    Text mode prints a formatted warning to stderr. JSON mode leaves printing
    to the caller so a single ``api_json`` / ``json.dumps`` line is emitted.

    Returns a dict with ``status='error'`` (item-2 compat) and ``warnings[]``.
    """
    body: dict[str, Any] = {
        "status": "error",
        "message": warning.message,
        "warnings": [warning.to_dict()],
        **payload,
    }
    if output_format != "json":
        print_warning_structured(warning)
    return body
