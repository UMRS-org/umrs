"""Helpers for download_results readiness checks.

Detects whether PROCESS_OBJECTS work for a given output bucket is still
incomplete, so the client can warn or block instead of silently downloading
empty results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Terminal statuses for PROCESS_OBJECTS thread entries.
_TERMINAL_STATUSES = frozenset({"Done", "Idle"})


@dataclass(frozen=True)
class BucketWorkStatus:
    """Parsed PROCESS_OBJECTS readiness for one output bucket."""

    ready: bool
    """True when no incomplete work targets this bucket."""

    status_known: bool
    """False when the server has no usable PROCESS_OBJECTS status."""

    incomplete_statuses: tuple[str, ...] = ()
    """Distinct non-terminal statuses for matching work entries."""

    message: str = ""
    """Human-readable summary suitable for CLI warnings."""

    raw_message: str | None = None
    """Server payload message when status is missing/error."""

    matching_entries: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _bucket_matches(work_bucket: str, full_bucket_name: str) -> bool:
    """Match input or output bucket naming used by PROCESS_OBJECTS status."""
    if not work_bucket:
        return False
    return work_bucket == full_bucket_name or full_bucket_name == f"{work_bucket}-output"


def _extract_thread_entries(parsed: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return thread status entries from a STATUS payload, or None if unusable."""
    if not isinstance(parsed, dict):
        return None

    message = parsed.get("message")
    if isinstance(message, str) and (
        message == "No status entries" or message.startswith("No status for ")
    ):
        return None

    summary = parsed.get("summary")
    if not isinstance(summary, dict):
        return None

    data = summary.get("data")
    if data is None:
        return None
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return list(data)
    return None


def parse_process_objects_readiness(
    parsed: dict[str, Any] | None,
    full_bucket_name: str,
) -> BucketWorkStatus:
    """Interpret a PROCESS_OBJECTS status payload for *full_bucket_name*.

    Missing / unusable status is treated as **not ready** (not done), so
    download_results does not silently succeed with an empty bucket.
    """
    if parsed is None:
        return BucketWorkStatus(
            ready=False,
            status_known=False,
            message=(
                f"Cannot determine task status for bucket '{full_bucket_name}' "
                "(no status response from server)."
            ),
        )

    raw_message = parsed.get("message") if isinstance(parsed, dict) else None
    if isinstance(raw_message, str) and raw_message == "":
        raw_message = None

    entries = _extract_thread_entries(parsed)
    if entries is None:
        detail = raw_message or "no PROCESS_OBJECTS status available"
        return BucketWorkStatus(
            ready=False,
            status_known=False,
            message=(
                f"Cannot download results for bucket '{full_bucket_name}': "
                f"task status is unknown ({detail})."
            ),
            raw_message=raw_message if isinstance(raw_message, str) else None,
        )

    matching: list[dict[str, Any]] = []
    incomplete_statuses: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        work_list = entry.get("work") or []
        if not isinstance(work_list, list):
            continue
        related = any(
            isinstance(w, dict) and _bucket_matches(str(w.get("bucket", "")), full_bucket_name)
            for w in work_list
        )
        if not related:
            continue
        matching.append(entry)
        status = str(entry.get("status", "Unknown"))
        if status not in _TERMINAL_STATUSES:
            incomplete_statuses.append(status)

    if not matching:
        # No work targets this bucket → nothing incomplete (same as legacy
        # download_results: done when no non-terminal matching entries).
        return BucketWorkStatus(
            ready=True,
            status_known=True,
            message=(
                f"No in-progress PROCESS_OBJECTS work for bucket "
                f"'{full_bucket_name}'."
            ),
            raw_message=raw_message if isinstance(raw_message, str) else None,
            matching_entries=(),
        )

    if incomplete_statuses:
        unique = tuple(dict.fromkeys(incomplete_statuses))
        status_list = ", ".join(unique)
        return BucketWorkStatus(
            ready=False,
            status_known=True,
            incomplete_statuses=unique,
            message=(
                f"Cannot download results for bucket '{full_bucket_name}': "
                f"task is still incomplete (status: {status_list})."
            ),
            raw_message=raw_message if isinstance(raw_message, str) else None,
            matching_entries=tuple(matching),
        )

    statuses = tuple(
        dict.fromkeys(str(e.get("status", "Unknown")) for e in matching)
    )
    status_list = ", ".join(statuses) if statuses else "Done"
    return BucketWorkStatus(
        ready=True,
        status_known=True,
        incomplete_statuses=(),
        message=(
            f"Results ready for bucket '{full_bucket_name}' "
            f"(status: {status_list})."
        ),
        raw_message=raw_message if isinstance(raw_message, str) else None,
        matching_entries=tuple(matching),
    )


def format_wait_progress(
    readiness: BucketWorkStatus,
    *,
    elapsed_sec: int,
) -> str:
    """One-line progress message while blocking for incomplete work."""
    mins, secs = divmod(max(0, elapsed_sec), 60)
    if readiness.incomplete_statuses:
        status = ", ".join(readiness.incomplete_statuses)
    elif not readiness.status_known:
        status = "unknown"
    else:
        status = "pending"
    return (
        f"Waiting for results (status: {status}, "
        f"elapsed: {mins}m {secs}s)..."
    )


def task_status_field(readiness: BucketWorkStatus) -> list[str] | str:
    """Value for the ``task_status`` field in CLI JSON responses."""
    if readiness.incomplete_statuses:
        return list(readiness.incomplete_statuses)
    if not readiness.status_known:
        return "unknown"
    return "pending"


def incomplete_error_payload(
    readiness: BucketWorkStatus,
    *,
    full_bucket_name: str,
    subdir: str,
    modality: str,
    output_format: str,
    elapsed_sec: int | None = None,
    timed_out: bool = False,
    max_wait_sec: int | None = None,
) -> dict[str, Any]:
    """Build the error dict returned when download cannot proceed."""
    if timed_out and max_wait_sec is not None:
        message = (
            f"Timed out after {max_wait_sec}s waiting for results. "
            f"{readiness.message}"
        )
    else:
        message = readiness.message
    payload: dict[str, Any] = {
        "status": "error",
        "message": message,
        "bucket": full_bucket_name,
        "subdir": subdir,
        "modality": modality,
        "task_status": task_status_field(readiness),
        "output_format": output_format,
    }
    if elapsed_sec is not None:
        payload["elapsed_sec"] = elapsed_sec
    return payload


def classify_wait_reply(reply: str) -> str:
    """Classify a ``wait_for_new_data`` reply.

    Returns one of: ``timeout``, ``update``, ``hard_error``, ``unexpected``.
    """
    if reply == "TIMEOUT":
        return "timeout"
    if reply == "UPDATE":
        return "update"
    if reply in ("NO_SUCH_BUCKET", "SERVER_OVERLOADED"):
        return "hard_error"
    return "unexpected"

