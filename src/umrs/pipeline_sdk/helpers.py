"""Pipeline SDK helpers — polling, logging, formatting utilities."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .base import Result

logger = logging.getLogger(__name__)


def poll_until_done(
    probe: Callable[[], str | None],
    success_values: set = frozenset({"success", "done"}),
    failure_values: set = frozenset({"failure", "error", "cancelled"}),
    poll_interval: float = 5.0,
    timeout: float = 300.0,
) -> str:
    """Generic poller.

    Calls *probe()* repeatedly until it returns a value present in
    *success_values* or *failure_values*.

    Args:
        probe: Callable returning a status string (or None to keep waiting).
        success_values: Values that mean "finished successfully".
        failure_values: Values that mean "finished with error".
        poll_interval: Seconds between polls.
        timeout: Maximum seconds to wait.

    Returns:
        The final status string.

    Raises:
        TimeoutError: If timeout exceeded before a terminal status.
    """
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            raise TimeoutError(f"Polling timed out after {timeout:.0f}s")

        status = probe()
        if status is None:
            time.sleep(poll_interval)
            continue

        status_lower = status.lower()
        if status_lower in success_values:
            return status
        if status_lower in failure_values:
            return status

        logger.debug("Poll status: %s (%.0fs elapsed)", status, elapsed)
        time.sleep(poll_interval)


def format_results_table(results: list[Result]) -> str:
    """Format a list of Results as a human-readable table.

    Returns a multi-line string.
    """
    if not results:
        return "(no results)"

    lines = []
    lines.append("")
    lines.append(f"{'#':>3}  {'Status':<10}  {'Endpoint':<20}  {'Key':<30}  Details")
    lines.append(f"{'─' * 3}  {'─' * 10}  {'─' * 20}  {'─' * 30}  {'─' * 20}")

    for i, r in enumerate(results, 1):
        key = ""
        if r.task_handle and r.task_handle.data_item:
            key = r.task_handle.data_item.key
        endpoint = r.task_handle.endpoint if r.task_handle else ""

        # Build a short detail string from result data
        detail_parts = []
        if r.error:
            detail_parts.append(f"error={r.error[:40]}")
        elif r.data:
            for k in ("num_objects", "mae_loss", "cls_shape", "task"):
                if k in r.data:
                    detail_parts.append(f"{k}={r.data[k]}")
        detail = ", ".join(detail_parts) or "—"

        lines.append(f"{i:>3}  {r.status:<10}  {endpoint:<20}  {key:<30}  {detail}")

    successes = sum(1 for r in results if r.status == "success")
    failures = sum(1 for r in results if r.status == "failure")
    lines.append("")
    lines.append(
        f"Total: {len(results)} result(s), {successes} success, {failures} failure"
    )
    return "\n".join(lines)
