"""Export current tab data to JSON or CSV.

Export files are written to the current working directory as
``umrs-monitor-export-{tab}-{timestamp}.{ext}``.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def export_tab(
    tab: str,
    data: Any,
    *,
    fmt: str = "json",
    out_dir: str | None = None,
) -> str:
    """Export ``data`` for ``tab`` in ``fmt`` (``json`` or ``csv``).

    Returns the path to the written file.
    """
    ext = "json" if fmt == "json" else "csv"
    out = os.path.join(out_dir or os.getcwd(), f"umrs-monitor-export-{tab}-{_timestamp()}.{ext}")
    if fmt == "json":
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    else:
        rows = _to_rows(data)
        if rows:
            with open(out, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    return out


def _to_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("jobs", "endpoints", "models", "workflows", "tasks"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    return []
