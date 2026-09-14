"""Unified structured JSON logging for UMRS services.

Provides a :class:`StructuredJSONFormatter` that emits one JSON object
per log line, including:

- ``ts``         — ISO-8601 timestamp
- ``level``      — log level name
- ``service``    — service identifier (from env or default)
- ``request_id`` — correlation ID from the current request context
- ``logger``     — logger name (module path)
- ``message``    — formatted log message
- ``details``    — extra structured data (if provided via ``extra``)

Enable via env:

    UMRS_LOG_FORMAT=json        # default is "text" (standard Python format)
    UMRS_LOG_LEVEL=INFO         # root log level
    UMRS_SERVICE_NAME=umrs      # service tag in JSON output

Usage in application code (no changes needed — standard logging):

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Endpoint provisioned", extra={"details": {"model": "qwen-7b"}})

The formatter automatically injects ``request_id`` from the correlation
middleware when available.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


class StructuredJSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Automatically enriches records with:
    - ``service`` from ``UMRS_SERVICE_NAME`` env var
    - ``request_id`` from the correlation middleware context
    """

    def __init__(self, service: str = "umrs") -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject correlation ID from middleware context if available
        request_id = ""
        try:
            from umrs.server.correlation import get_request_id

            request_id = get_request_id() or ""
        except Exception:
            pass
        if request_id:
            entry["request_id"] = request_id

        # Include structured details if provided via extra={"details": {...}}
        details = getattr(record, "details", None)
        if details and isinstance(details, dict):
            entry["details"] = details

        # Include exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def configure_structured_logging() -> None:
    """Configure the root logger based on ``UMRS_LOG_FORMAT`` env var.

    When ``UMRS_LOG_FORMAT=json``, replaces the root handler with
    :class:`StructuredJSONFormatter`. Otherwise uses the default
    Python text format.

    Safe to call multiple times — removes existing handlers first.
    """
    log_format = os.environ.get("UMRS_LOG_FORMAT", "text").lower().strip()
    log_level = os.environ.get("UMRS_LOG_LEVEL", "INFO").upper().strip()
    service_name = os.environ.get("UMRS_SERVICE_NAME", "umrs").strip()

    root = logging.getLogger()
    level = getattr(logging, log_level, logging.INFO)
    root.setLevel(level)

    if log_format == "json":
        # Remove existing handlers to avoid duplicate output
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJSONFormatter(service=service_name))
        root.addHandler(handler)
