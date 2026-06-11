"""
Plugin audit logging (X-106).

Tracks plugin execution events for security and debugging.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import PluginAuditEntry

logger = logging.getLogger(__name__)


class PluginAuditLog:
    """
    Audit logger for plugin executions (X-106).

    Logs events to:
    - Structured JSON logs
    - Optional PostgreSQL persistence
    - Optional file-based audit trail
    """

    def __init__(
        self,
        db_url: str | None = None,
        file_path: str | None = None,
        log_to_stdout: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            db_url: Optional PostgreSQL connection URL.
            file_path: Optional file path for audit log.
            log_to_stdout: Whether to log to stdout.
        """
        self.db_url = db_url or os.environ.get("PLUGIN_AUDIT_DB")
        self.file_path = Path(file_path) if file_path else None
        self.log_to_stdout = log_to_stdout

        self._db_pool = None
        self._audit_logger = logging.getLogger("plugin.audit")

        if self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def _get_db_pool(self):
        """Get or create database connection pool."""
        if self._db_pool is None and self.db_url:
            try:
                import asyncpg

                self._db_pool = await asyncpg.create_pool(self.db_url)

                # Ensure table exists
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS plugin_audit_log (
                            id SERIAL PRIMARY KEY,
                            plugin_id TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            user_id TEXT,
                            project_id TEXT,
                            details JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        );

                        CREATE INDEX IF NOT EXISTS idx_plugin_audit_plugin_id
                        ON plugin_audit_log(plugin_id);

                        CREATE INDEX IF NOT EXISTS idx_plugin_audit_user_id
                        ON plugin_audit_log(user_id);

                        CREATE INDEX IF NOT EXISTS idx_plugin_audit_timestamp
                        ON plugin_audit_log(timestamp);
                    """)
            except ImportError:
                logger.warning("asyncpg not available, audit logging to DB disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize audit DB: {e}")

        return self._db_pool

    async def log_event(
        self,
        plugin_id: str,
        event_type: str,
        user_id: str | None = None,
        project_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> PluginAuditEntry:
        """
        Log an audit event.

        Args:
            plugin_id: Plugin execution ID.
            event_type: Type of event (submitted, started, output_written, succeeded, failed).
            user_id: Optional user ID.
            project_id: Optional project ID.
            details: Optional event details.

        Returns:
            Created audit entry.
        """
        entry = PluginAuditEntry(
            plugin_id=plugin_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            project_id=project_id,
            details=details or {},
        )

        # Log to stdout
        if self.log_to_stdout:
            self._audit_logger.info(json.dumps(entry.to_dict(), default=str))

        # Log to file
        if self.file_path:
            await self._log_to_file(entry)

        # Log to database
        if self.db_url:
            await self._log_to_db(entry)

        return entry

    async def _log_to_file(self, entry: PluginAuditEntry) -> None:
        """Log entry to file."""
        try:
            with open(self.file_path, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write audit log to file: {e}")

    async def _log_to_db(self, entry: PluginAuditEntry) -> None:
        """Log entry to database."""
        pool = await self._get_db_pool()
        if not pool:
            return

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO plugin_audit_log
                    (plugin_id, event_type, timestamp, user_id, project_id, details)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    entry.plugin_id,
                    entry.event_type,
                    entry.timestamp,
                    entry.user_id,
                    entry.project_id,
                    json.dumps(entry.details),
                )
        except Exception as e:
            logger.warning(f"Failed to write audit log to DB: {e}")

    async def get_events(
        self,
        plugin_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[PluginAuditEntry]:
        """
        Query audit events.

        Args:
            plugin_id: Filter by plugin ID.
            user_id: Filter by user ID.
            project_id: Filter by project ID.
            event_type: Filter by event type.
            since: Filter events after this time.
            until: Filter events before this time.
            limit: Maximum number of events.

        Returns:
            List of matching audit entries.
        """
        pool = await self._get_db_pool()
        if not pool:
            return []

        # Build query
        conditions = []
        params = []
        param_idx = 1

        if plugin_id:
            conditions.append(f"plugin_id = ${param_idx}")
            params.append(plugin_id)
            param_idx += 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if project_id:
            conditions.append(f"project_id = ${param_idx}")
            params.append(project_id)
            param_idx += 1

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        if since:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(since)
            param_idx += 1

        if until:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(until)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT plugin_id, event_type, timestamp, user_id, project_id, details
            FROM plugin_audit_log
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            return [
                PluginAuditEntry(
                    plugin_id=row["plugin_id"],
                    event_type=row["event_type"],
                    timestamp=row["timestamp"],
                    user_id=row["user_id"],
                    project_id=row["project_id"],
                    details=json.loads(row["details"]) if row["details"] else {},
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to query audit log: {e}")
            return []

    async def get_plugin_history(
        self,
        plugin_id: str,
    ) -> list[PluginAuditEntry]:
        """Get full audit history for a plugin."""
        return await self.get_events(plugin_id=plugin_id, limit=1000)

    async def close(self) -> None:
        """Close database connections."""
        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None
