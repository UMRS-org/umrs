"""Connector sync runner (P2) — durable, isolated, deduplicating intake.

Executes one connector sync run end-to-end:

1. Load durable checkpoint (``SyncCheckpoint.cursor`` + metadata) from the
   sync-state store (migration 0063) — survives restart.
2. Stream ``Connector.fetch`` documents, applying **content-hash dedup**
   (``sha256(source_id + content)`` against ``connector_seen_hashes``).
3. Forward unseen documents to the ingest callback (knowledge ingestion or
   an operator-provided sink; default collects without side effects).
4. Persist cursor, stats, errors, and dead letters (DLQ JSONB) — one
   failing connector never blocks others; failures are replayable rows.

No knowledge-job dependency: the runner is usable stand-alone and by the
future durable connector_sync job executor (I2).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from . import Connector, ConnectorConfig, Document, SyncCheckpoint
from .intake_gate import IntakeGate, IntakePolicy
from .sync_state import SyncStateRecord

logger = logging.getLogger(__name__)

_DEAD_LETTER_CAP = 500  # keep DLQ rows bounded; oldest dropped first


@dataclass
class SyncRunReport:
    """Outcome of one connector sync run."""

    connector_name: str
    status: str  # "ok" | "error"
    docs_seen: int = 0
    docs_synced: int = 0
    docs_skipped_dup: int = 0
    docs_failed: int = 0
    docs_blocked: int = 0
    new_cursor: str | None = None
    dead_letters: list[dict[str, str]] = field(default_factory=list)
    scan_patterns: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_name": self.connector_name,
            "status": self.status,
            "docs_seen": self.docs_seen,
            "docs_synced": self.docs_synced,
            "docs_skipped_dup": self.docs_skipped_dup,
            "docs_failed": self.docs_failed,
            "docs_blocked": self.docs_blocked,
            "new_cursor": self.new_cursor,
            "dead_letters": list(self.dead_letters),
            "scan_patterns": dict(self.scan_patterns),
            "error": self.error,
        }


def _doc_hash(doc: Document) -> str:
    """Deterministic dedup key: identity (source id) + content payload."""
    return hashlib.sha256(f"{doc.source_id}:{doc.content}".encode()).hexdigest()


def _emit_metrics(
    config: ConnectorConfig, report: SyncRunReport, elapsed: float
) -> None:
    """Record X1 connector metrics; instrumentation never breaks a sync."""
    try:
        from ..metrics import METRICS

        ctype = config.connector_type.value
        METRICS.connector_sync(ctype, report.status, elapsed)
        METRICS.connector_docs(
            ctype,
            synced=report.docs_synced,
            dup=report.docs_skipped_dup,
            failed=report.docs_failed,
            blocked=report.docs_blocked,
        )
    except Exception:  # noqa: BLE001 — metrics must not break syncs
        logger.debug("connector metrics emission failed", exc_info=True)


class ConnectorSyncRunner:
    """Executes connector syncs with durable checkpoints and dedup."""

    def __init__(self, store, policy: IntakePolicy | None = None) -> None:
        self._store = store
        self._gate = IntakeGate(policy)

    def run(
        self,
        connector: Connector,
        config: ConnectorConfig,
        *,
        ingest: Callable[[Document], None] | None = None,
        docs_iter: Iterator[Document] | None = None,
    ) -> SyncRunReport:
        """Run one sync; ingest callback receives unseen docs only.

        ``docs_iter`` is a test seam: when provided it is used instead of
        ``connector.fetch`` (avoids mocking connector internals).
        """
        name = config.name
        report = SyncRunReport(connector_name=name, status="running")
        prior = self._store.get(name)
        checkpoint = _checkpoint_from(prior, name)
        ingest_fn = ingest or (lambda doc: None)

        if config.enabled is False:
            report.status = "skipped_disabled"
            _emit_metrics(config, report, 0.0)
            return report

        import time as _time

        start = _time.monotonic()

        # Provisional row first: seen-hashFK requires the state row to exist
        # before any dedup insert lands (also makes the run visible mid-sync).
        self._persist(name, config, checkpoint, prior, report)

        try:
            source = (
                docs_iter
                if docs_iter is not None
                else connector.fetch(config, checkpoint)
            )
            self._consume(source, name, ingest_fn, report)
        except Exception as exc:  # noqa: BLE001 — isolation: mark, don't raise
            logger.warning("connector %s sync failed: %s", name, exc)
            report.status = "error"
            report.error = f"{exc.__class__.__name__}: {exc}"
        else:
            report.status = "ok"

        elapsed = _time.monotonic() - start
        self._persist(name, config, checkpoint, prior, report)
        _emit_metrics(config, report, elapsed)
        return report

    # ── internals ─────────────────────────────────────────────────────────

    def _consume(self, source, name: str, ingest_fn, report: SyncRunReport) -> None:
        """Stream documents; gate first (I2), then dedup, ingest, collect DLQ."""
        for doc in source:
            report.docs_seen += 1
            decision = self._gate.inspect(doc, doc_index=report.docs_seen - 1)
            if decision.action != "allow":
                report.docs_blocked += 1
                pat = decision.pattern or "gate:block"
                report.scan_patterns[pat] = report.scan_patterns.get(pat, 0) + 1
                report.dead_letters.append(
                    {
                        "source_id": doc.source_id,
                        "blocked_by": pat,
                        "reason": decision.reason,
                    }
                )
                continue
            try:
                digest = _doc_hash(doc)
                if self._store.seen(name, digest):
                    report.docs_skipped_dup += 1
                    continue
                ingest_fn(doc)
                self._store.mark_seen(name, digest)
                report.docs_synced += 1
            except Exception as exc:  # noqa: BLE001 — dead-letter and continue
                report.docs_failed += 1
                report.dead_letters.append(
                    {
                        "source_id": doc.source_id,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )

    def _persist(
        self,
        name: str,
        config: ConnectorConfig,
        checkpoint: SyncCheckpoint,
        prior: SyncStateRecord | None,
        report: SyncRunReport,
    ) -> None:
        """Write cursor + stats + DLQ back to the durable store."""
        dlq = ((prior.dlq if prior else []) + report.dead_letters)[_DLQ_CAP_DROP:]
        checkpoint_payload = dict((prior.checkpoint_json if prior else {}))
        checkpoint_payload.update(checkpoint.metadata)
        if report.docs_blocked:
            checkpoint_payload["last_scan"] = {
                "blocked": report.docs_blocked,
                "patterns": dict(report.scan_patterns),
            }
        self._store.put(
            SyncStateRecord(
                name=name,
                connector_type=config.connector_type.value,
                cursor=report.new_cursor
                or checkpoint.cursor
                or (prior.cursor if prior else None),
                last_sync_at=datetime.now(timezone.utc).isoformat(),
                last_status=report.status,
                docs_synced=(prior.docs_synced if prior else 0) + report.docs_synced,
                errors_count=(prior.errors_count if prior else 0) + report.docs_failed,
                dlq=dlq,
                checkpoint_json=checkpoint_payload,
                last_error=report.error,
            )
        )


_DLQ_CAP_DROP = -_DEAD_LETTER_CAP


def _checkpoint_from(prior: SyncStateRecord | None, name: str) -> SyncCheckpoint:
    """Reconstruct the SyncCheckpoint a connector expects."""
    if prior is None:
        return SyncCheckpoint(connector_name=name)
    return SyncCheckpoint(
        connector_name=prior.name,
        last_sync=_parse_dt(prior.last_sync_at),
        cursor=prior.cursor,
        docs_synced=prior.docs_synced,
        errors=prior.errors_count,
        metadata=dict(prior.checkpoint_json or {}),
    )


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = ["ConnectorSyncRunner", "SyncRunReport"]
