"""Stamp the canonical identity on natively ingested artifacts (ADR-032
addendum 7, precondition #1 v1 leg).

Ingest stamps new rows; this runner stamps rows ingested before that. It
reads each unstamped artifact's bytes by ``source_ref`` and adds
``metadata.canonical_identity`` — ids are never rewritten. A row whose bytes
are unreadable (or over the read cap) is left unstamped: freezing the weaker
``physical`` identity would hide a later text match; the export keeps
computing it on the fly instead.

Usage (dry-run first, then --apply; DSN and S3 from the server's env)::

    python -m umrs.knowledge.identity_backfill --project-id <project> [--apply]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import canonical_identity

logger = logging.getLogger("umrs.knowledge.identity_backfill")

_BATCH = 200


@dataclass
class BackfillOutcome:
    scanned: int = 0
    stamped: int = 0
    unreadable: int = 0
    by_normalization: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "stamped": self.stamped,
            "unreadable": self.unreadable,
            "by_normalization": dict(sorted(self.by_normalization.items())),
        }


def _read(reader: Callable[[str], bytes | None], row: dict[str, Any]) -> bytes | None:
    if int(row.get("size_bytes") or 0) > canonical_identity.READ_CAP:
        return None
    return reader(str(row.get("source_ref") or ""))


async def _stamp_row(
    store: Any,
    reader: Callable[[str], bytes | None],
    row: dict[str, Any],
    outcome: BackfillOutcome,
    apply: bool,
) -> None:
    raw = _read(reader, row)
    if raw is None:
        outcome.unreadable += 1
        return
    block = canonical_identity.identity_of(raw, str(row.get("content_hash") or ""))
    norm = block.normalization
    outcome.by_normalization[norm] = outcome.by_normalization.get(norm, 0) + 1
    if apply and await store.stamp_canonical_identity(
        str(row["artifact_id"]), block.to_dict()
    ):
        outcome.stamped += 1


async def backfill(
    store: Any,
    reader: Callable[[str], bytes | None],
    *,
    project_id: str | None = None,
    apply: bool = False,
    batch: int = _BATCH,
) -> BackfillOutcome:
    """Stamp every unstamped active artifact (dry-run unless ``apply``)."""
    outcome = BackfillOutcome()
    after = ""
    while True:
        rows = await store.artifacts_missing_identity(project_id, after, batch)
        for row in rows:
            outcome.scanned += 1
            await _stamp_row(store, reader, row, outcome, apply)
        if len(rows) < batch:
            return outcome
        after = str(rows[-1]["artifact_id"])


async def _run(project_id: str | None, apply: bool) -> BackfillOutcome:
    from umrs.jobs.knowledge_adapters import _lazy_artifact_reader, _open_pool

    from .store import KnowledgeStore

    async with _open_pool() as pool:
        if pool is None:
            raise SystemExit("knowledge database unavailable (check the DSN env)")
        return await backfill(
            KnowledgeStore(pool),
            _lazy_artifact_reader(),
            project_id=project_id,
            apply=apply,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-id", default=None, help="one project (default: all)")
    parser.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    outcome = asyncio.run(_run(args.project_id, args.apply))
    print(json.dumps({"apply": args.apply, **outcome.to_dict()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI path
    raise SystemExit(main())
