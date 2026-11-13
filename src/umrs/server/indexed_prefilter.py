"""Skip objects that are already indexed before PROCESS_OBJECTS_BATCH runs.

Port of upstream 85a1cf06 / 8f9fe001 / b0c2d345 (plan U1 A5). An object is
skipped only when its bucket's own table already holds it
(``s3_object = "<bucket>/<key>"``) with the object's current S3 ETag as the
checksum and, where the table records one, the bucket's current model on
every row. Scope rules:

* only clients the ownership rule admits for every bucket they name are
  filtered (``processing._refused_clients``); a refused client passes
  through untouched so ``prepare_model_groups`` still refuses it, and
  another tenant's index never answers for it;
* only plain embedding buckets (store_embeddings on, the model gets
  pre-loaded objects): user scripts, workflows, file-producing and
  direct-compute models always run, since a re-upload may be a re-trigger;
* any S3 or DB error for a bucket means nothing is skipped there.

``UMRS_PREFILTER_INDEXED=0`` turns the filter off.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from psycopg2 import sql as pgsql

logger = logging.getLogger(__name__)

_HEAD_WORKERS = 10
_OWN_IO_MODEL_TYPES = frozenset({"user_script", "user_workflow", "bentoml"})
_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def prefilter_enabled() -> bool:
    """``UMRS_PREFILTER_INDEXED`` (default on)."""
    raw = os.getenv("UMRS_PREFILTER_INDEXED", "1").strip().lower()
    return raw not in _OFF_VALUES


@dataclass
class PrefilterResult:
    """What is left to process, and what the filter took out (per bucket)."""

    entries: list
    submitted: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    done_clients: list[str] = field(default_factory=list)

    def fully_skipped(self) -> dict[str, int]:
        """Buckets whose every submitted object was already indexed."""
        return {
            bucket: count for bucket, count in self.skipped.items()
            if count and count == self.submitted.get(bucket)
        }


def filter_already_indexed(buckets_cfg, client_entries) -> PrefilterResult:
    """``client_entries`` (``[(client_id, [s3_uri, ...])]``) minus indexed objects."""
    from umrs.server.processing import _refused_clients

    refused = _refused_clients(client_entries)
    candidates = _candidate_keys(client_entries, refused)
    skip: set[tuple[str, str]] = set()
    for bucket, keys in candidates.items():
        indexed = _indexed_keys(buckets_cfg.get(bucket), bucket, sorted(keys))
        skip.update((bucket, key) for key in indexed)
    result = _split_entries(client_entries, refused, skip)
    result.submitted = {bucket: len(keys) for bucket, keys in candidates.items()}
    if skip:
        logger.info("[PROCESS] skipping %d already-indexed object(s): %s",
                    len(skip), result.skipped)
    return result


def _candidate_keys(client_entries, refused) -> dict[str, set[str]]:
    """Unique keys per bucket from the admitted clients."""
    from umrs.server.processing import _split_s3_object

    candidates: dict[str, set[str]] = {}
    for client_id, objects in client_entries:
        if client_id in refused:
            continue
        for s3_uri in objects:
            bucket, key = _split_s3_object(s3_uri)
            candidates.setdefault(bucket, set()).add(key)
    return candidates


def _split_entries(client_entries, refused, skip) -> PrefilterResult:
    from umrs.server.processing import _split_s3_object

    result = PrefilterResult(entries=[])
    for client_id, objects in client_entries:
        if client_id in refused:
            result.entries.append((client_id, objects))
            continue
        kept = [obj for obj in objects if _split_s3_object(obj) not in skip]
        if kept:
            result.entries.append((client_id, kept))
        else:
            result.done_clients.append(client_id)
    for bucket, _key in skip:
        result.skipped[bucket] = result.skipped.get(bucket, 0) + 1
    return result


def plain_embedding_bucket(entry) -> bool:
    """Plain embedding bucket whose objects are pre-loaded for the model."""
    if entry is None or not getattr(entry, "store_embeddings_flag", False):
        return False
    model = getattr(entry, "model", None)
    if model is None or getattr(model, "return_type", None) == "file":
        return False
    if getattr(model, "type", None) in _OWN_IO_MODEL_TYPES:
        return False
    from umrs.inference.embeddings import is_direct_compute

    return not is_direct_compute(model, entry.modality_name, entry.task)


def _indexed_keys(entry, bucket, keys) -> set[str]:
    """Keys whose stored rows match the current ETag (and model)."""
    if not keys or not plain_embedding_bucket(entry):
        return set()
    try:
        etags = _head_etags(entry.s3_connection, bucket, keys)
        stored = _stored_rows(entry.data_base, entry.table_name, bucket, keys)
    except Exception:  # fail open: nothing skipped, all indexed
        logger.warning("[PROCESS] indexed pre-filter failed for bucket %s; "
                       "indexing every object", bucket, exc_info=True)
        return set()
    model_name = getattr(entry.model, "model_name", None)
    return {
        key for key in keys
        if _matches(etags.get(key), stored.get(f"{bucket}/{key}"), model_name)
    }


def _matches(etag, rows, model_name) -> bool:
    """Every stored row carries this ETag (and, when recorded, this model)."""
    if not etag or not rows:
        return False
    return all(
        row[0] == etag and (len(row) == 1 or row[1] == model_name) for row in rows
    )


def _head_etags(s3, bucket, keys) -> dict[str, str | None]:
    def _head(key):
        try:
            return key, s3.head_object(Bucket=bucket, Key=key)["ETag"].strip('"')
        except Exception:  # noqa: BLE001 — a missing/unreadable object is just not skipped
            return key, None

    with ThreadPoolExecutor(max_workers=min(_HEAD_WORKERS, len(keys))) as pool:
        return dict(pool.map(_head, keys))


def _stored_rows(data_base, table_name, bucket, keys) -> dict[str, set[tuple]]:
    """``"<bucket>/<key>"`` -> set of (checksum[, model]) rows in the table."""
    columns = ["s3_object", "checksum"]
    if data_base.has_column(table_name, "model"):
        columns.append("model")
    query = pgsql.SQL("SELECT {} FROM {} WHERE s3_object = ANY(%s)").format(
        pgsql.SQL(", ").join(pgsql.Identifier(c) for c in columns),
        pgsql.Identifier(table_name),
    )
    db_keys = [f"{bucket}/{key}" for key in keys]
    conn = data_base.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (db_keys,))
            rows = cur.fetchall()
    finally:
        # b0c2d345: back to the pool on every path (the pool also rolls
        # back a failed transaction on put).
        data_base.put_conn(conn)
    stored: dict[str, set[tuple]] = {}
    for row in rows:
        stored.setdefault(row[0], set()).add(tuple(row[1:]))
    return stored
