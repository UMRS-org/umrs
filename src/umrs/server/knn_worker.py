"""Worker for the KNN command.

Loads the query object from S3 (reuses the cached embedding when the ETag
matches, otherwise computes it via the bucket's model), performs a
cosine-distance kNN search via pgvector, and sends the JSON-encoded result
back to the caller over ZMQ.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time

from psycopg2 import sql

from umrs.core.TaskCmdStatus import TaskStatus

logger = logging.getLogger(__name__)


def knn_search_worker(
    active_threads,
    user,
    n,
    s3_object,
    bucket_name,
    buckets_cfg,
    control_mq,
    zmq_sender_id,
    key,
    subdir=None,
    update_status_callback=None,
):
    """Worker function for the KNN command, runs in a thread pool."""
    from umrs.shared.storage.s3 import load_data_from_s3
    from umrs.shared.pretty_print import pretty_print_knn

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    t0 = time.perf_counter()
    entry = buckets_cfg.get(bucket_name)
    if entry is None:
        result = json.dumps({"status": "error", "message": f"No bucket entry for: {bucket_name}"}, ensure_ascii=False)
        with contextlib.suppress(RuntimeError):
            control_mq.send_string(result, zmq_sender_id)
        if update_status_callback:
            update_status_callback(TaskStatus.ERROR)
        return

    table_name = entry.table_name
    model = entry.model
    s3 = entry.s3_connection
    modality_name = entry.modality_name

    # Load object from S3 to get ETag + raw data
    try:
        objects, _, checksums, sizes, _ = load_data_from_s3(s3, bucket_name, [s3_object], modality_name)
        etag = checksums[0]
        obj_data = objects[0]
        query_size_bytes = sizes[0]
    except Exception as exc:
        logger.exception("KNN: failed to load %s from S3: %s", s3_object, exc)
        result = json.dumps({"status": "error", "message": f"S3 load failed: {exc}"}, ensure_ascii=False)
        with contextlib.suppress(RuntimeError):
            control_mq.send_string(result, zmq_sender_id)
        if update_status_callback:
            update_status_callback(TaskStatus.ERROR)
        return

    conn = entry.data_base.get_conn()
    try:
        # Fast path: reuse cached embedding if ETag matches.
        # DB stores s3_object as "bucket_name/s3_key" (no s3:// prefix).
        db_s3_object = f"{bucket_name}/{s3_object}"
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT embedding FROM {} WHERE s3_object = %s AND checksum = %s LIMIT 1")
                .format(sql.Identifier(table_name)),
                (db_s3_object, etag),
            )
            row = cur.fetchone()

        cached = row is not None
        if cached:
            query_embedding = row[0]
        else:
            embeddings = model.compute_embeddings([obj_data])
            emb = embeddings[0]
            query_embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        # Cosine kNN — the HNSW index uses vector_cosine_ops so <=> is indexed.
        with conn.cursor() as cur:
            if subdir:
                prefix = f"{bucket_name}/{subdir}/%"
                cur.execute(
                    sql.SQL("""
                        SELECT s3_object, 1.0 - (embedding <=> %s::vector) AS score
                        FROM {}
                        WHERE s3_object LIKE %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """).format(sql.Identifier(table_name)),
                    (query_embedding, prefix, query_embedding, n),
                )
            else:
                cur.execute(
                    sql.SQL("""
                        SELECT s3_object, 1.0 - (embedding <=> %s::vector) AS score
                        FROM {}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """).format(sql.Identifier(table_name)),
                    (query_embedding, query_embedding, n),
                )
            rows = cur.fetchall()

        results_list = [{"object": r[0], "score": float(r[1])} for r in rows]

        if subdir:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {} WHERE s3_object LIKE %s")
                    .format(sql.Identifier(table_name)),
                    (f"{bucket_name}/{subdir}/%",),
                )
                n_indexed = cur.fetchone()[0]
        else:
            n_indexed = entry.num_objects

        duration_s = round(time.perf_counter() - t0, 3)
        result = json.dumps(
            {
                "status": "ok",
                "n": n,
                "query": s3_object,
                "results": results_list,
                "duration_s": duration_s,
                "cached": cached,
                "n_indexed": n_indexed,
                "summary": pretty_print_knn(s3_object, bucket_name, n, results_list, duration_s, cached, n_indexed),
            },
            ensure_ascii=False,
        )
        if update_status_callback:
            update_status_callback(TaskStatus.COMPLETED)

    except Exception as exc:
        logger.exception("KNN search failed for %s: %s", s3_object, exc)
        result = json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)
        if update_status_callback:
            update_status_callback(TaskStatus.ERROR)
    finally:
        entry.data_base.put_conn(conn)

    with contextlib.suppress(RuntimeError):
        control_mq.send_string(result, zmq_sender_id)

    active_threads.update_status_to_done()
    active_threads.record_status_for_users_by_key([user], key)
    active_threads.remove_from_running_list()
