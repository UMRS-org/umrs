"""Canonical mint for RAG uploads (ADR-027 G5 / P1b wave 3).

Ownership (G1'): the rag upload EXECUTOR (``umrs.api.rag_service``) owns the
canonical write — only the executor holds the computed embeddings, so only
it can write node-keyed vectors into migration-0075 rows. The API proxy's
P1a mint (`api.rag._mint_knowledge_nodes`) stays in place for one release;
ids are identical and every write here is upsert-shaped, so the two paths
are idempotent under double-write during the transition.

Contract: best-effort. The caller wraps this in try/except and logs — a
mint failure must never fail an otherwise-successful upload (mirrors the
proxy's `_mint_knowledge_nodes_best_effort` contract).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from umrs.knowledge.graph_registry import default_graph_id

logger = logging.getLogger(__name__)

_MINT_CAP = 100  # mirror the proxy cap: oversized uploads mint the first N chunks


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _upsert_artifact(
    cur: Any,
    *,
    artifact_id: str,
    content_hash: str,
    project_id: str,
    document_id: str,
    filename: str,
    mime_type: str,
) -> None:
    import json

    ts = _iso_now()
    cur.execute(
        """
        INSERT INTO knowledge_artifacts
            (artifact_id, content_hash, kind, project_id, source_ref,
             size_bytes, mime_type, status, metadata, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s::jsonb, %s, %s)
        ON CONFLICT (artifact_id) DO UPDATE
        SET updated_at = %s, status = 'active'
        """,
        (
            artifact_id,
            content_hash,
            "document",
            project_id,
            f"rag://{document_id}",
            0,
            mime_type,
            json.dumps(
                {
                    "origin": "rag.upload",
                    "filename": filename,
                    "document_id": document_id,
                }
            ),
            ts,
            ts,
            ts,
        ),
    )


def _upsert_node(
    cur: Any,
    *,
    node_id: str,
    project_id: str,
    label: str,
    node_type: str,
    content: dict[str, Any],
    content_hash: str,
    graph_id: str,
) -> None:
    import json

    ts = _iso_now()
    cur.execute(
        """
        INSERT INTO knowledge_nodes
            (node_id, project_id, label, node_type, content, content_hash,
             source_ir, created_at, updated_at, graph_id)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, NULL, %s, %s, %s)
        ON CONFLICT (node_id) DO UPDATE
        SET content = %s::jsonb, content_hash = %s, source_ir = NULL, updated_at = %s
        """,
        (
            node_id,
            project_id,
            label,
            node_type,
            json.dumps(content),
            content_hash,
            ts,
            ts,
            graph_id,
            json.dumps(content),
            content_hash,
            ts,
        ),
    )


def _upsert_part_of_edge(
    cur: Any, *, chunk_id: str, doc_node_id: str, project_id: str, graph_id: str
) -> None:
    import json

    cur.execute(
        """
        INSERT INTO knowledge_edges
            (edge_id, project_id, source_id, target_id, relation_type, content, graph_id)
        VALUES (%s, %s, %s, %s, 'part_of', %s::jsonb, %s)
        ON CONFLICT (edge_id) DO UPDATE
        SET content = %s::jsonb
        """,
        (
            f"{chunk_id}-part-of",
            project_id,
            chunk_id,
            doc_node_id,
            json.dumps({"kind": "part_of", "origin": "rag.upload"}),
            graph_id,
            json.dumps({"kind": "part_of", "origin": "rag.upload"}),
        ),
    )


def _upsert_embedding(
    cur: Any,
    *,
    chunk_id: str,
    model_name: str,
    embedding: Any,
    content_hash: str,
) -> None:
    vector_literal = "[" + ",".join(repr(float(v)) for v in embedding) + "]"
    cur.execute(
        """
        INSERT INTO knowledge_node_embeddings
            (node_id, model_name, dimension, embedding, content_hash)
        VALUES (%s, %s, %s, %s::vector, %s)
        ON CONFLICT (node_id, model_name) DO UPDATE SET
            dimension    = EXCLUDED.dimension,
            embedding    = EXCLUDED.embedding,
            content_hash = EXCLUDED.content_hash,
            updated_at   = now()
        """,
        (chunk_id, model_name, len(embedding), vector_literal, content_hash),
    )


def mint_rag_upload(
    pool: Any,
    *,
    project_id: str,
    document_id: str,
    filename: str,
    mime_type: str,
    chunks: list[dict[str, Any]],
    embeddings: list[Any],
    model_name: str | None,
) -> int:
    """Mint artifact + doc node + chunk nodes/part_of edges + node vectors.

    Args:
        pool: sync psycopg2 pool against the CENTRAL umrs database (where the
              canonical 0055/0069/0075 tables live).
        chunks/embeddings: the same lists the executor just persisted into the
              rag corpus table — chunk-node content follows the proxy scheme.

    Returns the number of minted chunk nodes. Skips (returns 0) when the pool
    or model name is unavailable — the upload itself already succeeded.
    """
    if pool is None or not model_name:
        return 0

    gid = default_graph_id(project_id)
    doc_node_id = f"ragdoc-{document_id}"
    content_hash = hashlib.sha256(
        f"{document_id}:{filename}".encode("utf-8")
    ).hexdigest()[:16]
    artifact_id = f"artifact-rag-{document_id}"
    minted = 0

    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            _upsert_artifact(
                cur,
                artifact_id=artifact_id,
                content_hash=content_hash,
                project_id=project_id,
                document_id=document_id,
                filename=filename,
                mime_type=mime_type,
            )
            _upsert_node(
                cur,
                node_id=doc_node_id,
                project_id=project_id,
                label=filename,
                node_type="file",
                content={
                    "kind": "document",
                    "document_id": document_id,
                    "filename": filename,
                    "origin": "rag.upload",
                    "artifact_id": artifact_id,
                },
                content_hash=content_hash,
                graph_id=gid,
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                if i >= _MINT_CAP:
                    break
                chunk_id = f"{doc_node_id}-chunk{i}"
                chunk_md5 = str(chunk.get("md5") or "")
                _upsert_node(
                    cur,
                    node_id=chunk_id,
                    project_id=project_id,
                    label=f"chunk {i} of {filename}",
                    node_type="entity",
                    content={
                        "kind": "chunk",
                        "document_id": document_id,
                        "chunk_index": i,
                        "chunk_start": chunk.get("start"),
                        "chunk_end": chunk.get("end"),
                        "chunk_md5": chunk_md5,
                        "origin": "rag.upload",
                    },
                    content_hash=chunk_md5,
                    graph_id=gid,
                )
                _upsert_part_of_edge(
                    cur,
                    chunk_id=chunk_id,
                    doc_node_id=doc_node_id,
                    project_id=project_id,
                    graph_id=gid,
                )
                _upsert_embedding(
                    cur,
                    chunk_id=chunk_id,
                    model_name=model_name,
                    embedding=emb,
                    content_hash=chunk_md5,
                )
                minted += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
    logger.info("minted %d canonical nodes (with vectors) for %s", minted, document_id)
    return minted
