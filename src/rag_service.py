import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import BaseRetriever, Document

from umrs.config import (
    get_default_json_server_file,
    get_default_umrs_cfg_dir,
    get_umrs_cfg_env_var_name,
)
from umrs.db_store import store_db_model  # noqa: F401  # imported for side effects / type hints
from umrs.json_utils import load_server_cfg_file
from umrs.sql import ensure_pgvector_enabled  # noqa: F401

import server_umrs


class RAGSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query to search with")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    user: Optional[str] = Field(None, description="Optional UMRS username filter")
    modality: Optional[str] = Field("text", description="Data modality (e.g., 'text')")
    bucket_alias: Optional[str] = Field(
        None, description="Optional bucket alias to scope the search"
    )
    xp_name: Optional[str] = Field(
        None, description="Optional experience name to scope the search"
    )


class RAGSearchResultItem(BaseModel):
    id: int
    score: float
    content: str
    metadata: Dict[str, Any]


class RAGSearchResponse(BaseModel):
    results: List[RAGSearchResultItem]


def _resolve_server_config_path() -> str:
    """Resolve the UMRS server configuration file path.

    This mirrors the logic in server_umrs.main() but without depending on CLI.
    """

    umrs_cfg_dir = os.getenv(get_umrs_cfg_env_var_name())
    if umrs_cfg_dir and os.path.isdir(umrs_cfg_dir):
        return os.path.join(umrs_cfg_dir, get_default_json_server_file())

    default_dir = get_default_umrs_cfg_dir()
    if os.path.isdir(default_dir):
        return os.path.join(default_dir, get_default_json_server_file())

    raise RuntimeError(
        f"No valid UMRS configuration directory. Expected either {default_dir} "
        f"or the directory pointed to by {get_umrs_cfg_env_var_name()}"
    )


def _build_cfg_store(configure_file: str) -> Dict[str, Any]:
    """Load UMRS configuration and build the cfg_store structure.

    This is a lightweight variant of perform_initial_config that:
    - Loads the server JSON configuration.
    - Runs configure_user for each user to populate db_model_store.
    It does NOT start message queues or background tasks.
    """

    if not os.path.exists(configure_file):
        raise RuntimeError(f"Configuration file '{configure_file}' not found.")

    _admin_s3_configuration_file, users, _server_cfg = load_server_cfg_file(
        configure_file
    )

    cfg_store: Dict[str, Any] = defaultdict(dict)
    for user in users:
        server_umrs.configure_user(user, cfg_store)
    return cfg_store


class UMRSPgVectorRetriever(BaseRetriever):
    """LangChain-compatible retriever over UMRS pgvector tables.

    For now this retriever assumes a single matching table per query,
    identified by (user, xp_name, bucket_alias, modality).
    """

    def __init__(
        self,
        *,
        db_model_store: Dict[str, Dict[str, Any]],
        embeddings: SentenceTransformerEmbeddings,
        user: Optional[str] = None,
        xp_name: Optional[str] = None,
        bucket_alias: Optional[str] = None,
        modality: Optional[str] = "text",
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.db_model_store = db_model_store
        self.embeddings = embeddings
        self.user = user
        self.xp_name = xp_name
        self.bucket_alias = bucket_alias
        self.modality = modality
        self.top_k = top_k

    def _select_entry(self) -> Dict[str, Any]:
        """Select a single matching entry from db_model_store.

        We currently require that filters resolve to exactly one entry to
        keep SQL simpler. This can be generalized later.
        """

        matches: List[Dict[str, Any]] = []
        for _path, entry in self.db_model_store.items():
            if self.user and entry.get("s3_username") != self.user:
                continue
            if self.xp_name and entry.get("xp_name") != self.xp_name:
                continue
            if self.bucket_alias and entry.get("bucket_alias") != self.bucket_alias:
                continue
            if self.modality and entry.get("modality_name") != self.modality:
                continue
            if not entry.get("store_embeddings_flag", False):
                continue
            matches.append(entry)

        if not matches:
            raise ValueError(
                "No matching embedding table found for the given filters. "
                "Please check user/xp_name/bucket_alias/modality."
            )
        if len(matches) > 1:
            raise ValueError(
                "Multiple embedding tables match the given filters; "
                "please narrow the query (e.g. by xp_name or bucket_alias)."
            )
        return matches[0]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        from psycopg2 import sql

        if not self.db_model_store:
            raise ValueError("UMRS db_model_store is empty; server may not be configured.")

        entry = self._select_entry()
        data_base = entry["data_base"]
        table_name = entry["table_name"]
        xp_name = entry.get("xp_name")
        username = entry.get("s3_username")
        bucket_alias = entry.get("bucket_alias")

        # Compute query embedding using LangChain
        query_vec = self.embeddings.embed_query(query)

        db_pool = data_base.db_pool
        conn = db_pool.getconn()
        cur = conn.cursor()
        try:
            # For text modalities we expect _content and chunk_* columns.
            has_content = data_base.has_column(table_name, "_content")

            if has_content:
                query_sql = sql.SQL(
                    """
                    SELECT
                        id,
                        s3_object,
                        embedding <-> %s AS distance,
                        _content,
                        chunk_start,
                        chunk_end,
                        chunk_md5,
                        checksum,
                        model
                    FROM {table}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> %s
                    LIMIT %s;
                    """
                ).format(table=sql.Identifier(table_name))
            else:
                query_sql = sql.SQL(
                    """
                    SELECT
                        id,
                        s3_object,
                        embedding <-> %s AS distance,
                        NULL AS _content,
                        NULL AS chunk_start,
                        NULL AS chunk_end,
                        NULL AS chunk_md5,
                        checksum,
                        model
                    FROM {table}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> %s
                    LIMIT %s;
                    """
                ).format(table=sql.Identifier(table_name))

            cur.execute(query_sql, (query_vec, query_vec, self.top_k))
            rows = cur.fetchall()
        finally:
            cur.close()
            db_pool.putconn(conn)

        docs: List[Document] = []
        for row in rows:
            (
                row_id,
                s3_object,
                distance,
                content,
                chunk_start,
                chunk_end,
                chunk_md5,
                checksum,
                model_name,
            ) = row

            # Convert distance to a score where higher is better.
            # pgvector <-> is a distance (lower is better), so we invert.
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0

            page_content = content or ""
            metadata: Dict[str, Any] = {
                "id": row_id,
                "s3_object": s3_object,
                "distance": float(distance) if distance is not None else None,
                "checksum": checksum,
                "model": model_name,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "chunk_md5": chunk_md5,
                "user": username,
                "bucket_alias": bucket_alias,
                "xp_name": xp_name,
            }

            docs.append(Document(page_content=page_content, metadata=metadata))

        return docs


app = FastAPI(title="UMRS RAG Service", version="0.1.0")


DB_MODEL_STORE: Dict[str, Dict[str, Any]] = {}
EMBEDDINGS: Optional[SentenceTransformerEmbeddings] = None


@app.on_event("startup")
def startup_event() -> None:
    global DB_MODEL_STORE, EMBEDDINGS

    config_path = _resolve_server_config_path()
    cfg_store = _build_cfg_store(config_path)
    DB_MODEL_STORE = cfg_store.get("db_model_store", {})

    if not DB_MODEL_STORE:
        raise RuntimeError(
            "UMRS configuration loaded but db_model_store is empty. "
            "Ensure the server configuration lists at least one user/experience "
            "with store_embeddings = 'yes'."
        )

    model_name = os.getenv(
        "UMRS_RAG_EMBEDDING_MODEL",
        # Default to a commonly used sentence-transformer; this should match
        # the model used for ingestion when possible.
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    EMBEDDINGS = SentenceTransformerEmbeddings(model_name=model_name)


@app.post("/rag/search", response_model=RAGSearchResponse)
def rag_search(payload: RAGSearchRequest) -> RAGSearchResponse:
    if EMBEDDINGS is None:
        raise HTTPException(status_code=500, detail="Embeddings backend not initialized")

    try:
        retriever = UMRSPgVectorRetriever(
            db_model_store=DB_MODEL_STORE,
            embeddings=EMBEDDINGS,
            user=payload.user,
            xp_name=payload.xp_name,
            bucket_alias=payload.bucket_alias,
            modality=payload.modality,
            top_k=payload.top_k,
        )
        docs = retriever.get_relevant_documents(payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results: List[RAGSearchResultItem] = []
    for doc in docs:
        meta = dict(doc.metadata)
        row_id = int(meta.get("id", 0))
        # Reconstruct score from stored distance where possible
        distance = meta.get("distance")
        score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
        results.append(
            RAGSearchResultItem(
                id=row_id,
                score=score,
                content=doc.page_content,
                metadata=meta,
            )
        )

    return RAGSearchResponse(results=results)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("rag_service:app", host="0.0.0.0", port=8081, reload=False)
