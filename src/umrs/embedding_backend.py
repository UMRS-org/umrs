import uuid
import json
from typing import List, Any, Optional

import zmq

from umrs.embeddings import compute_and_store_embedding
from umrs.sql import store_embeddings_in_db


class EmbeddingBackend:
    """Abstract backend for computing and storing embeddings.

    Subclasses must implement `compute_and_store`.
    """

    def compute_and_store(
        self,
        path_to_bucket: str,
        s3_keys: List[str],
        data_base: Any,
        table_name: str,
        model: Any,
        device: str,
        modality_name: str,
        s3_connection: Any,
    ) -> None:
        raise NotImplementedError


class LocalEmbeddingBackend(EmbeddingBackend):
    """Local backend that uses the existing in-process implementation.

    This preserves current behavior by delegating directly to
    `compute_and_store_embedding`.
    """

    def compute_and_store(
        self,
        path_to_bucket: str,
        s3_keys: List[str],
        data_base: Any,
        table_name: str,
        model: Any,
        device: str,
        modality_name: str,
        s3_connection: Any,
    ) -> None:
        compute_and_store_embedding(
            path_to_bucket,
            s3_keys,
            data_base,
            table_name,
            model,
            device,
            modality_name,
            s3_connection,
        )


class RemoteEmbeddingBackend(EmbeddingBackend):
    """Remote backend that delegates embedding computation to a worker.

    The worker is a separate process reachable over ZeroMQ that performs
    compute-only embedding work and returns embeddings + metadata.
    This backend is responsible for storing results in the database.
    """

    def __init__(
        self,
        worker_address: str,
        timeout_sec: float,
        local_fallback: Optional[EmbeddingBackend] = None,
    ) -> None:
        self._worker_address = worker_address
        self._timeout_sec = timeout_sec
        self._local_fallback = local_fallback

        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)

        # Configure timeouts in milliseconds
        timeout_ms = int(timeout_sec * 1000)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        socket.connect(worker_address)

        self._context = context
        self._socket = socket

    def compute_and_store(
        self,
        path_to_bucket: str,
        s3_keys: List[str],
        data_base: Any,
        table_name: str,
        model: Any,
        device: str,
        modality_name: str,
        s3_connection: Any,
    ) -> None:
        job_id = uuid.uuid4().hex

        payload = {
            "job_id": job_id,
            "command": "compute_embeddings",
            "path_to_bucket": path_to_bucket,
            "s3_keys": s3_keys,
            "modality_name": modality_name,
            "model_name": getattr(model, "model_name", None),
        }

        try:
            self._socket.send_json(payload)
            reply = self._socket.recv_json()
        except Exception as exc:
            if self._local_fallback is not None:
                # Fallback to local backend on communication failure
                self._local_fallback.compute_and_store(
                    path_to_bucket,
                    s3_keys,
                    data_base,
                    table_name,
                    model,
                    device,
                    modality_name,
                    s3_connection,
                )
                return
            raise RuntimeError(f"RemoteEmbeddingBackend communication error: {exc}") from exc

        if not isinstance(reply, dict):
            raise RuntimeError("RemoteEmbeddingBackend: invalid reply type from worker (expected dict)")

        if reply.get("job_id") != job_id:
            raise RuntimeError("RemoteEmbeddingBackend: job_id mismatch in worker reply")

        status = reply.get("status")
        if status != "ok":
            error_msg = reply.get("error", "unknown error")
            if self._local_fallback is not None:
                self._local_fallback.compute_and_store(
                    path_to_bucket,
                    s3_keys,
                    data_base,
                    table_name,
                    model,
                    device,
                    modality_name,
                    s3_connection,
                )
                return
            raise RuntimeError(f"RemoteEmbeddingBackend worker error: {error_msg}")

        embeddings = reply.get("embeddings")
        checksums = reply.get("checksums")
        metadata = reply.get("metadata")

        if embeddings is None or checksums is None or metadata is None:
            raise RuntimeError("RemoteEmbeddingBackend: reply missing required fields")

        if not (
            isinstance(embeddings, list)
            and isinstance(checksums, list)
            and isinstance(metadata, list)
        ):
            raise RuntimeError("RemoteEmbeddingBackend: embeddings/checksums/metadata must be lists")

        if not (
            len(embeddings) == len(checksums) == len(metadata)
        ):
            raise RuntimeError("RemoteEmbeddingBackend: mismatched lengths in reply data")

        store_embeddings_in_db(
            data_base,
            embeddings,
            table_name,
            path_to_bucket,
            s3_keys,
            modality_name,
            checksums,
            getattr(model, "model_name", None),
            metadata_list=metadata,
        )


def create_embedding_backend_from_config(cfg: dict) -> EmbeddingBackend:
    """Factory to create an embedding backend from server configuration.

    The configuration is expected to contain optional keys:
    - embedding_backend: "local" (default) or "remote"
    - embedding_worker_address: ZeroMQ address of the worker
    - embedding_worker_timeout_sec: timeout in seconds
    """

    backend_type = cfg.get("embedding_backend", "local").lower()

    if backend_type == "remote":
        addr = cfg.get("embedding_worker_address", "tcp://umrs-embed-worker:6000")
        timeout = cfg.get("embedding_worker_timeout_sec", 600)
        # Use a local backend as a safe fallback when configured
        local_backend = LocalEmbeddingBackend()
        return RemoteEmbeddingBackend(
            worker_address=addr,
            timeout_sec=float(timeout),
            local_fallback=local_backend,
        )

    return LocalEmbeddingBackend()
