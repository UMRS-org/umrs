import uuid
import json
import logging
from typing import List, Any, Optional

import zmq
import numpy as np

from umrs.embeddings import compute_and_store_embedding
from umrs.sql import store_embeddings_in_db
from umrs.s3_presign import generate_presigned_urls_for_keys

logger = logging.getLogger(__name__)

# avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from server_umrs import ServerCfg


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
    `compute_and_store_embedding` or 'compute_and_return_file'
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
        output_bucket: str,
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
            output_bucket,
        )




################# TODO (EJ)
# Now computing embedings needs the output bucket path  (set to `None` if there is no output generated)
# So the following code should be updated accordingly as above for local


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
        output_bucket: str,
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

        # Generate presigned GET URLs so the worker can download objects
        # without needing direct S3 credentials.
        try:
            presigned_urls = generate_presigned_urls_for_keys(
                s3_connection, path_to_bucket, s3_keys, operation="get",
            )
            payload["presigned_urls"] = presigned_urls
        except Exception as exc:
            logger.warning(
                "Failed to generate presigned URLs, worker will fall back "
                "to direct S3: %s", exc,
            )

        try:
            self._socket.send_json(payload)

            # The worker may reply with either a JSON error (single frame)
            # or a multipart message with a JSON header + binary payloads.
            frames = self._socket.recv_multipart()
            if len(frames) == 1:
                reply = json.loads(frames[0].decode("utf-8"))
                is_binary = False
            elif len(frames) == 4:
                header = json.loads(frames[0].decode("utf-8"))
                reply = header
                is_binary = True
            else:
                raise RuntimeError(
                    f"RemoteEmbeddingBackend: unexpected number of frames ({len(frames)}) from worker"
                )
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

        # For JSON replies (local/legacy), embeddings/checksums/metadata are lists
        # and everything is already present in `reply`.
        # For binary replies, we reconstruct the embeddings array and decode the
        # checksums/metadata from the additional frames.
        if 'is_binary' in locals() and is_binary:
            # frames[1] = embeddings binary, frames[2] = checksums JSON, frames[3] = metadata JSON
            embeddings_bytes = frames[1]
            checksums = json.loads(frames[2].decode("utf-8"))
            metadata = json.loads(frames[3].decode("utf-8"))

            shape = reply.get("shape")
            dtype = reply.get("dtype", "float32")
            if not isinstance(shape, list) or len(shape) != 2:
                raise RuntimeError(
                    f"RemoteEmbeddingBackend: invalid shape in header: {shape}"
                )
            arr = np.frombuffer(embeddings_bytes, dtype=dtype)
            try:
                embeddings = arr.reshape(shape).tolist()
            except Exception as exc:
                raise RuntimeError(
                    f"RemoteEmbeddingBackend: could not reshape embeddings buffer to {shape}: {exc}"
                )
        else:
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

def create_embedding_backend_from_config(cfg: "ServerCfg") -> EmbeddingBackend:
    """Factory to create an embedding backend from server configuration.

    The configuration is expected to contain optional keys:
    - embedding_backend: "local" (default) or "remote"
    - embedding_worker_address: ZeroMQ address of the worker
    - embedding_worker_timeout_sec: timeout in seconds
    """

    backend_type = cfg.embedding_backend.lower()

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
