from typing import Any

from umrs.inference.embeddings import compute_and_store_embedding, compute_and_store_embedding_batch



class LocalEmbeddingBackend:
    """Embedding backend that computes embeddings in-process.

    Delegates to ``compute_and_store_embedding`` which handles data
    loading from S3, model inference, and storing results in pgvector.
    """

    def compute_and_store(
        self,
        path_to_bucket: str,
        s3_keys: list[str],
        data_base: Any,
        table_name: str,
        model: Any,
        device: str,
        modality_name: str,
        s3_connection: Any,
        output_bucket: str,
        task: str = None,
        skip_existing: bool = True,
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
            task=task,
            skip_existing=skip_existing,
        )

    def compute_and_store_batch(self, items: list) -> list[bool]:
        """Batch variant: single model forward pass across multiple clients.

        Each element of `items` is a dict as expected by
        ``compute_and_store_embedding_batch``.  All items must share the same
        model.  Returns a parallel list of per-item success bools.
        """
        return compute_and_store_embedding_batch(items)
