from umrs.inference.embeddings import compute_and_store_embedding_batch


class LocalEmbeddingBackend:
    """Embedding backend that computes embeddings in-process.

    Its one entry point, ``compute_and_store_batch``, goes through
    ``compute_and_store_embedding_batch`` (S3 load, model inference, pgvector
    store), so batching and pipelining (``UMRS_INFERENCE_BATCH_SIZE``) apply.
    The single-item ``compute_and_store`` had no caller (processing.py only
    batches) and was removed (review U1).
    """

    def compute_and_store_batch(self, items: list) -> list[bool]:
        """Batch variant: single model forward pass across multiple clients.

        Each element of `items` is a dict as expected by
        ``compute_and_store_embedding_batch``.  All items must share the same
        model.  Returns a parallel list of per-item success bools.
        """
        return compute_and_store_embedding_batch(items)
