from typing import List, Any

from umrs.embeddings import compute_and_store_embedding


class LocalEmbeddingBackend:
    """Embedding backend that computes embeddings in-process.

    Delegates to ``compute_and_store_embedding`` which handles data
    loading from S3, model inference, and storing results in pgvector.
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
        task: str = None,
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
        )
