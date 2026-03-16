"""BentoML compute_embeddings adapters for UMRS vision services.

Bridges the BentoMLTaskClient async task API into the synchronous
compute_embeddings interface expected by EmbeddingModel / embeddings.py.

Each adapter returns a function with the signature:
    func(self, s3, s3_keys, path_to_bucket, output_bucket) -> (results, checksums, metadata, saved_keys)

matching the contract of the local HMAE and SAM3 compute functions.
"""

import io
import logging

logger = logging.getLogger(__name__)


def make_bentoml_hmae_compute(client, endpoint_name, mode):
    """Create a compute_embeddings closure for HMAE via BentoML.

    Args:
        client: BentoMLTaskClient connected to the vision service.
        endpoint_name: BentoML task endpoint (e.g. "hmae_reconstruct").
        mode: HMAE inference mode — "reconstruct", "mae", or "features".

    Returns:
        A method suitable for binding to EmbeddingModel.compute_embeddings.
    """

    def _compute(self, s3, s3_keys, path_to_bucket, output_bucket):
        cls_embeddings = []
        checksums = []
        metadata_list = []
        saved_keys = []

        for idx, s3_key in enumerate(s3_keys):
            logger.info("[%d/%d] BentoML HMAE %s: %s",
                        idx + 1, len(s3_keys), mode, s3_key)

            # Retrieve checksum and metadata from S3
            obj = s3.head_object(Bucket=path_to_bucket, Key=s3_key)
            checksums.append(obj["ETag"].strip('"'))
            metadata_list.append(obj.get("Metadata", {}))

            # Submit task to BentoML vision service
            image_url = f"s3://{path_to_bucket}/{s3_key}"
            task_id = client.submit(endpoint_name, image_url=image_url, mode=mode)
            result = client.wait_for_result(endpoint_name, task_id)

            if "error" in result:
                raise RuntimeError(
                    f"BentoML HMAE task failed for {s3_key}: {result['error']}"
                )

            # Extract output S3 key and embedding
            output_s3_key = result.get("output_s3_key")
            saved_keys.append(output_s3_key)

            # For features mode, cls_token is produced as embedding; for
            # reconstruct/mae modes there is no embedding vector.
            cls_embeddings.append(None)

        return cls_embeddings, checksums, metadata_list, saved_keys

    return _compute


def make_bentoml_sam3_compute(client, endpoint_name):
    """Create a compute_embeddings closure for SAM3 via BentoML.

    Args:
        client: BentoMLTaskClient connected to the vision service.
        endpoint_name: BentoML task endpoint (e.g. "sam3_segment").

    Returns:
        A method suitable for binding to EmbeddingModel.compute_embeddings.
    """

    def _compute(self, s3, s3_keys, path_to_bucket, output_bucket,
                 text_prompt="object"):
        results_list = []
        checksums = []
        metadata_list = []
        saved_keys = []

        for idx, s3_key in enumerate(s3_keys):
            logger.info("[%d/%d] BentoML SAM3 segment: %s",
                        idx + 1, len(s3_keys), s3_key)

            # Retrieve checksum and metadata from S3
            obj = s3.head_object(Bucket=path_to_bucket, Key=s3_key)
            checksums.append(obj["ETag"].strip('"'))
            metadata_list.append(obj.get("Metadata", {}))

            # Submit task to BentoML vision service
            image_url = f"s3://{path_to_bucket}/{s3_key}"
            task_id = client.submit(endpoint_name,
                                    image_url=image_url, prompt=text_prompt)
            result = client.wait_for_result(endpoint_name, task_id)

            if "error" in result:
                raise RuntimeError(
                    f"BentoML SAM3 task failed for {s3_key}: {result['error']}"
                )

            results_list.append(result)
            saved_keys.append(result.get("overlay_s3_key"))

        return results_list, checksums, metadata_list, saved_keys

    return _compute
