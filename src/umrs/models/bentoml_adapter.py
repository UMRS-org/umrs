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


def _copy_to_output_bucket(s3, s3_uri, output_bucket):
    """Copy an object from its current bucket to the output bucket.

    Args:
        s3: boto3 S3 client.
        s3_uri: Full S3 URI like ``s3://bucket/key``.
        output_bucket: Target output bucket name.

    Returns:
        New S3 URI in the output bucket, or the original if copy is skipped.
    """
    if not output_bucket or not s3_uri:
        return s3_uri
    stripped = s3_uri.replace("s3://", "", 1)
    src_bucket, key = stripped.split("/", 1)
    if src_bucket == output_bucket:
        return s3_uri
    copy_source = {"Bucket": src_bucket, "Key": key}
    s3.copy_object(CopySource=copy_source, Bucket=output_bucket, Key=key)
    new_uri = f"s3://{output_bucket}/{key}"
    logger.info("Copied %s -> %s", s3_uri, new_uri)
    return new_uri


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

            # Extract output S3 key and copy to output bucket
            output_s3_key = result.get("output_s3_key")
            output_s3_key = _copy_to_output_bucket(s3, output_s3_key, output_bucket)
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
            overlay_s3_key = result.get("overlay_s3_key")
            overlay_s3_key = _copy_to_output_bucket(s3, overlay_s3_key, output_bucket)
            saved_keys.append(overlay_s3_key)

        return results_list, checksums, metadata_list, saved_keys

    return _compute
