"""BentoML compute_embeddings adapters for UMRS vision services.

Bridges the BentoMLTaskClient async task API into the synchronous
compute_embeddings interface expected by EmbeddingModel / embeddings.py.

The generic factory ``make_bentoml_vision_compute`` creates a closure for
**any** model type supported by the generic ``/infer`` endpoint.  The old
model-specific factories are kept as thin wrappers for backward compatibility.
"""

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


def make_bentoml_vision_compute(client, model_type, model_config):
    """Create a compute_embeddings closure for any vision model via BentoML.

    Submits tasks to the generic ``/infer`` endpoint and polls for results.

    Args:
        client: ``BentoMLTaskClient`` connected to the vision service.
        model_type: Adapter key (e.g. ``"sam3"``, ``"hmae"``).
        model_config: Adapter-specific config dict forwarded to the service.

    Returns:
        A method suitable for binding to ``EmbeddingModel.compute_embeddings``.
    """

    def _compute(self, s3, s3_keys, path_to_bucket, output_bucket):
        results = []
        checksums = []
        metadata_list = []
        saved_keys = []

        for idx, s3_key in enumerate(s3_keys):
            logger.info("[%d/%d] BentoML %s: %s",
                        idx + 1, len(s3_keys), model_type, s3_key)

            obj = s3.head_object(Bucket=path_to_bucket, Key=s3_key)
            checksums.append(obj["ETag"].strip('"'))
            metadata_list.append(obj.get("Metadata", {}))

            image_url = f"s3://{path_to_bucket}/{s3_key}"
            task_id = client.submit(
                "infer",
                model_type=model_type,
                model_config=model_config,
                image_url=image_url,
                output_bucket=output_bucket,
            )
            result = client.wait_for_result("infer", task_id)

            if "error" in result:
                raise RuntimeError(
                    f"BentoML {model_type} task failed for {s3_key}: "
                    f"{result['error']}"
                )

            results.append(result)

            output_s3_key = (result.get("output_s3_key")
                             or result.get("overlay_s3_key"))
            output_s3_key = _copy_to_output_bucket(s3, output_s3_key,
                                                   output_bucket)
            saved_keys.append(output_s3_key)

        return results, checksums, metadata_list, saved_keys

    return _compute


# ── Backward-compatible wrappers ────────────────────────────────────────


def make_bentoml_hmae_compute(client, endpoint_name, mode):
    """Backward-compatible wrapper — delegates to the generic factory."""
    return make_bentoml_vision_compute(
        client, model_type="hmae", model_config={"mode": mode},
    )


def make_bentoml_sam3_compute(client, endpoint_name):
    """Backward-compatible wrapper — delegates to the generic factory."""
    return make_bentoml_vision_compute(
        client, model_type="sam3", model_config={},
    )
