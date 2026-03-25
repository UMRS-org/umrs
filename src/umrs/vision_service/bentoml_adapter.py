"""BentoML compute_embeddings adapters for UMRS vision services.

Bridges the ``bentoml.SyncHTTPClient`` task API into the synchronous
compute_embeddings interface expected by EmbeddingModel / embeddings.py.

The generic factory ``make_bentoml_vision_compute`` creates a closure for
**any** model type supported by the generic ``/infer`` endpoint.
"""

import logging
import time

import bentoml

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


def make_bentoml_vision_compute(client: bentoml.SyncHTTPClient, model_type, model_config,
                                poll_interval=2.0, timeout=1200.0):
    """Create a compute_embeddings closure for any vision model via BentoML.

    Submits tasks to the ``/infer`` endpoint using BentoML's native client
    and polls for results.

    Args:
        client: ``bentoml.SyncHTTPClient`` connected to the vision service.
        model_type: Adapter key (e.g. ``"sam3"``, ``"hmae"``).
        model_config: Adapter-specific config dict forwarded to the service.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait per task.

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
            task = client.infer.submit(
                model_type=model_type,
                inference_config=model_config,
                image_url=image_url,
                output_bucket=output_bucket,
            )
            logger.info("Task submitted: infer -> %s", task.id)

            start = time.monotonic()
            while True:
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Task {task.id} did not complete within {timeout}s"
                    )
                status = task.get_status()
                if status.value == "completed":
                    result = task.get()
                    break
                elif status.value == "failed":
                    raise RuntimeError(f"Task {task.id} failed")
                elif status.value == "canceled":
                    raise RuntimeError(f"Task {task.id} was canceled")
                time.sleep(poll_interval)

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


# ---------------------------------------------------------------------------
# Jens's standalone ``mini-rivault-vision`` adapter
# ---------------------------------------------------------------------------

def make_jens_vision_compute(client: bentoml.SyncHTTPClient, model_type, model_config,
                             auth_token=None, poll_interval=2.0, timeout=1200.0):
    """Create a compute_embeddings closure for Jens's ``/vision_task`` endpoint.

    Jens's VisionFrontend accepts:
        image (str): s3:// URI or URL
        model (str): "hmae" or "sam3"
        task  (str): inference mode (e.g. "reconstruct", "mae", "features")
        out   (str): "raw" or s3:// output URI

    Returns pickled dicts.
    """
    import pickle

    # Map UMRS model_config to Jens's task parameter.
    # For HMAE the task is the inference mode; for SAM3 it's typically empty.
    task_param = model_config.get("mode", "reconstruct")

    def _compute(self, s3, s3_keys, path_to_bucket, output_bucket):
        results = []
        checksums = []
        metadata_list = []
        saved_keys = []

        for idx, s3_key in enumerate(s3_keys):
            logger.info("[%d/%d] Jens BentoML %s: %s",
                        idx + 1, len(s3_keys), model_type, s3_key)

            obj = s3.head_object(Bucket=path_to_bucket, Key=s3_key)
            checksums.append(obj["ETag"].strip('"'))
            metadata_list.append(obj.get("Metadata", {}))

            image_url = f"s3://{path_to_bucket}/{s3_key}"
            out_uri = f"s3://{output_bucket}/{s3_key}" if output_bucket else "raw"

            task = client.vision_task.submit(
                image=image_url,
                model=model_type,
                task=task_param,
                out=out_uri,
            )
            logger.info("Task submitted: vision_task -> %s", task.id)

            start = time.monotonic()
            while True:
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Task {task.id} did not complete within {timeout}s"
                    )
                status = task.get_status()
                if status.value == "completed":
                    raw_result = task.get()
                    break
                elif status.value == "failed":
                    raise RuntimeError(f"Task {task.id} failed")
                elif status.value == "canceled":
                    raise RuntimeError(f"Task {task.id} was canceled")
                time.sleep(poll_interval)

            # Jens's endpoint returns pickled bytes.
            if isinstance(raw_result, bytes):
                result = pickle.loads(raw_result)  # noqa: S301
            else:
                result = raw_result

            if result.get("status") not in ("ok", "success", None):
                if result["status"] in ("missing_or_invalid_api_key",
                                        "image_not_found", "model_not_found"):
                    raise RuntimeError(
                        f"Jens vision_task error for {s3_key}: {result['status']}"
                    )

            results.append(result)

            # For Jens's stack, the output is written directly to the S3 bucket
            # by the service itself when out=s3://..., so we just record the key.
            output_s3_key = f"s3://{output_bucket}/{s3_key}" if output_bucket else None
            output_s3_key = _copy_to_output_bucket(s3, output_s3_key,
                                                   output_bucket)
            saved_keys.append(output_s3_key)

        return results, checksums, metadata_list, saved_keys

    return _compute