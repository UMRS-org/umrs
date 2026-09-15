from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Tuple

from umrs.server.rivault_launcher import launch_rivault_model

logger = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_TYPE = "bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit"


def get_rivault_model(
    vision_model_type: str,
    *,
    task: str = "",
    model_name: str = "",
    model_config: Dict[str, Any] | None = None,
) -> Any:
    config = dict(model_config or {})
    cache_key = json.dumps(
        {
            "vision_model_type": vision_model_type,
            "task": task,
            "model_name": model_name,
            "model_config": config,
        },
        sort_keys=True,
    )

    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is not None:
            return model

        launch_cfg: Dict[str, Any] = {
            "model_name": model_name or vision_model_type.upper(),
            "model_type": _MODEL_TYPE,
            "vision_model_type": vision_model_type,
            "vision_model_config": config,
            "vector_size": 0,
            "return_type": "file",
            "input_type": "image",
        }
        if task:
            launch_cfg["_task"] = task

        logger.info(
            "Launching RiVault vision model %s (task=%s)",
            vision_model_type,
            task or "default",
        )
        model = launch_rivault_model(launch_cfg)
        _MODEL_CACHE[cache_key] = model
        return model


def split_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    bucket, key = s3_uri[5:].split("/", 1)
    return bucket, key
