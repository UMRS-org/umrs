"""RiVault mock service used by the embedding worker to fetch model configs.

This FastAPI application exposes a lightweight OpenAPI surface that mimics an
external "RiVault" model registry. The service simply keeps a curated catalog
of model configuration objects (compatible with ``umrs.models_loading``) and
returns them to clients over HTTP.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(__file__)
CATALOG_PATH = os.environ.get(
    "RIVAULT_MODELS_FILE",
    os.path.join(BASE_DIR, "rivault_models.json"),
)
WEIGHTS_ROOT = os.environ.get(
    "RIVAULT_MODELS_ROOT",
    os.path.join(BASE_DIR, "rivault_models_store"),
)

app = FastAPI(title="RiVault Mock", version="0.1.0")


class ModelList(BaseModel):
    models: List[str]


class ModelResponse(BaseModel):
    model_name: str
    config: Dict[str, Any]


def _model_dir(model_name: str) -> str:
    safe_name = model_name.replace("/", "__")
    return os.path.join(WEIGHTS_ROOT, safe_name)


def _weights_path(model_name: str) -> str:
    return os.path.join(_model_dir(model_name), "weights.tar")


@lru_cache(maxsize=1)
def _load_catalog(file_path: str | None = None) -> Dict[str, Dict[str, Any]]:
    path = file_path or CATALOG_PATH
    if not os.path.exists(path):
        raise RuntimeError(f"RiVault catalog file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    models = raw.get("models", {})
    if not isinstance(models, dict):
        raise RuntimeError("Invalid RiVault catalog structure: 'models' must be a dict")

    return models


@app.get("/healthz")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models", response_model=ModelList)
def list_models() -> ModelList:
    catalog = _load_catalog()
    return ModelList(models=sorted(catalog.keys()))


@app.get("/api/models/{model_name}", response_model=ModelResponse)
def get_model(model_name: str) -> ModelResponse:
    catalog = _load_catalog()
    config = catalog.get(model_name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return ModelResponse(model_name=model_name, config=config)


@app.put("/api/models/{model_name}/weights")
async def upload_model_weights(model_name: str, file: UploadFile = File(...)) -> Dict[str, str]:
    os.makedirs(WEIGHTS_ROOT, exist_ok=True)
    model_dir = _model_dir(model_name)
    os.makedirs(model_dir, exist_ok=True)
    dest_path = _weights_path(model_name)

    try:
        with open(dest_path, "wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store weights: {exc}") from exc

    return {"status": "ok", "model_name": model_name, "path": dest_path}


@app.get("/api/models/{model_name}/weights")
def download_model_weights(model_name: str) -> StreamingResponse:
    path = _weights_path(model_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Weights for model '{model_name}' not found")

    def iter_file():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(iter_file(), media_type="application/octet-stream")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("RiVault Mock Service")
    parser.add_argument(
        "--host",
        default=os.getenv("RIVAULT_HOST", "0.0.0.0"),
        help="Host interface for uvicorn to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RIVAULT_PORT", "8000")),
        help="Port for uvicorn to expose",
    )
    parser.add_argument(
        "--models-file",
        default=CATALOG_PATH,
        help="Path to the RiVault models catalog (JSON)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Persist the selected catalog path so subsequent calls reuse it.
    global CATALOG_PATH
    CATALOG_PATH = args.models_file
    os.environ["RIVAULT_MODELS_FILE"] = CATALOG_PATH
    _load_catalog(CATALOG_PATH)

    import uvicorn  # Local import so that CLI usage works even without FastAPI server

    uvicorn.run("rivault_service:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
