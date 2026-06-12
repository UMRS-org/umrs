"""Gateway-facing vision inference API.

UMRS owns endpoint discovery, presigned URL creation, and BentoML request
forwarding so external clients never receive presigned URLs directly.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from umrs.api.files import S3Config, get_project_id, get_s3_config, validate_key
from umrs.orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
    get_registry,
)
from umrs.shared.storage.s3_presign import (
    generate_presigned_get,
    generate_presigned_put,
)
from umrs.vision_service.output_specs import describe_output_artifacts

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/vision", tags=["vision"])


class VisionInferSubmitRequest(BaseModel):
    """Submit a BentoML vision inference task through UMRS."""

    image_uri: str = Field(
        ..., description="S3 URI or project-scoped object key for the input image"
    )
    model: str | None = Field(None, description="Endpoint model name to use")
    endpoint_id: str | None = Field(None, description="Specific endpoint job_id to use")
    adapter: str | None = Field(None, description="BentoML vision adapter name")
    model_type: str | None = Field(
        None, description="Alias for adapter, matching BentoML payload naming"
    )
    inference_config: dict[str, Any] = Field(default_factory=dict)
    output_bucket: str | None = None
    output_artifacts: dict[str, str] | None = Field(
        None,
        description="Optional output object keys mapped to content types for custom adapters",
    )
    ttl_sec: int | None = Field(
        None, ge=60, le=86400, description="Presigned URL TTL for internal BentoML use"
    )


class VisionOutputArtifact(BaseModel):
    """Output artifact planned for a vision task."""

    bucket: str
    key: str
    uri: str
    content_type: str


class VisionSubmitResponse(BaseModel):
    """Submission response with endpoint and output metadata."""

    endpoint_id: str
    endpoint_model: str
    adapter: str
    task: dict[str, Any]
    outputs: list[VisionOutputArtifact] = Field(default_factory=list)


def _parse_s3_uri(image_uri: str, project_id: str, config: S3Config) -> tuple[str, str]:
    if image_uri.startswith("s3://"):
        path = image_uri[5:]
        parts = path.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise HTTPException(status_code=400, detail="Invalid image_uri S3 format")
        bucket, key = parts
    else:
        bucket, key = config.bucket, image_uri
    return bucket, validate_key(key, project_id)


def _has_project_access(endpoint: RegisteredEndpoint, project_id: str) -> bool:
    owner_type = str(
        endpoint.metadata.get("owner_type") or endpoint.metadata.get("tier") or ""
    )
    return endpoint.project_id == project_id or owner_type == "system"


def _is_vision_endpoint(endpoint: RegisteredEndpoint) -> bool:
    metadata = endpoint.metadata
    return (
        metadata.get("model_type") == "vision"
        or metadata.get("service_kind") == "bentoml_vision"
        or metadata.get("serving_backend") == "bentoml"
    )


def _resolve_endpoint(
    registry: EndpointRegistry,
    project_id: str,
    *,
    endpoint_id: str | None = None,
    model: str | None = None,
) -> RegisteredEndpoint:
    if endpoint_id:
        endpoint = registry.get_by_job_id(endpoint_id)
        if (
            endpoint
            and endpoint.is_usable
            and _has_project_access(endpoint, project_id)
        ):
            return endpoint
        raise HTTPException(
            status_code=404, detail=f"No ready vision endpoint: {endpoint_id}"
        )

    statuses = [EndpointStatus.READY, EndpointStatus.DEGRADED]
    candidates = registry.list_all(status_filter=statuses)
    matches = [
        endpoint
        for endpoint in candidates
        if _has_project_access(endpoint, project_id)
        and _is_vision_endpoint(endpoint)
        and (model is None or endpoint.model == model)
    ]
    if matches:
        matches.sort(key=lambda ep: 0 if ep.project_id == project_id else 1)
        return matches[0]

    target = model or "vision"
    raise HTTPException(
        status_code=404, detail=f"No ready vision endpoint for {target}"
    )


def _adapter_name(
    request: VisionInferSubmitRequest, endpoint: RegisteredEndpoint
) -> str:
    adapter = (
        request.adapter
        or request.model_type
        or endpoint.metadata.get("vision_adapter")
        or endpoint.model
    )
    return str(adapter)


def _endpoint_base_url(endpoint: RegisteredEndpoint) -> str:
    base_url = str(endpoint.metadata.get("service_url") or endpoint.litellm_url).rstrip(
        "/"
    )
    invoke_base_path = endpoint.metadata.get("invoke_base_path")
    if invoke_base_path:
        return f"{base_url}/{str(invoke_base_path).strip('/')}"
    return base_url


def _resolve_output_artifacts(
    request: VisionInferSubmitRequest,
    *,
    adapter: str,
    input_key: str,
    project_id: str,
    output_bucket: str,
    config: S3Config,
) -> tuple[dict[str, str], list[VisionOutputArtifact]]:
    raw_outputs = request.output_artifacts or describe_output_artifacts(
        adapter,
        input_key,
        request.inference_config,
    )
    presigned_urls: dict[str, str] = {}
    artifacts: list[VisionOutputArtifact] = []
    if not raw_outputs:
        return presigned_urls, artifacts

    ttl_sec = min(request.ttl_sec or config.default_ttl, config.max_ttl)
    s3_client = config.get_client()
    for raw_key, content_type in raw_outputs.items():
        key = validate_key(raw_key, project_id)
        media_type = content_type or "application/octet-stream"
        presigned_urls[key] = generate_presigned_put(
            s3_client,
            output_bucket,
            key,
            content_type=media_type,
            ttl=ttl_sec,
            use_external_endpoint=False,
        )
        artifacts.append(
            VisionOutputArtifact(
                bucket=output_bucket,
                key=key,
                uri=f"s3://{output_bucket}/{key}",
                content_type=media_type,
            )
        )
    return presigned_urls, artifacts


async def _request_endpoint(
    endpoint: RegisteredEndpoint,
    method: str,
    path: str,
    *,
    json_payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    url = f"{_endpoint_base_url(endpoint)}{path}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0)) as client:
        response = await client.request(method, url, json=json_payload, params=params)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response


def _response_from_downstream(response: httpx.Response) -> Response:
    content_type = response.headers.get("content-type", "application/octet-stream")
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=content_type,
    )


@router.post("/infer/submit", response_model=VisionSubmitResponse)
async def submit_inference(
    request: VisionInferSubmitRequest,
    x_verified_project_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    registry: EndpointRegistry = Depends(get_registry),
    config: S3Config = Depends(get_s3_config),
) -> VisionSubmitResponse:
    project_id = get_project_id(None, x_verified_project_id, x_project_id)
    endpoint = _resolve_endpoint(
        registry,
        project_id,
        endpoint_id=request.endpoint_id,
        model=request.model,
    )
    adapter = _adapter_name(request, endpoint)
    input_bucket, input_key = _parse_s3_uri(request.image_uri, project_id, config)
    ttl_sec = min(request.ttl_sec or config.default_ttl, config.max_ttl)
    s3_client = config.get_client()
    input_url = generate_presigned_get(
        s3_client,
        input_bucket,
        input_key,
        ttl=ttl_sec,
        use_external_endpoint=False,
    )
    output_bucket = request.output_bucket or input_bucket
    output_urls, outputs = _resolve_output_artifacts(
        request,
        adapter=adapter,
        input_key=input_key,
        project_id=project_id,
        output_bucket=output_bucket,
        config=config,
    )

    payload = {
        "model_type": adapter,
        "inference_config": request.inference_config,
        "image_url": f"s3://{input_bucket}/{input_key}",
        "output_bucket": output_bucket,
        "presigned_input_url": input_url,
        "presigned_output_urls": output_urls,
    }
    response = await _request_endpoint(
        endpoint, "POST", "/infer/submit", json_payload=payload
    )
    try:
        task = response.json()
    except ValueError:
        task = {"raw": response.text}

    registry.mark_used(endpoint.job_id)
    return VisionSubmitResponse(
        endpoint_id=endpoint.job_id,
        endpoint_model=endpoint.model,
        adapter=adapter,
        task=task,
        outputs=outputs,
    )


@router.get("/infer/status")
async def get_inference_status(
    task_id: str = Query(...),
    model: str | None = Query(None),
    endpoint_id: str | None = Query(None),
    project_id: str = Depends(get_project_id),
    registry: EndpointRegistry = Depends(get_registry),
) -> Response:
    endpoint = _resolve_endpoint(
        registry, project_id, endpoint_id=endpoint_id, model=model
    )
    response = await _request_endpoint(
        endpoint, "GET", "/infer/status", params={"task_id": task_id}
    )
    return _response_from_downstream(response)


@router.get("/infer/get")
async def get_inference_result(
    task_id: str = Query(...),
    model: str | None = Query(None),
    endpoint_id: str | None = Query(None),
    project_id: str = Depends(get_project_id),
    registry: EndpointRegistry = Depends(get_registry),
) -> Response:
    endpoint = _resolve_endpoint(
        registry, project_id, endpoint_id=endpoint_id, model=model
    )
    response = await _request_endpoint(
        endpoint, "GET", "/infer/get", params={"task_id": task_id}
    )
    registry.mark_used(endpoint.job_id)
    return _response_from_downstream(response)


@router.post("/infer/cancel")
async def cancel_inference(
    task_id: str = Query(...),
    model: str | None = Query(None),
    endpoint_id: str | None = Query(None),
    project_id: str = Depends(get_project_id),
    registry: EndpointRegistry = Depends(get_registry),
) -> Response:
    endpoint = _resolve_endpoint(
        registry, project_id, endpoint_id=endpoint_id, model=model
    )
    response = await _request_endpoint(
        endpoint, "POST", "/infer/cancel", params={"task_id": task_id}
    )
    return _response_from_downstream(response)
