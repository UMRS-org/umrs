from __future__ import annotations

import configparser
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import boto3

from umrs.pipeline_sdk import StepContext


@dataclass(frozen=True)
class InputImage:
    bucket: str
    key: str

    @property
    def s3_url(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def resolve_bentoml_url(config: Dict[str, Any]) -> str:
    configured = _configured_bentoml_url(config)
    if configured:
        return configured

    api_base = _configured_api_base(config)
    if not api_base:
        return "http://localhost:11434/vision"

    return _normalise_vision_base(api_base)


def _configured_bentoml_url(config: Dict[str, Any]) -> str:
    return str(config.get("bentoml_url") or "").strip().rstrip("/")


def _configured_api_base(config: Dict[str, Any]) -> str:
    routed = config.get("_routing") if isinstance(config.get("_routing"), dict) else {}
    return str(
        config.get("api_base") or routed.get("api_base") or os.environ.get("BENTOML_URL") or ""
    ).strip()


def _normalise_vision_base(api_base: str) -> str:
    parsed = urlparse(api_base)
    path = parsed.path.rstrip("/")
    if "/vision" in path:
        path = path[: path.index("/vision") + len("/vision")]
    else:
        if path.endswith("/v1"):
            path = path[: -len("/v1")]
        path = f"{path}/vision" if path else "/vision"
    return parsed._replace(path=path).geturl().rstrip("/")


def resolve_input_images(ctx: StepContext) -> List[InputImage]:
    if ctx.input_keys:
        return [InputImage(bucket=ctx.input_bucket, key=key) for key in ctx.input_keys]

    params = _workflow_params(ctx)
    bucket = _configured_input_bucket(ctx, params)
    values = _configured_input_values(ctx, params)
    images = [_parse_input_image(value, bucket) for value in values]
    if images:
        return images

    raise RuntimeError(
        "No input files resolved. Upload images first, then trigger the workflow "
        "with input_files and input_bucket runtime parameters."
    )


def download_s3_bytes(s3_url: str) -> bytes:
    image = _parse_input_image(s3_url, "")
    client = _build_s3_client()
    obj = client.get_object(Bucket=image.bucket, Key=image.key)
    return obj["Body"].read()


def _workflow_params(ctx: StepContext) -> Dict[str, Any]:
    params = ctx.config.get("_workflow_params") if isinstance(ctx.config, dict) else {}
    if isinstance(params, dict):
        return params
    return {}


def _configured_input_bucket(ctx: StepContext, params: Dict[str, Any]) -> str:
    return str(params.get("input_bucket") or ctx.config.get("input_bucket") or "").strip()


def _configured_input_values(ctx: StepContext, params: Dict[str, Any]) -> List[str]:
    for key in ("input_files", "input_keys"):
        if key in params and params[key] is not None:
            return _normalise_input_values(params[key])
    for key in ("input_files", "input_keys"):
        if key in ctx.config and ctx.config[key] is not None:
            return _normalise_input_values(ctx.config[key])
    return []


def _normalise_input_values(raw_inputs: Any) -> List[str]:
    if raw_inputs is None:
        return []
    if isinstance(raw_inputs, str):
        return _values_from_string(raw_inputs)
    if isinstance(raw_inputs, Iterable):
        return _values_from_iterable(raw_inputs)
    return [str(raw_inputs).strip()]


def _values_from_string(raw_inputs: str) -> List[str]:
    text = raw_inputs.strip()
    if not text:
        return []
    decoded = _decode_json_list(text)
    if decoded is not None:
        return decoded
    return [item.strip() for item in text.split(",") if item.strip()]


def _decode_json_list(text: str) -> List[str] | None:
    if not text.startswith("["):
        return None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list):
        return None
    return [str(item).strip() for item in decoded if str(item).strip()]


def _values_from_iterable(raw_inputs: Iterable[Any]) -> List[str]:
    values: List[str] = []
    for item in raw_inputs:
        text = str(item).strip()
        if text:
            values.append(text)
    return values


def _parse_input_image(value: str, fallback_bucket: str) -> InputImage:
    if value.startswith("s3://"):
        remainder = value[5:]
        bucket, sep, key = remainder.partition("/")
        if not bucket or not sep or not key:
            raise RuntimeError(f"Invalid S3 object reference: {value}")
        return InputImage(bucket=bucket, key=key)
    if not fallback_bucket:
        raise RuntimeError(
            f"Input file '{value}' is missing an S3 bucket. Pass input_bucket when triggering the workflow."
        )
    return InputImage(bucket=fallback_bucket, key=value)


def _build_s3_client():
    endpoint, access_key, secret_key = _s3_client_settings()

    if not endpoint or not access_key or not secret_key:
        raise RuntimeError(
            "S3 client is not configured. Set UMRS_S3_ENDPOINT/AWS credentials or provide ~/.config/umrs/s3cfg."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _s3_client_settings() -> tuple[str | None, str | None, str | None]:
    env_settings = _env_s3_settings()
    if all(env_settings):
        return env_settings

    cfg_path = os.environ.get("UMRS_S3CFG") or os.path.expanduser("~/.config/umrs/s3cfg")
    cfg_settings = _s3_settings_from_config(cfg_path)
    return _merge_s3_settings(env_settings, cfg_settings)


def _env_s3_settings() -> tuple[str | None, str | None, str | None]:
    endpoint = os.environ.get("UMRS_S3_ENDPOINT") or os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    return endpoint, access_key, secret_key


def _merge_s3_settings(
    env_settings: tuple[str | None, str | None, str | None],
    cfg_settings: tuple[str | None, str | None, str | None],
) -> tuple[str | None, str | None, str | None]:
    return tuple(env or cfg for env, cfg in zip(env_settings, cfg_settings, strict=False))


def _s3_settings_from_config(cfg_path: str) -> tuple[str | None, str | None, str | None]:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(cfg_path)
    if not parser.has_section("default"):
        return None, None, None

    protocol = "https" if parser.getboolean("default", "use_https", fallback=False) else "http"
    endpoint = f"{protocol}://{parser.get('default', 'host_base')}"
    access_key = parser.get("default", "access_key", fallback=None)
    secret_key = parser.get("default", "secret_key", fallback=None)
    return endpoint, access_key, secret_key
