"""``VisionStep`` — built-in step that produces a ``VisionPerception``.

The step calls a vision tool (VLM, SAM3, OCR backend) and writes a
serialised :class:`VisionPerception` to ``StepResult.extra["perception"]``
plus an S3 object so downstream LLM steps can read it.

Modes
-----
- ``caption``: short caption from a VLM.
- ``dense``: long, dense description from a VLM.
- ``detect``: object detection (boxes + labels).
- ``ocr``: text extraction.
- ``segment``: SAM3 mask generation (mask URIs go to ``meta``).
- ``embed``: image-embedding extraction (URI to ``embeddings_ref``).

The step is *transport agnostic*: when ``mode in {caption, dense, detect, ocr}``
it talks to the platform Gateway's VLM endpoint via
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``.  When ``mode in {segment, embed}``
it delegates to the local Vision MCP server (which already wraps SAM3
and the image-guard sandbox).

If the environment variable ``UMRS_VISION_MOCK=1`` is set, the step
returns a deterministic mock perception — used by unit/E2E tests that
do not bring up a real VLM.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.schemas.vision import (
    DetectedObject,
    ImageInfo,
    ModelInfo,
    OCRItem,
    VisionPerception,
)

logger = logging.getLogger(__name__)

VALID_MODES = {"caption", "dense", "detect", "ocr", "segment", "embed"}


class VisionStep(Step):
    """Run a vision model and produce a :class:`VisionPerception`."""

    name = "vision"

    def setup(self, config: Dict[str, Any]) -> None:
        self.mode: str = str(config.get("mode", "dense")).lower()
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"VisionStep mode must be one of {sorted(VALID_MODES)}, "
                f"got {self.mode!r}"
            )

        self.model: str = str(config.get("model", "rivault-vlm-static"))
        self.image_param: str = str(config.get("image_param", "image"))
        self.prompt: Optional[str] = config.get("prompt")
        self.max_chars: int = int(config.get("max_chars", 2000))
        self.api_base: Optional[str] = (
            config.get("api_base")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("R4_GATEWAY_URL")
        )
        self.api_key: Optional[str] = (
            config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        )
        self.timeout_sec: float = float(config.get("timeout_sec", 60.0))
        # Per-execution cache key (set by run() so retries are cheap).
        self._cache: Dict[str, VisionPerception] = {}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, ctx: StepContext) -> StepResult:
        image_ref = self._resolve_image(ctx)
        if image_ref is None:
            return StepResult(extra={"error": "No image provided to VisionStep"})

        cache_key = image_ref.cache_key()
        if cache_key in self._cache:
            perception = self._cache[cache_key]
        else:
            perception = self._invoke(ctx, image_ref)
            self._cache[cache_key] = perception

        # Persist perception JSON to S3 so other steps can pick it up.
        out_key = f"{ctx.run_id or 'run'}/{ctx.config.get('step_name', 'vision')}/perception.json"
        try:
            payload = json.dumps(perception.to_dict(), ensure_ascii=False).encode("utf-8")
            ctx.upload_bytes(out_key, payload, content_type="application/json")
            output_keys: List[str] = [out_key]
        except Exception as exc:  # noqa: BLE001 — runtime path, just log and skip
            logger.warning("VisionStep: failed to upload perception JSON: %s", exc)
            output_keys = []

        ctx.report_progress(100.0, f"vision_step({self.mode}) complete")

        extra: Dict[str, Any] = {
            "perception": perception.to_dict(),
            "mode": self.mode,
        }
        if perception.dense_description:
            extra["dense_description"] = perception.dense_description
        if perception.caption:
            extra["caption"] = perception.caption
        if perception.ocr:
            extra["ocr_text"] = "\n".join(item.text for item in perception.ocr)
        if perception.objects:
            extra["object_labels"] = [obj.label for obj in perception.objects]

        return StepResult(output_keys=output_keys, extra=extra)

    # ------------------------------------------------------------------
    # Image resolution
    # ------------------------------------------------------------------

    def _resolve_image(self, ctx: StepContext) -> Optional["_ImageRef"]:
        # Priority 1: explicit config (e.g. image_path).
        image_value = ctx.config.get(self.image_param)
        if image_value:
            return _ImageRef.from_value(str(image_value), ctx)

        # Priority 2: first input key from upstream step.
        if ctx.input_keys:
            return _ImageRef.from_value(ctx.input_keys[0], ctx)

        # Priority 3: parent step result (e.g. previous VisionStep wrote source).
        for parent in ctx.parent_results.values():
            extra = getattr(parent, "extra", {}) or {}
            value = extra.get("image_source")
            if value:
                return _ImageRef.from_value(str(value), ctx)
        return None

    # ------------------------------------------------------------------
    # Vision invocation
    # ------------------------------------------------------------------

    def _invoke(self, ctx: StepContext, image_ref: "_ImageRef") -> VisionPerception:
        if os.environ.get("UMRS_VISION_MOCK", "0") in ("1", "true", "TRUE"):
            return self._mock_perception(image_ref)

        if self.mode in {"caption", "dense", "detect", "ocr"}:
            try:
                return self._invoke_vlm(image_ref)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "VisionStep: VLM call failed (%s), falling back to mock perception",
                    exc,
                )
                return self._mock_perception(image_ref, fallback=True, error=str(exc))

        if self.mode in {"segment", "embed"}:
            try:
                return self._invoke_vision_mcp(image_ref)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "VisionStep: Vision MCP call failed (%s), falling back to mock", exc
                )
                return self._mock_perception(image_ref, fallback=True, error=str(exc))

        return self._mock_perception(image_ref, fallback=True, error="unknown mode")

    def _invoke_vlm(self, image_ref: "_ImageRef") -> VisionPerception:
        """Call an OpenAI-compatible VLM via the Gateway."""
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - hard dep elsewhere
            raise RuntimeError("httpx is required for VLM mode") from exc

        if not self.api_base:
            raise RuntimeError(
                "VisionStep requires api_base (OPENAI_BASE_URL / R4_GATEWAY_URL) "
                "for VLM modes"
            )

        prompt = self.prompt or _DEFAULT_PROMPTS[self.mode]
        url = f"{self.api_base.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        image_url = image_ref.to_data_url()
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": self.max_chars,
        }

        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()

        text = ""
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = ""
        if isinstance(text, list):
            # OpenAI multimodal response — concatenate text parts.
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))

        return self._parse_vlm_text(text, image_ref)

    def _invoke_vision_mcp(self, image_ref: "_ImageRef") -> VisionPerception:
        """Call the local Vision MCP server (SAM3 / embeddings)."""
        # The Vision MCP runs inside mini-rivault and is not directly
        # reachable from a step process in every environment.  When
        # available the path is exposed via UMRS_VISION_MCP_URL; if not,
        # we fall back to a mock perception (callers can opt into the
        # heavier path via that env).
        mcp_url = os.environ.get("UMRS_VISION_MCP_URL")
        if not mcp_url:
            raise RuntimeError(
                "UMRS_VISION_MCP_URL not configured for segment/embed mode"
            )
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("httpx is required for Vision MCP mode") from exc

        tool = "extract_features" if self.mode == "embed" else "segment_image"
        payload = {"tool": tool, "args": {"image_path": image_ref.local_path()}}
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(mcp_url.rstrip("/") + "/invoke", json=payload)
            response.raise_for_status()
            data = response.json()

        meta = {"raw_mcp_response": data, "tool": tool}
        embeddings_ref = None
        if self.mode == "embed":
            embeddings_ref = data.get("embeddings_uri")
        return VisionPerception(
            embeddings_ref=embeddings_ref,
            image=image_ref.image_info(),
            model=ModelInfo(name=tool, version="vision-mcp"),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Output parsing / mocking
    # ------------------------------------------------------------------

    def _parse_vlm_text(self, text: str, image_ref: "_ImageRef") -> VisionPerception:
        text = text.strip()
        info = image_ref.image_info()
        model = ModelInfo(name=self.model, version="vlm")

        if self.mode == "caption":
            return VisionPerception(caption=text, image=info, model=model)
        if self.mode == "dense":
            return VisionPerception(dense_description=text, image=info, model=model)
        if self.mode == "ocr":
            ocr_items: List[OCRItem] = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    ocr_items.append(OCRItem(text=line))
            return VisionPerception(ocr=ocr_items, image=info, model=model)
        if self.mode == "detect":
            objects = self._parse_detection_text(text)
            return VisionPerception(objects=objects, image=info, model=model)
        return VisionPerception(image=info, model=model, meta={"raw_text": text})

    @staticmethod
    def _parse_detection_text(text: str) -> List[DetectedObject]:
        """Best-effort parse of detection output.

        We try JSON first (preferred VLM format); fall back to a simple
        ``label, score`` line parser otherwise.  Unparseable text becomes
        an empty list.
        """
        text = text.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict) and "objects" in data:
                data = data["objects"]
            if isinstance(data, list):
                out: List[DetectedObject] = []
                for item in data:
                    if isinstance(item, dict) and item.get("label"):
                        try:
                            out.append(DetectedObject.from_dict(item))
                        except (KeyError, TypeError, ValueError):
                            continue
                return out
        # Fallback: each non-empty line is one object label.
        objects: List[DetectedObject] = []
        for line in text.splitlines():
            line = line.strip().lstrip("-").strip()
            if not line:
                continue
            label, _, rest = line.partition(",")
            score = 1.0
            rest = rest.strip()
            if rest:
                try:
                    score = float(rest.split()[0])
                except (ValueError, IndexError):
                    score = 1.0
            objects.append(DetectedObject(label=label.strip(), score=score))
        return objects

    def _mock_perception(
        self,
        image_ref: "_ImageRef",
        *,
        fallback: bool = False,
        error: Optional[str] = None,
    ) -> VisionPerception:
        info = image_ref.image_info()
        model = ModelInfo(name="mock-vlm", version="0.1")
        meta: Dict[str, Any] = {"mock": True, "mode": self.mode}
        if fallback:
            meta["fallback"] = True
        if error:
            meta["error"] = error

        if self.mode == "caption":
            return VisionPerception(
                caption="A scientific figure showing data trends.",
                image=info,
                model=model,
                meta=meta,
            )
        if self.mode == "dense":
            return VisionPerception(
                dense_description=(
                    "A two-panel figure with line plots and axis labels. "
                    "The left panel shows a monotonic increase; the right panel "
                    "shows oscillatory behaviour."
                ),
                image=info,
                model=model,
                meta=meta,
            )
        if self.mode == "ocr":
            return VisionPerception(
                ocr=[
                    OCRItem(text="Figure 1"),
                    OCRItem(text="Time (s)"),
                    OCRItem(text="Amplitude"),
                ],
                image=info,
                model=model,
                meta=meta,
            )
        if self.mode == "detect":
            return VisionPerception(
                objects=[
                    DetectedObject(label="axis", score=0.95),
                    DetectedObject(label="legend", score=0.91),
                    DetectedObject(label="line_plot", score=0.97),
                ],
                image=info,
                model=model,
                meta=meta,
            )
        if self.mode == "embed":
            return VisionPerception(
                embeddings_ref="s3://mock-bucket/mock-embeddings.npy",
                image=info,
                model=ModelInfo(name="mock-embed", version="0.1"),
                meta=meta,
            )
        # segment
        meta = {**meta, "masks_uri": "s3://mock-bucket/mock-masks.json", "num_masks": 3}
        return VisionPerception(
            image=info,
            model=ModelInfo(name="mock-sam3", version="0.1"),
            meta=meta,
        )


# Default per-mode prompts when the user does not override.
_DEFAULT_PROMPTS = {
    "caption": "Provide a single concise caption for this image.",
    "dense": (
        "Describe this image in detail.  Include layout, axes, key features, "
        "and any text visible in the figure."
    ),
    "detect": (
        "List the salient objects visible in this image as a JSON array of "
        '{"label", "score", "bbox":[x1,y1,x2,y2]} entries. '
        "Use pixel coordinates."
    ),
    "ocr": "Transcribe ALL visible text in the image, one item per line.",
}


# ----------------------------------------------------------------------
# Image reference helper
# ----------------------------------------------------------------------


class _ImageRef:
    """Lightweight handle over an image identified by S3 key, path, or URL."""

    def __init__(
        self,
        *,
        bytes_data: Optional[bytes] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        s3_key: Optional[str] = None,
    ) -> None:
        self.bytes_data = bytes_data
        self.path = path
        self.url = url
        self.s3_key = s3_key

    # ----- factory ---------------------------------------------------

    @classmethod
    def from_value(cls, value: str, ctx: StepContext) -> "_ImageRef":
        if value.startswith(("http://", "https://", "data:")):
            return cls(url=value)
        if value.startswith("file://"):
            path = value[len("file://") :]
            return cls(path=path)
        path_obj = Path(value)
        if path_obj.exists():
            return cls(path=str(path_obj.resolve()))
        # Fall back to treating value as an S3 key in the input bucket.
        try:
            data = ctx.download_bytes(value)
            return cls(bytes_data=data, s3_key=value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VisionStep: cannot fetch image %s: %s", value, exc)
            return cls(s3_key=value)

    # ----- helpers ---------------------------------------------------

    def cache_key(self) -> str:
        if self.bytes_data:
            return hashlib.sha256(self.bytes_data).hexdigest()
        if self.path:
            return f"path:{self.path}"
        if self.url:
            return f"url:{self.url}"
        return f"s3:{self.s3_key or '?'}"

    def image_info(self) -> ImageInfo:
        sha = None
        if self.bytes_data:
            sha = hashlib.sha256(self.bytes_data).hexdigest()
        source = self.path or self.url or self.s3_key
        return ImageInfo(sha256=sha, source=source)

    def local_path(self) -> str:
        if self.path:
            return self.path
        raise RuntimeError("Image not available as a local path")

    def to_data_url(self) -> str:
        if self.url:
            return self.url
        data = self.bytes_data
        if data is None and self.path:
            try:
                with open(self.path, "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                raise RuntimeError(f"Cannot read image file {self.path}: {exc}") from exc
        if data is None:
            raise RuntimeError("No image bytes available for VLM call")
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
