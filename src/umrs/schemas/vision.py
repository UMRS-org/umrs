"""``VisionPerception`` — unified perception envelope for vision steps.

This is the **only** contract used between vision tools (VLM, SAM3, OCR)
and downstream LLM reasoning steps.  Any vision tool must emit a
``VisionPerception`` instance (or its ``to_dict()`` form) so that
``LLMStep`` can render it into a prompt without knowing which model
produced it.

The envelope is intentionally compact:

- Heavy artefacts (segmentation masks, raw embeddings) are *not* inlined.
  Tools should write them to S3 and put the URI in ``embeddings_ref`` /
  ``meta``.
- Spatial annotations use pixel coordinates relative to the original
  image (``image.width`` × ``image.height``).
- All numeric scores are floats in ``[0.0, 1.0]``.

The class deliberately uses ``dataclass`` (no Pydantic dependency) so it
is usable from lightweight step processes that don't pull the full
UMRS stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Type alias for an axis-aligned bounding box: (x1, y1, x2, y2) in pixels.
BoundingBox = tuple[float, float, float, float]


@dataclass
class DetectedObject:
    """One detected object with optional grounding box."""

    label: str
    score: float = 1.0
    bbox: BoundingBox | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label, "score": float(self.score)}
        if self.bbox is not None:
            out["bbox"] = list(self.bbox)
        if self.attributes:
            out["attributes"] = self.attributes
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectedObject:
        bbox = data.get("bbox")
        if bbox is not None:
            bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        return cls(
            label=str(data["label"]),
            score=float(data.get("score", 1.0)),
            bbox=bbox,
            attributes=dict(data.get("attributes", {})),
        )


@dataclass
class OCRItem:
    """One OCR-extracted text snippet with optional grounding."""

    text: str
    score: float = 1.0
    bbox: BoundingBox | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"text": self.text, "score": float(self.score)}
        if self.bbox is not None:
            out["bbox"] = list(self.bbox)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRItem:
        bbox = data.get("bbox")
        if bbox is not None:
            bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        return cls(
            text=str(data["text"]),
            score=float(data.get("score", 1.0)),
            bbox=bbox,
        )


@dataclass
class ImageInfo:
    """Provenance and dimensions of the source image."""

    width: int = 0
    height: int = 0
    sha256: str | None = None
    source: str | None = None  # original path, S3 URI, or presigned URL hash

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"width": int(self.width), "height": int(self.height)}
        if self.sha256:
            out["sha256"] = self.sha256
        if self.source:
            out["source"] = self.source
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageInfo:
        return cls(
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            sha256=data.get("sha256"),
            source=data.get("source"),
        )


@dataclass
class ModelInfo:
    """Identifier of the model that produced this perception slice."""

    name: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        return cls(
            name=str(data.get("name", "")),
            version=str(data.get("version", "")),
        )


@dataclass
class VisionPerception:
    """Unified envelope for vision-step output.

    All fields are optional so a tool can fill in only what it produces
    (e.g. an OCR-only tool sets ``ocr`` and leaves ``objects`` empty).
    Multiple ``VisionPerception`` instances may be merged via
    :meth:`merge` when several tools run on the same image.
    """

    caption: str | None = None
    dense_description: str | None = None
    objects: list[DetectedObject] = field(default_factory=list)
    ocr: list[OCRItem] = field(default_factory=list)
    embeddings_ref: str | None = None  # S3 URI; raw embeddings are never inlined.
    image: ImageInfo = field(default_factory=ImageInfo)
    model: ModelInfo = field(default_factory=ModelInfo)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.caption is not None:
            out["caption"] = self.caption
        if self.dense_description is not None:
            out["dense_description"] = self.dense_description
        if self.objects:
            out["objects"] = [o.to_dict() for o in self.objects]
        if self.ocr:
            out["ocr"] = [o.to_dict() for o in self.ocr]
        if self.embeddings_ref is not None:
            out["embeddings_ref"] = self.embeddings_ref
        out["image"] = self.image.to_dict()
        out["model"] = self.model.to_dict()
        if self.meta:
            out["meta"] = self.meta
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionPerception:
        return cls(
            caption=data.get("caption"),
            dense_description=data.get("dense_description"),
            objects=[DetectedObject.from_dict(o) for o in data.get("objects", [])],
            ocr=[OCRItem.from_dict(o) for o in data.get("ocr", [])],
            embeddings_ref=data.get("embeddings_ref"),
            image=ImageInfo.from_dict(data.get("image", {})),
            model=ModelInfo.from_dict(data.get("model", {})),
            meta=dict(data.get("meta", {})),
        )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, other: VisionPerception) -> VisionPerception:
        """Return a new perception combining ``self`` and ``other``.

        Scalar fields prefer the non-empty value from ``other`` (the
        more recent tool's output wins).  List fields are concatenated.
        ``meta`` is merged with ``other`` taking precedence.
        """
        return VisionPerception(
            caption=other.caption if other.caption else self.caption,
            dense_description=(
                other.dense_description if other.dense_description else self.dense_description
            ),
            objects=list(self.objects) + list(other.objects),
            ocr=list(self.ocr) + list(other.ocr),
            embeddings_ref=other.embeddings_ref or self.embeddings_ref,
            image=other.image if other.image.width or other.image.height else self.image,
            model=other.model if other.model.name else self.model,
            meta={**self.meta, **other.meta},
        )

    # ------------------------------------------------------------------
    # Prompt rendering helpers (used by LLMStep)
    # ------------------------------------------------------------------

    def to_prompt_text(self, max_chars: int = 4000) -> str:
        """Render the perception as compact text for inclusion in an LLM prompt.

        The format is intentionally line-oriented and human-readable
        rather than raw JSON so smaller models reason about it more
        reliably.  Truncation guards against excessively large outputs.
        """
        parts: list[str] = []
        if self.caption:
            parts.append(f"caption: {self.caption}")
        if self.dense_description:
            parts.append(f"description: {self.dense_description}")
        if self.objects:
            obj_lines = []
            for obj in self.objects[:50]:
                bbox_str = ""
                if obj.bbox is not None:
                    bbox_str = f" bbox={list(obj.bbox)}"
                obj_lines.append(f"  - {obj.label} (score={obj.score:.2f}){bbox_str}")
            parts.append("objects:\n" + "\n".join(obj_lines))
        if self.ocr:
            ocr_lines = [f"  - {item.text}" for item in self.ocr[:50]]
            parts.append("ocr:\n" + "\n".join(ocr_lines))
        if self.image.width and self.image.height:
            parts.append(f"image_size: {self.image.width}x{self.image.height}")
        if self.model.name:
            parts.append(f"model: {self.model.name}")
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text
