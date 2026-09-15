"""Tests for VisionPerception schema (umrs.schemas.vision)."""

from __future__ import annotations

from umrs.schemas.vision import (
    DetectedObject,
    ImageInfo,
    ModelInfo,
    OCRItem,
    VisionPerception,
)


def test_round_trip_dict():
    perception = VisionPerception(
        caption="A plot of x vs y",
        dense_description="A line plot showing increasing x.",
        objects=[
            DetectedObject(label="axis", score=0.9, bbox=(1.0, 2.0, 3.0, 4.0)),
            DetectedObject(label="legend", score=0.7),
        ],
        ocr=[OCRItem(text="Time (s)", score=0.95, bbox=(0, 0, 100, 20))],
        embeddings_ref="s3://bucket/emb.npy",
        image=ImageInfo(width=512, height=384, sha256="deadbeef", source="img.png"),
        model=ModelInfo(name="qwen-vl", version="2.5"),
        meta={"latency_ms": 123},
    )

    payload = perception.to_dict()
    restored = VisionPerception.from_dict(payload)

    assert restored.caption == "A plot of x vs y"
    assert restored.dense_description.startswith("A line plot")
    assert len(restored.objects) == 2
    assert restored.objects[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert restored.ocr[0].text == "Time (s)"
    assert restored.embeddings_ref == "s3://bucket/emb.npy"
    assert restored.image.width == 512 and restored.image.sha256 == "deadbeef"
    assert restored.model.name == "qwen-vl"
    assert restored.meta == {"latency_ms": 123}


def test_merge_combines_lists_and_prefers_other_scalars():
    a = VisionPerception(caption="cap-a", objects=[DetectedObject(label="x")])
    b = VisionPerception(
        caption="cap-b",
        dense_description="desc-b",
        objects=[DetectedObject(label="y")],
        meta={"k": "v"},
    )

    merged = a.merge(b)

    assert merged.caption == "cap-b"  # other wins when both set
    assert merged.dense_description == "desc-b"
    assert [obj.label for obj in merged.objects] == ["x", "y"]
    assert merged.meta == {"k": "v"}


def test_to_prompt_text_truncates_and_lists():
    perception = VisionPerception(
        caption="cap",
        dense_description="desc",
        objects=[DetectedObject(label="line", score=0.5, bbox=(0, 0, 10, 10))],
        ocr=[OCRItem(text="hello"), OCRItem(text="world")],
        image=ImageInfo(width=100, height=50),
        model=ModelInfo(name="qwen", version="vlm"),
    )
    text = perception.to_prompt_text()
    assert "caption: cap" in text
    assert "description: desc" in text
    assert "- line (score=0.50)" in text
    assert "- hello" in text and "- world" in text
    assert "image_size: 100x50" in text
    assert "model: qwen" in text


def test_to_prompt_text_obeys_max_chars():
    perception = VisionPerception(dense_description="x" * 10_000)
    text = perception.to_prompt_text(max_chars=200)
    assert len(text) <= 200
    assert text.endswith("...")
