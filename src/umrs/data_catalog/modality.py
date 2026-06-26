"""Modality detection from extension and magic bytes."""

from __future__ import annotations

from pathlib import PurePosixPath

_IMAGE_MAGIC = {
    b"\xff\xd8\xff": "image",
    b"\x89PNG\r\n\x1a\n": "image",
    b"GIF87a": "image",
    b"GIF89a": "image",
    b"RIFF": "video",  # may be audio/webp; refined below
}
_TEXT_EXTENSIONS = {".txt", ".md", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml"}
_STRUCTURED_EXTENSIONS = {".json", ".csv", ".tsv", ".yaml", ".yml"}
_DOC_EXTENSIONS = {".pdf", ".pptx", ".ppt"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def detect_modality(filename: str, content: bytes) -> str:
    """Return a coarse modality label for catalog metadata."""
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix in _TEXT_EXTENSIONS and suffix not in {".csv", ".tsv", ".json"}:
        return "text"
    if suffix in _STRUCTURED_EXTENSIONS:
        return "structured"
    if suffix in _DOC_EXTENSIONS:
        return "document"
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"

    head = content[:16]
    for magic, label in _IMAGE_MAGIC.items():
        if head.startswith(magic):
            if label == "video" and head[8:12] == b"WEBP":
                return "image"
            if label == "video" and suffix in _AUDIO_EXTENSIONS:
                return "audio"
            return label
    if head.startswith(b"%PDF"):
        return "document"
    if head.startswith(b"PK\x03\x04"):
        return "document"
    try:
        content[:4096].decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        pass
    return "binary"


def filename_signals(filename: str) -> list[str]:
    """Extract searchable tokens from a filename."""
    name = PurePosixPath(filename).name
    stem = PurePosixPath(name).stem
    parts = [p for p in stem.replace("-", "_").split("_") if p]
    return [name, *parts]
