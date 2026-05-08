"""Whisper STT adapter — speech-to-text using OpenAI Whisper models.

This adapter wraps Whisper models for transcription. The actual
whisper library is an optional dependency.

No dependency on r4.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from . import (
    STTAdapter,
    STTResult,
    VoiceConfig,
    register_voice_adapter,
)

logger = logging.getLogger(__name__)

_WHISPER_MODELS = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large",
    "whisper-large-v3",
]


class WhisperAdapter(STTAdapter):
    """Whisper-based speech-to-text adapter."""

    def __init__(self) -> None:
        self._model: object = None
        self._model_name: str = ""

    @property
    def name(self) -> str:
        return "whisper"

    @property
    def supported_models(self) -> List[str]:
        return list(_WHISPER_MODELS)

    def load_model(self, config: VoiceConfig) -> None:
        model_name = config.model_name or "whisper-base"
        # Strip "whisper-" prefix for the library
        size = model_name.replace("whisper-", "")
        try:
            import whisper  # type: ignore[import-untyped]

            self._model = whisper.load_model(size)
            self._model_name = model_name
            logger.info("Loaded Whisper model: %s", model_name)
        except ImportError:
            logger.warning("whisper library not installed")
            raise

    def unload_model(self) -> None:
        self._model = None
        self._model_name = ""

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> STTResult:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import tempfile
        import os

        # Whisper expects a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            result = self._model.transcribe(  # type: ignore[union-attr]
                tmp_path,
                language=language,
            )
            segments = [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                }
                for s in result.get("segments", [])
            ]
            return STTResult(
                text=result.get("text", ""),
                language=result.get("language", language or ""),
                confidence=1.0,
                segments=segments,
            )
        finally:
            os.unlink(tmp_path)

    def health_check(self) -> bool:
        return self._model is not None


# Auto-register
try:
    register_voice_adapter(WhisperAdapter())
except Exception:
    pass
