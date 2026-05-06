"""UMRS Voice adapters — STT/TTS via BentoML adapter pattern.

Follows the same adapter registry pattern as the vision service.
Each voice adapter wraps a specific STT/TTS model and exposes a
uniform interface. Adapters are auto-discovered and registered.

No dependency on r4.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VoiceTaskType(Enum):
    """Types of voice tasks."""

    STT = "speech_to_text"
    TTS = "text_to_speech"


@dataclass
class VoiceConfig:
    """Configuration for a voice adapter."""

    model_name: str = ""
    language: str = "en"
    sample_rate: int = 16000
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class STTResult:
    """Result from speech-to-text."""

    text: str = ""
    language: str = ""
    confidence: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class TTSResult:
    """Result from text-to-speech."""

    audio_bytes: bytes = b""
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    format: str = "wav"  # wav, mp3, ogg


class VoiceAdapter(ABC):
    """Abstract adapter for voice models (STT/TTS).

    Follows the same pattern as umrs.vision_service adapters.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name."""

    @property
    @abstractmethod
    def task_type(self) -> VoiceTaskType:
        """Whether this adapter does STT or TTS."""

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of model names this adapter supports."""

    @abstractmethod
    def load_model(self, config: VoiceConfig) -> None:
        """Load the model into memory."""

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources."""

    def health_check(self) -> bool:
        """Check if the adapter is ready."""
        return True


class STTAdapter(VoiceAdapter):
    """Abstract adapter for speech-to-text."""

    @property
    def task_type(self) -> VoiceTaskType:
        return VoiceTaskType.STT

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> STTResult:
        """Transcribe audio to text."""


class TTSAdapter(VoiceAdapter):
    """Abstract adapter for text-to-speech."""

    @property
    def task_type(self) -> VoiceTaskType:
        return VoiceTaskType.TTS

    @abstractmethod
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize text to audio."""


# ── Adapter registry ────────────────────────────────────────────────────────


class VoiceAdapterRegistry:
    """Registry for voice adapters.

    Mirrors the pattern from umrs.vision_service.adapters.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, VoiceAdapter] = {}

    def register(self, adapter: VoiceAdapter) -> None:
        self._adapters[adapter.name] = adapter
        logger.info("Registered voice adapter: %s", adapter.name)

    def get(self, name: str) -> Optional[VoiceAdapter]:
        return self._adapters.get(name)

    def find_for_model(self, model_name: str) -> Optional[VoiceAdapter]:
        """Find an adapter that supports the given model."""
        for adapter in self._adapters.values():
            if model_name in adapter.supported_models:
                return adapter
        return None

    def list_stt(self) -> List[STTAdapter]:
        return [
            a for a in self._adapters.values()
            if isinstance(a, STTAdapter)
        ]

    def list_tts(self) -> List[TTSAdapter]:
        return [
            a for a in self._adapters.values()
            if isinstance(a, TTSAdapter)
        ]

    def list_all(self) -> List[VoiceAdapter]:
        return list(self._adapters.values())


# Global registry
_registry = VoiceAdapterRegistry()


def get_voice_registry() -> VoiceAdapterRegistry:
    return _registry


def register_voice_adapter(adapter: VoiceAdapter) -> None:
    _registry.register(adapter)
