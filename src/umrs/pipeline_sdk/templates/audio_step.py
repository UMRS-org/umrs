"""``AudioStep`` — built-in step that transcribes audio files.

The step reads an audio file from S3, transcribes it using an
OpenAI-compatible Whisper endpoint (or falls back to a mock), and
writes the transcript as a JSON document.

Configuration keys:
    - ``whisper_url``: URL of the Whisper transcription endpoint.
    - ``language``: Language code (e.g. ``"en"``, ``"ja"``).
      Leave empty for auto-detection.
    - ``model``: Whisper model name (default: ``"whisper-1"``).
    - ``timeout_sec``: Per-request timeout (default: 300).

Example ``workflow.json`` snippet::

    {
      "name": "audio-transcribe",
      "entrypoint": "umrs.pipeline_sdk.templates.audio_step:AudioStep",
      "config": {
        "whisper_url": "http://whisper:9000/v1/audio/transcriptions",
        "model": "whisper-1"
      }
    }
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any

from umrs.pipeline_sdk.step import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


class AudioStep(Step):
    """Transcribe audio files using a Whisper endpoint."""

    name = "audio"

    def setup(self, config: dict[str, Any]) -> None:
        self.whisper_url: str | None = config.get("whisper_url")
        self.language: str | None = config.get("language")
        self.model: str = str(config.get("model", "whisper-1"))
        self.timeout_sec: float = float(config.get("timeout_sec", 300.0))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, ctx: StepContext) -> StepResult:
        out_keys: list[str] = []
        metadata_list: list[dict[str, Any]] = []

        for key in ctx.input_keys:
            try:
                raw = ctx.download_bytes(key)
            except Exception as exc:
                logger.warning("AudioStep: failed to read %s: %s", key, exc)
                continue

            transcript = self._transcribe(raw, key)

            out_key = self._write_transcript(transcript, key, ctx)
            out_keys.append(out_key)
            metadata_list.append({
                "source_key": key,
                "language": transcript.get("language"),
                "duration_sec": transcript.get("duration_sec"),
                "num_segments": len(transcript.get("segments", [])),
            })

        return StepResult(
            output_keys=out_keys,
            metadata=metadata_list,
            extra={"step": "audio"},
        )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe(self, raw: bytes, source_key: str) -> dict[str, Any]:
        """Transcribe audio bytes and return a transcript dict."""
        mock = os.environ.get("UMRS_AUDIO_MOCK", "0") in ("1", "true", "TRUE")
        if mock or not self.whisper_url:
            return self._mock_transcript(source_key)
        try:
            return self._call_whisper(raw, source_key)
        except Exception as exc:
            logger.warning(
                "AudioStep: Whisper call failed (%s), returning mock transcript",
                exc,
            )
            return self._mock_transcript(source_key, error=str(exc))

    def _call_whisper(self, raw: bytes, source_key: str) -> dict[str, Any]:
        """Call the Whisper endpoint and parse the response."""
        import httpx
        from pathlib import Path

        ext = Path(source_key).suffix or ".wav"
        filename = Path(source_key).name

        files = {"file": (filename, raw, f"audio/{ext.lstrip('.')}")}
        data: dict[str, str] = {"model": self.model}
        if self.language:
            data["language"] = self.language
        data["response_format"] = "verbose_json"

        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(self.whisper_url, files=files, data=data)
            response.raise_for_status()
            return response.json()

    def _mock_transcript(
        self,
        source_key: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Return a mock transcript for testing."""
        return {
            "text": f"[mock transcript for {source_key}]",
            "language": self.language or "en",
            "duration_sec": 0.0,
            "segments": [],
            "source_key": source_key,
            "mock": True,
            "error": error,
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_transcript(
        self,
        transcript: dict[str, Any],
        source_key: str,
        ctx: StepContext,
    ) -> str:
        """Write transcript as JSON to S3."""
        base = source_key.rsplit(".", 1)[0] if "." in source_key else source_key
        out_key = f"audio/{base}.json"
        data = json.dumps(transcript, indent=2).encode("utf-8")
        ctx.upload_bytes(out_key, data, content_type="application/json")
        return out_key
