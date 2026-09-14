"""
Certification — model validation, smoke test, and certification tracking.

A CertificationResult binds a ModelDefinition to a specific LauncherStack,
recording the tested runtime configuration (parsers, topology, context limits)
along with evidence (smoke test output, timing).

Certifications are invalidated when the LauncherStack changes (SIF upgrade).
The ``certified`` flag on model_catalog_entries is set based on the latest
CertificationResult for the current LauncherStack.

Schema (new table, added in schema v7):
    model_certifications (
        cert_id, model_id, launcher_fingerprint, sif_version,
        certified_config JSONB, smoke_test_result JSONB,
        certified_at, expires_at, tester
    )
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CertificationResult:
    """Evidence that a model was tested and works on a specific LauncherStack."""

    # Identity
    cert_id: str = ""  # cert-xxxxxxxxxxxx
    model_name: str = ""  # Registered model name
    model_id: str = ""  # model_catalog_entries.model_id

    # Launcher stack binding
    launcher_fingerprint: str = ""  # LauncherStack.fingerprint
    sif_version: str = ""  # "v4"
    sif_sha256: str = ""  # Content-addressable identity

    # Certified configuration (what was tested and proven)
    certified_config: dict[str, Any] = field(default_factory=dict)
    # {
    #   "tool_call_parser": "glm47",
    #   "reasoning_parser": "glm45",
    #   "kv_cache_dtype": null,
    #   "enforce_eager": true,
    #   "flashinfer_moe_fp16": false,
    #   "validated_topologies": {
    #     "default": {"tp": 4, "pp": 4, "nodes": 4, "ctx": 202752, "tested": true},
    #     "tep-256k": {"tp": 8, "pp": 1, "nodes": 2, "ctx": 262144, "tested": false}
    #   }
    # }

    # Smoke test evidence
    smoke_test_passed: bool = False
    smoke_test_duration_ms: float = 0.0
    smoke_test_output_hash: str = ""  # SHA256 of expected output
    smoke_test_prompt: str = ""  # The prompt used for testing
    smoke_test_messages: list[dict] = field(default_factory=list)

    # Metadata
    certified_at: str = ""  # ISO timestamp
    tester: str = ""  # Who ran the certification
    notes: str = ""

    # Expiration
    invalidated_at: str | None = None  # Set when superseded or SIF changes
    invalidated_reason: str = ""

    def is_valid_for(self, launcher_fingerprint: str) -> bool:
        """Check if this certification is valid for a given LauncherStack."""
        if self.invalidated_at:
            return False
        return self.launcher_fingerprint == launcher_fingerprint

    def invalidate(self, reason: str) -> None:
        """Mark this certification as invalid (e.g., SIF upgraded)."""
        self.invalidated_at = datetime.now(timezone.utc).isoformat()
        self.invalidated_reason = reason

    @property
    def is_active(self) -> bool:
        return self.invalidated_at is None and self.smoke_test_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "cert_id": self.cert_id,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "launcher_fingerprint": self.launcher_fingerprint,
            "sif_version": self.sif_version,
            "sif_sha256": self.sif_sha256,
            "certified_config": self.certified_config,
            "smoke_test_passed": self.smoke_test_passed,
            "smoke_test_duration_ms": self.smoke_test_duration_ms,
            "smoke_test_output_hash": self.smoke_test_output_hash,
            "smoke_test_prompt": self.smoke_test_prompt,
            "certified_at": self.certified_at,
            "tester": self.tester,
            "notes": self.notes,
            "invalidated_at": self.invalidated_at,
            "invalidated_reason": self.invalidated_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CertificationResult:
        return cls(
            cert_id=data.get("cert_id", ""),
            model_name=data.get("model_name", ""),
            model_id=data.get("model_id", ""),
            launcher_fingerprint=data.get("launcher_fingerprint", ""),
            sif_version=data.get("sif_version", ""),
            sif_sha256=data.get("sif_sha256", ""),
            certified_config=data.get("certified_config", {}),
            smoke_test_passed=data.get("smoke_test_passed", False),
            smoke_test_duration_ms=data.get("smoke_test_duration_ms", 0.0),
            smoke_test_output_hash=data.get("smoke_test_output_hash", ""),
            smoke_test_prompt=data.get("smoke_test_prompt", ""),
            smoke_test_messages=data.get("smoke_test_messages", []),
            certified_at=data.get("certified_at", ""),
            tester=data.get("tester", ""),
            notes=data.get("notes", ""),
            invalidated_at=data.get("invalidated_at"),
            invalidated_reason=data.get("invalidated_reason", ""),
        )

    @classmethod
    def from_row(cls, row: tuple) -> CertificationResult:
        """Create from database row."""
        certified_config = row[6] if len(row) > 6 else {}
        smoke_result = row[7] if len(row) > 7 else {}
        return cls(
            cert_id=row[0],
            model_id=row[1],
            model_name=row[2] if len(row) > 2 else "",
            launcher_fingerprint=row[3] if len(row) > 3 else "",
            sif_version=row[4] if len(row) > 4 else "",
            sif_sha256=row[5] if len(row) > 5 else "",
            certified_config=certified_config
            if isinstance(certified_config, dict)
            else {},
            smoke_test_passed=smoke_result.get("passed", False)
            if isinstance(smoke_result, dict)
            else False,
            smoke_test_duration_ms=smoke_result.get("duration_ms", 0.0)
            if isinstance(smoke_result, dict)
            else 0.0,
            smoke_test_output_hash=smoke_result.get("output_hash", "")
            if isinstance(smoke_result, dict)
            else "",
            smoke_test_prompt=smoke_result.get("prompt", "")
            if isinstance(smoke_result, dict)
            else "",
            smoke_test_messages=smoke_result.get("messages", [])
            if isinstance(smoke_result, dict)
            else [],
            certified_at=str(row[8] or "") if len(row) > 8 else "",
            tester=row[9] if len(row) > 9 else "",
            notes=row[10] if len(row) > 10 else "",
            invalidated_at=str(row[11]) if len(row) > 11 and row[11] else None,
            invalidated_reason=row[12] if len(row) > 12 else "",
        )


# ── Smoke test prompts per model family ──────────────────────────────────────

_SMOKE_TEST_PROMPTS: dict[str, str] = {
    "glm": "What is 2+2? Answer with just the number.",
    "deepseek": "Say 'hello world' in exactly those words.",
    "deepseek_v4": "Say 'hello world' in exactly those words.",
    "kimi": "Say 'hello' exactly.",
    "kimi_k2": "Say 'hello' exactly.",
    "llm_jp": "こんにちはと言ってください。",
    "gpt_oss": "Say 'hello' exactly.",
    "gemma4": "Say 'hello' exactly.",
    "diffusion_gemma": "Say 'hello' exactly.",
    "unknown": "Say 'hello' exactly.",
}


def smoke_test_output_hash(response_text: str) -> str:
    """Hash the smoke test response for evidence tracking."""
    return hashlib.sha256(response_text.encode()).hexdigest()[:16]


def get_smoke_prompt(model_family: str) -> str:
    """Get the smoke test prompt for a model family."""
    for key in (model_family, model_family.lower(), "unknown"):
        if key in _SMOKE_TEST_PROMPTS:
            return _SMOKE_TEST_PROMPTS[key]
    return _SMOKE_TEST_PROMPTS["unknown"]
