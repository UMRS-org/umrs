"""
Launcher Stack — discover and version the runtime environment inside a SIF image.

A LauncherStack captures the complete runtime identity of a mini-RiVault SIF:
vLLM version, CUDA version, available parsers, FlashInfer/Triton/Torch versions,
and Python version.  This identity is used to:

1. Bind CertificationResults to a specific stack version (invalidated on upgrade)
2. Preflight-validate parser availability before sbatch
3. Generate reproducible runtime fingerprints for audit/debugging

The stack is discovered by executing Python introspection commands inside the
SIF via ``singularity exec`` on the remote AI4S login node (SSH).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from umrs.orchestrator.ssh_control import SSHControlManager

logger = logging.getLogger(__name__)

#: Python one-liner that prints JSON describing the runtime stack.
_LAUNCHER_STACK_PROBE = r"""
import json, sys
info = {"error": None}
try:
    import vllm
    info["vllm_version"] = vllm.__version__
except Exception as e:
    info["vllm_version"] = None
    info["error"] = info["error"] or f"vllm: {e}"

try:
    import torch
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
except Exception as e:
    info["torch_version"] = None
    info["cuda_version"] = None

try:
    import flashinfer
    info["flashinfer_version"] = getattr(flashinfer, "__version__", "unknown")
except Exception:
    info["flashinfer_version"] = None

try:
    import triton
    info["triton_version"] = getattr(triton, "__version__", "unknown")
except Exception:
    info["triton_version"] = None

try:
    info["python_version"] = sys.version.split()[0]
except Exception:
    info["python_version"] = None

# Discover available tool-call parsers
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    parsers = []
    if hasattr(ToolParserManager, "tool_parsers"):
        parsers = list(ToolParserManager.tool_parsers.keys())
    elif hasattr(ToolParserManager, "get_parser"):
        # Fallback: try known parser names
        known = ["hermes", "deepseek_v3", "deepseek_v4", "glm45", "glm47",
                  "kimi_k2", "gemma4", "gpt_oss", "llama3", "mistral"]
        parsers = [p for p in known if _parser_available(ToolParserManager, p)]
    info["available_tool_parsers"] = sorted(parsers)
except Exception:
    info["available_tool_parsers"] = None

# Discover available reasoning parsers
try:
    from vllm.reasoning import ReasoningParserManager
    r_parsers = []
    if hasattr(ReasoningParserManager, "reasoning_parsers"):
        r_parsers = list(ReasoningParserManager.reasoning_parsers.keys())
    info["available_reasoning_parsers"] = sorted(r_parsers)
except Exception:
    info["available_reasoning_parsers"] = None

def _parser_available(mgr, name):
    try:
        mgr.get_parser(name)
        return True
    except Exception:
        return False

print(json.dumps(info, default=str))
"""


@dataclass
class LauncherStack:
    """Discovered runtime identity of a mini-RiVault SIF image.

    All fields except ``sif_path`` and ``sif_sha256`` are discovered by
    executing the probe script inside the SIF.  Fields can be ``None`` when
    the corresponding package is not installed or the probe fails.
    """

    sif_path: str
    sif_sha256: str = ""

    vllm_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    python_version: str | None = None
    flashinfer_version: str | None = None
    triton_version: str | None = None

    available_tool_parsers: list[str] | None = None
    available_reasoning_parsers: list[str] | None = None

    probe_error: str | None = None

    #: Opaque fingerprint for quick equality checks.
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Deterministic fingerprint of this stack identity."""
        key_parts = [
            self.sif_sha256 or self.sif_path,
            self.vllm_version or "?",
            self.cuda_version or "?",
            self.flashinfer_version or "?",
            self.triton_version or "?",
            self.python_version or "?",
        ]
        raw = "|".join(key_parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def has_parser(self, parser_name: str) -> bool:
        """Check whether a tool-call parser is available."""
        if self.available_tool_parsers is None:
            return False  # Unknown — assume not available
        return parser_name in self.available_tool_parsers

    def has_reasoning_parser(self, parser_name: str) -> bool:
        """Check whether a reasoning parser is available."""
        if self.available_reasoning_parsers is None:
            return False
        return parser_name in self.available_reasoning_parsers

    def to_dict(self) -> dict[str, Any]:
        return {
            "sif_path": self.sif_path,
            "sif_sha256": self.sif_sha256,
            "vllm_version": self.vllm_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "python_version": self.python_version,
            "flashinfer_version": self.flashinfer_version,
            "triton_version": self.triton_version,
            "available_tool_parsers": self.available_tool_parsers,
            "available_reasoning_parsers": self.available_reasoning_parsers,
            "probe_error": self.probe_error,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LauncherStack":
        return cls(
            sif_path=data.get("sif_path", ""),
            sif_sha256=data.get("sif_sha256", ""),
            vllm_version=data.get("vllm_version"),
            torch_version=data.get("torch_version"),
            cuda_version=data.get("cuda_version"),
            python_version=data.get("python_version"),
            flashinfer_version=data.get("flashinfer_version"),
            triton_version=data.get("triton_version"),
            available_tool_parsers=data.get("available_tool_parsers"),
            available_reasoning_parsers=data.get("available_reasoning_parsers"),
            probe_error=data.get("probe_error"),
            fingerprint=data.get("fingerprint", ""),
        )

    @classmethod
    async def discover(
        cls,
        sif_path: str,
        ssh_manager: "SSHControlManager",
        *,
        sif_sha256: str = "",
    ) -> "LauncherStack":
        """Discover the runtime stack by probing the SIF on the remote host.

        Executes ``singularity exec <sif> python -c "..."`` via SSH to
        extract version information from the installed packages.

        Args:
            sif_path: Absolute path to the SIF on the remote host.
            ssh_manager: Connected SSHControlManager to the remote host.
            sif_sha256: Pre-computed SHA256 (used for fingerprinting).

        Returns:
            A LauncherStack with discovered information populated.

        Raises:
            RuntimeError: If the SSH command fails or the SIF cannot be
                executed (missing singularity, corrupt SIF, etc.).
        """
        # Sanitize the probe script for shell injection safety.
        # The probe is self-contained Python with single-quoted JSON;
        # we pass it via stdin to avoid shell escaping issues.
        probe_cmd = (
            f"singularity exec --cleanenv {_sh_quote(sif_path)} "
            f"python -c {_sh_quote(_LAUNCHER_STACK_PROBE)}"
        )

        result = await ssh_manager.run_remote(probe_cmd, timeout=90)

        if not result.ok:
            return cls(
                sif_path=sif_path,
                sif_sha256=sif_sha256,
                probe_error=f"SSH probe failed (rc={result.returncode}): {result.stderr[:500]}",
            )

        try:
            raw = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return cls(
                sif_path=sif_path,
                sif_sha256=sif_sha256,
                probe_error=f"Probe output not valid JSON: {exc}",
            )

        return cls(
            sif_path=sif_path,
            sif_sha256=sif_sha256,
            vllm_version=raw.get("vllm_version"),
            torch_version=raw.get("torch_version"),
            cuda_version=raw.get("cuda_version"),
            python_version=raw.get("python_version"),
            flashinfer_version=raw.get("flashinfer_version"),
            triton_version=raw.get("triton_version"),
            available_tool_parsers=raw.get("available_tool_parsers"),
            available_reasoning_parsers=raw.get("available_reasoning_parsers"),
            probe_error=raw.get("error"),
        )


def _sh_quote(value: str) -> str:
    """Single-quote a string for safe use in a shell command."""
    # Python's shlex.quote is safer but may not be available on remote.
    # Single-quoting handles most cases; the probe script has no single quotes.
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"
