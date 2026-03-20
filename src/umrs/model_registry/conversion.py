"""
Model Conversion Pipeline (MM-004).

NOTE: This module defines conversion specifications and job models.
      The actual conversion backends (calling torch, auto-gptq, etc.)
      are not yet implemented. Specs are ready for integration when
      conversion execution is needed.

Supports:
- FP16/BF16 conversion
- FP8 quantization
- AWQ quantization
- GPTQ quantization
- GGUF conversion for llama.cpp

Conversions run as durable jobs via the UMRS job system.
See DEVELOPMENT_OBJECTIVES.md Section 8.1 for design details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversionType(Enum):
    """Supported model conversion types."""

    # Precision conversions
    FP32_TO_FP16 = "fp32_to_fp16"
    FP32_TO_BF16 = "fp32_to_bf16"
    FP16_TO_FP8 = "fp16_to_fp8"

    # Quantization
    AWQ = "awq"              # 4-bit Activation-aware Weight Quantization
    GPTQ = "gptq"            # 4-bit GPTQ quantization
    GGUF = "gguf"            # llama.cpp format
    GGUF_Q4_K_M = "gguf_q4_k_m"  # Specific GGUF quantization
    GGUF_Q5_K_M = "gguf_q5_k_m"
    GGUF_Q8_0 = "gguf_q8_0"

    # Adapter merging
    MERGE_LORA = "merge_lora"
    MERGE_QLORA = "merge_qlora"

    # Format conversions
    SAFETENSORS = "safetensors"  # Convert to safetensors format


class ConversionStatus(Enum):
    """Status of a conversion job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ConversionConfig:
    """
    Configuration for model conversion.

    Specifies conversion parameters based on type.
    """

    conversion_type: ConversionType

    # Common parameters
    output_name: Optional[str] = None  # Override output model name
    keep_original: bool = True         # Keep source model after conversion

    # Quantization parameters
    bits: int = 4                      # For AWQ/GPTQ: 4 or 8
    group_size: int = 128              # Quantization group size
    use_exllama: bool = True           # Use exllama kernels for faster inference
    desc_act: bool = False             # Order activations by decreasing order

    # AWQ specific
    awq_version: str = "gemm"          # "gemm" or "gemv"

    # GPTQ specific
    gptq_damp_percent: float = 0.01    # Dampening percentage
    gptq_sym: bool = True              # Symmetric quantization

    # GGUF specific
    gguf_type: str = "q4_k_m"          # Quantization type for GGUF
    gguf_vocab_type: Optional[str] = None  # Override vocab type

    # LoRA merge parameters
    lora_adapter_path: Optional[str] = None
    lora_scale: float = 1.0

    # Resource requirements
    gpu_required: bool = True
    min_gpu_memory_gb: float = 24.0
    estimated_time_minutes: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for job spec."""
        return {
            "conversion_type": self.conversion_type.value,
            "output_name": self.output_name,
            "keep_original": self.keep_original,
            "bits": self.bits,
            "group_size": self.group_size,
            "use_exllama": self.use_exllama,
            "desc_act": self.desc_act,
            "awq_version": self.awq_version,
            "gptq_damp_percent": self.gptq_damp_percent,
            "gptq_sym": self.gptq_sym,
            "gguf_type": self.gguf_type,
            "gguf_vocab_type": self.gguf_vocab_type,
            "lora_adapter_path": self.lora_adapter_path,
            "lora_scale": self.lora_scale,
            "gpu_required": self.gpu_required,
            "min_gpu_memory_gb": self.min_gpu_memory_gb,
            "estimated_time_minutes": self.estimated_time_minutes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversionConfig":
        """Create from dictionary."""
        return cls(
            conversion_type=ConversionType(data["conversion_type"]),
            output_name=data.get("output_name"),
            keep_original=data.get("keep_original", True),
            bits=data.get("bits", 4),
            group_size=data.get("group_size", 128),
            use_exllama=data.get("use_exllama", True),
            desc_act=data.get("desc_act", False),
            awq_version=data.get("awq_version", "gemm"),
            gptq_damp_percent=data.get("gptq_damp_percent", 0.01),
            gptq_sym=data.get("gptq_sym", True),
            gguf_type=data.get("gguf_type", "q4_k_m"),
            gguf_vocab_type=data.get("gguf_vocab_type"),
            lora_adapter_path=data.get("lora_adapter_path"),
            lora_scale=data.get("lora_scale", 1.0),
            gpu_required=data.get("gpu_required", True),
            min_gpu_memory_gb=data.get("min_gpu_memory_gb", 24.0),
            estimated_time_minutes=data.get("estimated_time_minutes", 60),
        )


@dataclass
class ConversionJob:
    """
    Model conversion job specification.

    Submitted as a durable job to UMRS job system.
    """

    job_id: str
    source_model_id: str
    source_storage_ref: str
    target_storage_ref: str
    config: ConversionConfig

    # Ownership
    project_id: str
    user_id: str

    # Status
    status: ConversionStatus = ConversionStatus.PENDING
    progress_percent: int = 0
    error_message: Optional[str] = None

    # Result
    output_model_id: Optional[str] = None
    output_size_bytes: Optional[int] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "job_id": self.job_id,
            "source_model_id": self.source_model_id,
            "source_storage_ref": self.source_storage_ref,
            "target_storage_ref": self.target_storage_ref,
            "config": self.config.to_dict(),
            "project_id": self.project_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "error_message": self.error_message,
            "output_model_id": self.output_model_id,
            "output_size_bytes": self.output_size_bytes,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ConversionPipeline:
    """
    Model conversion pipeline orchestrator.

    Validates conversion requests and generates job specifications.
    """

    # Conversion compatibility matrix
    SUPPORTED_CONVERSIONS = {
        ConversionType.FP32_TO_FP16: {"source_formats": ["fp32"], "requires_gpu": False},
        ConversionType.FP32_TO_BF16: {"source_formats": ["fp32"], "requires_gpu": False},
        ConversionType.FP16_TO_FP8: {"source_formats": ["fp16", "bf16"], "requires_gpu": True},
        ConversionType.AWQ: {"source_formats": ["fp16", "bf16"], "requires_gpu": True},
        ConversionType.GPTQ: {"source_formats": ["fp16", "bf16"], "requires_gpu": True},
        ConversionType.GGUF: {"source_formats": ["fp16", "bf16", "fp32"], "requires_gpu": False},
        ConversionType.GGUF_Q4_K_M: {"source_formats": ["fp16", "bf16"], "requires_gpu": False},
        ConversionType.GGUF_Q5_K_M: {"source_formats": ["fp16", "bf16"], "requires_gpu": False},
        ConversionType.GGUF_Q8_0: {"source_formats": ["fp16", "bf16"], "requires_gpu": False},
        ConversionType.MERGE_LORA: {"source_formats": ["*"], "requires_gpu": True},
        ConversionType.MERGE_QLORA: {"source_formats": ["*"], "requires_gpu": True},
        ConversionType.SAFETENSORS: {"source_formats": ["*"], "requires_gpu": False},
    }

    # Size reduction estimates
    SIZE_MULTIPLIERS = {
        ConversionType.FP32_TO_FP16: 0.5,
        ConversionType.FP32_TO_BF16: 0.5,
        ConversionType.FP16_TO_FP8: 0.5,
        ConversionType.AWQ: 0.25,  # 4-bit
        ConversionType.GPTQ: 0.25,  # 4-bit
        ConversionType.GGUF_Q4_K_M: 0.28,
        ConversionType.GGUF_Q5_K_M: 0.35,
        ConversionType.GGUF_Q8_0: 0.55,
    }

    @classmethod
    def validate_conversion(
        cls,
        conversion_type: ConversionType,
        source_format: Optional[str] = None,
    ) -> bool:
        """Validate if conversion is supported."""
        if conversion_type not in cls.SUPPORTED_CONVERSIONS:
            return False

        spec = cls.SUPPORTED_CONVERSIONS[conversion_type]
        if source_format and spec["source_formats"] != ["*"]:
            if source_format not in spec["source_formats"]:
                return False

        return True

    @classmethod
    def estimate_output_size(
        cls,
        conversion_type: ConversionType,
        source_size_bytes: int,
    ) -> int:
        """Estimate output model size after conversion."""
        multiplier = cls.SIZE_MULTIPLIERS.get(conversion_type, 1.0)
        return int(source_size_bytes * multiplier)

    @classmethod
    def estimate_time(
        cls,
        conversion_type: ConversionType,
        source_size_gb: float,
    ) -> int:
        """Estimate conversion time in minutes."""
        base_times = {
            ConversionType.FP32_TO_FP16: 5,
            ConversionType.AWQ: 60,
            ConversionType.GPTQ: 120,
            ConversionType.GGUF: 30,
            ConversionType.MERGE_LORA: 10,
        }
        base = base_times.get(conversion_type, 30)
        # Scale roughly with model size
        return int(base * (source_size_gb / 10.0))

    @classmethod
    def generate_container_command(cls, config: ConversionConfig) -> List[str]:
        """Generate container command for conversion."""
        base_cmd = ["python", "-m", "model_converter"]

        if config.conversion_type == ConversionType.AWQ:
            return base_cmd + [
                "awq",
                "--bits", str(config.bits),
                "--group-size", str(config.group_size),
                "--version", config.awq_version,
            ]
        elif config.conversion_type == ConversionType.GPTQ:
            return base_cmd + [
                "gptq",
                "--bits", str(config.bits),
                "--group-size", str(config.group_size),
                "--damp-percent", str(config.gptq_damp_percent),
                "--sym" if config.gptq_sym else "--asym",
            ]
        elif config.conversion_type in [
            ConversionType.GGUF,
            ConversionType.GGUF_Q4_K_M,
            ConversionType.GGUF_Q5_K_M,
            ConversionType.GGUF_Q8_0,
        ]:
            return base_cmd + [
                "gguf",
                "--type", config.gguf_type,
            ]
        elif config.conversion_type == ConversionType.MERGE_LORA:
            return base_cmd + [
                "merge-lora",
                "--adapter", config.lora_adapter_path or "",
                "--scale", str(config.lora_scale),
            ]
        elif config.conversion_type == ConversionType.FP32_TO_FP16:
            return base_cmd + ["fp16"]
        elif config.conversion_type == ConversionType.FP32_TO_BF16:
            return base_cmd + ["bf16"]
        else:
            return base_cmd + [config.conversion_type.value]
