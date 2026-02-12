"""
Model Conversion Pipeline (MM-004).

Handles model format conversion and optimization:
- GGUF quantization for llama.cpp
- AWQ/GPTQ quantization for vLLM
- SafeTensors conversion
- LoRA adapter merging
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization formats."""

    # GGUF quantizations (for llama.cpp)
    GGUF_Q4_0 = "Q4_0"
    GGUF_Q4_K_M = "Q4_K_M"
    GGUF_Q5_K_M = "Q5_K_M"
    GGUF_Q6_K = "Q6_K"
    GGUF_Q8_0 = "Q8_0"

    # vLLM-compatible quantizations
    AWQ = "awq"
    GPTQ = "gptq"
    FP8 = "fp8"

    # No quantization
    FP16 = "fp16"
    BF16 = "bf16"


class ConversionFormat(Enum):
    """Target format for conversion."""

    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    VLLM = "vllm"


class ConversionStatus(Enum):
    """Conversion job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""

    # Paths
    work_dir: str = "/tmp/model-conversion"
    output_dir: str = "/shared/models/converted"

    # llama.cpp paths for GGUF conversion
    llamacpp_path: Optional[str] = None

    # AWQ paths
    awq_config: Optional[str] = None

    # Optimization settings
    use_gpu: bool = True
    gpu_memory_gb: float = 24.0
    num_threads: int = 8

    # Timeouts
    conversion_timeout_seconds: int = 7200

    # Cleanup
    keep_intermediate: bool = False


@dataclass
class ConversionJob:
    """Represents a conversion job."""

    id: str
    source_path: str
    target_format: ConversionFormat
    quantization: Optional[QuantizationType]
    status: ConversionStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    output_path: Optional[str] = None
    output_size_bytes: int = 0
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


@dataclass
class ConversionResult:
    """Result of a conversion operation."""

    success: bool
    output_path: str
    output_format: ConversionFormat
    quantization: Optional[QuantizationType]
    source_size_bytes: int
    output_size_bytes: int
    duration_seconds: float
    compression_ratio: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelConverter:
    """
    Model Conversion Pipeline (MM-004).

    Converts models between formats and applies quantization
    for optimized inference.
    """

    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize converter."""
        self.config = config or ConversionConfig()
        self._jobs: Dict[str, ConversionJob] = {}

        # Ensure directories exist
        Path(self.config.work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    async def convert_to_gguf(
        self,
        source_path: str,
        output_name: str,
        quantization: QuantizationType = QuantizationType.GGUF_Q4_K_M,
    ) -> ConversionResult:
        """
        Convert model to GGUF format for llama.cpp.

        Args:
            source_path: Path to source model (HF format)
            output_name: Name for output file
            quantization: GGUF quantization type

        Returns:
            Conversion result
        """
        start_time = datetime.now()
        source = Path(source_path)

        if not source.exists():
            return ConversionResult(
                success=False,
                output_path="",
                output_format=ConversionFormat.GGUF,
                quantization=quantization,
                source_size_bytes=0,
                output_size_bytes=0,
                duration_seconds=0,
                compression_ratio=0,
                error=f"Source path not found: {source_path}",
            )

        source_size = self._get_dir_size(source)

        try:
            # Step 1: Convert to FP16 GGUF
            work_path = Path(self.config.work_dir) / output_name
            work_path.mkdir(parents=True, exist_ok=True)

            fp16_path = work_path / f"{output_name}-fp16.gguf"
            await self._run_gguf_convert(source, fp16_path)

            # Step 2: Quantize
            output_path = Path(self.config.output_dir) / f"{output_name}-{quantization.value}.gguf"
            await self._run_gguf_quantize(fp16_path, output_path, quantization)

            output_size = output_path.stat().st_size
            duration = (datetime.now() - start_time).total_seconds()

            # Cleanup
            if not self.config.keep_intermediate:
                shutil.rmtree(work_path)

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                output_format=ConversionFormat.GGUF,
                quantization=quantization,
                source_size_bytes=source_size,
                output_size_bytes=output_size,
                duration_seconds=duration,
                compression_ratio=source_size / output_size if output_size > 0 else 0,
                metadata={
                    "gguf_version": "3",
                    "vocab_type": "auto",
                },
            )

        except Exception as e:
            logger.error("GGUF conversion failed: %s", e)
            return ConversionResult(
                success=False,
                output_path="",
                output_format=ConversionFormat.GGUF,
                quantization=quantization,
                source_size_bytes=source_size,
                output_size_bytes=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                compression_ratio=0,
                error=str(e),
            )

    async def convert_to_safetensors(
        self,
        source_path: str,
        output_name: str,
    ) -> ConversionResult:
        """
        Convert model to SafeTensors format.

        Args:
            source_path: Path to source model
            output_name: Name for output directory

        Returns:
            Conversion result
        """
        start_time = datetime.now()
        source = Path(source_path)
        source_size = self._get_dir_size(source)

        try:
            from safetensors.torch import save_file
            import torch

            output_path = Path(self.config.output_dir) / output_name
            output_path.mkdir(parents=True, exist_ok=True)

            # Find and convert .bin files
            bin_files = list(source.glob("*.bin"))
            for bin_file in bin_files:
                state_dict = torch.load(bin_file, map_location="cpu")

                safetensor_name = bin_file.stem + ".safetensors"
                save_file(state_dict, output_path / safetensor_name)

                del state_dict

            # Copy non-.bin files
            for f in source.iterdir():
                if f.suffix not in [".bin", ".pt", ".pth"]:
                    shutil.copy2(f, output_path / f.name)

            output_size = self._get_dir_size(output_path)
            duration = (datetime.now() - start_time).total_seconds()

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                output_format=ConversionFormat.SAFETENSORS,
                quantization=None,
                source_size_bytes=source_size,
                output_size_bytes=output_size,
                duration_seconds=duration,
                compression_ratio=source_size / output_size if output_size > 0 else 1,
            )

        except Exception as e:
            logger.error("SafeTensors conversion failed: %s", e)
            return ConversionResult(
                success=False,
                output_path="",
                output_format=ConversionFormat.SAFETENSORS,
                quantization=None,
                source_size_bytes=source_size,
                output_size_bytes=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                compression_ratio=0,
                error=str(e),
            )

    async def quantize_awq(
        self,
        source_path: str,
        output_name: str,
        calibration_data: Optional[str] = None,
    ) -> ConversionResult:
        """
        Apply AWQ quantization for vLLM.

        Args:
            source_path: Path to source model
            output_name: Output directory name
            calibration_data: Path to calibration dataset

        Returns:
            Conversion result
        """
        start_time = datetime.now()
        source = Path(source_path)
        source_size = self._get_dir_size(source)

        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer

            # Load model
            model = AutoAWQForCausalLM.from_pretrained(
                str(source),
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(str(source))

            # Quantization config
            quant_config = {
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM",
            }

            # Quantize
            model.quantize(tokenizer, quant_config=quant_config)

            # Save
            output_path = Path(self.config.output_dir) / output_name
            model.save_quantized(str(output_path))
            tokenizer.save_pretrained(str(output_path))

            output_size = self._get_dir_size(output_path)
            duration = (datetime.now() - start_time).total_seconds()

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                output_format=ConversionFormat.VLLM,
                quantization=QuantizationType.AWQ,
                source_size_bytes=source_size,
                output_size_bytes=output_size,
                duration_seconds=duration,
                compression_ratio=source_size / output_size if output_size > 0 else 0,
                metadata={
                    "w_bit": 4,
                    "group_size": 128,
                },
            )

        except ImportError:
            return ConversionResult(
                success=False,
                output_path="",
                output_format=ConversionFormat.VLLM,
                quantization=QuantizationType.AWQ,
                source_size_bytes=source_size,
                output_size_bytes=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                compression_ratio=0,
                error="AWQ library not installed. Run: pip install autoawq",
            )
        except Exception as e:
            logger.error("AWQ quantization failed: %s", e)
            return ConversionResult(
                success=False,
                output_path="",
                output_format=ConversionFormat.VLLM,
                quantization=QuantizationType.AWQ,
                source_size_bytes=source_size,
                output_size_bytes=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                compression_ratio=0,
                error=str(e),
            )

    async def merge_lora(
        self,
        base_model_path: str,
        lora_path: str,
        output_name: str,
    ) -> ConversionResult:
        """
        Merge LoRA adapter into base model.

        Args:
            base_model_path: Path to base model
            lora_path: Path to LoRA adapter
            output_name: Output directory name

        Returns:
            Conversion result
        """
        start_time = datetime.now()
        base_size = self._get_dir_size(Path(base_model_path))

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load base model and LoRA
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype="auto",
                device_map="cpu",
            )
            model = PeftModel.from_pretrained(base_model, lora_path)

            # Merge
            merged_model = model.merge_and_unload()

            # Save
            output_path = Path(self.config.output_dir) / output_name
            merged_model.save_pretrained(str(output_path))

            # Copy tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            tokenizer.save_pretrained(str(output_path))

            output_size = self._get_dir_size(output_path)
            duration = (datetime.now() - start_time).total_seconds()

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                output_format=ConversionFormat.PYTORCH,
                quantization=None,
                source_size_bytes=base_size,
                output_size_bytes=output_size,
                duration_seconds=duration,
                compression_ratio=1.0,
                metadata={
                    "lora_merged": True,
                    "lora_path": lora_path,
                },
            )

        except ImportError:
            return ConversionResult(
                success=False,
                output_path="",
                output_format=ConversionFormat.PYTORCH,
                quantization=None,
                source_size_bytes=base_size,
                output_size_bytes=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                compression_ratio=0,
                error="PEFT library not installed. Run: pip install peft",
            )
        except Exception as e:
            logger.error("LoRA merge failed: %s", e)
            return ConversionResult(
                success=False,
                output_path="",
                output_format=ConversionFormat.PYTORCH,
                quantization=None,
                source_size_bytes=base_size,
                output_size_bytes=0,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                compression_ratio=0,
                error=str(e),
            )

    async def estimate_output_size(
        self,
        source_path: str,
        quantization: QuantizationType,
    ) -> int:
        """
        Estimate output size after conversion.

        Args:
            source_path: Path to source model
            quantization: Target quantization

        Returns:
            Estimated size in bytes
        """
        source_size = self._get_dir_size(Path(source_path))

        # Compression ratios are approximate
        ratios = {
            QuantizationType.GGUF_Q4_0: 4.0,
            QuantizationType.GGUF_Q4_K_M: 3.5,
            QuantizationType.GGUF_Q5_K_M: 3.0,
            QuantizationType.GGUF_Q6_K: 2.5,
            QuantizationType.GGUF_Q8_0: 2.0,
            QuantizationType.AWQ: 4.0,
            QuantizationType.GPTQ: 4.0,
            QuantizationType.FP8: 2.0,
            QuantizationType.FP16: 1.0,
            QuantizationType.BF16: 1.0,
        }

        ratio = ratios.get(quantization, 1.0)
        return int(source_size / ratio)

    def get_supported_conversions(self) -> Dict[str, List[str]]:
        """Get supported conversion paths."""
        return {
            "huggingface": [
                "gguf",
                "safetensors",
                "awq",
                "gptq",
            ],
            "pytorch": [
                "gguf",
                "safetensors",
            ],
            "safetensors": [
                "gguf",
            ],
            "lora": [
                "merged",
            ],
        }

    # =========================================================================
    # Private helpers
    # =========================================================================

    async def _run_gguf_convert(
        self,
        source: Path,
        output: Path,
    ) -> None:
        """Run llama.cpp convert script."""
        llamacpp = self.config.llamacpp_path or os.environ.get("LLAMA_CPP_PATH", "")

        if not llamacpp:
            # Try to find in common locations
            candidates = [
                "/opt/llama.cpp",
                os.path.expanduser("~/llama.cpp"),
                "/usr/local/llama.cpp",
            ]
            for c in candidates:
                if Path(c).exists():
                    llamacpp = c
                    break

        if not llamacpp or not Path(llamacpp).exists():
            raise RuntimeError(
                "llama.cpp not found. Set LLAMA_CPP_PATH environment variable."
            )

        convert_script = Path(llamacpp) / "convert_hf_to_gguf.py"

        cmd = [
            "python",
            str(convert_script),
            str(source),
            "--outfile",
            str(output),
            "--outtype",
            "f16",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=self.config.conversion_timeout_seconds,
        )

        if process.returncode != 0:
            raise RuntimeError(f"GGUF convert failed: {stderr.decode()}")

    async def _run_gguf_quantize(
        self,
        source: Path,
        output: Path,
        quantization: QuantizationType,
    ) -> None:
        """Run llama.cpp quantize."""
        llamacpp = self.config.llamacpp_path or os.environ.get("LLAMA_CPP_PATH", "")

        quantize_bin = Path(llamacpp) / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = Path(llamacpp) / "quantize"

        cmd = [
            str(quantize_bin),
            str(source),
            str(output),
            quantization.value,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=self.config.conversion_timeout_seconds,
        )

        if process.returncode != 0:
            raise RuntimeError(f"GGUF quantize failed: {stderr.decode()}")

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory."""
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size

        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total
