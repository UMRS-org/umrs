# Mini-RiVault

**vLLM + LiteLLM inference runtime** for the AI-for-Science Platform.

## Overview

Mini-RiVault is a self-contained inference container that runs:
- **vLLM** — High-performance LLM inference engine (:8000)
- **LiteLLM** — OpenAI-compatible API routing layer (:11434)

Each Slurm job or Kubernetes pod runs its own Mini-RiVault instance with
pre-staged model weights.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Mini-RiVault Container (Singularity/Podman)                            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LiteLLM (:11434)                         ← Gateway Target      │    │
│  │  • OpenAI-compatible API                                       │    │
│  │  • Model name mapping                                          │    │
│  │  • Static model name: rivault-gpt-oss-static                   │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                             │                                           │
│  ┌──────────────────────────▼──────────────────────────────────────┐    │
│  │  vLLM (:8000)                              ← Private, not exposed│    │
│  │  • Tensor parallel inference                                    │    │
│  │  • Tool calling support                                         │    │
│  │  • Model: /models/... (pre-staged)                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Mount Points:                                                          │
│    /models/      ← Pre-staged model weights (read-only)                │
│    /scratch/     ← Ephemeral workspace (read-write)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Usage

### HPC (Slurm + Singularity)

```bash
# Submit inference job
sbatch --time=01:00:00 -N 1 -p a100 serve-model.slurm

# Check endpoint
MODEL_ENDPOINT=$(scontrol show job $SLURM_JOB_ID | grep NodeList | awk -F'=' '{print $2}')
curl -H "Authorization: Bearer $LITELLM_KEY" \
     "http://${MODEL_ENDPOINT}:11434/v1/models"
```

### Local Development (Docker Compose)

```bash
# Set model path
export MODEL_PATH=/path/to/models/Qwen2.5-7B-Instruct

# Start with GPU
docker compose --env-file env.mini-rivault \
    -f compose.yaml \
    -f compose.nvidia.yaml \
    up -d

# Test
curl -H "Authorization: Bearer sk-demo-key" \
     http://localhost:11434/v1/models
```

## Configuration

### Environment Variables

```bash
# vLLM
VLLM_MAX_MODEL_LEN=4096        # Max context length
VLLM_MAX_NUM_SEQS=4            # Max concurrent sequences
VLLM_ENABLE_TOOL_CALLING=true  # Enable function calling
VLLM_TENSOR_PARALLEL_SIZE=1    # GPU parallelism
VLLM_MODEL_PATH=/models/...    # Pre-staged model path
VLLM_PORT=8000                 # Internal port

# LiteLLM
LITELLM_MASTER_KEY=sk-ll-xxxx  # Auth key
LITELLM_PORT=11434             # API port

# Offline mode (REQUIRED for inference containers)
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

### Model Configuration (litellm_fs/litellm_config.yaml)

```yaml
model_list:
  - model_name: rivault-gpt-oss-static
    litellm_params:
      model: openai/local
      api_base: http://127.0.0.1:8000/v1
```

## Files

| File | Purpose |
|------|---------|
| `serve-model.slurm` | Slurm batch script for HPC |
| `serve-model.yaml` | Configuration template |
| `compose.yaml` | Docker Compose base |
| `compose.nvidia.yaml` | NVIDIA GPU override |
| `litellm_fs/` | LiteLLM configuration |
| `coredns_fs/` | CoreDNS for service discovery |
| `nginx_fs/` | Optional NGINX layer |

## Key Invariants

1. **No model downloads** — Models must be pre-staged at `$VLLM_MODEL_PATH`
2. **Offline mode** — Set `TRANSFORMERS_OFFLINE=1` to fail fast on network access
3. **Static model name** — Always use `rivault-gpt-oss-static` for consistency
4. **Gateway access only** — Users access via Platform Gateway, never directly

## See Also

- [../ARCHITECTURE.md](../ARCHITECTURE.md) - Platform architecture
- [../demos/](../demos/) - Demo scripts
- [../gateway/](../gateway/) - Platform Gateway
