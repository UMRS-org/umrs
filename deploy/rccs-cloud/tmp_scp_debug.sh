#!/bin/bash
set -e
REMOTE="rccs-cloud:umrs-deploy"
LOCAL="/Users/mnagaso/workspace/RCCS/umrs"

scp "${LOCAL}/src/umrs/mini-rivault-llm/serve-model.slurm" "${REMOTE}/src/umrs/mini-rivault-llm/serve-model.slurm"
echo "1-OK inner serve-model.slurm"

scp "${LOCAL}/src/umrs/mini-rivault-llm/serve-model.yaml" "${REMOTE}/src/umrs/mini-rivault-llm/serve-model.yaml"
echo "2-OK serve-model.yaml"

scp "${LOCAL}/src/umrs/vision_service/service.py" "${REMOTE}/src/umrs/vision_service/service.py"
echo "3-OK service.py"

scp "${LOCAL}/src/umrs/vision_service/bentoml_adapter.py" "${REMOTE}/src/umrs/vision_service/bentoml_adapter.py"
echo "4-OK bentoml_adapter.py"

scp "${LOCAL}/src/umrs/vision_service/default_adapters/sam3_adapter.py" "${REMOTE}/src/umrs/vision_service/default_adapters/sam3_adapter.py"
echo "5-OK sam3_adapter.py"

scp "${LOCAL}/src/umrs/vision_service/default_adapters/hmae_adapter.py" "${REMOTE}/src/umrs/vision_service/default_adapters/hmae_adapter.py"
echo "6-OK hmae_adapter.py"

scp "${LOCAL}/deploy/rccs-cloud/serve-model.slurm" "${REMOTE}/deploy/rccs-cloud/serve-model.slurm"
echo "7-OK outer serve-model.slurm"

echo "ALL_SCP_DONE"
