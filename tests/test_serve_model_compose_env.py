"""serve-model.yaml: every variable the vLLM command expands in the CONTAINER
shell must be declared in the service environment — an undeclared one is
silently empty (the VLLM_REASONING_PARSER and VLLM_SPECULATIVE_CONFIG drops)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "src/umrs/mini-rivault-llm/serve-model.yaml"


def _vllm_service():
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    return next(s for s in services.values() if "vllm serve" in str(s.get("command", "")))


def test_every_container_expanded_var_is_declared():
    svc = _vllm_service()
    command = str(svc["command"])
    # $$VAR / $${VAR...} escape compose interpolation → the container shell expands them
    used = set(re.findall(r"\$\$\{?([A-Z][A-Z0-9_]*)", command))
    declared = set(svc.get("environment") or {})
    missing = sorted(v for v in used if v.startswith("VLLM_") and v not in declared)
    assert not missing, f"container-expanded but undeclared: {missing}"


def test_speculative_config_reaches_the_container():
    assert "VLLM_SPECULATIVE_CONFIG" in (_vllm_service().get("environment") or {})
