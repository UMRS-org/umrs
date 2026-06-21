"""Cost-model for LLM usage billing.

Maps token classes (prompt, completion, reasoning, cached reads/writes,
audio) to USD charge totals per user/project.

Token-class overlaps
--------------------
The OpenAI/vLLM ``usage`` object has two important subset relationships:

- ``prompt_tokens`` INCLUDES ``cached_read_tokens``.
- ``completion_tokens`` INCLUDES ``reasoning_tokens``.

To avoid double-counting we compute:

- ``non_cached_prompt`` = ``prompt_tokens`` - ``cached_read_tokens``
- ``visible_completion`` = ``completion_tokens`` - ``reasoning_tokens``

Then bill:

- non-cached prompt @ ``prompt`` rate
- cached reads @ ``cached_read`` rate (cache hits — cheaper)
- visible completion @ ``completion`` rate
- reasoning @ ``reasoning`` rate (may equal completion rate)
- cache writes @ ``cached_write`` rate
- audio @ ``audio`` rate

Rates are USD per **million** tokens (Anthropic-style convention).

Configuration
-------------
Rates come from (highest precedence first):

1. ``RIVAULT_BILLING_RATES`` env var (JSON string)
2. ``billing/pricing.yaml`` next to this module
3. Hard-coded ``DEFAULT_RATES``

Per-model overrides are supported via the ``models`` key in
``pricing.yaml``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Rate dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenRates:
    """USD per million tokens for each token class."""

    prompt: float = 1.00
    completion: float = 3.00
    cached_read: float = 0.10
    cached_write: float = 1.25
    reasoning: float = 3.00
    audio: float = 2.00


# Sensible defaults (USD per million tokens).
# These mirror common industry pricing: cache hits at 10% of prompt,
# cache writes at 1.25x prompt (Anthropic-style), reasoning at completion
# rate, audio at a midpoint between prompt and completion.
DEFAULT_RATES: TokenRates = TokenRates()


@dataclass
class ModelPricing:
    """Per-model pricing with fallback to defaults."""

    defaults: TokenRates = field(default_factory=TokenRates)
    overrides: dict[str, TokenRates] = field(default_factory=dict)

    def rates_for(self, model: str) -> TokenRates:
        """Return the rates for *model*, falling back to defaults."""
        return self.overrides.get(model, self.defaults)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _rates_from_dict(d: dict) -> TokenRates:
    """Build a :class:`TokenRates` from a dict, falling back to defaults."""
    base = DEFAULT_RATES
    return TokenRates(
        prompt=float(d.get("prompt", base.prompt)),
        completion=float(d.get("completion", base.completion)),
        cached_read=float(d.get("cached_read", base.cached_read)),
        cached_write=float(d.get("cached_write", base.cached_write)),
        reasoning=float(d.get("reasoning", base.reasoning)),
        audio=float(d.get("audio", base.audio)),
    )


def load_pricing() -> ModelPricing:
    """Load the pricing model from env / config file.

    Order of precedence:

    1. ``RIVAULT_BILLING_RATES`` env var (JSON) — overrides everything.
    2. ``pricing.yaml`` beside this module — per-model overrides.
    3. Hard-coded :data:`DEFAULT_RATES`.
    """
    here = Path(__file__).parent
    yaml_path = here / "pricing.yaml"

    # Defaults
    defaults = DEFAULT_RATES
    overrides: dict[str, TokenRates] = {}

    # 2. YAML config
    if yaml_path.exists():
        try:
            import yaml

            cfg = yaml.safe_load(yaml_path.read_text()) or {}
            if isinstance(cfg.get("defaults"), dict):
                defaults = _rates_from_dict(cfg["defaults"])
            models = cfg.get("models") or {}
            if isinstance(models, dict):
                for name, rates in models.items():
                    if isinstance(rates, dict):
                        overrides[name] = _rates_from_dict(rates)
        except Exception:  # pragma: no cover — config is best-effort
            pass

    # 1. Env var overrides defaults only (not per-model)
    env_rates = os.environ.get("RIVAULT_BILLING_RATES")
    if env_rates:
        try:
            d = json.loads(env_rates)
            if isinstance(d, dict):
                defaults = _rates_from_dict(d)
        except (json.JSONDecodeError, TypeError):
            pass

    return ModelPricing(defaults=defaults, overrides=overrides)


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def compute_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_read_tokens: int = 0,
    cached_write_tokens: int = 0,
    reasoning_tokens: int = 0,
    audio_tokens: int = 0,
    rates: TokenRates | None = None,
) -> dict:
    """Compute USD cost from token counts and rates.

    Token-class overlaps (important — avoids double-counting):

    - ``prompt_tokens`` INCLUDES ``cached_read_tokens``.
    - ``completion_tokens`` INCLUDES ``reasoning_tokens``.

    Returns a dict with per-class and total cost (USD, 6 dp).
    """
    rates = rates or DEFAULT_RATES

    # Deduct cache hits from prompt to avoid double-counting
    non_cached_prompt = max(0, prompt_tokens - cached_read_tokens)
    # Deduct reasoning from completion to get visible output
    visible_completion = max(0, completion_tokens - reasoning_tokens)

    prompt_cost = non_cached_prompt * rates.prompt / 1_000_000
    completion_cost = visible_completion * rates.completion / 1_000_000
    cached_read_cost = cached_read_tokens * rates.cached_read / 1_000_000
    cached_write_cost = cached_write_tokens * rates.cached_write / 1_000_000
    reasoning_cost = reasoning_tokens * rates.reasoning / 1_000_000
    audio_cost = audio_tokens * rates.audio / 1_000_000

    total = (
        prompt_cost
        + completion_cost
        + cached_read_cost
        + cached_write_cost
        + reasoning_cost
        + audio_cost
    )

    return {
        "prompt_cost": round(prompt_cost, 6),
        "completion_cost": round(completion_cost, 6),
        "cached_read_cost": round(cached_read_cost, 6),
        "cached_write_cost": round(cached_write_cost, 6),
        "reasoning_cost": round(reasoning_cost, 6),
        "audio_cost": round(audio_cost, 6),
        "total_cost": round(total, 6),
    }
