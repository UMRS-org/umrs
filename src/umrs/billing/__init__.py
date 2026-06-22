"""Billing and cost-computation module.

Maps token classes (prompt, completion, reasoning, cached reads/writes,
audio) to USD charge totals per user/project.
"""

from __future__ import annotations

from .cost_model import (
    DEFAULT_RATES,
    ModelPricing,
    TokenRates,
    compute_cost,
    load_pricing,
)

__all__ = [
    "DEFAULT_RATES",
    "ModelPricing",
    "TokenRates",
    "compute_cost",
    "load_pricing",
]
