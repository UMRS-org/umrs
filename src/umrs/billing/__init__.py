"""**EXPERIMENTAL / DEFERRED** — not wired into request flows (X3, 2026-08-02).

Billing and cost-computation module.

Maps token classes (prompt, completion, reasoning, cached reads/writes,
audio) to USD charge totals per user/project. The cost model is in
memory only; no UMRS route, job, or ledger persists billing records.
X3 decision: billing remains deferred — do not build against this API
until a durable ledger batch lands.
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
