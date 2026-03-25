"""UMRS Vision Service — BentoML-based async task queue for vision models.

Submodules:
    adapters        — Pluggable adapter registry (ABC, auto-discovery)
    bentoml_client  — submit_and_wait helper for BentoML task endpoints
    bentoml_adapter — compute_embeddings bridge for EmbeddingModel
    model_cache     — Thread-safe LRU model cache with GPU eviction
    service         — BentoML service (VisionModelService)
    default_adapters — Built-in SAM3 and HMAE adapters
"""
