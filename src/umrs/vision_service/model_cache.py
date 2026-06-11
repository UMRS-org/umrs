"""Thread-safe LRU model cache with GPU memory eviction."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_CacheEntry = tuple[Any, float]  # (model_object, last_used_timestamp)


class ModelCache:
    """LRU cache that keeps up to ``max_models`` loaded vision models.

    When the cache is full and a new model needs loading, the
    least-recently-used entry is evicted and its GPU memory released.

    The ``max_models`` limit is read from the ``MAX_VISION_MODELS``
    environment variable (default **3**).
    """

    def __init__(self, max_models: int | None = None):
        if max_models is None:
            max_models = int(os.environ.get("MAX_VISION_MODELS", "3"))
        self._max = max_models
        self._lock = threading.Lock()
        self._entries: dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of model objects this cache can retain."""
        return self._max

    def get_or_load(self, adapter, config: dict) -> Any:
        """Return a cached model or load a new one via *adapter*.

        Args:
            adapter: A ``VisionModelAdapter`` instance.
            config: Model configuration dict (forwarded to adapter).

        Returns:
            The model object ready for inference.
        """
        key = adapter.cache_key(config)

        with self._lock:
            if key in self._entries:
                model, _ = self._entries[key]
                self._entries[key] = (model, time.monotonic())
                logger.debug("Cache hit: %s", key)
                return model

        # Load outside the lock (may be slow / GPU-heavy)
        logger.info("Cache miss; loading model: %s", key)
        model = adapter.load(config)

        # Run warmup/self-test if the adapter supports it
        if hasattr(adapter, "warmup"):
            try:
                ok = adapter.warmup(model, config)
                if not ok:
                    logger.warning(
                        "Warmup returned False for %s; model may be unhealthy", key
                    )
            except Exception as exc:
                logger.warning("Warmup raised for %s: %s; continuing anyway", key, exc)

        with self._lock:
            # Double-check: another thread may have loaded it concurrently
            if key in self._entries:
                logger.debug("Another thread loaded %s while we waited", key)
                model, _ = self._entries[key]
                self._entries[key] = (model, time.monotonic())
                return model

            # Evict if at capacity
            while len(self._entries) >= self._max:
                self._evict_lru()

            self._entries[key] = (model, time.monotonic())
        return model

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry.  Must be called under lock."""
        if not self._entries:
            return
        lru_key = min(self._entries, key=lambda k: self._entries[k][1])
        model, ts = self._entries.pop(lru_key)
        logger.info(
            "Evicting model %s (last used %.1fs ago)", lru_key, time.monotonic() - ts
        )
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
