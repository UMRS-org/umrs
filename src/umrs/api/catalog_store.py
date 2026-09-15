"""Global catalog metadata store dependency injection."""

from __future__ import annotations

import logging
from typing import Any

from umrs.data_catalog.pg_store import CatalogMetaStore

logger = logging.getLogger(__name__)

_catalog_meta_store: CatalogMetaStore | None = None


class CatalogStoreUnavailable(RuntimeError):
    """Raised when catalog metadata store was never initialized in this process."""


def set_catalog_meta_store(store: CatalogMetaStore | None) -> None:
    """Set the global catalog metadata store."""
    global _catalog_meta_store
    _catalog_meta_store = store


def init_catalog_meta_store(db_pool: Any) -> CatalogMetaStore | None:
    """Initialize the process-global catalog store from a Postgres pool (idempotent)."""
    if _catalog_meta_store is not None:
        logger.debug("Catalog meta store already initialized; skipping")
        return _catalog_meta_store
    try:
        store = CatalogMetaStore(db_pool)
        set_catalog_meta_store(store)
        logger.info("Catalog meta store initialized")
        return store
    except Exception:
        logger.exception("Failed to initialize catalog meta store")
        return None


def get_catalog_meta_store() -> CatalogMetaStore:
    """Get the global catalog metadata store."""
    if _catalog_meta_store is None:
        raise CatalogStoreUnavailable("Catalog store not initialized")
    return _catalog_meta_store


def get_optional_catalog_meta_store() -> CatalogMetaStore | None:
    """Return the configured catalog store, if this process owns one."""
    return _catalog_meta_store
