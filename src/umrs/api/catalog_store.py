"""Global catalog metadata store dependency injection.

ADR-053 (1.1): the singleton home moved to
`umrs.shared.catalog_store_access` (jobs→api cycle closure). This module is
its api-layer re-export preserving every api caller byte-for-byte.
"""

from umrs.shared.catalog_store_access import (
    CatalogStoreUnavailable,
    get_catalog_meta_store,
    get_optional_catalog_meta_store,
    init_catalog_meta_store,
    set_catalog_meta_store,
)

__all__ = [
    "CatalogStoreUnavailable",
    "get_catalog_meta_store",
    "get_optional_catalog_meta_store",
    "init_catalog_meta_store",
    "set_catalog_meta_store",
]
