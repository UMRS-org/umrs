"""Scientific Catalog foundation package (store only; routes in umrs.api)."""

from .store import (  # noqa: F401 — re-exported for convenience
    CatalogNotFound,
    CatalogSpec,
    ScientificCatalogStore,
)
