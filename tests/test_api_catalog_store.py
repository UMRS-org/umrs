"""Unit tests for catalog store dependency injection."""

from __future__ import annotations

import pytest
from catalog_store_fake import FakeCatalogMetaStore
from umrs.api.catalog_store import (
    CatalogStoreUnavailable,
    get_catalog_meta_store,
    get_optional_catalog_meta_store,
    init_catalog_meta_store,
    set_catalog_meta_store,
)

# NOTE: real package import (2026-09-13 repair). This file previously loaded a
# standalone module object into ``sys.modules['umrs.api.catalog_store']``,
# which replaced the package for every later suite in the same process and
# poisoned routes bound after it (catalog stats 503 in full-suite runs when
# `tests/api/catalog...` imported the route through the shadow).


@pytest.fixture(autouse=True)
def _reset_catalog_store():
    prior = get_optional_catalog_meta_store()
    set_catalog_meta_store(None)
    yield
    set_catalog_meta_store(prior)


def test_get_catalog_meta_store_raises_when_uninitialized():
    with pytest.raises(CatalogStoreUnavailable, match="not initialized"):
        get_catalog_meta_store()


def test_get_optional_returns_none_when_uninitialized():
    assert get_optional_catalog_meta_store() is None


def test_init_catalog_meta_store_is_idempotent():
    fake_pool = object()
    first = init_catalog_meta_store(fake_pool)
    second = init_catalog_meta_store(object())
    assert first is second
    assert get_catalog_meta_store() is first


def test_set_catalog_meta_store_for_tests():
    store = FakeCatalogMetaStore()
    set_catalog_meta_store(store)
    assert get_catalog_meta_store() is store
