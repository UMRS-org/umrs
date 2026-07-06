"""Unit tests for catalog store dependency injection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from catalog_store_fake import FakeCatalogMetaStore

_api_dir = Path(__file__).resolve().parents[1] / "src" / "umrs" / "api"
_spec = importlib.util.spec_from_file_location(
    "umrs.api.catalog_store",
    _api_dir / "catalog_store.py",
)
assert _spec and _spec.loader
_catalog_store = importlib.util.module_from_spec(_spec)
sys.modules["umrs.api.catalog_store"] = _catalog_store
_spec.loader.exec_module(_catalog_store)

CatalogStoreUnavailable = _catalog_store.CatalogStoreUnavailable
get_catalog_meta_store = _catalog_store.get_catalog_meta_store
get_optional_catalog_meta_store = _catalog_store.get_optional_catalog_meta_store
init_catalog_meta_store = _catalog_store.init_catalog_meta_store
set_catalog_meta_store = _catalog_store.set_catalog_meta_store


@pytest.fixture(autouse=True)
def _reset_catalog_store():
    set_catalog_meta_store(None)
    yield
    set_catalog_meta_store(None)


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
