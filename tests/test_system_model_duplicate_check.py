"""create_system_model's duplicate check (2026-09-25).

It called `self._check_duplicate(...)`, which is defined nowhere, so every
system-model registration died with AttributeError before writing anything.
The BDRT login-node checkout carried the fix as an un-upstreamed hotfix
(found while healing that checkout); this pins it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from umrs.model_registry.models import ModelScope
from umrs.model_registry.store.acl import AccessContext
from umrs.model_registry.store.crud import DuplicateModelError, ModelStore


def _store(existing):
    store = ModelStore.__new__(ModelStore)
    store.path_validator = SimpleNamespace(validate=lambda _ref: None)
    calls = []

    def find(name, scope, project_id=None):
        calls.append((name, scope, project_id))
        return existing

    store._find_model_by_name = find
    return store, calls


def test_duplicate_system_model_is_refused_by_name_not_attribute_error():
    store, calls = _store(existing=object())
    with pytest.raises(DuplicateModelError):
        store.create_system_model(
            "qwen", "/models/qwen", ctx=AccessContext(user_id="admin", is_admin=True)
        )
    assert calls == [("qwen", ModelScope.SYSTEM, None)]


def test_the_check_is_scope_wide_not_per_user():
    # System models are global: the owner of the ctx must not change the answer.
    store, calls = _store(existing=object())
    for user in ("admin", "other-admin"):
        with pytest.raises(DuplicateModelError):
            store.create_system_model(
                "qwen", "/models/qwen", ctx=AccessContext(user_id=user, is_admin=True)
            )
    assert {c[1] for c in calls} == {ModelScope.SYSTEM}
