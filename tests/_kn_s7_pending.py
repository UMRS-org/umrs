"""Skip-until-implemented marker for the KN-S7 worker queue (ADR-081 W2a).

The KN-S7 seam commit ships stub functions that raise
``NotImplementedError("kn-s7-NN")`` together with their full tests. Until a
stub is implemented, the tests that call it skip, so the suite stays green.
``KN_S7_STRICT=1`` (set by every worker's validate.sh) turns the skip off,
so a worker's own tests fail on the stub.

Integration (orchestrator): once every kn-s7 task has landed, delete this
module and every ``@pending(...)`` decorator that uses it.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any

import pytest

_MARK = 'NotImplementedError("kn-s7-'


def _stubbed(fn: Callable[..., Any]) -> bool:
    try:
        return _MARK in inspect.getsource(fn)
    except (OSError, TypeError):
        return False


def pending(*fns: Callable[..., Any]) -> pytest.MarkDecorator:
    """Skip while any of ``fns`` is still a kn-s7 stub (unless strict)."""
    strict = os.environ.get("KN_S7_STRICT") == "1"
    stubs = sorted({fn.__qualname__ for fn in fns if _stubbed(fn)})
    return pytest.mark.skipif(
        bool(stubs) and not strict,
        reason=f"KN-S7 stub not implemented yet: {', '.join(stubs)}",
    )
