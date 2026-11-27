"""Skip-until-implemented marker for the KN-S6 worker tasks (integrator: remove).

``pending("kn-s6-01", fn, ...)`` skips a test while any of ``fn`` still
raises its ``NotImplementedError("kn-s6-01")`` stub. ``KN_S6_STRICT=1`` (set
by every task's validate.sh) never skips, so an unimplemented stub fails.
Once all KN-S6 tasks have landed, delete this module and its uses.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any

import pytest


def _is_stub(task_id: str, fn: Callable[..., Any]) -> bool:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return f'NotImplementedError("{task_id}")' in source


def pending(task_id: str, *fns: Callable[..., Any]) -> pytest.MarkDecorator:
    stub = any(_is_stub(task_id, fn) for fn in fns)
    strict = os.environ.get("KN_S6_STRICT") == "1"
    return pytest.mark.skipif(
        stub and not strict, reason=f"{task_id} not implemented yet (KN-S6 queue)"
    )
