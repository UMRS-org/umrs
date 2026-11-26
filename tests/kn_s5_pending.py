"""Skip-until-implemented for slice KN-S5's stubbed modules (one helper).

``until_implemented(task_id, probe)`` skips a test while ``probe()`` still
raises ``NotImplementedError(task_id)``. With ``KN_S5_REQUIRE`` naming the
task id (comma-separated list allowed) it never skips, so a task's check
fails on the stub instead. The integrator deletes this module and its uses
once every task has landed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest


def _still_stub(task_id: str, probe: Callable[[], Any]) -> bool:
    try:
        probe()
    except NotImplementedError as exc:
        return str(exc) == task_id
    except Exception:  # noqa: BLE001 — an implementation that fails the probe input
        return False
    return False


def until_implemented(task_id: str, probe: Callable[[], Any]) -> Any:
    required = {t.strip() for t in os.environ.get("KN_S5_REQUIRE", "").split(",")}
    if task_id in required:
        return pytest.mark.usefixtures()
    return pytest.mark.skipif(
        _still_stub(task_id, probe), reason=f"{task_id} not implemented yet"
    )
