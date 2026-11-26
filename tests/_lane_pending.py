"""Skip-until-implemented marks for queued lane stubs (S-D39a).

INTEGRATOR: delete this module and every ``pending(...)`` mark once the
queued tasks have landed.

``pending(task_id, probe)`` skips a test while ``probe()`` still raises the
stub's ``NotImplementedError``. ``LANE_PENDING_ENFORCE=1`` turns every mark
off, so a task's validate.sh runs the tests against the stub (they fail)
and against the implementation (they pass).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest


def _still_stubbed(probe: Callable[[], Any]) -> bool:
    try:
        probe()
    except NotImplementedError:
        return True
    except Exception:  # noqa: BLE001 — implemented; bad probe args are fine
        return False
    return False


def pending(task_id: str, probe: Callable[[], Any]) -> pytest.MarkDecorator:
    """A skip mark while ``task_id``'s stub is unimplemented."""
    stubbed = os.environ.get("LANE_PENDING_ENFORCE") != "1" and _still_stubbed(probe)
    return pytest.mark.skipif(stubbed, reason=f"{task_id} not implemented yet")
