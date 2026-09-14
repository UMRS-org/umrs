"""Persistent monitor preferences.

Preferences are persisted to ``<umrs_cfg_dir>/monitor.json`` so the user's
last active tab, filters, sort order, and interval survive across
invocations.  The path uses
:func:`umrs.shared.config.get_resolved_umrs_cfg_dir` to honour
``UMRS_CONFIG`` / XDG overrides (avoids the config-shadowing class of bug
flagged in the branch-separation instructions).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from umrs.shared.config import get_resolved_umrs_cfg_dir


@dataclass
class MonitorPrefs:
    """User preferences for the monitor TUI."""

    active_tab: str = "jobs"
    status_filter: str | None = None
    project_filter: str | None = None
    job_type_filter: str | None = None
    model_scope_filter: str | None = None
    workflow_scope_filter: str | None = None
    interval: float = 3.0
    compact: bool = True
    sort_keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | None = None) -> MonitorPrefs:
        """Load prefs from disk, returning defaults if missing/invalid."""
        p = Path(path) if path else cls._default_path()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return cls()

    def save(self, path: str | None = None) -> None:
        """Persist prefs to disk."""
        p = Path(path) if path else self._default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def _default_path() -> Path:
        cfg_dir = get_resolved_umrs_cfg_dir()
        return Path(cfg_dir) / "monitor.json"
