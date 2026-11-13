"""``import umrs_protocol`` loads no submodule (U2.0 spec §7).

The package ``__init__`` is a docstring only. An eager import would load the
7,000-line gate (and every other module) into each process that only needs,
say, the handshake, and would make a missing module fail at package import.
Checked in an isolated interpreter (``python -I``: no site, no user path,
no PYTHONPATH), with this checkout's ``src`` put on ``sys.path`` explicitly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

_PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
import umrs_protocol
loaded = sorted(m for m in sys.modules if m.startswith("umrs_protocol."))
print(json.dumps({"file": umrs_protocol.__file__, "loaded": loaded}))
"""


def _probe() -> dict[str, object]:
    out = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE, str(SRC)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(out.stdout)


def test_import_loads_no_submodule() -> None:
    result = _probe()
    assert Path(str(result["file"])).resolve() == SRC / "umrs_protocol" / "__init__.py"
    assert result["loaded"] == []


def test_init_defines_no_version() -> None:
    """The pyproject is the one version source (importlib.metadata)."""
    import umrs_protocol

    assert not hasattr(umrs_protocol, "__version__")
