"""umrs-r4 test configuration: the core suite's hermetic baseline.

Mirrors ``umrs/tests/conftest.py`` (H-3.6 hermetic HOME, the explicit mvp
auth baseline); this directory is outside that conftest's reach.
"""

import atexit
import os
import shutil
import tempfile

# One pytest session may also load umrs/tests/conftest.py (both directories
# on one command line); its scratch HOME then stays, never a second one.
_INHERITED = os.path.basename(os.environ.get("HOME", "")).startswith("umrs-tests-home-")
TEST_HOME = os.environ["HOME"] if _INHERITED else tempfile.mkdtemp(prefix="umrs-r4-tests-home-")
os.environ["HOME"] = TEST_HOME
for _xdg_var, _xdg_sub in (
    ("XDG_CONFIG_HOME", ".config"),
    ("XDG_DATA_HOME", ".local/share"),
    ("XDG_CACHE_HOME", ".cache"),
    ("XDG_STATE_HOME", ".local/state"),
):
    os.environ[_xdg_var] = os.path.join(TEST_HOME, _xdg_sub)
for _path_var in ("UMRS_CONFIG", "RIVAULT_OUTPUT_DIR", "SLURM_JOB_INFO_DIR"):
    os.environ.pop(_path_var, None)
os.environ["UMRS_TRANSPORT_KEY_DIR"] = os.path.join(TEST_HOME, "transport-keys")
if not _INHERITED:
    atexit.register(shutil.rmtree, TEST_HOME, True)

# The suites were written against the historic mvp baseline; strict cases opt
# in per test with monkeypatch.setenv("UMRS_AUTH_MODE", "strict").
os.environ.setdefault("UMRS_AUTH_MODE", "mvp")
