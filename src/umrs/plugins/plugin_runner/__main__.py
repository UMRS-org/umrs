"""``python -m umrs.plugins.plugin_runner`` support."""

import sys

from .entrypoint import main

sys.exit(main())
