"""``python -m umrs.plugins.plugin_runner`` support."""
from .entrypoint import main
import sys

sys.exit(main())
