"""
UMRS REST API module.

This module provides FastAPI routers for UMRS functionality.
The main server (server_umrs.py) can mount these routers alongside
the existing ZeroMQ-based messaging.

Note: Context/memory routers excluded from selective merge.
"""

from .endpoints import router as endpoints_router
from .files import router as files_router
from .jobs import router as jobs_router
from .models import router as models_router
from .models import policy_router
from .pipelines import router as pipelines_router  # api/pipelines.py router (not the old umrs.pipelines package)

__all__ = [
    "endpoints_router",
    "files_router",
    "jobs_router",
    "models_router",
    "policy_router",
    "pipelines_router",
]
