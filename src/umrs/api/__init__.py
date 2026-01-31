"""
UMRS REST API module.

This module provides FastAPI routers for UMRS functionality.
The main server (server_umrs.py) can mount these routers alongside
the existing ZeroMQ-based messaging.
"""

from .endpoints import router as endpoints_router

__all__ = ["endpoints_router"]
