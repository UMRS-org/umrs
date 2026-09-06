"""
UMRS REST API module.

This module provides FastAPI routers for UMRS functionality.
The main server (server_umrs.py) can mount these routers alongside
the existing ZeroMQ-based messaging.

Note: Context/memory routers excluded from selective merge.
"""

from .a2l import router as a2l_router
from .a2l_ledger import router as a2l_ledger_router
from .artifact_routes import router as artifact_router
from .assets import router as assets_router
from .bundle_restore import router as bundle_restore_router
from .catalog import router as catalog_router
from .chat import router as chat_router
from .companion_sessions import router as companion_sessions_router
from .egress_audit import router as egress_audit_router
from .endpoints import router as endpoints_router
from .evidence import router as evidence_router
from .experiment_registry import router as experiment_registry_router
from .files import router as files_router
from .grants import router as grants_router
from .intake_uploads import router as intake_uploads_router
from .jobs import router as jobs_router
from .knowledge import router as knowledge_router
from .knowledge_graph import router as knowledge_graph_router
from .knowledge_graphs import router as knowledge_graphs_router
from .knowledge_lineage import router as knowledge_lineage_router
from .knowledge_proposals import router as knowledge_proposals_router
from .knowledge_query import router as knowledge_query_router
from .knowledge_v2 import router as knowledge_v2_router
from .llm import router as llm_router
from .loop import router as loops_router
from .mcp import router as mcp_router
from .models import policy_router
from .models import router as models_router
from .ops import router as ops_router
from .pipelines import (
    router as pipelines_router,  # api/pipelines.py router (not the old umrs.pipelines package)
)
from .provenance import router as provenance_router
from .rag import router as rag_router
from .reflective_memory import router as reflective_memory_router
from .research_sessions import router as research_sessions_router
from .research_sessions import set_session_store
from .scientific_memory import router as scientific_memory_router
from .skills import router as skills_router
from .traces import router as traces_router
from .usage import router as usage_router
from .verification import router as verification_router
from .vision import router as vision_router
from .workflows import router as workflows_router
from .zmq_sessions import router as zmq_sessions_router

__all__ = [
    "a2l_ledger_router",
    "a2l_router",
    "artifact_router",
    "assets_router",
    "bundle_restore_router",
    "catalog_router",
    "chat_router",
    "companion_sessions_router",
    "egress_audit_router",
    "endpoints_router",
    "evidence_router",
    "experiment_registry_router",
    "files_router",
    "grants_router",
    "intake_uploads_router",
    "jobs_router",
    "knowledge_graph_router",
    "knowledge_graphs_router",
    "knowledge_lineage_router",
    "knowledge_proposals_router",
    "knowledge_query_router",
    "knowledge_router",
    "knowledge_v2_router",
    "llm_router",
    "loops_router",
    "mcp_router",
    "models_router",
    "ops_router",
    "pipelines_router",
    "policy_router",
    "provenance_router",
    "rag_router",
    "reflective_memory_router",
    "research_sessions_router",
    "scientific_memory_router",
    "set_session_store",
    "skills_router",
    "traces_router",
    "usage_router",
    "verification_router",
    "vision_router",
    "workflows_router",
    "zmq_sessions_router",
]
