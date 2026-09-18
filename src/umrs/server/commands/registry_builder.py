"""registry_builder.py — assembles the CommandRegistry for the UMRS server.

This is the single place where every TaskCmd is mapped to its handler.
Adding a new command requires:
  1. Creating a handler class in its own module.
  2. Adding one ``r.register(...)`` line in ``build_registry()`` below.
  3. Nothing else — the event loop in server_umrs.py is untouched.
"""

from __future__ import annotations

from umrs.core.TaskCmdStatus import TaskCmd

from .enroll_user import EnrollUserHandler
from .export_embeddings import ExportEmbeddingsHandler
from .file_upload import FileHandler
from .job import JobCancelHandler, JobListHandler, JobSubmitHandler
from .maintenance import CheckHandler, RecomputeHandler, ReloadHandler, UpdateHandler
from .model import ModelDownloadHandler, ModelListHandler, ModelRegisterHandler
from .process_objects import ProcessObjectsBatchHandler
from .purge_buckets import PurgeBucketsHandler
from .registry import CommandRegistry
from .upload_workflow import UploadWorkflowHandler
from .wait_output import WaitOutputHandler
from .workflow_commands import (
    ListWorkflowsHandler,
    RunWorkflowHandler,
    WorkflowCancelHandler,
    WorkflowErrorsHandler,
    WorkflowResultsHandler,
    WorkflowStatusHandler,
)


def build_registry() -> CommandRegistry:
    """Return a fully wired CommandRegistry.

    Called once at server startup.  Handlers are stateless objects —
    they receive the ``HandlerContext`` at dispatch time, not here.
    """
    r = CommandRegistry()

    # --- Ingestion --------------------------------------------------------
    r.register(TaskCmd.PROCESS_OBJECTS_BATCH, ProcessObjectsBatchHandler())

    # --- Maintenance (admin-only: RECOMPUTE, UPDATE, CHECK, RELOAD) -------
    r.register(TaskCmd.RECOMPUTE, RecomputeHandler())
    r.register(TaskCmd.UPDATE, UpdateHandler())
    r.register(TaskCmd.CHECK, CheckHandler())
    r.register(TaskCmd.RELOAD, ReloadHandler())

    # --- File management --------------------------------------------------
    r.register(TaskCmd.FILE, FileHandler())
    r.register(TaskCmd.EXPORT_EMBEDDINGS, ExportEmbeddingsHandler())
    r.register(TaskCmd.WAIT_OUTPUT, WaitOutputHandler())

    # --- User management (admin-only) -------------------------------------
    r.register(TaskCmd.ENROLL_USER, EnrollUserHandler())

    # --- Model registry ---------------------------------------------------
    r.register(TaskCmd.MODEL_REGISTER, ModelRegisterHandler())
    r.register(TaskCmd.MODEL_LIST, ModelListHandler())
    r.register(TaskCmd.MODEL_DOWNLOAD, ModelDownloadHandler())

    # --- Job tracking -----------------------------------------------------
    r.register(TaskCmd.JOB_SUBMIT, JobSubmitHandler())
    r.register(TaskCmd.JOB_CANCEL, JobCancelHandler())
    r.register(TaskCmd.JOB_LIST, JobListHandler())

    # --- Bucket management ------------------------------------------------
    r.register(TaskCmd.PURGE_BUCKETS, PurgeBucketsHandler())

    # --- Workflows --------------------------------------------------------
    r.register(TaskCmd.UPLOAD_WORKFLOW, UploadWorkflowHandler())
    r.register(TaskCmd.LIST_WORKFLOWS, ListWorkflowsHandler())
    r.register(TaskCmd.RUN_WORKFLOW, RunWorkflowHandler())
    r.register(TaskCmd.WORKFLOW_STATUS, WorkflowStatusHandler())
    r.register(TaskCmd.WORKFLOW_RESULTS, WorkflowResultsHandler())
    r.register(TaskCmd.WORKFLOW_CANCEL, WorkflowCancelHandler())
    r.register(TaskCmd.WORKFLOW_ERRORS, WorkflowErrorsHandler())

    return r
