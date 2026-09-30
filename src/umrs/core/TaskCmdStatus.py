# here are values and methods to assocaite tasks with keys to be stored in dictionnaries and to get status
class TaskCmd:
    PROCESS_OBJECTS = "PROCESS_OBJECTS"
    PROCESS_OBJECTS_BATCH = "PROCESS_OBJECTS_BATCH"  # internal: batched multi-client
    WAIT_OUTPUT = "WAIT_OUTPUT"
    RECOMPUTE = "RECOMPUTE"
    UPDATE = "UPDATE"
    CHECK = "CHECK"
    RELOAD = "RELOAD"
    FILE = "FILE"
    # HANDSHAKE and HAND_CHECK are redundant readiness pings that both expect a
    # "READY!" reply and map to _parse_handshake in MessageQueue._CONTROL_PARSERS.
    # HANDSHAKE is sent by the in-repo client (MessageQueue.is_server_ready);
    # HAND_CHECK is sent by the r4 SDK transport (r4/src/r4/umrs_transport._ping).
    # Keep BOTH: r4 and umrs are separate packages on the ZMQ wire and may deploy
    # at different times, so neither alias can be dropped without lockstep release.
    HANDSHAKE = "HANDSHAKE"
    HAND_CHECK = "HAND_CHECK"
    CREATE_DATA_QUEUE = "CREATE_DATA_QUEUE"
    STATUS = "STATUS"
    STATUS_ALL = "STATUS_ALL"  # server-wide snapshot: all clients in one response
    EXPORT_EMBEDDINGS = "EXPORT_EMBEDDINGS"
    KNN = "KNN"
    # Model registry commands
    MODEL_REGISTER = "MODEL_REGISTER"
    MODEL_LIST = "MODEL_LIST"
    MODEL_DOWNLOAD = "MODEL_DOWNLOAD"  # server-side download (background thread)
    # Job tracking commands
    JOB_SUBMIT = "JOB_SUBMIT"  # durable submission with retry config
    JOB_CANCEL = "JOB_CANCEL"  # cancel a submitted job by ID
    JOB_LIST = "JOB_LIST"  # list jobs with optional filters
    # Workflow upload command
    UPLOAD_WORKFLOW = "UPLOAD_WORKFLOW"  # upload a workflow zip to server
    # Workflow registry commands
    LIST_WORKFLOWS = "LIST_WORKFLOWS"  # list registered workflows
    RUN_WORKFLOW = "RUN_WORKFLOW"  # trigger execution of a registered workflow
    # Workflow execution commands (unified under `workflow` subcommand)
    WORKFLOW_STATUS = "WORKFLOW_STATUS"  # show execution status for a workflow run
    WORKFLOW_RESULTS = "WORKFLOW_RESULTS"  # fetch results for a completed workflow run
    WORKFLOW_CANCEL = "WORKFLOW_CANCEL"  # cancel a running workflow execution
    WORKFLOW_ERRORS = "WORKFLOW_ERRORS"  # show error logs for a failed workflow execution
    WORKFLOW_LOGS = "WORKFLOW_LOGS"  # stream execution logs for a workflow run
    # Admin enrollment
    ENROLL_USER = "ENROLL_USER"  # admin-only: register a new user
    # Bootstrap context
    GET_BOOTSTRAP_CONTEXT = "GET_BOOTSTRAP_CONTEXT"  # return versioned user context
    # Bucket management
    PURGE_BUCKETS = "PURGE_BUCKETS"  # delete user-owned extra S3 buckets
    # Meta-key: request all status entries
    ALL = "all"

    @classmethod
    def all_cmd(cls):
        return tuple(v for k, v in vars(cls).items() if not k.startswith("__") and isinstance(v, str))

    @classmethod
    def status_cmd(cls):
        """Lowercase CLI choices for the status command (argparse-facing)."""
        return (
            cls.PROCESS_OBJECTS.lower(),
            cls.WAIT_OUTPUT.lower(),
            cls.RECOMPUTE.lower(),
            cls.UPDATE.lower(),
            cls.CHECK.lower(),
            cls.RELOAD.lower(),
            cls.ALL,  # already "all"
        )


class TaskStatus:
    SUBMITTED = "Submitted"
    RUNNING = "Running"
    COMPLETED = "Completed"
    NO_SUCH_BUCKET = "No Such Bucket"
    UNKNOWN = "Unknown"
    ABORTED = "Aborted"
    FAILED = "Failed"
    OVERRIDDEN = "Overridden"
    ERROR = "Error"
    LAUNCHED = "Launched"
    INITIALIZING = "Initializing"

    @classmethod
    def all_status(cls):
        return [
            v for k, v in vars(cls).items()
            if not k.startswith("__") and isinstance(v, str)
        ]
