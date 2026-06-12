# here are values and methods to assocaite tasks with keys to be stored in dictionnaries and to get status
class TaskCmd:
    PROCESS_OBJECTS = "PROCESS_OBJECTS"
    WAIT_OUTPUT = "WAIT_OUTPUT"
    RECOMPUTE = "RECOMPUTE"
    UPDATE = "UPDATE"
    CHECK = "CHECK"
    RELOAD = "RELOAD"
    FILE = "FILE"
    HAND_CHECK = "HAND_CHECK"
    CREATE_DATA_QUEUE = "CREATE_DATA_QUEUE"
    STATUS = "STATUS"
    EXPORT_EMBEDDINGS = "EXPORT_EMBEDDINGS"
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
    # Admin enrollment
    ENROLL_USER = "ENROLL_USER"  # admin-only: register a new user
    # Bootstrap context
    GET_BOOTSTRAP_CONTEXT = "GET_BOOTSTRAP_CONTEXT"  # return versioned user context
    # Bucket management
    PURGE_BUCKETS = "PURGE_BUCKETS"  # delete user-owned extra S3 buckets

    @classmethod
    def all_cmd(cls):
        # Collect all class attributes that are strings (ignore methods and magic attributes)
        keys = [v for k, v in vars(cls).items() if isinstance(v, str)]
        # Optionally add extra keys that are not class constants
        keys.append("all")
        return tuple(keys)

    @classmethod
    def status_cmd(cls):
        return (
            cls.PROCESS_OBJECTS.lower(),
            cls.WAIT_OUTPUT.lower(),
            cls.RECOMPUTE.lower(),
            cls.UPDATE.lower(),
            cls.CHECK.lower(),
            cls.RELOAD.lower(),
            "all",
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
