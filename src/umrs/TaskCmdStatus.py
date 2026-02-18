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
            "all"
        )
    
class TaskStatus:
    SUBMITTED = "Submitted"
    RUNNING = "Running"
    COMPLETED = "Completed"
    NO_SUCH_BUCKET = "No Such Bucket"
    UNKNOWN = "Unknown"
    ABORTED = "Aborted"
    OVERRIDDEN = "Overridden"
    ERROR = "Error"