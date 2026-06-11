import threading

from umrs.core.UMRSTask import UMRSTask
from umrs.core.UMRSThreads import UMRSThreads


class TaskManager:
    def __init__(self, status_store, thread_dict, db_path, pool=None):
        self.active_threads = UMRSThreads(status_store, thread_dict, pool=pool)
        self.tasks_status = UMRSTask(db_path)
        self.task_to_thread = {}  # UMRSTask_id -> thread_ident
        self.thread_to_task = {}  # thread_ident -> UMRSTask_id
        self.lock = threading.Lock()

    def associate_thread_with_task(self, thread_ident, task_id):
        """Associate a thread with a task"""
        with self.lock:
            self.task_to_thread[task_id] = thread_ident
            self.thread_to_task[thread_ident] = task_id

    def get_thread_from_task(self, task_id):
        """Return the thread ID handling this task."""
        with self.lock:
            return self.task_to_thread.get(task_id)

    def get_task_from_thread(self, thread_ident):
        """Return the task ID handled by this thread."""
        with self.lock:
            return self.thread_to_task.get(thread_ident)

    def update_status_by_threads_ident_list(self, threads_ident_list, status):
        for thread_ident in threads_ident_list:
            task_id = self.get_task_from_thread(thread_ident)
            self.tasks_status.update_status(task_id, status)
