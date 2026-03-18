# A class to manage server threads

import threading
import time
from collections import defaultdict
from umrs.pretty_print import pretty_print_thread_status
from datetime import datetime
import sys
from diskcache import Cache
from umrs.TaskCmdStatus import TaskStatus

class ThreadStatus:
    RUNNING = "Running"
    DONE = "Done"

class UMRSThreads:
    """
    UMRSThreads: A class for managing asynchronous threads executed in server threads.

    This class handles the lifecycle of threads submitted to server threads, including:

    1. Launching threads:
    - Supports processing S3 objects or other work asynchronously in separate threads.
    - Launching thread for waiting an udate of a table in the database
    - Tracks each thread and associates it with the work being done.

    2. Thread bookkeeping:
    - Maintains a thread-safe dictionary of active thread statuses (running/done) using a lock.
    - Assigns a unique thread ID for each thread.
    - Allows filtering of thread status per user/client.

    3. Status tracking and persistence:
    - Uses a disk-backed cache (via `diskcache.Cache`) to store thread and thread status.
    - Supports recording summaries and JSON-safe representations for external systems (e.g., ZeroMQ).

    4. Cleanup and synchronization:
    - Waits for all threads to finish or periodically cleans up finished threads.
    - Ensures that resources (thread cache, status store) can be safely closed and cleared.

    5. Utilities:
    - Pretty-prints running thread statuses.
    - Converts internal thread status data into JSON-safe formats.
    - Provides helper methods for per-user views and status recording.

    Key components:
    - `thread_list`: list of currently running `threading.Thread` objects.
    - `thread_dict`: disk-backed cache storing per-thread info, work, and status.
    - `status_store`: disk-backed cache storing per-user thread summaries, shared with other processes (e.g., ZeroMQ).
    - `lock`: a `threading.Lock` ensuring thread-safe updates to shared resources.

    Dependencies:
    - `diskcache.Cache` for persistent shared state.
    - `threading` and `time` for thread execution and synchronization.
    - Database query helpers (`query_db_model`) and pretty-print helpers (`pretty_print_thread_status`).
    """
    def __init__(self, status_store, thread_dict):

        if not isinstance(status_store, Cache) or not isinstance(thread_dict, Cache) :
            sys.exit("Passed variables of UMRSThreads constructor must be Cache objects")
        
   
        self.thread_id = 0
        
        # for now restart from sctatch enev in case of crash
        # this could be changed later by updatring self.thread_id accordingly
        status_store.clear()
        thread_dict.clear()

        self.thread_list = []
        self.lock = threading.Lock()
        self.thread_dict = thread_dict
        self.status_store = status_store # shared with ZeroMQ. Here we write. ZerMQ reads it
            
    
    def _wait_for_key_in_dict(self, dict, key, check_interval=0.05):
        while True:
            with self.lock:
                if key in dict:
                    break
            time.sleep(check_interval)  # wait for 50ms before checking again
        # key is now valid in dictionary


    def launch_s3_threads(self, s3_objects, buckets_cfg, device,  embedding_backend, key, update_status_callback=None):
        from server_umrs import process_s3_objects
        
        thread = threading.Thread(target=process_s3_objects, args=(self, s3_objects, buckets_cfg, device,  embedding_backend, key, True, update_status_callback))
        
        if update_status_callback is not None:
            update_status_callback(TaskStatus.LAUNCHED)
        
        thread.start()
        
        self._wait_for_key_in_dict(self.thread_dict, thread.ident) # wait for the thread status to be updated
        self.thread_list.append(thread)
        return thread.ident
        

    def launch_export_embeddings(self, user, export_args, buckets_cfg, control_mq, zmq_sender_id, key, update_status_callback=None):
        from umrs.sql import export_embeddings
        
        thread = threading.Thread(target=export_embeddings, 
                                  args=(self, user, export_args, buckets_cfg, control_mq, zmq_sender_id, key, update_status_callback))

        if update_status_callback is not None:
            update_status_callback(TaskStatus.LAUNCHED)
        
        thread.start()

        # `export_args`` is ``"path_to_bucket;format"
        path_to_bucket = export_args.split(";", 1)[0]
        keys_per_bucket = {}
        keys_per_bucket[path_to_bucket] = None

        self.update_thread_dict(buckets_cfg, keys_per_bucket, key, thread_ident=thread.ident)
        self.thread_list.append(thread)
        return thread.ident
    
    def launch_wait_table_modification_on_db(self, path_to_bucket, buckets_cfg, user, data_base, table_name, timeout, control_mq, zmq_sender_id, key, update_status_callback=None):
        from umrs.sql import wait_table_modification_on_db
        from collections import defaultdict
                
        thread = threading.Thread(target=wait_table_modification_on_db, args=(self, user, key, data_base, table_name, timeout, control_mq, zmq_sender_id, update_status_callback))
       
        if update_status_callback is not None:
            update_status_callback(TaskStatus.LAUNCHED)
        
        thread.start()
        keys_per_bucket = {}
        keys_per_bucket[path_to_bucket] = None

        self.update_thread_dict(buckets_cfg, keys_per_bucket, key, thread_ident=thread.ident)

        self.thread_list.append(thread)
        return thread.ident
        
    def launch_check_objects(self, buckets_cfg, key, verbose, log_output, update_status_callback=None) : 
        from server_umrs import check_objects
     
        thread = threading.Thread(target=check_objects, args=(self, buckets_cfg, key, verbose, log_output, update_status_callback))
       
        if update_status_callback is not None:
            update_status_callback(TaskStatus.LAUNCHED)
        
        thread.start()
        
        keys_per_bucket = {}
        for path_to_bucket, entry in buckets_cfg.items():
            keys_per_bucket[path_to_bucket] = entry.num_objects
        
        self.update_thread_dict(buckets_cfg, keys_per_bucket, key, thread_ident=thread.ident)

        self.thread_list.append(thread)
        return thread.ident
        
    def launch_update_objects(self, client_id, buckets_cfg, device, embedding_backend, key, verbose, log_output, update_status_callback=None) :
        from server_umrs import update_objects
  

        thread = threading.Thread(target=update_objects, args=(self, client_id, buckets_cfg, device, embedding_backend, key, verbose, log_output, update_status_callback))
        
        if update_status_callback is not None:
            update_status_callback(TaskStatus.LAUNCHED)
        
        thread.start()
        
        keys_per_bucket = {}
        for path_to_bucket, entry in buckets_cfg.items():
            keys_per_bucket[path_to_bucket] = entry.num_objects
        
        self.update_thread_dict(buckets_cfg, keys_per_bucket, key, thread_ident=thread.ident)

        self.thread_list.append(thread)
        return thread.ident
        

    def launch_recompute_all_objects(self, client_id,  buckets_cfg, device, embedding_backend, key, log_output, update_status_callback=None):
        from server_umrs import recompute_all_objects
  
        thread = threading.Thread(target=recompute_all_objects, args=(self, client_id, buckets_cfg, device, embedding_backend, key, log_output, update_status_callback))
       
        if update_status_callback is not None:
            update_status_callback(TaskStatus.LAUNCHED)
        
        thread.start()
        
        keys_per_bucket = {}
        for path_to_bucket, entry in buckets_cfg.items():
            keys_per_bucket[path_to_bucket] = entry.num_objects
        
        self.update_thread_dict(buckets_cfg, keys_per_bucket, key, thread_ident=thread.ident)

        self.thread_list.append(thread)
        return thread.ident
        

    def update_thread_dict(self, buckets_cfg, keys_per_bucket, key, thread_ident = "self"):
        array_of_works = []
        for path_to_bucket in keys_per_bucket.keys():
            selected_fields = ["s3_username", "table_name","modality_name", "db_name"]
            user, table_name,  modality_name, db_name = buckets_cfg.get(path_to_bucket, filter=selected_fields)
            s3_keys = keys_per_bucket[path_to_bucket]
            work = {
                "user": user,
                "data_base": db_name,
                "table_name": table_name,
                "modality_name": modality_name,
                "bucket": path_to_bucket,
                "work_type": key,
                "nb_objects": s3_keys if isinstance(s3_keys, int) else (len(s3_keys) if s3_keys is not None else "N/A"),
            }
            array_of_works.append(work)
        with self.lock:
            thread_id = self.thread_id
            self.thread_id += 1
        if thread_ident == "self" : #we are inside a thread thread, we need to get its ident 
            current_thread = threading.current_thread()
            thread_ident = current_thread.ident
        with self.lock:
            self.thread_dict[thread_ident] = {
                "thread_id" : thread_id,
                "work": array_of_works,
                "status": ThreadStatus.RUNNING,
            }

        # print(f"{thread_ident=}\n\t{thread_id=}\n\t{ThreadStatus.RUNNING}" )
        self.record_status_for_users_by_key([user], key)
    
    def is_worker_thread(self):
        current_thread = threading.current_thread()
        thread_ident = current_thread.ident
        return thread_ident in self.thread_dict
    
    def remove_from_running_list(self):
        current_thread = threading.current_thread()
        with self.lock:
            if current_thread in self.thread_list:
                self.thread_list.remove(current_thread)

    def update_status_to_done(self):
        current_thread = threading.current_thread()
        new_key = None
        with self.lock:
            thread_ident = current_thread.ident
            if thread_ident in self.thread_dict:
                entry = self.thread_dict.pop(thread_ident)  # remove  get the full dict
                entry["status"] = ThreadStatus.DONE    # modify it
                new_key = f"{thread_ident}-{entry['thread_id']}" # done thread keys are prepended with their ID
                self.thread_dict[new_key] = entry    # reassign back to cache
            else:
                print(f"Error: cannot find '{thread_ident}' entry for {self.thread_dict=}")
    
    def clean_finished_threads(self):
        from server_umrs import ENABLE_PRETTY_PRINT
        with self.lock:
            running_threads = []
            finished_threads = []
            for thread in self.thread_list:
                print(f"{thread.ident=}")
                if not thread.is_alive():
                    finished_threads.append(thread.ident)
                    thread.join()
                else:
                    running_threads.append(thread)
            print (f"==> Running threads: {len(running_threads)}")
            # No need to cleanup old entries as we use diskcache to cache them on dsk if necessary
            pretty_print_thread_status(self.thread_dict, running_only=True, print_output=ENABLE_PRETTY_PRINT)
            self.thread_list = running_threads
            return finished_threads

    def waitall(self):
        finish_threads = []
        for thread in self.thread_list:
            finish_threads.append(thread.ident)
            thread.join()
            self.thread_list.remove(thread)
        return finish_threads


    def filter_thread_status_for_user(self, client_id, key):
        """Build a filtered view of global_thread_status containing only works for a given client_id."""
        filtered = {}
        with self.lock:
            for thread_ident in self.thread_dict:
                info = self.thread_dict[thread_ident]
                works = info.get("work", [])
                user_works = [w for w in works if w.get("user") == client_id and w.get("work_type") == key]
                if user_works:
                    # Shallow copy with filtered works
                    filtered[thread_ident] = {
                        "thread_id": info.get("thread_id"),
                        "status": info.get("status"),
                        "work": user_works,
                    }
        return filtered



    def get_status_store(self):
        return self.status_store

    def record_status(self, client_id, key, payload):
        # record entry about the status of the thread for a given used in a status_stire (shared with ZeoMQ)
        def _sanitize_status_data(data):
            if isinstance(data, dict):
                return {k: _sanitize_status_data(v) for k, v in data.items()}
            if isinstance(data, (list, tuple)):
                return [_sanitize_status_data(item) for item in data]
            if hasattr(data, "model_name"):
                return getattr(data, "model_name")
            if isinstance(data, (str, int, float, bool)) or data is None:
                return data
            return str(data)


        if self.status_store is None:
            return

        owner = client_id or "system"
        entry = {
            "summary": payload.get("summary", ""),
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
        if "data" in payload:
            entry["data"] = _sanitize_status_data(payload["data"])
        if "requires_update" in payload:
            entry["requires_update"] = payload["requires_update"]
        if "error" in payload:
            entry["error"] = payload["error"]

        # print(f"[{owner}]: {key} -> {entry}")

        # get owner dict or empty dict
        owner_dict = self.status_store.get(owner, {})
        owner_dict[key] = entry
        self.status_store[owner] = owner_dict


    def _json_safe_thread_status(self, ts_dict):
        """Convert thread-keyed dict into a JSON-serializable list of thread info dicts, avoiding key collisions."""
        safe = []
        for _, info in ts_dict.items():
            safe.append({
                "thread_id": info.get("thread_id"),
                "status": info.get("status"),
                "work": info.get("work", []),
            })
        return safe


    
    def record_status_for_users_by_key(self, users, key):
        """Record 'threads' status summaries for the provided user list into the global status store."""
        if self.status_store is None:
            return
        for user in users:
            filtered = self.filter_thread_status_for_user(user, key)
            # Include both Running and Done threads, preserving status column
            summary = pretty_print_thread_status(filtered, running_only=False, print_output=False)
            # print(f"{summary=}")
            json_safe = self._json_safe_thread_status(filtered)
            self.record_status(user, key, {"summary": summary, "data": json_safe})


    def close(self):
        try:
            # if using diskcache.Cache
            self.thread_dict.close()   # closes the cache
            self.status_store.close()   # closes the cache
            # optionally clear on disk
            self.thread_dict.clear()
            self.status_store.clear()
            print("thread_dict and status store dictionnary deleted")
        except Exception as e:
            print(f"Cannot close thread_dict/status_store dictionary properly: {e}")
