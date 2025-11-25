# A class to manage server threads

import threading
import time 
from collections import defaultdict
from umrs.db_store import store_db_model, query_db_model
from umrs.pretty_print import pretty_print_thread_status
from datetime import datetime
import sys
from diskcache import Cache

class ThreadStatus:
    RUNNING = "Running"
    DONE = "Done"



class UMRSTasks:
    def __init__(self, status_store, thread_dict):

        if not isinstance(status_store, Cache) or not isinstance(thread_dict, Cache) :
            sys.exit("Passed variables of UMRSTasks constructor must be Cache objects")
        self.thread_id = 0
        self.thread_list = []
        self.lock = threading.Lock()
        self.thread_dict = thread_dict
        self.status_store = status_store # shared with ZeroMQ. Here we write. ZerMQ reads it
            
    
    def launch_s3_tasks(self, s3_objects, db_model_store, device):
        from server_umrs import process_s3_objects
        def _wait_for_key_in_dict( dict, key, check_interval=0.05):
            while True:
                with self.lock:
                    if key in dict:
                        break
                time.sleep(check_interval)  # wait for 50ms before checking again
            # key is now valid in dictionary

        thread = threading.Thread(target=process_s3_objects, args=(self, s3_objects, db_model_store, device))
        thread.start()
        _wait_for_key_in_dict(self.thread_dict, thread.ident) # wait for th thread status to be updated
        self.thread_list.append(thread)
        

    def update_thread_dict(self, db_model_store, keys_per_bucket):
        current_thread = threading.current_thread()
        array_of_works = []
        for path_to_bucket in keys_per_bucket.keys():
            user, _, table_name, _, _, modality_name, _, data_base = query_db_model(db_model_store, path_to_bucket)
            s3_keys = keys_per_bucket[path_to_bucket]
            work = {
                "user": user,
                "data_base": data_base,
                "table_name": table_name,
                "modality_name": modality_name,
                "bucket": path_to_bucket,
                "nb_objects": len(s3_keys),
            }
            array_of_works.append(work)
        with self.lock:
            thread_id = self.thread_id
            self.thread_id += 1
        self.thread_dict[current_thread.ident] = {
            "thread_id" : thread_id,
            "work": array_of_works,
            "status": ThreadStatus.RUNNING,
        }
    
    def update_status_to_done(self):
        current_thread = threading.current_thread()
        with self.lock:
            thread_id = current_thread.ident
            if thread_id in self.thread_dict:
                entry = self.thread_dict[thread_id]  # get the full dict
                entry["status"] = ThreadStatus.DONE    # modify it
                self.thread_dict[thread_id] = entry    # reassign back to cache
                
    def clean_finished_tasks(self):
        from server_umrs import ENABLE_PRETTY_PRINT
        with self.lock:  
            running_threads = []
            for thread in self.thread_list:
                if not thread.is_alive():
                    thread.join()
                else:
                    running_threads.append(thread)
            print (f"==> Running threads: {len(running_threads)}")
            # No need to cleanup old entries as we use diskcache to cache them on dsk if necessary
            pretty_print_thread_status(self.thread_dict, running_only=True, print_output=ENABLE_PRETTY_PRINT)
            self.thread_list = running_threads

    def waitall(self):
        for thread in self.thread_list:
            thread.join()
            self.thread_list.remove(thread)


    def filter_thread_status_for_user(self, client_id):
        """Build a filtered view of global_thread_status containing only works for a given client_id."""
        filtered = {}
        with self.lock:
            for thread_ident in self.thread_dict:
                info = self.thread_dict[thread_ident] 
                works = info.get("work", [])
                user_works = [w for w in works if w.get("user") == client_id]
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
        # record entry about teh status of the thread for a given used in a status_stire (shared with ZeoMQ)
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

        # get owner doct or empty dict
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


    
    def record_tasks_status_for_users(self, users):
        """Record 'tasks' status summaries for the provided user list into the global status store."""
        if self.status_store is None:
            return
        for user in users:
            filtered = self.filter_thread_status_for_user(user)
            # Include both Running and Done tasks, preserving status column
            summary = pretty_print_thread_status(filtered, running_only=False, print_output=False)
            json_safe = self._json_safe_thread_status(filtered)
            self.record_status(user, "tasks", {"summary": summary, "data": json_safe})

    
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
    