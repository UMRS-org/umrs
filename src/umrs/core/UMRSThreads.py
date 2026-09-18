# Manages server background tasks submitted to a ThreadPoolExecutor

import logging
import sys
import threading
import time
from datetime import datetime

from diskcache import Cache

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.core.thread_pool import ThreadPoolManager
from umrs.shared.pretty_print import pretty_print_thread_status

logger = logging.getLogger(__name__)


class ThreadStatus:
    RUNNING = "Running"
    DONE = "Done"


class UMRSThreads:
    """
    Manages background tasks submitted by the server command handlers.

    All tasks are submitted to a bounded ThreadPoolExecutor via _submit_to_pool();
    thread_list holds the resulting Future objects (no raw threading.Thread).

    Responsibilities:
    1. Task submission: _submit_to_pool() submits a callable to the pool, attaches
       a thread_ident to the Future for identity tracking, and appends it to thread_list.

    2. Lifecycle tracking: thread_dict (disk-backed Cache) stores per-task metadata
       (command, status, user, timestamps). Entries survive the task lifetime for
       client STATUS queries.

    3. Status store: status_store (disk-backed Cache) is written here and read by
       the ZeroMQ layer to answer STATUS commands without hitting the pool.

    4. Cleanup: clean_finished_threads() is called each iteration of the server
       event loop; it removes done Futures from thread_list and logs any exceptions.

    5. Per-user views: filter_thread_status_for_user() scopes status data to a
       single client_id for multi-tenant isolation.

    Key attributes:
    - thread_list  : list[Future] — active (not yet done) submitted tasks.
    - thread_dict  : diskcache.Cache — per-task metadata keyed by thread_ident.
    - status_store : diskcache.Cache — per-user summaries shared with ZeroMQ.
    - lock         : threading.Lock — guards thread_list and thread_dict updates.
    - pool         : ThreadPoolManager — bounded executor (max_workers configurable).
    """

    def __init__(self, status_store, thread_dict, pool=None):

        if not isinstance(status_store, Cache) or not isinstance(thread_dict, Cache):
            sys.exit(
                "Passed variables of UMRSThreads constructor must be Cache objects"
            )

        self.thread_id = 0

        # for now restart from sctatch enev in case of crash
        # this could be changed later by updatring self.thread_id accordingly
        status_store.clear()
        thread_dict.clear()

        self.thread_list = []
        self.lock = threading.Lock()
        self.thread_dict = thread_dict
        self.status_store = (
            status_store  # shared with ZeroMQ. Here we write. ZerMQ reads it
        )

        # Thread pool for bounded concurrency (falls back to unbounded threads if None)
        if pool is None:
            self.pool = ThreadPoolManager()
        else:
            self.pool = pool

    def _now_iso(self):
        return datetime.utcnow().isoformat() + "Z"

    def _wait_for_key_in_dict(self, cache, key, check_interval=0.05, timeout=60):
        start = time.time()
        while True:
            with self.lock:
                if key in cache:
                    break
            if time.time() - start > timeout:
                raise RuntimeError(f"Timeout ({timeout}s) waiting for thread key {key}")
            time.sleep(check_interval)  # wait for 50ms before checking again
        # key is now valid in dictionary

    def _submit_to_pool(self, target, args, key, update_status_callback=None):
        """Submit *target* to the thread pool and do standard bookkeeping.

        Returns the ``thread_ident`` of the worker thread (available after
        the task starts executing inside the pool).

        Raises ``ServerOverloadedError`` if the pool is saturated.
        """
        if update_status_callback is not None:
            update_status_callback(TaskStatus.LAUNCHED)

        # We need the worker thread's ident for the existing bookkeeping.
        # Wrap *target* so the first thing the worker does is publish its
        # ident into a threading.Event we can wait on.
        ident_holder: dict = {}
        ident_ready = threading.Event()

        def _wrapper(*a, **kw):
            ident_holder["ident"] = threading.current_thread().ident
            ident_ready.set()
            return target(*a, **kw)

        future = self.pool.submit(_wrapper, *args)

        # Wait for the worker to start so we know its ident.
        if not ident_ready.wait(timeout=60):
            raise RuntimeError(f"Pool worker did not start within 60 s for '{key}'")

        thread_ident = ident_holder["ident"]
        future.ident = (
            thread_ident  # mirror Thread.ident on the Future so thread_list items
        )
        # and ident-based lookups share the same key

        # Keep a reference so waitall / clean_finished_threads still works.
        with self.lock:
            self.thread_list.append(future)

        return thread_ident, future

    def launch_s3_threads(
        self,
        s3_objects,
        buckets_cfg,
        device,
        embedding_backend,
        key,
        update_status_callback=None,
        client_id=None,
    ):
        return self.launch_s3_threads_batch(
            [(client_id, s3_objects)],
            buckets_cfg,
            device,
            embedding_backend,
            key,
            client_callbacks={client_id: update_status_callback} if update_status_callback else {},
        )

    def launch_s3_threads_batch(
        self,
        client_entries,
        buckets_cfg,
        device,
        embedding_backend,
        key,
        client_callbacks=None,
    ):
        """Submit one pool thread per model group across all client entries.

        ``prepare_model_groups`` resolves bucket configs and groups items by
        model synchronously (before pool submission), so each submitted thread
        is independent and can run in parallel — Rivault nodes on different
        GPUs, or simply different local models.
        """
        from umrs.server.processing import prepare_model_groups, process_model_group

        if client_callbacks:
            for cb in client_callbacks.values():
                cb(TaskStatus.LAUNCHED)

        groups = prepare_model_groups(
            self, client_entries, buckets_cfg, device, key, client_callbacks,
        )

        logger.info(
            "[PROCESS] %d model group(s) to process for key='%s'",
            len(groups), key,
        )

        for items, group_callbacks, group_users in groups:
            model_name = getattr(items[0].get("model"), "model_name", str(items[0].get("model")))
            logger.info(
                "[PROCESS] Submitting pool thread — model='%s' buckets=%d clients=%s",
                model_name,
                len(items),
                sorted(group_callbacks.keys()),
            )
            thread_ident, _ = self._submit_to_pool(
                process_model_group,
                (self, items, group_callbacks, buckets_cfg, embedding_backend, key, group_users),
                key,
                update_status_callback=None,
            )
            try:
                self._wait_for_key_in_dict(self.thread_dict, thread_ident)
            except RuntimeError as exc:
                logger.info(
                    "WARNING: Thread %s did not register in time for '%s': %s",
                    thread_ident, key, exc,
                )

    def launch_export_embeddings(
        self,
        user,
        export_args,
        buckets_cfg,
        control_mq,
        zmq_sender_id,
        key,
        update_status_callback=None,
    ):
        from umrs.server.sql import export_embeddings

        thread_ident, future = self._submit_to_pool(
            export_embeddings,
            (
                self,
                user,
                export_args,
                buckets_cfg,
                control_mq,
                zmq_sender_id,
                key,
                update_status_callback,
            ),
            key,
            update_status_callback=update_status_callback,
        )

        # `export_args`` is ``"path_to_bucket;format"
        path_to_bucket = export_args.split(";", 1)[0]
        keys_per_bucket = {}
        keys_per_bucket[path_to_bucket] = None

        self.update_thread_dict(
            buckets_cfg, keys_per_bucket, key, thread_ident=thread_ident
        )
        return thread_ident

    def launch_wait_table_modification_on_db(
        self,
        path_to_bucket,
        buckets_cfg,
        user,
        data_base,
        table_name,
        timeout,
        control_mq,
        zmq_sender_id,
        key,
        update_status_callback=None,
    ):
        from umrs.server.sql import wait_table_modification_on_db

        thread_ident, future = self._submit_to_pool(
            wait_table_modification_on_db,
            (
                self,
                user,
                key,
                data_base,
                table_name,
                timeout,
                control_mq,
                zmq_sender_id,
                update_status_callback,
            ),
            key,
            update_status_callback=update_status_callback,
        )

        keys_per_bucket = {}
        keys_per_bucket[path_to_bucket] = None

        self.update_thread_dict(
            buckets_cfg, keys_per_bucket, key, thread_ident=thread_ident
        )

        return thread_ident

    def launch_check_objects(
        self,
        client_id,
        buckets_cfg,
        key,
        verbose,
        log_output,
        update_status_callback=None,
    ):
        from umrs.server.maintenance import check_objects

        thread_ident, future = self._submit_to_pool(
            check_objects,
            (
                self,
                client_id,
                buckets_cfg,
                key,
                verbose,
                log_output,
                update_status_callback,
            ),
            key,
            update_status_callback=update_status_callback,
        )

        keys_per_bucket = {}
        for path_to_bucket, entry in buckets_cfg.items():
            if client_id is None or entry.s3_username == client_id:
                keys_per_bucket[path_to_bucket] = entry.num_objects

        self.update_thread_dict(
            buckets_cfg, keys_per_bucket, key, thread_ident=thread_ident
        )

        return thread_ident

    def launch_update_objects(
        self,
        client_id,
        buckets_cfg,
        device,
        embedding_backend,
        key,
        verbose,
        log_output,
        update_status_callback=None,
    ):
        from umrs.server.maintenance import update_objects

        thread_ident, future = self._submit_to_pool(
            update_objects,
            (
                self,
                client_id,
                buckets_cfg,
                device,
                embedding_backend,
                key,
                verbose,
                log_output,
                update_status_callback,
            ),
            key,
            update_status_callback=update_status_callback,
        )

        keys_per_bucket = {}
        for path_to_bucket, entry in buckets_cfg.items():
            if client_id is None or entry.s3_username == client_id:
                keys_per_bucket[path_to_bucket] = entry.num_objects

        self.update_thread_dict(
            buckets_cfg, keys_per_bucket, key, thread_ident=thread_ident
        )

        return thread_ident

    def launch_recompute_all_objects(
        self,
        client_id,
        buckets_cfg,
        device,
        embedding_backend,
        key,
        log_output,
        update_status_callback=None,
    ):
        from umrs.server.maintenance import recompute_all_objects

        thread_ident, future = self._submit_to_pool(
            recompute_all_objects,
            (
                self,
                client_id,
                buckets_cfg,
                device,
                embedding_backend,
                key,
                log_output,
                update_status_callback,
            ),
            key,
            update_status_callback=update_status_callback,
        )

        keys_per_bucket = {}
        for path_to_bucket, entry in buckets_cfg.items():
            if client_id is None or entry.s3_username == client_id:
                keys_per_bucket[path_to_bucket] = entry.num_objects

        self.update_thread_dict(
            buckets_cfg, keys_per_bucket, key, thread_ident=thread_ident
        )

        return thread_ident

    def update_thread_dict(
        self, buckets_cfg, keys_per_bucket, key, thread_ident="self"
    ):
        array_of_works = []
        for path_to_bucket in keys_per_bucket:
            selected_fields = ["s3_username", "table_name", "modality_name", "db_name"]
            result = buckets_cfg.get(path_to_bucket, fields=selected_fields)
            if result is None:
                logger.info(
                    f"WARNING: No bucket config for '{path_to_bucket}', skipping"
                )
                continue
            user, table_name, modality_name, db_name = result
            s3_keys = keys_per_bucket[path_to_bucket]
            work = {
                "user": user,
                "data_base": db_name,
                "table_name": table_name,
                "modality_name": modality_name,
                "bucket": path_to_bucket,
                "work_type": key,
                "nb_objects": s3_keys
                if isinstance(s3_keys, int)
                else (len(s3_keys) if s3_keys is not None else "N/A"),
            }
            array_of_works.append(work)
        with self.lock:
            thread_id = self.thread_id
            self.thread_id += 1
        if (
            thread_ident == "self"
        ):  # we are inside a thread thread, we need to get its ident
            current_thread = threading.current_thread()
            thread_ident = current_thread.ident
        with self.lock:
            self.thread_dict[thread_ident] = {
                "thread_id": thread_id,
                "work": array_of_works,
                "status": ThreadStatus.RUNNING,
                "updated_at": self._now_iso(),
            }

        # print(f"{thread_ident=}\n\t{thread_id=}\n\t{ThreadStatus.RUNNING}" )
        users = sorted({work["user"] for work in array_of_works if work.get("user")})
        if users:
            self.record_status_for_users_by_key(users, key)

    def is_worker_thread(self):
        current_thread = threading.current_thread()
        thread_ident = current_thread.ident
        return thread_ident in self.thread_dict

    def remove_from_running_list(self):
        # No-op: pool-managed futures are cleaned up by clean_finished_threads.
        pass

    def update_status_to_done(self):
        current_thread = threading.current_thread()
        new_key = None
        with self.lock:
            thread_ident = current_thread.ident
            if thread_ident in self.thread_dict:
                entry = self.thread_dict.pop(thread_ident)  # remove  get the full dict
                entry["status"] = ThreadStatus.DONE  # modify it
                entry["updated_at"] = self._now_iso()
                new_key = f"{thread_ident}-{entry['thread_id']}"  # done thread keys are prepended with their ID
                self.thread_dict[new_key] = entry  # reassign back to cache
            else:
                logger.info(
                    f"Error: cannot find '{thread_ident}' entry for {self.thread_dict=}"
                )

    def clean_finished_threads(self):
        from umrs.server.config import ENABLE_PRETTY_PRINT

        with self.lock:
            running = []
            finished_idents = []
            for item in self.thread_list:
                if item.done():
                    exc = item.exception()
                    if exc is not None:
                        logger.error("Pool task raised: %s", exc)
                    finished_idents.append(id(item))
                else:
                    running.append(item)
            pretty_print_thread_status(
                self.thread_dict, running_only=True, print_output=ENABLE_PRETTY_PRINT
            )
            self.thread_list = running
            return finished_idents

    def waitall(self):
        finished_idents = []
        with self.lock:
            items = list(self.thread_list)
            self.thread_list.clear()
        for item in items:
            try:
                item.result()  # blocks until done
            except Exception as e:
                logger.error("Pool task raised during waitall: %s", e)
            finished_idents.append(id(item))
        return finished_idents

    def filter_thread_status_for_user(self, client_id, key):
        """Build a filtered view of global_thread_status containing only works for a given client_id."""
        filtered = {}
        with self.lock:
            for thread_ident in self.thread_dict:
                info = self.thread_dict[thread_ident]
                works = info.get("work", [])
                user_works = [
                    w
                    for w in works
                    if w.get("user") == client_id and w.get("work_type") == key
                ]
                if user_works:
                    # Shallow copy with filtered works
                    filtered[thread_ident] = {
                        "thread_id": info.get("thread_id"),
                        "status": info.get("status"),
                        "updated_at": info.get("updated_at"),
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
                return data.model_name
            if isinstance(data, (str, int, float, bool)) or data is None:
                return data
            return str(data)

        if self.status_store is None:
            return

        owner = client_id or "system"
        entry = {
            "summary": payload.get("summary", ""),
            "generated_at": self._now_iso(),
        }
        if "last_updated_at" in payload:
            entry["last_updated_at"] = payload["last_updated_at"]
        if "data" in payload:
            entry["data"] = _sanitize_status_data(payload["data"])
        if "requires_update" in payload:
            entry["requires_update"] = payload["requires_update"]
        if "error" in payload:
            entry["error"] = payload["error"]
        if "s3cfg_content" in payload:
            entry["s3cfg_content"] = payload["s3cfg_content"]

        # print(f"[{owner}]: {key} -> {entry}")

        # get owner dict or empty dict
        owner_dict = self.status_store.get(owner, {})
        owner_dict[key] = entry
        self.status_store[owner] = owner_dict

    def _json_safe_thread_status(self, ts_dict):
        """Convert thread-keyed dict into a JSON-serializable list of thread info dicts, avoiding key collisions."""
        safe = []
        for _, info in ts_dict.items():
            safe.append(
                {
                    "thread_id": info.get("thread_id"),
                    "status": info.get("status"),
                    "updated_at": info.get("updated_at"),
                    "work": info.get("work", []),
                }
            )
        return safe

    def _derive_last_updated_at(self, data):
        timestamps = [
            entry.get("updated_at") for entry in data if entry.get("updated_at")
        ]
        return max(timestamps) if timestamps else None

    def record_status_for_users_by_key(self, users, key):
        """Record 'threads' status summaries for the provided user list into the global status store."""
        if self.status_store is None:
            return
        for user in users:
            filtered = self.filter_thread_status_for_user(user, key)
            # Include both Running and Done threads, preserving status column
            summary = pretty_print_thread_status(
                filtered, running_only=False, print_output=False
            )
            json_safe = self._json_safe_thread_status(filtered)
            payload = {"summary": summary, "data": json_safe}
            last_updated_at = self._derive_last_updated_at(json_safe)
            if last_updated_at:
                payload["last_updated_at"] = last_updated_at
            self.record_status(user, key, payload)

    def close(self):
        try:
            self.pool.shutdown(wait=True)
        except Exception as e:
            logger.info(f"Warning: pool shutdown error: {e}")
        try:
            # if using diskcache.Cache
            self.thread_dict.close()  # closes the cache
            self.status_store.close()  # closes the cache
            # optionally clear on disk
            self.thread_dict.clear()
            self.status_store.clear()
            logger.info("thread_dict and status store dictionnary deleted")
        except Exception as e:
            logger.info(
                f"Cannot close thread_dict/status_store dictionary properly: {e}"
            )
